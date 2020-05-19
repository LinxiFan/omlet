import os
import socket
import torch
import torch.distributed as _dist
import torch.utils.data
import torch.backends.cudnn as cudnn
from typing import Union, Optional


def optimize_cudnn():
    cudnn.benchmark = True


def _get_group(group):
    if group is None:
        return _dist.group.WORLD
    else:
        return group


get_rank = _dist.get_rank
get_world_size = _dist.get_world_size


def is_master(group=None):
    return _dist.get_rank(_get_group(group)) == 0


def get_backend(group=None):
    return _dist.get_backend(group)


def current_device():
    return torch.cuda.current_device()


def _get_reduce_op(op):
    OPS = {
        'sum': _dist.ReduceOp.SUM,
        'mean': _dist.ReduceOp.SUM,  # no buildin for mean, we handle it ourselves
        'product': _dist.ReduceOp.PRODUCT,
        'max': _dist.ReduceOp.MAX,
        'min': _dist.ReduceOp.MIN,
    }
    op = op.lower()
    assert op in OPS, 'invalid op: '+op
    return OPS[op]


def reduce(tensor, op='sum', group=None,
           broadcast=True, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        tensor
        broadcast: True for all-reduce, False for master-only
        out:
            - if the same tensor as input, will mutate input
            - None: create a new tensor and leave the input unchanged
    """
    assert torch.is_tensor(tensor)
    if out is None:
        out = tensor.clone().detach()
    else:
        assert tensor.size() == out.size() and tensor.dtype == out.dtype
    group = _get_group(group)
    opcode = _get_reduce_op(op)
    if broadcast:
        _dist.all_reduce(out, op=opcode, group=group)
    else:
        _dist.reduce(out, dst=0, op=opcode, group=group)
    if op.lower() == 'mean' and (broadcast or is_master(group)):
        out.div_(1. * _dist.get_world_size(group))
    return out


def all_reduce_(tensor, op='sum', group=None):
    return reduce(
        tensor, op=op, group=group, broadcast=True, out=tensor
    )


def all_reduce(tensor, op='sum', group=None):
    return reduce(
        tensor, op=op, group=group, broadcast=True, out=None
    )


def master_reduce_(tensor, op='sum', group=None):
    return reduce(
        tensor, op=op, group=group, broadcast=False, out=tensor
    )


def master_reduce(tensor, op='sum', group=None):
    return reduce(
        tensor, op=op, group=group, broadcast=False, out=None
    )


def reduce_scalar(scalar, op='sum', *,
                  broadcast=True, storage=None, group=None, device=None):
    """
    Args:
        scalar: single python float
        broadcast: True to use all_reduce, else master_reduce
        storage: if you call this method repeatedly, consider allocating
            a persistent CUDA tensor in advance to provide temporary storage
            if None, allocates new CUDA memory every time
        device: if None, defaults to torch.cuda.current_device()
    Returns:
        python float
    """
    scalar = float(scalar)
    group = _get_group(group)
    is_nccl = get_backend(group) == 'nccl'
    if storage is None:
        s = torch.tensor([scalar], dtype=torch.float32)
        if is_nccl:
            if device is None:
                device = torch.cuda.current_device()
            s = s.cuda(device)
    else:
        assert torch.is_tensor(storage)
        assert storage.numel() > 0, 'storage must have at least 1 element'
        if is_nccl:
            assert storage.is_cuda
        storage[0] = scalar
        s = storage
    reduce(s, op=op, group=group, broadcast=broadcast, out=s)
    return float(s[0])


def reduce_scalars(scalars, op='sum', *,
                   broadcast=True, storage=None, group=None, device=None):
    """
    Args:
        scalars: list, tuple or dict of python floats
        broadcast: True to use all_reduce, else master_reduce
        storage: if you call this method repeatedly, consider allocating
            a persistent CUDA tensor in advance to provide temporary storage
            if None, allocates new CUDA memory every time
        device: if None, defaults to torch.cuda.current_device()
    Returns:
        list, tuple or dict of reduced floats
    """
    if not isinstance(scalars, (tuple, list, dict)):
        # singleton falls back to reduce_scalar()
        return reduce_scalar(
            scalars, op=op, broadcast=broadcast, storage=storage,
            group=group, device=device
        )

    group = _get_group(group)
    is_nccl = get_backend(group) == 'nccl'
    idx_map = None
    if isinstance(scalars, dict):
        values = []
        idx_map = {}
        for i, (k, v) in enumerate(scalars.items()):
            idx_map[k] = i
            values.append(float(v))
    else:
        values = [float(v) for v in scalars]
    numel = len(values)

    values = torch.tensor(values, dtype=torch.float32)
    if storage is None:
        if is_nccl:
            if device is None:
                device = torch.cuda.current_device()
            values = values.cuda(device)
    else:
        assert torch.is_tensor(storage)
        assert storage.numel() >= numel, 'storage not enough'
        if is_nccl:
            assert storage.is_cuda
        storage[:numel] = values
        values = storage
    reduce(values, op=op, group=group, broadcast=broadcast, out=values)

    if isinstance(scalars, dict):
        return {k: float(values[idx_map[k]]) for k in idx_map}
    else:
        return type(scalars)(float(v) for v in values[:numel])


def random_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port
