import os
import numpy as np
import torch
import random
import time
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Union


class Swish(nn.Module):
    def __init__(self, learnable_beta=False):
        super().__init__()
        self.learnable_beta = learnable_beta
        if learnable_beta:
            self.beta = nn.Parameter(torch.ones(()), requires_grad=True)
        else:
            self.beta = None

    def forward(self, x):
        if self.learnable_beta:
            return x * torch.sigmoid(self.beta * x)
        else:
            return x * torch.sigmoid(x)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def set_seed_everywhere(seed: Optional[Union[int, str]], deterministic=False):
    """
    seed:
        None, "time", "sys": use scrambled system time
        int < 0, "noop", "skip": do nothing
        int >= 0: set seed
    """
    if seed is None or seed in ['time', 'sys']:
        # https://stackoverflow.com/questions/27276135/python-random-system-time-seed
        t = int(time.time() * 10000)
        seed = (((t & 0xff000000) >> 24) +
                ((t & 0x00ff0000) >> 8) +
                ((t & 0x0000ff00) << 8) +
                ((t & 0x000000ff) << 24))
    if seed < 0 or seed in ['noop', 'skip']:
        return -1
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    return seed


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


# ========== module operations =========
def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def freeze_params(model):
    set_requires_grad(model, False)


def unfreeze_params(model):
    set_requires_grad(model, True)


def clip_grad_value(model, max_value):
    with torch.no_grad():
        nn.utils.clip_grad_value_(model.parameters(), max_value)


def clip_grad_norm(model, max_norm, norm_type=2):
    """
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    with torch.no_grad():
        return nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type
        )


def count_parameters(model):
    return sum(x.numel() for x in model.parameters())


def get_model_device(model):
    """
    Returns:
        first model parameter's device
    """
    return next(model.parameters()).device


def clone_model(model):
    # with torch.no_grad():
    new_model = deepcopy(model).to(get_model_device(model))
    print('clone no grad')
        # new_model.load_state_dict(model.state_dict())
    return new_model


def update_soft_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)
