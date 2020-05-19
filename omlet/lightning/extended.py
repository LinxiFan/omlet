"""
Quick hack: enhanced pytorch-lightning mixin
"""

import argparse
import functools
from typing import Union, Dict, List, Optional, Any
from pprint import pprint
import omlet.utils as U
import omlet.utils.distributed as dist
import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
import logging
from . import omlet_logger as _log


STAGES = ('train', 'val', 'test')


class ExtendedModule(LightningModule):
    """
    hparams should have the following keys:
        - train_metrics
        - val_metrics
        - test_metrics (defaults to val_metrics if unspecified)
        - best_metrics
        - batch_size or global_batch_size
        - eval_batch_size or global_eval_batch_size (defaults to `batch_size` if unspecified)
        - num_workers or global_num_workers

    Useful attributes:
        - hparams
        - use_ddp, use_dp, use_ddp2
        - global_step
        - current_epoch
        - hparams: a plain dictionary
            LightningModule can only handle dumping dict
        - conf: OmegaConf of the hyperparameters
        - C: alias for conf

    Patched methods:
        - {training|validation|test}_step()
        - {training|validation|test}_epoch_end()

    Overrideable method:
        - get_batch_size(batch): current returns batch[0].size(0)
            override if you have a more complicated batch structure

    best_metrics:
        {'val/acc1': 'max', 'test/acc5': 'max', 'val/loss': 'min'}
    """
    def __new__(cls, *args, **kwargs):
        """
        Class-level patch to avoid multiprocessing pickle error:
            _pickle.PicklingError: Can't pickle <function .. at ..>: it's not
            the same object as __main__.<obj>
        """
        obj = super().__new__(cls)
        # patch up subclass methods:
        #   - [train|validation|test]_step
        #   - [train|validation|test]_epoch_end
        for stage in STAGES:
            cls._patch_pl_step(stage)
            cls._patch_pl_epoch_end(stage)
        return obj

    def __init__(self, hparams: Union[dict, argparse.Namespace, DictConfig]):
        super().__init__()

        if not isinstance(hparams, DictConfig):
            if isinstance(hparams, argparse.Namespace):
                hparams = vars(hparams)
            else:
                raise NotImplementedError(f'Unrecognized hparams format: {type(self.pl.hparams)}')
            hparams = OmegaConf.create(hparams)
        self.conf = hparams
        self.hparams = OmegaConf.to_container(self.conf, resolve=True)

        train_metrics: List[str] = self._check_hparams('train_metrics')
        val_metrics: List[str] = self._check_hparams('val_metrics')
        test_metrics: List[str] = self._check_hparams('test_metrics', default=val_metrics)
        best_metrics: Dict[str, str] = self._check_hparams('best_metrics', default={})
        self.debug = self._check_hparams('debug', default=False)

        self._metrics_meter = {}
        self._train_step_meter = {}  # show fine-grained step logs for training
        self._metrics_history = []  # list of {'train/loss': .., 'val/acc1': ...}
        self._best_metrics_values = {}  # {'val/acc1': {'value': 76.1, 'epoch': 87, 'step': 900}}
        self._best_metrics_spec = best_metrics

        for stage, names in zip(STAGES, (train_metrics, val_metrics, test_metrics)):
            self._metrics_meter[stage] = {m: U.AverageMeter() for m in names}
            if stage == 'train':
                self._train_step_meter = {m: U.AverageMeter() for m in names}

        for name, value in best_metrics.items():
            assert '/' in name
            assert value in ['min', 'max']

        self._is_training_started = False
        # to be propagated to children processes
        self._global_logging_level = U.get_logging_level()

    def _check_hparams(self, key, default: Any = '__required__'):
        if key not in self.conf and default == '__required__':
            raise KeyError(f'required key {key} not found in hparams')
        return self.conf.get(key, default)

    @property
    def C(self):
        return self.conf

    @property
    def num_training_batches(self):
        return self.trainer.num_training_batches
    @property
    def num_val_batches(self):
        return self.trainer.num_val_batches
    @property
    def num_test_batches(self):
        return self.trainer.num_test_batches
    @property
    def rank(self):
        return self.trainer.proc_rank
    @property
    def world_size(self):
        return self.trainer.world_size
    @property
    def device(self):
        return self.trainer.root_gpu
    @property
    def num_gpus(self):
        return self.trainer.num_gpus
    @property
    def batch_idx(self):
        return self.trainer.batch_idx

    # ==================== subclass overrideable ====================
    def get_batch_size(self, batch):
        """
        This function can be overridden in subclass pl_module
        if you have a complicated batch data structure
        """
        images, *_ = batch
        return int(images.size(0))

    # ==================== Data loaders ====================
    def _divide_by_gpu(self, name, default='__required__'):
        """
        check for the following configs:
        - batch_size or global_batch_size
        - eval_batch_size or global_eval_batch_size
        - num_workers or global_num_workers

        local version = global version / num_gpus
        """
        C = self.conf
        if name in C:
            local_value = C[name]
            assert local_value > 0, f'{name} must > 0'
        elif f'global_{name}' in C:
            global_value = C[f'global_{name}']
            assert global_value % self.num_gpus == 0,\
                f'global_{name} {global_value} must divide number of GPUs {self.num_gpus}'
            local_value = global_value // self.num_gpus
        else:
            if default == '__required__':
                raise KeyError(f'You must specify at least one of {name} or global_{name}')
            else:
                local_value = default
        return local_value

    def get_dataloader(self, dataset, stage):
        assert stage in STAGES
        num_workers = self._divide_by_gpu('num_workers', default=8)
        if stage == 'train':
            return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self._divide_by_gpu('batch_size'),
                shuffle=not self.use_ddp,
                num_workers=num_workers
            )
        else:
            eval_batch_size = self._divide_by_gpu('eval_batch_size', default=-1)
            if eval_batch_size == -1:
                # defaults to training batch_size
                eval_batch_size = self._divide_by_gpu('batch_size')
            return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers
            )

    # ================ Patch [train|validation|test]_step() ===================
    @classmethod
    def _patch_pl_step(cls, stage):
        pl_method_name = {
            'train': 'training_step',
            'val': 'validation_step',
            'test': 'test_step'
        }[stage]
        pl_step_method = getattr(cls, pl_method_name)

        @functools.wraps(pl_step_method)
        def _wrapped(self, batch, batch_idx, *args, **kwargs):
            output = pl_step_method(self, batch, batch_idx, *args, **kwargs)
            return self._after_step(stage, batch, output)
        # monkey patch
        setattr(cls, pl_method_name, _wrapped)

    def _after_step(self, stage, batch, output: Dict):
        assert not self.use_dp or self.use_ddp2, \
            'self does not yet support DP or DDP2, only single GPU or DDP'
        batch_size = self.get_batch_size(batch)
        self._update_epoch_metrics(stage, output, batch_size)
        if 'progress_bar' not in output:
            output['progress_bar'] = {}
        output['progress_bar'].update(self._get_avg_epoch_metrics(stage))
        output['progress_bar'].pop('loss', None)  # hack: PTL already adds loss to progress bar
        if stage == 'train':
            # record log at every training batch step
            # for test and val, we don't record log at every batch step
            # only one summary statistic at step end
            # training stats are averaged over trainer.row_log_interval
            # (how often we write to TB logs)
            if 'log' not in output:
                output['log'] = {}

            self._update_train_step_metrics(output, batch_size)
            output['log'].update(self._get_avg_train_step_metrics(
                name_template='train/stepwise_{name}'
            ))
            self._add_extended_log(output['log'])
            # we don't do (self.batch_idx+1) because we need to sync with PL
            if self.batch_idx % self.trainer.row_log_interval == 0:
                self._reset_train_step_metrics()
        return output

    # ============ Patch [train|validation|test]_epoch_end() ==================
    @classmethod
    def _patch_pl_epoch_end(cls, stage):
        "we don't need outputs because values are stored in self._metrics_sum"
        pl_method_name = {
            'train': 'training_epoch_end',
            'val': 'validation_epoch_end',
            'test': 'test_epoch_end'
        }[stage]
        pl_method = getattr(cls, pl_method_name)

        @functools.wraps(pl_method)
        def _wrapped(self, outputs):
            pl_method(self, outputs)
            return self._epoch_end(stage)

        setattr(cls, pl_method_name, _wrapped)

    def _epoch_end(self, stage):
        """
        Collect stats from all processes at the end of an epoch
        """
        _info = self._get_avg_epoch_metrics(stage, name_template='{name}')
        # cumulative batch size on all GPUs should be exactly the same
        # so we can do a simple mean
        info_short_name = self.reduce(_info, op='mean')
        # add long name (train/acc1)
        info_long_name = {f'{stage}/{name}': v for name, v in info_short_name.items()}
        pbar = info_long_name.copy()
        log = info_long_name.copy()

        if self._is_training_started:  # avoid record in val sanity check
            if len(self._metrics_history) == 0 or self._metrics_history[-1]['epoch'] != self.current_epoch:
                new_epoch_info = {
                    'epoch': self.current_epoch,
                    'train': {}, 'val': {}, 'test': {}
                }
                new_epoch_info[stage].update(info_short_name)
                self._metrics_history.append(new_epoch_info)
            else:
                self._metrics_history[-1][stage].update(info_short_name)

            for m, new_value in info_long_name.items():
                # m has format `val/acc1`
                if m in self._best_metrics_spec:
                    self._update_best_metrics(m, new_value)
                    # change name to `val/best_acc1` and add to log for TensorBoard
                    _stage, _name = m.split('/')
                    assert stage == _stage, f'metric {m} does not conform to stage {stage}'
                    best_name = f'{_stage}/best_{_name}'
                    # pbar[best_name] = self._get_best_str(m)  # pbar has too much info
                    log[best_name] = self._get_best_value(m)

        # self.log_debug(stage, 'history', self._metrics_history)
        # self.log_debug(stage, 'best', self._best_metrics_values)

        log['step'] = self.current_epoch  # set TB x-axis to epoch
        self._add_extended_log(log)
        return {
            'progress_bar': pbar,
            'log': log
        }

    # ==================== distributed ====================
    def is_master(self):
        if self.use_ddp:
            assert dist.get_rank() == self.rank
        return self.rank == 0

    def is_worker(self):
        if self.use_ddp:
            assert dist.get_rank() == self.rank
        return self.rank != 0

    def reduce(self, values: Union[float, Dict[str, float]], op: str) \
            -> Union[float, Dict[str, float]]:
        if self.use_ddp:
            return dist.reduce_scalars(
                values, broadcast=True, device=self.device, op=op
            )
        else:
            return values

    # ==================== Metrics book-keeping ====================
    def _reset_epoch_metrics(self, stages='all'):
        for stage in STAGES if stages == 'all' else [stages]:
            for meter in self._metrics_meter[stage].values():
                meter.reset()

    def _reset_train_step_metrics(self, stages='all'):
        for meter in self._train_step_meter.values():
            meter.reset()

    def _get_avg_epoch_metrics(self, stage, name_template='{name}'):
        return {
            name_template.format(stage=stage, name=name): meter.value
            for name, meter in self._metrics_meter[stage].items()
        }

    def _get_avg_train_step_metrics(self, name_template='{name}'):
        return {
            name_template.format(name=name): meter.value
            for name, meter in self._train_step_meter.items()
        }
    
    def _update_epoch_metrics(self, stage, output, batch_size):
        for key, value in output.items():
            if key in self._metrics_meter[stage]:
                self._metrics_meter[stage][key].update(value, batch_size)

    def _update_train_step_metrics(self, output, batch_size):
        for key, value in output.items():
            if key in self._train_step_meter:
                self._train_step_meter[key].update(value, batch_size)

    def _update_best_metrics(self, metric_name, new_value):
        assert metric_name in self._best_metrics_spec
        is_max = self._best_metrics_spec[metric_name] == 'max'
        if metric_name in self._best_metrics_values:
            info = self._best_metrics_values[metric_name]
            if (is_max and float(new_value) >= info['value']
                    or not is_max and float(new_value) <= info['value']):
                info['value'] = new_value
                info['epoch'] = self.current_epoch
        else:
            self._best_metrics_values[metric_name] = {
                'value': new_value, 'epoch': self.current_epoch
            }

    def _get_best_value(self, name):
        return self._best_metrics_values[name]['value']

    def _get_best_str(self, name):
        assert name in self._best_metrics_values
        info = self._best_metrics_values[name]
        return f'{info["value"]:.2f} @ep{info["epoch"]}'

    # ==================== Recording ====================
    def _add_extended_log(self, log_dict):
        # please use together with ExtendedTB and ExtendedWandb
        log_dict['system/global_step'] = self.global_step
        log_dict['system/epoch'] = self.current_epoch

    # ==================== Override hooks ====================
    def on_epoch_start(self):
        self._reset_epoch_metrics()
        self._reset_train_step_metrics()
        self._is_training_started = True  # avoid sanity check

    def on_save_checkpoint(self, checkpoint):
        # patch pl
        extended = {
            'metrics_history': self._metrics_history,
            'best_metrics': self._best_metrics_values
        }
        checkpoint['extended'] = extended
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        # patch pl
        extended = checkpoint['extended']
        self._metrics_history = extended['metrics_history']
        self._best_metrics_values = extended['best_metrics']

    def init_ddp_connection(
            self,
            proc_rank: int,
            world_size: int,
            is_slurm_managing_tasks: bool = True
    ):
        """
        Override this method to add features to DDP children processes
        Callbacks can optionally define a method:
            on_ddp_connection(proc_rank, world_size)
        """
        super().init_ddp_connection(
            proc_rank=proc_rank, world_size=world_size,
            is_slurm_managing_tasks=is_slurm_managing_tasks
        )
        # set global level for children processes
        U.set_logging_level(self._global_logging_level)
        for callback in self.trainer.callbacks:
            if hasattr(callback, 'on_ddp_connection'):
                callback.on_ddp_connection(proc_rank, world_size)

    # ================= Default test same as val ==================
    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_dataloader(self):
        return self.val_dataloader()

    def test_epoch_end(self, *args, **kwargs):
        return self.validation_epoch_end(*args, **kwargs)

    # ==================== debugging ====================
    @rank_zero_only  # equivalent to `if self.is_master()`
    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def _log_write(self, level, *args, **kwargs):
        if _log.isEnabledFor(level):
            if 'end' not in kwargs:
                kwargs['end'] = ''
            s = U.print_str(*args, **kwargs)
            _log.log(level, s)

    @rank_zero_only
    def log_debug(self, *args, **kwargs):
        # DEBUG2 = logging.DEBUG + 2, we log higher to avoid debug messages from other apps
        self._log_write(logging.DEBUG2, *args, **kwargs)

    @rank_zero_only
    def log_info(self, *args, **kwargs):
        self._log_write(logging.INFO, *args, **kwargs)

    @rank_zero_only
    def log_infov(self, *args, **kwargs):
        # INFOV = logging.INFO - 2, more verbose
        self._log_write(logging.INFOV, *args, **kwargs)

    @rank_zero_only
    def log_warn(self, *args, **kwargs):
        self._log_write(logging.WARNING, *args, **kwargs)

    @rank_zero_only
    def log_error(self, *args, **kwargs):
        self._log_write(logging.ERROR, *args, **kwargs)

    @rank_zero_only
    def log_critical(self, *args, **kwargs):
        self._log_write(logging.CRITICAL, *args, **kwargs)

