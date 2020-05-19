import os
import re
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import omlet.utils as U
from typing import Optional
from . import omlet_logger as _log


class ExtendedCheckpoint(ModelCheckpoint):
    def __init__(self,
                 save_dir,
                 filename_template: str = '{epoch}',
                 best_filename_template: str = 'best/{epoch}',
                 monitor_metric: str = 'val/loss',
                 monitor_metric_mode: str = 'auto',
                 save_top_k: int = 1,
                 save_epoch_interval: int = 1,
                 always_save_last: bool = True,
                 ):
        """
        Periodic checkpoint save every `period`, in addition to standard `best` save
        `epoch` is always actual epoch index + 1 (number of epochs so far)

        Verbose is controlled by global logging level.
            set logging level to INFOV (i.e. INFO - 2)
        """
        self.save_dir = os.path.expanduser(save_dir)
        if not filename_template:
            filename_template = '{epoch}'
        self.ckpt_path = os.path.join(self.save_dir, filename_template)
        if not best_filename_template:
            best_filename_template = 'best-{epoch}'
        self.best_ckpt_path = os.path.join(self.save_dir, best_filename_template)
        os.makedirs(self.save_dir, exist_ok=True)
        self.always_save_last = always_save_last

        super().__init__(
            filepath=self.ckpt_path, monitor=monitor_metric, verbose=False,
            save_top_k=save_top_k, save_weights_only=False,
            mode=monitor_metric_mode, period=save_epoch_interval, prefix=''
        )

    def format_checkpoint_name(self, epoch, metrics, ver=None, ckpt_type='best'):
        """Override

        Example::

            >>> tmpdir = os.path.dirname(__file__)
            >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{epoch}'))
            >>> os.path.basename(ckpt.format_checkpoint_name(0, {}))
            'epoch=0.ckpt'
            >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{epoch:03d}'))
            >>> os.path.basename(ckpt.format_checkpoint_name(5, {}))
            'epoch=005.ckpt'
            >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{epoch}-{val_loss:.2f}'))
            >>> os.path.basename(ckpt.format_checkpoint_name(2, dict(val_loss=0.123456)))
            'epoch=2-val_loss=0.12.ckpt'
            >>> ckpt = ModelCheckpoint(os.path.join(tmpdir, '{missing:d}'))
            >>> os.path.basename(ckpt.format_checkpoint_name(0, {}))
            'missing=0.ckpt'
        """
        if ckpt_type == 'best':
            fpath = self.best_ckpt_path
        else:
            fpath = self.ckpt_path
        # check if user passed in keys to the string
        groups = re.findall(r'(\{.*?)[:\}]', fpath)
        assert len(groups) > 0, \
            f'INTERNAL ERROR: file name template {fpath} should have at least one {{...}}'

        metrics = {k.replace('/', '_'): v for k, v in metrics.items()}
        metrics['epoch'] = epoch + 1

        for tmp in groups:
            name = tmp[1:]
            name = name.replace('/', '_')
            fpath = fpath.replace(tmp, name + '={' + name)
            if name not in metrics:
                metrics[name] = 0
        fpath = fpath.format(**metrics)
        if fpath.endswith('.ckpt'):
            fpath = fpath[:-len('.ckpt')]
        str_ver = f'_v{ver}' if ver is not None else ''
        return fpath + str_ver + '.ckpt'

    def check_monitor_top_k(self, current):
        """
        Override warning from super class
        """
        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        if not isinstance(current, torch.Tensor):
            current = torch.tensor(current)

        monitor_op = {
            "min": torch.lt,
            "max": torch.gt,
        }[self.mode]

        return monitor_op(current, self.best_k_models[self.kth_best_model])

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        """
        Mostly copied from pytorch_lightning.callbacks.ModelCheckpoint
        """
        # only run on main process
        if trainer.proc_rank != 0:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        _best_path = self._save_best(metrics, epoch)
        _periodic_path = self._save_periodic(metrics, epoch, _best_path)
        self._save_last(_periodic_path, _best_path)

    def _save_best(self, metrics, epoch):
        """
        Returns:
            filepath if the best is saved, None if nothing saved
        """
        if self.save_top_k <= 0:
            # no best saved
            return None

        filepath = self.format_checkpoint_name(epoch, metrics, ckpt_type='best')
        version_cnt = 1
        while os.path.isfile(filepath):
            filepath = self.format_checkpoint_name(epoch, metrics, ver=version_cnt, ckpt_type='best')
            # this epoch called before
            version_cnt += 1

        current = metrics.get(self.monitor)

        if current is None:
            rank_zero_warn(
                f'Can save best model only with {self.monitor} available, skipping.', RuntimeWarning
            )
        elif self.check_monitor_top_k(current):
            self._do_check_save(filepath, current, epoch)
            if version_cnt > 1:
                _log.warn(f'best ckpt filepath exists, saving to a different version: {filepath}')
            _log.infov(
                f'\nEpoch {epoch:05d}: {self.monitor} reached'
                f' {current:0.5f} (best {self.best:0.5f}), saving model to'
                f' {filepath} as top {self.save_top_k}')
            return filepath

        _log.infov(f'\nEpoch {epoch:03d}: {self.monitor} was not in top {self.save_top_k}')
        return None

    def _save_periodic(self, metrics, epoch, _best_save_path):
        """
        Don't duplicate save if the same ckpt has already been saved as a best model
        """
        if self.period <= 0 or not ((epoch + 1) % self.period) == 0:
            # skip if not this period
            return None

        filepath = self.format_checkpoint_name(epoch, metrics, ckpt_type='periodic')
        version_cnt = 1
        while os.path.isfile(filepath):
            filepath = self.format_checkpoint_name(epoch, metrics, ver=version_cnt, ckpt_type='periodic')
            # this epoch called before
            version_cnt += 1
        if version_cnt > 1:
            _log.warn(f'ckpt filepath exists, saving to a different version: {filepath}')

        _log.infov(f'\nEpoch {epoch:03d}: periodic saving model to {filepath}')

        if _best_save_path:
            # the destination shouldn't exist tho
            _log.debug2(f'\nEpoch {epoch:03d}: periodic saving copies from the best ckpt {_best_save_path}')
            U.f_copy(_best_save_path, filepath, exists_ok=True)
        else:
            self._save_model(filepath)
        return filepath

    def _save_last(self, _periodic_save_path, _best_save_path):
        if not self.always_save_last:
            return

        # always save a copy to the special `last.ckpt`
        last_path = os.path.join(self.save_dir, 'last.ckpt')
        if _periodic_save_path:
            U.f_copy(_periodic_save_path, last_path, exists_ok=True)
        elif _best_save_path:
            U.f_copy(_best_save_path, last_path, exists_ok=True)
        else:
            # neither best or periodic saved
            self._save_model(last_path)
