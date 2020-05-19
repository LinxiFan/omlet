import logging
import functools
from typing import Union, Dict, List, Optional
import omlet.utils as U

from pytorch_lightning.callbacks import Callback, ProgressBar
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from .extended import ExtendedModule, STAGES


class SummaryMessage(Callback):
    def __init__(self,
                 train_message: str = None,
                 val_message: str = None,
                 test_message: str = None,
                 ):
        self.timers = {}
        for stage in STAGES:
            self.timers[stage] = U.Timer()
        self._summary_messages = {}
        for stage, msg in zip(
                STAGES, (train_message, val_message, test_message)
        ):
            self._summary_messages[stage] = msg

    def on_epoch_start(self, trainer, pl_module):
        assert isinstance(pl_module, ExtendedModule)
        # on_train_start and on_train_end enclose the entire training, not a single epoch
        self.timers['train'].start()

    def on_validation_start(self, trainer, pl_module):
        self.timers['val'].start()

    def on_test_start(self, trainer, pl_module):
        self.timers['test'].start()

    def _print_summary_message(self, stage, pl: ExtendedModule):
        if not pl._is_training_started or not self._summary_messages[stage]:
            return
        elapsed = self.timers[stage].elapsed_str()
        msg = f'{stage.capitalize()} summary [{pl.current_epoch:>2}] ' + self._summary_messages[stage] + f' ({stage} time {elapsed})'
        msg = msg.format(
            epoch=pl.current_epoch,
            **pl._metrics_history[-1][stage]
        )
        pl.log_info(msg)

    def on_epoch_end(self, trainer, pl_module):
        # WARNING: on_train_end() is when the entire trainer.fit() finishes
        # PTL train_epoch_end is always after val_epoch_end
        self._print_summary_message('train', pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._print_summary_message('val', pl_module)

    def on_test_end(self, trainer, pl_module):
        self._print_summary_message('test', pl_module)


class ExtendedProgressBar(ProgressBar):
    """
    Changes compared to the default one:
        - option to remove `v_num`

    """
    def __init__(self, remove_v_num=True,
                 exclude_metrics: List[str] = None,
                 formatter: Dict[str, str] = None,
                 use_short_name=True,
                 *args, **kwargs):
        """
        Args:
            use_short_name: True for "val/acc1" -> "v:acc1"

        Super class progress bar args:
            refresh_rate: int = 1
            process_position: int = 0
        """
        super().__init__(*args, **kwargs)
        self._remove_v_num = remove_v_num
        if exclude_metrics is None:
            exclude_metrics = []
        self._exclude_metrics = exclude_metrics
        self._use_short_name = use_short_name
        if formatter is None:
            formatter = {}
        self._formatter = formatter

    def _process_progress_dict(self, trainer):
        progress_dict = trainer.progress_bar_dict.copy()

        if self._remove_v_num:
            if 'v_num' in progress_dict:
                progress_dict.pop('v_num')
        for k, v in progress_dict.copy().items():
            if k in self._formatter:
                progress_dict[k] = self._formatter[k].format(v)
            # handle exclusions
            if k in self._exclude_metrics:
                progress_dict.pop(k)
            elif '/' in k and self._use_short_name:
                prefix, mname = k.split('/', 1)
                progress_dict[f'{prefix[0]}:{mname}'] = progress_dict[k]
                progress_dict.pop(k)
        # self.main_progress_bar.set_description(desc)
        return progress_dict

    def on_batch_end(self, trainer, pl_module):
        progress_dict = self._process_progress_dict(trainer)
        # super().on_batch_end(trainer, pl_module)
        # directly copied from pytorch_lightning/callbacks/progress
        if self.is_enabled and self.train_batch_idx % self.refresh_rate == 0:
            self.main_progress_bar.update(self.refresh_rate)
            self.main_progress_bar.set_postfix(**progress_dict)

    def on_validation_end(self, trainer, pl_module):
        progress_dict = self._process_progress_dict(trainer)
        # directly copied from pytorch_lightning/callbacks/progress
        self.main_progress_bar.set_postfix(**progress_dict)
        self.val_progress_bar.close()


class FileLogger(Callback):
    FLAG = '_LIGHTNING_HAS_FILE_LOGGER'

    def __init__(self, file_path, logger_name=None):
        """
        logger_name: if None, defaults to root logger
        """
        self._file_path = file_path
        self._logger_name = logger_name
        self.add_handler()

    def add_handler(self):
        logger = logging.getLogger(self._logger_name)
        # hack for multiprocessing, avoid adding duplicate FileHandler
        has_handler = False
        for h in logger.handlers:
            if hasattr(h, self.FLAG):
                has_handler = True
                break
        if not has_handler:
            handler = U.create_logging_file_handler(self._file_path)
            setattr(handler, self.FLAG, True)
            logger.addHandler(handler)

    def on_init_start(self, trainer):
        self.add_handler()

    def on_sanity_check_start(self, trainer, pl_module):
        if trainer.proc_rank == 0:
            self.add_handler()

    def on_train_start(self, trainer, pl_module):
        if trainer.proc_rank == 0:
            self.add_handler()

