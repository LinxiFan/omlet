from . import omlet_logger as _log
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from typing import Dict, Optional


class ExtendedTensorBoardLogger(TensorBoardLogger):
    """
    Tensorboard allows multiple global steps, as long as they are consistent
    """
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        metrics = metrics.copy()
        if 'system/epoch' in metrics:
            if 'epoch' in metrics:
                del metrics['epoch']
            if step == metrics['system/epoch']:
                # only keep `system/global_step` against step (== epoch)
                del metrics['system/epoch']
            elif step == metrics['system/global_step']:
                # only keep `system/epoch` against step (== global_step)
                del metrics['system/global_step']
            else:
                raise ValueError(f'INTERNAL: step {step} is neither '
                                 f'global_step {metrics["system/global_step"]} '
                                 f'nor epoch {metrics["system/epoch"]}')
        super().log_metrics(metrics, step)
        # _log.debug2(f'Extended TB: {metrics}  step={step}')


class ExtendedWandbLogger(WandbLogger):
    """
    wandb only allows a single `global_step`, so we ignore any other `step`
    """
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        metrics = metrics.copy()
        if 'system/global_step' in metrics:
            step = metrics.pop('system/global_step')
            metrics['epoch'] = metrics.pop('system/epoch')
        super().log_metrics(metrics, step)
        # _log.debug2(f'Extended Wandb: {metrics}  step={step}')
