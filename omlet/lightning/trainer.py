"""
Easily configure pl.Trainer with Hydra
"""
import os
import time
import inspect

import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
# from pytorch_lightning.loggers import *
from .loggers import *
from omegaconf import DictConfig
import omlet.utils as U
from typing import Optional, Union, Dict, Any, List, Callable
from . import omlet_logger as _log, override_loggers
from .callbacks import FileLogger
from .checkpoint import ExtendedCheckpoint


_DEFAULT_OS_ENVS = {
    # https://discuss.pytorch.org/t/issue-with-multiprocessing-semaphore-tracking/22943/4
    'PYTHONWARNINGS' : 'ignore:semaphore_tracker:UserWarning',
    'OMP_NUM_THREADS': 1,
}


def set_trainer_os_envs(envs: Optional[Dict[str, Any]] = None):
    # update environment variables
    U.set_os_envs(_DEFAULT_OS_ENVS)
    U.set_os_envs(envs)


def _check_run_name(run_name):
    for special_char in '\\/$#&|"\'~!^*:<>':
        assert special_char not in run_name, \
            f'Run name "{run_name}" cannot have special character {special_char}'


def configure_trainer(
        *,
        root_dir,  # experiment root folder
        run_name,  # should be unique across different runs
        epochs: int,
        gpus: Union[int, List[int]],
        fp16: bool = False,
        # checkpointing
        resume: Union[str, int, bool] = False,
        monitor_metric: str,
        monitor_metric_mode: str,  # min or max
        save_top_k: int = 3,
        save_epoch_interval: int = 5,
        always_save_last: bool = True,
        best_filename_template: Optional[str] = None,
        # distributed environment variables
        master_addr='localhost',
        master_port='auto',
        node_rank=0,
        distributed_backend='ddp',  # the only thing we support now
        # callbacks
        log_file: str = 'log.txt',
        enable_wandb: bool = False,
        wandb: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callback]] = None,
        **extra_trainer_kwargs
):
    """
    always_save_last:
        True to always save `last.ckpt` regardless of regular epoch intervals
        `last.ckpt` is for resuming and will be replaced every epoch
    resume:
        - 'last': resume from last.ckpt checkpoint in the folder
        - 'best/epoch=1': any relative path within the run folder
        - 4 (int): defaults to 'epoch={N}.ckpt'
        - '~/my/checkpoint/file.ckpt': full path
    wandb:
        project name defaults to the name of last subfolder in `root_dir`
    """
    _check_run_name(run_name)
    _log.info(f'Run name: {run_name}')

    if distributed_backend == 'ddp':
        # update env variables for distributed use case
        if master_port == 'auto':
            master_port = U.random_free_tcp_port()
            _log.info(f'MASTER_PORT env not specified. Randomly picks a free port: {master_port}')
        else:
            master_port = int(master_port)
        _envs = {
            'MASTER_ADDR': master_addr,
            'NODE_RANK': int(node_rank),
            'MASTER_PORT': master_port
        }
        U.set_os_envs(_envs)
    else:
        raise NotImplementedError('distributed_backend=ddp is the only mode supported for now')

    root_dir = os.path.expanduser(root_dir)
    assert os.path.isdir(root_dir)
    exp_dir = U.f_join(root_dir, run_name)
    is_exp_dir_exists = os.path.exists(exp_dir)
    assert '/' in monitor_metric

    if not monitor_metric_mode:
        monitor_metric_mode = 'auto'
    assert monitor_metric_mode in ['min', 'max', 'auto']

    if not best_filename_template:
        best_filename_template = 'best/{epoch}-{' + monitor_metric + ':.2f}'

    ckpt_dir = U.f_join(exp_dir, 'ckpt')
    checkpoint_callback = ExtendedCheckpoint(
        ckpt_dir,
        filename_template='{epoch}',
        best_filename_template=best_filename_template,
        monitor_metric=monitor_metric,
        monitor_metric_mode=monitor_metric_mode,
        save_top_k=save_top_k,
        save_epoch_interval=save_epoch_interval,
        always_save_last=always_save_last
    )
    _log.info(f'checkpoint dir: {ckpt_dir}')

    loggers = []
    tb_logger = ExtendedTensorBoardLogger(
        U.f_join(exp_dir, 'tb'),
        name='',
        version=''
    )
    loggers.append(tb_logger)

    if enable_wandb:
        _kwargs = {
            'save_dir': exp_dir,
            'name': run_name,
            'log_model': True,
            'project': os.path.basename(os.path.normpath(root_dir))
        }
        if wandb:
            _kwargs.update(wandb)
        wandb_logger = ExtendedWandbLogger(**_kwargs)
        loggers.append(wandb_logger)

    if callbacks is None:
        callbacks = []
    if log_file:
        callbacks.append(FileLogger(U.f_join(exp_dir, log_file)))

    if resume:
        assert isinstance(resume, (int, str, bool))
        if resume is True:
            resume = U.f_join(ckpt_dir, 'last.ckpt')
        elif isinstance(resume, int):
            resume = U.f_join(ckpt_dir, f'epoch={resume}.ckpt')
        elif os.path.isabs(resume):
            resume = os.path.expanduser(resume)
            if is_exp_dir_exists:
                _log.warn(f'The destination experiment dir already exists: {exp_dir}, make sure you do not unintentionally overwrite old checkpoints and logs')
                # give the user a bit time to terminate
                time.sleep(2)
        else:
            resume = U.f_join(ckpt_dir, resume)
        if not resume.endswith('.ckpt'):
            resume += '.ckpt'
        # check resume ckpt path must exist
        if not os.path.exists(resume):
            raise FileNotFoundError(f'Resume file {resume} does not exist')
        _log.info(f'Resuming run from checkpoint: {resume}')
    else:
        if is_exp_dir_exists:
            _log.warn(f'The destination experiment dir already exists: {exp_dir}, but `resume` option is not set. Make sure you do not unintentionally overwrite old checkpoints and logs')
            # give the user a bit time to terminate
            time.sleep(2)
        _log.info(f'Starting a new run from scratch: {run_name}')
        resume = None

    return pl.Trainer(
        gpus=gpus,
        max_epochs=epochs,
        distributed_backend=distributed_backend,
        precision=16 if fp16 else 32,
        checkpoint_callback=checkpoint_callback,
        callbacks=callbacks,
        logger=loggers,
        resume_from_checkpoint=resume,
        **extra_trainer_kwargs
    )


def hydra_trainer(pl_module_cls,
                  cfg: DictConfig,
                  run_name_generator: Optional[Callable[[DictConfig], str]] = None):
    """
    Args:
        run_name_generator: a function that takes hydra cfg as input and returns run name
            NOTE: any preceding or trailing '_' and space will be trimmed

    Config keys:
        - any parameters that fit configure_trainer() signature
        - log_level ("info"): sets global logging level
        - seed (None): None to use system time
        - eval (False): True for trainer.fit(); False for trainer.test()
        - os_envs (dict): sets extra os environment variables
        - callbacks (dict): instantiable callback configs
        - trainer (dict): extra pl.Trainer() kwargs

    Added config key:
        - override_name: from Hydra `override_dirname`. If you override configs on
            the command line, e.g. python main.py lr=0.2 dropout=0.7,
            override_name will be set to "lr=0.2_dropout=0.7"
            useful to configure your `run_name`
    """
    U.initialize_omlet_config(cfg)
    U.set_os_envs(_DEFAULT_OS_ENVS)
    U.set_os_envs(cfg.get('os_envs'))

    override_loggers(cfg.get('log_level', 'info'))
    seed = U.set_seed_everywhere(
        cfg.get('seed'), deterministic=cfg.get('deterministic', False)
    )
    if seed >= 0:
        _log.info(f'Random seed: {seed}')

    if run_name_generator is not None:
        cfg.run_name = run_name_generator(cfg)
    cfg.run_name = cfg.run_name.strip(' _')
    _check_run_name(cfg.run_name)

    model = pl_module_cls(cfg)

    callbacks = []
    for callback_cfg in cfg.get('callbacks', {}).values():
        callbacks.append(U.hydra_instantiate(callback_cfg))

    # inspect configure_trainer function and get its kwargs
    kwargs = {}
    for param in inspect.signature(configure_trainer).parameters.values():
        if param.name in cfg and param.kind != inspect.Parameter.VAR_KEYWORD:
            kwargs[param.name] = cfg[param.name]
    # override a few special ones
    # wandb dict can have `enable` key, which will be passed to `enable_kwargs`
    if 'enable_wandb' not in kwargs:
        kwargs['enable_wandb'] = kwargs.get('wandb', {}).pop('enable', False)

    kwargs['callbacks'] = callbacks

    # add extra args to be passed to pl.Trainer
    kwargs.update(cfg.get('trainer', {}))

    trainer = configure_trainer(**kwargs)

    if cfg.get('eval', False):
        trainer.test(model)
    else:
        trainer.fit(model)


