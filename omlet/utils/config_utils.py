import os
import re
import hydra
import pkg_resources
import fnmatch
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import OmegaConfBaseException
from typing import Optional, List


def _omlet_config_schema():
    return OmegaConf.create({
        'shorthand': {},
        'override_name': {
            'kv_sep': '=',
            'item_sep': ',',
            'use_shorthand': False,  # use shorthand names for override
            'include_keys': None,
            'exclude_keys': [],
        },
        'job': {},
        '_internal': {
            'is_initialized': False,
        },
    })


def initialize_omlet_config(cfg: DictConfig):
    """
    See _omlet_config_schema()
    """
    assert isinstance(cfg, DictConfig)
    OmegaConf.set_struct(cfg, False)
    om = _omlet_config_schema()
    om.update(cfg.get('omlet', {}))
    cfg.omlet = om  # this is a copy operation
    om = cfg.omlet

    om._internal.is_initialized = True
    assert is_omlet_initialized(cfg), 'INTERNAL'

    # process shorthands
    shortset = set()
    for original, short in om.shorthand.items():
        if short in shortset:
            raise KeyError(f'Shorthand {short} is duplicated')
        if short in cfg:
            # user has already passed the short value
            cfg[original] = cfg[short]  # create a redirection to short name
            del cfg[short]
        shortset.add(short)

    om.job.override_name = omlet_override_name(cfg)
    return cfg


def print_config(cfg: DictConfig):
    print(cfg.pretty(resolve=True))


def is_hydra_initialized():
    return hydra.utils.HydraConfig().cfg is not None


def is_omlet_initialized(cfg):
    try:
        return bool(cfg.omlet._internal.is_initialized)
    except OmegaConfBaseException:
        return False


def hydra_config():
    # https://github.com/facebookresearch/hydra/issues/377
    # HydraConfig() is a singleton
    if is_hydra_initialized():
        return hydra.utils.HydraConfig().cfg.hydra
    else:
        return None


def hydra_override_arg_list() -> List[str]:
    """
    Returns:
        list ["lr=0.2", "batch=64", ...]
    """
    if is_hydra_initialized():
        return hydra_config().overrides.task
    else:
        return []


def _match_patterns(patterns: List[str], key: str):
    for p in patterns:
        if fnmatch.fnmatch(key, p):
            return True
    return False


def hydra_override_name():
    if is_hydra_initialized():
        return hydra_config().job.override_dirname
    else:
        return ''


def omlet_override_name(cfg: Optional[DictConfig]) -> str:
    """
    Command line overrides, e.g. "gpus=4,arch=resnet18"

    If your cfg provides a top-level dict:

    omlet:
        override_name:
            kv_sep: =
            item_sep: ,
            include_keys: ['lr', 'momentum', 'model.*']  # supports patterns
            exclude_keys: ['lr', 'momentum', 'model.*']  # supports patterns

    `include_keys` takes precedence over `exclude_keys`
    """
    assert is_hydra_initialized() and is_omlet_initialized(cfg)

    override_cfg = cfg.omlet.override_name
    # cfg.omlet.shorthand maps original -> short
    # build a reverse lookup for short -> original
    longhand = {short: original for original, short in cfg.omlet.shorthand.items()}
    args = []
    for arg in hydra_override_arg_list():
        assert "=" in arg, f'INTERNAL ERROR: arg should contain "=": {arg}'
        key, value = arg.split('=', 1)
        append = False
        if key in longhand:  # key is a shorthand
            key = longhand[key]
        # first check include list
        if override_cfg.include_keys is not None:
            if _match_patterns(override_cfg.include_keys, key):
                append = True
        elif not _match_patterns(override_cfg.exclude_keys, key):
            append = True

        if append:
            # key is always a longhand by now
            if override_cfg.use_shorthand and key in cfg.omlet.shorthand:
                key = cfg.omlet.shorthand[key]
            args.append((key, value))
    args.sort()
    item_sep = override_cfg.item_sep
    kv_sep = override_cfg.kv_sep
    return item_sep.join(f'{key}{kv_sep}{value}' for key, value in args)


def hydra_exp_dir(*subpaths):
    return os.path.join(os.getcwd(), subpaths)


def hydra_original_dir(*subpaths):
    return os.path.join(hydra.utils.get_original_cwd(), *subpaths)


def resource_file(*subpaths):
    # return os.path.join(kitten.__path__[0], 'resources', *subpaths)
    return pkg_resources.resource_filename(
        'kitten', os.path.join('resources', *subpaths)
    )


def disable_unknown_key_check(cfg):
    OmegaConf.set_struct(cfg, False)


def hydra_instantiate(cfg: DictConfig):
    """
    If 'class' key exists but not 'params' key, rest of the keys will be treated as params
    """
    class_key = 'cls'
    assert class_key in cfg
    if 'params' in cfg:
        new_cfg = cfg
    else:
        new_cfg = OmegaConf.create()
        cls = cfg.pop(class_key)
        new_cfg[class_key] = cls
        new_cfg['params'] = cfg
    return hydra.utils.instantiate(new_cfg)


def get_global_cfg():
    """
    WARNING: does not work, hydra compose API seems to work only in Jupyter now
    Use experimental Hydra compose API
    """
    raise NotImplementedError
    from hydra.experimental import initialize, compose
    initialize(config_dir='../../conf', strict=True)
    return compose('config.yaml')
