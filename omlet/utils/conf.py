import os
import re
import hydra
import pkg_resources
import fnmatch
from omegaconf import OmegaConf, DictConfig
from typing import Optional, List


def is_hydra_initialized():
    return hydra.utils.HydraConfig().cfg is not None


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


def hydra_override_name(cfg: Optional[DictConfig] = None) -> str:
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
    if not is_hydra_initialized():
        return ''

    if cfg is None or cfg.get('omlet', {}).get('override_name', None) is None:
        return hydra_config().job.override_dirname

    override_cfg = cfg.omlet.override_name
    args = []
    for arg in hydra_override_arg_list():
        assert "=" in arg, f'INTERNAL ERROR: arg should contain "=": {arg}'
        key, value = arg.split('=', 1)
        # first check include list
        if override_cfg.get('include_keys', None) is not None:
            if _match_patterns(override_cfg.include_keys, key):
                args.append((key, value))
        elif not _match_patterns(override_cfg.get('exclude_keys', []), key):
            args.append((key, value))
    args.sort()
    item_sep = override_cfg.get('item_sep', ',')
    kv_sep = override_cfg.get('kv_sep', '=')
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
