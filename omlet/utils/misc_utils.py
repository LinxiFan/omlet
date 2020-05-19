import os
import numpy as np
from typing import Optional, Dict, Any


def get_human_readable_count(number: int, precision: int = 2):
    assert number >= 0
    labels = [' ', 'K', 'M', 'B', 'T']
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups,
                     len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10 ** shift)
    index = num_groups - 1
    rem = number - int(number)
    if precision > 0 and rem > 0.01:
        fmt = f'{{:.{precision}f}}'
        rem_str = fmt.format(rem).lstrip('0')
    else:
        rem_str = ''
    return f'{int(number):,d}{rem_str} {labels[index]}'


def set_os_envs(envs: Optional[Dict[str, Any]] = None):
    if envs is None:
        envs = {}
    os.environ.update({k: str(v) for k, v in envs.items()})


