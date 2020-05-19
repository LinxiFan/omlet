import logging
from omlet.utils.logging_utils import *

omlet_logger = get_logger("omlet")

# ================= Patch up PL lightning ================
import pytorch_lightning
# force remove lightning loggers and use ours
get_logger('lightning')
# set global logging level to at least INFO
set_logging_level(min(get_logging_level(), logging.INFO))

import omlet.utils.misc_utils

pytorch_lightning.core.memory.get_human_readable_count = lambda num: omlet.utils.misc_utils.get_human_readable_count(num, precision=2)
# ======================================

from .extended import ExtendedModule, STAGES
from .trainer import *
from .callbacks import *
from .checkpoint import ExtendedCheckpoint
