import os
import logging
from typing import Union

try:
    import colorlog
    IS_COLORLOG_INSTALLED = True
except ImportError:
    IS_COLORLOG_INSTALLED = False


__all__ = ['get_logger', 'set_logging_level', 'get_logging_level',
           'create_logging_file_handler', 'override_loggers']


# custom debugging level that's higher than usual to distinguish from
# debug logs from other applications
DEBUG2 = logging.DEBUG + 2
logging.DEBUG2 = DEBUG2

# custom info level that's lower than usual to enable more verbosity
INFOV = logging.INFO - 2
logging.INFOV = INFOV

# https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility
logging.addLevelName(DEBUG2, 'DEBUG2')
logging.addLevelName(INFOV, 'INFOV')


_DATEFMT = '%m-%d-%y %H:%M:%S'


def _debug2(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG2):
        # Yes, logger takes its '*args' as 'args'.
        self._log(DEBUG2, message, args, **kws)


def _infov(self, message, *args, **kws):
    if self.isEnabledFor(INFOV):
        # Yes, logger takes its '*args' as 'args'.
        self._log(INFOV, message, args, **kws)


logging.Logger.debug2 = _debug2
logging.Logger.infov = _infov


def create_colorlog_handler():
    if IS_COLORLOG_INSTALLED:
        # https://pypi.org/project/colorlog/
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] %(message)s",
            # available color prefixes: bold_, thin_, bg_, fg_
            # colors: black, red, green, yellow, blue, purple, cyan and white
            log_colors={
                'DEBUG': 'purple',
                'DEBUG2': 'purple,bg_yellow',
                'INFOV': 'thin_green',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'bold_red',
                'CRITICAL': 'red,bg_white',
            },
            datefmt=_DATEFMT
        ))
        return handler
    else:
        return None


def has_logger(name):
    return name in logging.Logger.manager.loggerDict


def get_logger(name, level=None):
    color_handler = create_colorlog_handler()
    exists = has_logger(name)
    logger = logging.getLogger(name)
    # if not exists:
    #     # defaults to at least INFO
    #     root_level = logging.getLogger().getEffectiveLevel()
    #     logger.setLevel(min(logging.INFO, root_level))

    if color_handler is not None:
        # root = logging.getLogger()
        logger.handlers = []
        logger.addHandler(color_handler)

    if level is not None:
        logger.setLevel(level)

    return logger


def create_logging_file_handler(file_path):
    file_path = os.path.expanduser(file_path)
    handler = logging.FileHandler(file_path)
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
        datefmt=_DATEFMT
    ))
    return handler


def get_logging_level(name=None):
    return logging.getLogger(name).getEffectiveLevel()


def set_logging_level(level: Union[int, str] = 'info', name=None):
    if isinstance(level, str):
        level = level.upper()
    logging.getLogger(name).setLevel(level)


def override_loggers(level=None):
    # force override Hydra's logging system
    if level is not None:
        set_logging_level(level)
    logging.getLogger().handlers = []
    get_logger('omlet')
    get_logger('lightning')

