import logging
from configparser import RawConfigParser
from logging.handlers import RotatingFileHandler
from pathlib import Path

# MLCF modules
from mlcf.envtools.pathtools import create_path


def init_logging(
    home_name: str, dir_pref: Path, config: RawConfigParser
) -> logging.Logger:

    log_dir = create_path(str(dir_pref.joinpath(config.get("Log", "DirLog"))))

    log_file_name_debug = config.get("Log", "BaseLogFileNameForDebug")
    path_log_file_name_debug = log_dir.joinpath(log_file_name_debug)

    log_file_name_info = config.get("Log", "BaseLogFileNameInfo")
    path_log_file_name_info = log_dir.joinpath(log_file_name_info)

    logger = logging.getLogger(home_name + "_logger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(config.get("Log", "LogFormat"))
    file_handler_debug = RotatingFileHandler(
        path_log_file_name_debug,
        mode="a",
        maxBytes=config.getint("Log", "MaxBytes"),
        backupCount=1,
    )
    file_handler_debug.setLevel(logging.DEBUG)
    file_handler_debug.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        path_log_file_name_info,
        mode="a",
        maxBytes=config.getint("Log", "MaxBytes"),
        backupCount=1,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler_debug)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
