import logging
import sys
from pathlib import Path
from typing import Optional

_LOGGERS = {}


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    if name not in _LOGGERS:
        return setup_logger(name)
    return _LOGGERS[name]
