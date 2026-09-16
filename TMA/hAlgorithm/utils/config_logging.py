import logging
import os
import sys

FMTDCIT = {
    "ERROR": "\033[0;37;41mERROR\033[0m",
    "INFO": "\033[33mINFO\033[0m",
    "DEBUG": "\033[32mDEBUG\033[0m",
    "WARN": "\033[31mWARN\033[0m",
    "WARNING": "\033[31mWARNING\033[0m",
    "CRITICAL": "\033[35mCRITICAL\033[0m",
}


class Filter(logging.Filter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def filter(self, record: logging.LogRecord) -> bool:
        record.levelname = FMTDCIT.get(record.levelname)
        return True


filter = Filter()


def config_logging(out_dir=None):
    file_level = logging.DEBUG
    console_level = logging.INFO
    # format = "%(asctime)s-%(levelname)s- %(filename)s - %(name)s - %(message)s"
    format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    log_formatter = logging.Formatter(format)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_logger.setLevel(min(file_level, console_level))

    if out_dir is not None:
        _logging_file = os.path.join(out_dir, "logging.log")
        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.addFilter(filter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Avoid pollution by packages
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)


def config_logging_v2(out_dir=None, console_level=logging.INFO, file_level=logging.DEBUG):
    # format = "%(asctime)s-%(levelname)s- %(filename)s - %(name)s - %(message)s"
    format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    log_formatter = logging.Formatter(format)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_logger.setLevel(min(file_level, console_level))

    if out_dir is not None:
        _logging_file = os.path.join(out_dir, "logging.log")
        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.addFilter(filter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Avoid pollution by packages
    logging.getLogger("PIL").setLevel(console_level)
    logging.getLogger("matplotlib").setLevel(console_level)