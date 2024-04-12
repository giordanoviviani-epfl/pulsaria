"""Utility functions for logging configurations."""

import atexit
import datetime as dt
import json
import logging
import logging.config
import logging.handlers
import os
from pathlib import Path
from typing import override

import yaml
from path_config import pulsaria_path

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def __init__(self, fmt_keys: dict[str, str] | None) -> None:
        """Initialize JSONFormatter.

        Parameters
        ----------
        fmt_keys : dict[str, str] | None
            Dictionary with keys and values to format the log message.
            The keys are the names of the fields in the log message and the
            values are the names of the attributes of the log record.
            If None, the default keys are:
                - "level": "levelname"
                - "logger": "name"
                - "timestamp": "asctime"
                - "message": "message"

        """
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict:
        always_fields = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created,
                tz=dt.UTC,
            ).isoformat(),
        }

        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    """Filter to log only messages with level <= INFO."""

    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= logging.INFO


def configure_logging() -> None:
    """Read and set logging configuration from logging_config.yaml."""
    logging_dir = Path(os.path.realpath(__file__)).parent
    config_file = logging_dir / "logging_config.yaml"

    save_log_dir = logging_dir.parent.parent / "logs"
    save_log_dir.mkdir(exist_ok=True)

    with config_file.open() as f_in:
        config = yaml.safe_load(f_in)

    for handler in config["handlers"].values():
        if handler.get("filename"):
            handler["filename"] = str(pulsaria_path.root / handler["filename"])

    logging.config.dictConfig(config)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)
