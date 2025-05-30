"""Logger configuration."""

import logging
import sys
from collections.abc import Mapping
from typing import Any

from loguru import logger
from loki_logger_handler.formatters.loguru_formatter import LoguruFormatter
from loki_logger_handler.loki_logger_handler import LokiLoggerHandler

from .config import settings

__all__ = ["config_logger"]


def config_logger() -> None:
    """Logger configuration."""
    is_production = settings.app_env == "production"

    if is_production:
        logging.root.handlers = [InterceptHandler()]
        logging.root.setLevel(logging.INFO)

        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger_instance in loggers:
            logger_instance.handlers = []
            logger_instance.propagate = True

        logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

    logger.remove()

    if not is_production:
        logger.add(
            settings.log_path,
            rotation=settings.rotation,
            format=_format_record,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            compression="zip",
            colorize=False,
        )

    logger.add(
        sys.stdout,
        format=_format_record,
        level=logging.INFO if is_production else settings.log_level,
        colorize=not is_production,
        enqueue=True,
        backtrace=not is_production,
        diagnose=not is_production,
    )

    if is_production:
        logger.add(
            LokiLoggerHandler(
                url="http://alloy:9999/loki/api/v1/push",
                labels={
                    "application": "fastapi",
                    "environment": settings.app_env,
                    "source": "fastapi",
                },
                timeout=10,
                enable_structured_loki_metadata=True,
                default_formatter=LoguruFormatter(),  # type: ignore[arg-type]
            ),
            serialize=True,
            enqueue=True,
            level=logging.INFO,
        )


class InterceptHandler(logging.Handler):
    @staticmethod
    def emit(record: logging.LogRecord) -> None:
        """Intercepts standard logging and sends it to Loguru."""
        if "changes detected" in record.getMessage():
            return

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_back and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _format_record(record: Mapping[str, Any]) -> str:
    ts = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    line = (
        f"<green>{ts}</green> | "
        f"<level>{record['level']:<8}</level> | "
        f"<cyan>{record['name']}:{record['function']}:{record['line']}</cyan> - "
        f"{record['message']}"
    )

    if record["extra"]:
        extras = " | ".join(
            f"<yellow>{k}</yellow>=<cyan>{v}</cyan>" for k, v in record["extra"].items()
        )
        line += f" | {extras}"

    return line + "\n"
