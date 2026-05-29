"""Package-wide logging helpers for llm_annotator."""

from __future__ import annotations

import logging
import sys
from typing import Literal

from colorama import Fore, Style, just_fix_windows_console


_LOGGER_NAME = "llm_annotator"
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LEVEL_STYLES: dict[int, str] = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}


class _ColorFormatter(logging.Formatter):
    """Terminal formatter with light ANSI color accents."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        color = _LEVEL_STYLES.get(record.levelno)
        if not color:
            return formatted
        return f"{color}{formatted}{Style.RESET_ALL}"


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    maybe_level = logging.getLevelName(level.upper())
    if isinstance(maybe_level, int):
        return maybe_level
    raise ValueError(
        "Unsupported log level. Use an int or one of "
        "DEBUG/INFO/WARNING/ERROR/CRITICAL."
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger scoped under the llm_annotator namespace."""
    if not name:
        return logging.getLogger(_LOGGER_NAME)

    normalized = name.strip(".")
    if not normalized:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{normalized}")


def configure_logging(
    *,
    enabled: bool = True,
    level: int | str = "INFO",
    style: Literal["pretty", "plain"] = "pretty",
) -> logging.Logger:
    """Configure package logging in a single call.

    Args:
        enabled: Whether package logs should be emitted.
        level: Logging level as int or name.
        style: Formatter style, either colorized ``"pretty"`` or ``"plain"``.

    Returns:
        The package root logger.
    """
    logger = get_logger()
    logger.propagate = False

    if not enabled:
        logger.disabled = True
        return logger

    logger.disabled = False
    logger.setLevel(_coerce_level(level))

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    formatter: logging.Formatter
    use_color = (
        style == "pretty"
        and hasattr(handler.stream, "isatty")
        and handler.stream.isatty()
    )
    if use_color:
        just_fix_windows_console()
        formatter = _ColorFormatter(_DEFAULT_FORMAT)
    else:
        formatter = logging.Formatter(_DEFAULT_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def set_log_level(level: int | str) -> None:
    """Update the package logger level.

    Args:
        level: Logging level as an integer or a standard level name.
    """
    get_logger().setLevel(_coerce_level(level))


__all__ = ["configure_logging", "get_logger", "set_log_level"]
