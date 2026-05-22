from __future__ import annotations

import logging
import sys
from io import StringIO

import pytest
from colorama import Fore, Style

from llm_annotator import logging_utils


def _reset_root_logger() -> logging.Logger:
    logger = logging_utils.get_logger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.disabled = False
    return logger


def test_coerce_level_accepts_int_and_string() -> None:
    # Verifies known int and string levels normalize correctly.
    assert logging_utils._coerce_level(logging.INFO) == logging.INFO
    assert logging_utils._coerce_level("debug") == logging.DEBUG


def test_coerce_level_rejects_unknown_level() -> None:
    # Verifies unknown level names raise a clear validation error.
    with pytest.raises(ValueError, match="Unsupported log level"):
        logging_utils._coerce_level("not-a-level")


def test_get_logger_namespace_normalization() -> None:
    # Verifies logger names are scoped and dot-normalized.
    assert logging_utils.get_logger().name == "llm_annotator"
    assert logging_utils.get_logger(".clients.openai.").name == (
        "llm_annotator.clients.openai"
    )
    assert logging_utils.get_logger("...").name == "llm_annotator"


def test_color_formatter_adds_color_for_known_level() -> None:
    # Verifies colorama color prefixes/suffixes are applied for known levels.
    formatter = logging_utils._ColorFormatter("%(message)s")
    record = logging.LogRecord(
        name="x",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert Fore.YELLOW in rendered
    assert rendered.endswith(Style.RESET_ALL)


def test_color_formatter_leaves_unknown_level_uncolored() -> None:
    # Verifies unknown log levels are rendered without ANSI color codes.
    formatter = logging_utils._ColorFormatter("%(message)s")
    record = logging.LogRecord(
        name="x",
        level=99,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert rendered == "hello"


def test_configure_logging_disabled() -> None:
    # Verifies disabling logging toggles logger.disabled and keeps logger object.
    logger = _reset_root_logger()
    configured = logging_utils.configure_logging(enabled=False)
    assert configured is logger
    assert logger.disabled is True


def test_configure_logging_plain_style(monkeypatch: pytest.MonkeyPatch) -> None:
    # Verifies plain style selects the standard formatter and requested level.
    _reset_root_logger()
    monkeypatch.setattr(sys, "stderr", StringIO())
    logger = logging_utils.configure_logging(
        enabled=True, level="WARNING", style="plain"
    )
    assert logger.level == logging.WARNING
    assert len(logger.handlers) == 1
    assert not isinstance(logger.handlers[0].formatter, logging_utils._ColorFormatter)


def test_configure_logging_pretty_style_posix_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies pretty style enables color formatter on POSIX TTY streams.
    class FakeTTY(StringIO):
        def isatty(self) -> bool:
            return True

    _reset_root_logger()
    monkeypatch.setattr(sys, "stderr", FakeTTY())

    logger = logging_utils.configure_logging(enabled=True, style="pretty")
    assert isinstance(logger.handlers[0].formatter, logging_utils._ColorFormatter)


def test_configure_logging_pretty_style_windows_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Verifies pretty style also uses color formatter on Windows-compatible streams.
    class FakeTTY(StringIO):
        def isatty(self) -> bool:
            return True

    _reset_root_logger()
    monkeypatch.setattr(sys, "stderr", FakeTTY())

    logger = logging_utils.configure_logging(enabled=True, style="pretty")
    assert isinstance(logger.handlers[0].formatter, logging_utils._ColorFormatter)


def test_set_log_level_updates_root_logger() -> None:
    # Verifies set_log_level mutates the already-configured root logger level.
    logger = _reset_root_logger()
    logging_utils.configure_logging(enabled=True, level="INFO")
    logging_utils.set_log_level("ERROR")
    assert logger.level == logging.ERROR
