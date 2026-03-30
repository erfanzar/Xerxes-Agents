"""Tests for calute.loggings module."""

import logging

from calute.logging.console import (
    COLORS,
    LEVEL_COLORS,
    CaluteLogger,
    ColorFormatter,
    get_logger,
    log_step,
    set_verbosity,
)


class TestColorFormatter:
    def test_format_info(self):
        formatter = ColorFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        assert "hello" in result

    def test_format_warning(self):
        formatter = ColorFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="warn msg",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        assert "warn msg" in result

    def test_format_multiline(self):
        formatter = ColorFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="line1\nline2",
            args=None,
            exc_info=None,
        )
        result = formatter.format(record)
        assert "line1" in result
        assert "line2" in result


class TestCaluteLogger:
    def test_singleton(self):
        l1 = CaluteLogger()
        l2 = CaluteLogger()
        assert l1 is l2

    def test_log_methods(self):
        logger = CaluteLogger()
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warning msg")
        logger.error("error msg")
        logger.critical("critical msg")

    def test_set_level(self):
        logger = CaluteLogger()
        logger.set_level("DEBUG")
        assert logger.logger.level == logging.DEBUG
        logger.set_level("INFO")


class TestGetLogger:
    def test_returns_calute_logger(self):
        logger = get_logger()
        assert isinstance(logger, CaluteLogger)


class TestSetVerbosity:
    def test_set_debug(self):
        set_verbosity("DEBUG")
        logger = get_logger()
        assert logger.logger.level == logging.DEBUG
        set_verbosity("INFO")


class TestLogStep:
    def test_basic(self):
        log_step("TEST", "testing step")

    def test_with_color(self):
        log_step("TEST", "testing green", color="GREEN")

    def test_no_description(self):
        log_step("TEST")


class TestColors:
    def test_colors_dict(self):
        assert "RED" in COLORS
        assert "GREEN" in COLORS
        assert "RESET" in COLORS
        assert "BOLD" in COLORS

    def test_level_colors(self):
        assert "DEBUG" in LEVEL_COLORS
        assert "INFO" in LEVEL_COLORS
        assert "WARNING" in LEVEL_COLORS
        assert "ERROR" in LEVEL_COLORS
        assert "CRITICAL" in LEVEL_COLORS
