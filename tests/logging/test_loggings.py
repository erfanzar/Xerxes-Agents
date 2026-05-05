# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for xerxes.loggings module."""

import logging

from xerxes.logging.console import (
    COLORS,
    LEVEL_COLORS,
    ColorFormatter,
    XerxesLogger,
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


class TestXerxesLogger:
    def test_singleton(self):
        l1 = XerxesLogger()
        l2 = XerxesLogger()
        assert l1 is l2

    def test_log_methods(self):
        logger = XerxesLogger()
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warning msg")
        logger.error("error msg")
        logger.critical("critical msg")

    def test_set_level(self):
        logger = XerxesLogger()
        logger.set_level("DEBUG")
        assert logger.logger.level == logging.DEBUG
        logger.set_level("INFO")


class TestGetLogger:
    def test_returns_xerxes_logger(self):
        logger = get_logger()
        assert isinstance(logger, XerxesLogger)


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
