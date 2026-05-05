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
"""Tests for xerxes.logging_config module."""

import logging

from xerxes.logging.structured import (
    XerxesLogger,
    configure_logging,
    get_logger,
)


class TestXerxesLogger:
    def test_init_defaults(self):
        logger = XerxesLogger(name="test_logger", level="INFO")
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert logger.log_file is None
        assert logger.enable_tracing is False

    def test_init_debug_level(self):
        logger = XerxesLogger(name="test_debug", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_log_function_call_success(self):
        logger = XerxesLogger(name="test_fc", level="WARNING")
        logger.log_function_call(
            agent_id="agent1",
            function_name="search",
            arguments={"query": "test"},
            result="found something",
            duration=0.5,
        )

    def test_log_function_call_error(self):
        logger = XerxesLogger(name="test_fc_err", level="WARNING")
        logger.log_function_call(
            agent_id="agent1",
            function_name="search",
            arguments={"query": "test"},
            error=ValueError("bad input"),
            duration=1.0,
        )

    def test_log_agent_switch(self):
        logger = XerxesLogger(name="test_switch", level="WARNING")
        logger.log_agent_switch(from_agent="agent1", to_agent="agent2", reason="task complete")

    def test_log_llm_request_success(self):
        logger = XerxesLogger(name="test_llm", level="WARNING")
        logger.log_llm_request(
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            duration=1.2,
        )

    def test_log_llm_request_error(self):
        logger = XerxesLogger(name="test_llm_err", level="WARNING")
        logger.log_llm_request(
            provider="openai",
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=0,
            duration=0.5,
            error=RuntimeError("API error"),
        )

    def test_log_memory_operation_add(self):
        logger = XerxesLogger(name="test_mem", level="WARNING")
        logger.log_memory_operation(
            operation="add",
            memory_type="short_term",
            agent_id="agent1",
            entry_count=3,
        )

    def test_log_memory_operation_remove(self):
        logger = XerxesLogger(name="test_mem_rm", level="WARNING")
        logger.log_memory_operation(
            operation="remove",
            memory_type="long_term",
            agent_id="agent1",
        )

    def test_log_memory_operation_error(self):
        logger = XerxesLogger(name="test_mem_err", level="WARNING")
        logger.log_memory_operation(
            operation="add",
            memory_type="short_term",
            agent_id="agent1",
            error=OSError("disk full"),
        )

    def test_span_no_tracing(self):
        logger = XerxesLogger(name="test_span", level="INFO", enable_tracing=False)
        with logger.span("test_operation", key="val") as span:
            assert span is None

    def test_get_metrics(self):
        logger = XerxesLogger(name="test_metrics", level="INFO")
        metrics = logger.get_metrics()
        assert isinstance(metrics, bytes)

    def test_with_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = XerxesLogger(name="test_file", level="INFO", log_file=log_file)
        assert logger.log_file == log_file

    def test_json_disabled(self):
        logger = XerxesLogger(name="test_no_json", level="INFO", enable_json=False)
        assert logger.enable_json is False


class TestGlobalLogger:
    def setup_method(self):
        import xerxes.logging.structured as mod

        mod._logger = None

    def test_get_logger_creates_default(self):
        logger = get_logger()
        assert isinstance(logger, XerxesLogger)

    def test_get_logger_returns_same(self):
        l1 = get_logger()
        l2 = get_logger()
        assert l1 is l2

    def test_configure_logging(self):
        configure_logging(name="custom", level="DEBUG")
        logger = get_logger()
        assert logger.name == "custom"
