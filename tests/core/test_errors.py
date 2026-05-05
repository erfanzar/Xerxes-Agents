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
"""Tests for xerxes.core.errors module."""

from xerxes.core.errors import (
    AgentError,
    ClientError,
    ConfigurationError,
    FunctionExecutionError,
    RateLimitError,
    ValidationError,
    XerxesError,
    XerxesMemoryError,
    XerxesTimeoutError,
)


class TestXerxesError:
    def test_basic(self):
        e = XerxesError("something went wrong")
        assert str(e) == "something went wrong"
        assert e.message == "something went wrong"
        assert e.details == {}

    def test_with_details(self):
        e = XerxesError("err", details={"key": "val"})
        assert e.details == {"key": "val"}

    def test_is_exception(self):
        assert issubclass(XerxesError, Exception)


class TestAgentError:
    def test_basic(self):
        e = AgentError("agent-1", "failed to execute")
        assert "agent-1" in str(e)
        assert e.agent_id == "agent-1"


class TestFunctionExecutionError:
    def test_basic(self):
        e = FunctionExecutionError("search", "timeout")
        assert "search" in str(e)
        assert e.function_name == "search"
        assert e.original_error is None

    def test_with_original_error(self):
        orig = ValueError("bad val")
        e = FunctionExecutionError("calc", "failed", original_error=orig)
        assert e.original_error is orig


class TestXerxesTimeoutError:
    def test_basic(self):
        e = XerxesTimeoutError("llm_call", 30.0)
        assert "30" in str(e)
        assert e.operation == "llm_call"
        assert e.timeout == 30.0


class TestValidationError:
    def test_basic(self):
        e = ValidationError("name", "required")
        assert "name" in str(e)
        assert e.field == "name"
        assert e.value is None

    def test_with_value(self):
        e = ValidationError("age", "must be positive", value=-1)
        assert e.value == -1


class TestRateLimitError:
    def test_basic(self):
        e = RateLimitError("api", 60, "minute")
        assert "60" in str(e)
        assert e.resource == "api"
        assert e.limit == 60
        assert e.window == "minute"
        assert e.retry_after is None

    def test_with_retry(self):
        e = RateLimitError("api", 100, "hour", retry_after=30.0)
        assert "30" in str(e)
        assert e.retry_after == 30.0


class TestXerxesMemoryError:
    def test_basic(self):
        e = XerxesMemoryError("store", "disk full")
        assert "store" in str(e)
        assert e.operation == "store"


class TestClientError:
    def test_basic(self):
        e = ClientError("openai", "rate limited")
        assert "openai" in str(e)
        assert e.client_type == "openai"
        assert e.original_error is None

    def test_with_original(self):
        orig = RuntimeError("connection reset")
        e = ClientError("anthropic", "failed", original_error=orig)
        assert e.original_error is orig


class TestConfigurationError:
    def test_basic(self):
        e = ConfigurationError("llm.api_key", "missing")
        assert "llm.api_key" in str(e)
        assert e.config_key == "llm.api_key"
