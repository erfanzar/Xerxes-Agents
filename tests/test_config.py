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
"""Tests for xerxes.core.config module."""

import json
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from xerxes.core.config import (
    EnvironmentType,
    ExecutorConfig,
    LLMConfig,
    LLMProvider,
    LoggingConfig,
    LogLevel,
    MemoryConfig,
    ObservabilityConfig,
    SecurityConfig,
    XerxesConfig,
    get_config,
    load_config,
    set_config,
)


class TestEnums:
    def test_log_level_values(self):
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"

    def test_environment_type_values(self):
        assert EnvironmentType.DEVELOPMENT == "development"
        assert EnvironmentType.TESTING == "testing"
        assert EnvironmentType.STAGING == "staging"
        assert EnvironmentType.PRODUCTION == "production"

    def test_llm_provider_values(self):
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.GEMINI == "gemini"
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.LOCAL == "local"


class TestExecutorConfig:
    def test_defaults(self):
        config = ExecutorConfig()
        assert config.default_timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.max_concurrent_executions == 10
        assert config.enable_metrics is True
        assert config.enable_caching is False
        assert config.cache_ttl == 3600

    def test_validation_bounds(self):
        with pytest.raises(ValidationError):
            ExecutorConfig(default_timeout=0.5)
        with pytest.raises(ValidationError):
            ExecutorConfig(max_retries=11)
        with pytest.raises(ValidationError):
            ExecutorConfig(max_concurrent_executions=0)


class TestMemoryConfig:
    def test_defaults(self):
        config = MemoryConfig()
        assert config.max_short_term == 10
        assert config.max_long_term == 1000
        assert config.enable_embeddings is False
        assert config.auto_consolidate is True

    def test_validation_bounds(self):
        with pytest.raises(ValidationError):
            MemoryConfig(max_short_term=0)
        with pytest.raises(ValidationError):
            MemoryConfig(consolidation_threshold=1.5)


class TestSecurityConfig:
    def test_defaults(self):
        config = SecurityConfig()
        assert config.enable_input_validation is True
        assert config.enable_rate_limiting is True
        assert config.rate_limit_per_minute == 60
        assert config.enable_authentication is False

    def test_validation_bounds(self):
        with pytest.raises(ValidationError):
            SecurityConfig(max_input_length=50)


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(api_key=None)
            assert config.api_key == "test-key-123"

    def test_api_key_explicit(self):
        config = LLMConfig(api_key="explicit-key")
        assert config.api_key == "explicit-key"

    def test_validation_bounds(self):
        with pytest.raises(ValidationError):
            LLMConfig(temperature=3.0)
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)


class TestLoggingConfig:
    def test_defaults(self):
        config = LoggingConfig()
        assert config.level == LogLevel.INFO
        assert config.enable_console is True
        assert config.enable_file is False


class TestObservabilityConfig:
    def test_defaults(self):
        config = ObservabilityConfig()
        assert config.enable_tracing is False
        assert config.enable_metrics is True
        assert config.service_name == "xerxes"


class TestXerxesConfig:
    def test_defaults(self):
        config = XerxesConfig()
        assert config.environment == EnvironmentType.DEVELOPMENT
        assert config.debug is False
        assert isinstance(config.executor, ExecutorConfig)
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.observability, ObservabilityConfig)
        assert config.features["enable_agent_switching"] is True

    def test_from_json_file(self, tmp_path):
        config_data = {
            "environment": "testing",
            "debug": True,
            "executor": {"default_timeout": 60.0},
            "llm": {"model": "gpt-3.5-turbo"},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = XerxesConfig.from_file(config_file)
        assert config.environment == EnvironmentType.TESTING
        assert config.debug is True
        assert config.executor.default_timeout == 60.0
        assert config.llm.model == "gpt-3.5-turbo"

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            XerxesConfig.from_file("/nonexistent/config.json")

    def test_from_file_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "config.toml"
        bad_file.write_text("key = 'value'")
        with pytest.raises(ValueError, match="Unsupported"):
            XerxesConfig.from_file(bad_file)

    def test_to_json_file(self, tmp_path):
        config = XerxesConfig(debug=True)
        config_file = tmp_path / "output.json"
        config.to_file(config_file)

        loaded = XerxesConfig.from_file(config_file)
        assert loaded.debug is True

    def test_to_file_unsupported_format(self, tmp_path):
        config = XerxesConfig()
        with pytest.raises(ValueError, match="Unsupported"):
            config.to_file(tmp_path / "config.toml")

    def test_merge(self):
        base = XerxesConfig(debug=False)
        override = XerxesConfig(debug=True, llm=LLMConfig(model="claude-3"))
        merged = base.merge(override)
        assert merged.debug is True
        assert merged.llm.model == "claude-3"

    def test_from_env(self):
        env_vars = {
            "XERXES_DEBUG": "true",
            "XERXES_ENVIRONMENT": "production",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = XerxesConfig.from_env()
            assert config.debug is True
            assert config.environment == EnvironmentType.PRODUCTION

    def test_from_env_nested(self):
        env_vars = {
            "XERXES_LLM_MODEL": "gpt-3.5-turbo",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = XerxesConfig.from_env()
            assert config.llm.model == "gpt-3.5-turbo"


class TestGlobalConfig:
    def setup_method(self):
        import xerxes.core.config as cfg_module

        cfg_module._config = None

    def test_get_config_creates_default(self):
        config = get_config()
        assert isinstance(config, XerxesConfig)

    def test_set_config(self):
        new_config = XerxesConfig(debug=True)
        set_config(new_config)
        assert get_config().debug is True

    def test_load_config_from_file(self, tmp_path):
        config_data = {"debug": True}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = load_config(config_file)
        assert config.debug is True

    def test_load_config_from_env_var(self, tmp_path):
        config_data = {"debug": True}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with patch.dict(os.environ, {"XERXES_CONFIG_FILE": str(config_file)}):
            config = load_config()
            assert config.debug is True

    def test_load_config_defaults_to_env(self):
        with patch.dict(os.environ, {}, clear=False):
            config = load_config()
            assert isinstance(config, XerxesConfig)
