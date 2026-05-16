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
"""Pydantic-modeled configuration for the legacy core stack.

Defines :class:`XerxesConfig` and its nested blocks (executor, memory,
security, LLM, logging, observability) plus loaders that read from YAML or
JSON files (:meth:`XerxesConfig.from_file`), expand ``XERXES_*`` environment
variables (:meth:`XerxesConfig.from_env`), and persist back to disk. A
process-wide :func:`get_config`/:func:`set_config` singleton lets older
subsystems read the active config without explicit wiring; modern code paths
use the runtime config context owned by ``xerxes.runtime.config_context``.
"""

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

yaml: Any = None
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class LogLevel(StrEnum):
    """Standard Python logging-module levels exposed as a string enum."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnvironmentType(StrEnum):
    """Logical deployment environment used to gate feature flags."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    """LLM providers known to the legacy core config layer."""

    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ExecutorConfig(BaseModel):
    """Limits and feature toggles for the function-execution subsystem."""

    default_timeout: float = Field(default=30.0, ge=1.0, le=600.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_concurrent_executions: int = Field(default=10, ge=1, le=100)
    enable_metrics: bool = True
    enable_caching: bool = False
    cache_ttl: int = Field(default=3600, ge=60, le=86400)


class MemoryConfig(BaseModel):
    """Sizing and persistence knobs for the four-tier memory subsystem."""

    max_short_term: int = Field(default=10, ge=1, le=1000)
    max_working: int = Field(default=5, ge=1, le=100)
    max_long_term: int = Field(default=1000, ge=100, le=100000)
    enable_embeddings: bool = False
    embedding_model: str | None = None
    enable_persistence: bool = False
    persistence_path: str | None = None
    auto_consolidate: bool = True
    consolidation_threshold: float = Field(default=0.8, ge=0.1, le=1.0)


class SecurityConfig(BaseModel):
    """Input/output guardrails, allow/block lists, and rate-limit knobs."""

    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    max_input_length: int = Field(default=10000, ge=100, le=1000000)
    max_output_length: int = Field(default=10000, ge=100, le=1000000)
    allowed_functions: list[str] | None = None
    blocked_functions: list[str] | None = None
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    rate_limit_per_hour: int = Field(default=1000, ge=10, le=10000)
    enable_authentication: bool = False
    api_key: str | None = None
    api_key_env_var: str = "XERXES_API_KEY"


class LLMConfig(BaseModel):
    """Provider-agnostic LLM client settings (provider, model, sampling, timeouts)."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4"
    api_key: str | None = None
    api_key_env_var: str = "OPENAI_API_KEY"
    base_url: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=100000)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0, le=100)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0)
    timeout: float = Field(default=60.0, ge=1.0, le=600.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    enable_streaming: bool = True
    enable_caching: bool = False

    @field_validator("api_key")
    def validate_api_key(cls, v, info):
        """Fall back to the env var named by ``api_key_env_var`` when ``v`` is None."""
        if v is None:
            env_var = info.data.get("api_key_env_var", "OPENAI_API_KEY")
            v = os.getenv(env_var)
        return v


class LoggingConfig(BaseModel):
    """Logging destinations, format, and rotation policy."""

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str | None = None
    enable_console: bool = True
    enable_file: bool = False
    max_file_size: int = Field(default=10485760, ge=1024, le=104857600)
    backup_count: int = Field(default=5, ge=1, le=100)
    enable_json_format: bool = False


class ObservabilityConfig(BaseModel):
    """Tracing, metrics, and profiling endpoints + per-feature toggles."""

    enable_tracing: bool = False
    enable_metrics: bool = True
    enable_profiling: bool = False
    trace_endpoint: str | None = None
    metrics_endpoint: str | None = None
    service_name: str = "xerxes"
    service_version: str = "0.2.0"
    enable_request_logging: bool = True
    enable_response_logging: bool = False
    enable_function_logging: bool = True


class XerxesConfig(BaseModel):
    """Composite root model bundling every subsystem config plus feature flags."""

    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    debug: bool = False
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    plugins: dict[str, Any] = Field(default_factory=dict)

    features: dict[str, bool] = Field(
        default_factory=lambda: {
            "enable_agent_switching": True,
            "enable_function_chaining": True,
            "enable_context_awareness": True,
            "enable_auto_retry": True,
            "enable_adaptive_timeout": False,
            "enable_smart_caching": False,
        }
    )

    @classmethod
    def from_file(cls, path: str | Path) -> "XerxesConfig":
        """Build a config from a YAML or JSON file.

        Raises:
            FileNotFoundError: When ``path`` doesn't exist.
            ImportError: When the file is YAML but PyYAML isn't installed.
            ValueError: For unsupported file extensions.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                if not HAS_YAML:
                    raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")

        return cls(**data)

    @classmethod
    def from_env(cls, prefix: str = "XERXES_") -> "XerxesConfig":
        """Materialise a config from ``prefix``-keyed env vars (underscore-nested dot paths)."""
        config_dict: dict[str, Any] = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()

                parts = config_key.split("_")
                current = config_dict

                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                try:
                    current[parts[-1]] = json.loads(value)
                except json.JSONDecodeError:
                    current[parts[-1]] = value

        return cls(**config_dict)

    def to_file(self, path: str | Path) -> None:
        """Persist the config to ``path`` (``.yaml``/``.yml``/``.json``).

        Falls back to JSON if PyYAML is missing while a YAML extension was
        requested; raises ``ValueError`` for unknown extensions.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump()

        with open(path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                if not HAS_YAML:
                    path = path.with_suffix(".json")
                    json.dump(data, f, indent=2)
                else:
                    yaml.safe_dump(data, f, default_flow_style=False)
            elif path.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")

    def merge(self, other: "XerxesConfig") -> "XerxesConfig":
        """Return a new config whose values come from ``other`` overlaid on ``self``."""
        self_dict = self.model_dump()
        other_dict = other.model_dump()

        def deep_merge(dict1: dict, dict2: dict) -> dict:
            """Recursive dict-overlay used by :meth:`merge`."""
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged = deep_merge(self_dict, other_dict)
        return XerxesConfig(**merged)


_config: XerxesConfig | None = None


def get_config() -> XerxesConfig:
    """Return the process-wide config, lazily building defaults on first call."""
    global _config
    if _config is None:
        _config = XerxesConfig()
    return _config


def set_config(config: XerxesConfig) -> None:
    """Replace the process-wide config singleton with ``config``."""
    global _config
    _config = config


def load_config(path: str | Path | None = None) -> XerxesConfig:
    """Load a config and publish it as the global singleton.

    Resolution order: explicit ``path``; ``XERXES_CONFIG_FILE``; the standard
    cwd/home search paths; falling back to :meth:`XerxesConfig.from_env`.
    """
    if path:
        config = XerxesConfig.from_file(path)
    elif os.getenv("XERXES_CONFIG_FILE"):
        _config_file = os.getenv("XERXES_CONFIG_FILE")
        assert _config_file is not None
        config = XerxesConfig.from_file(_config_file)
    else:
        from xerxes.core.paths import xerxes_home

        _home = xerxes_home()
        default_paths = [
            Path.cwd() / "xerxes.yaml",
            Path.cwd() / "xerxes.yml",
            Path.cwd() / "xerxes.json",
            _home / "config.yaml",
            _home / "config.yml",
            _home / "config.json",
        ]

        for default_path in default_paths:
            if default_path.exists():
                config = XerxesConfig.from_file(default_path)
                break
        else:
            config = XerxesConfig.from_env()

    set_config(config)
    return config
