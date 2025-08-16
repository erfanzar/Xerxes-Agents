# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Configuration management system for Calute."""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

from pydantic import BaseModel, Field, field_validator


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnvironmentType(str, Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ExecutorConfig(BaseModel):
    """Executor configuration."""

    default_timeout: float = Field(default=30.0, ge=1.0, le=600.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_concurrent_executions: int = Field(default=10, ge=1, le=100)
    enable_metrics: bool = True
    enable_caching: bool = False
    cache_ttl: int = Field(default=3600, ge=60, le=86400)


class MemoryConfig(BaseModel):
    """Memory configuration."""

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
    """Security configuration."""

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
    api_key_env_var: str = "CALUTE_API_KEY"


class LLMConfig(BaseModel):
    """LLM configuration."""

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
        """Validate or load API key from environment."""
        if v is None:
            env_var = info.data.get("api_key_env_var", "OPENAI_API_KEY")
            v = os.getenv(env_var)
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str | None = None
    enable_console: bool = True
    enable_file: bool = False
    max_file_size: int = Field(default=10485760, ge=1024, le=104857600)  # 10MB default
    backup_count: int = Field(default=5, ge=1, le=100)
    enable_json_format: bool = False


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    enable_tracing: bool = False
    enable_metrics: bool = True
    enable_profiling: bool = False
    trace_endpoint: str | None = None
    metrics_endpoint: str | None = None
    service_name: str = "calute"
    service_version: str = "0.0.18"
    enable_request_logging: bool = True
    enable_response_logging: bool = False
    enable_function_logging: bool = True


class CaluteConfig(BaseModel):
    """Main Calute configuration."""

    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    debug: bool = False
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Plugin configurations
    plugins: dict[str, Any] = Field(default_factory=dict)

    # Feature flags
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
    def from_file(cls, path: str | Path) -> "CaluteConfig":
        """Load configuration from file (JSON or YAML)."""
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
    def from_env(cls, prefix: str = "CALUTE_") -> "CaluteConfig":
        """Load configuration from environment variables."""
        config_dict = {}

        # Parse environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix) :].lower()

                # Handle nested configuration
                parts = config_key.split("_")
                current = config_dict

                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Try to parse the value
                try:
                    current[parts[-1]] = json.loads(value)
                except json.JSONDecodeError:
                    current[parts[-1]] = value

        return cls(**config_dict)

    def to_file(self, path: str | Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump()

        with open(path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                if not HAS_YAML:
                    # Fallback to JSON if YAML not available
                    path = path.with_suffix(".json")
                    json.dump(data, f, indent=2)
                else:
                    yaml.safe_dump(data, f, default_flow_style=False)
            elif path.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")

    def merge(self, other: "CaluteConfig") -> "CaluteConfig":
        """Merge with another configuration (other takes precedence)."""
        self_dict = self.model_dump()
        other_dict = other.model_dump()

        def deep_merge(dict1, dict2):
            """Deep merge two dictionaries."""
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged = deep_merge(self_dict, other_dict)
        return CaluteConfig(**merged)


# Global configuration instance
_config: CaluteConfig | None = None


def get_config() -> CaluteConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = CaluteConfig()
    return _config


def set_config(config: CaluteConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(path: str | Path | None = None) -> CaluteConfig:
    """Load configuration from file or environment."""
    if path:
        config = CaluteConfig.from_file(path)
    elif os.getenv("CALUTE_CONFIG_FILE"):
        config = CaluteConfig.from_file(os.getenv("CALUTE_CONFIG_FILE"))
    else:
        # Try to load from default locations
        default_paths = [
            Path.cwd() / "calute.yaml",
            Path.cwd() / "calute.yml",
            Path.cwd() / "calute.json",
            Path.home() / ".calute" / "config.yaml",
            Path.home() / ".calute" / "config.yml",
            Path.home() / ".calute" / "config.json",
        ]

        for default_path in default_paths:
            if default_path.exists():
                config = CaluteConfig.from_file(default_path)
                break
        else:
            # Load from environment or use defaults
            config = CaluteConfig.from_env()

    set_config(config)
    return config
