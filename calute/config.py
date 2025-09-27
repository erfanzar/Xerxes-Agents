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


"""Configuration management system for Calute.

This module provides a comprehensive configuration management system
for the Calute framework. It includes:
- Pydantic-based configuration models with validation
- Support for JSON and YAML configuration files
- Environment variable configuration loading
- Configuration merging and persistence
- Separate config sections for executor, memory, security, LLM, logging, and observability

The configuration system follows a hierarchical structure with sensible
defaults and extensive validation to ensure configuration integrity.

Example:
    >>> config = CaluteConfig.from_file("config.yaml")
    >>> config.llm.model = "gpt-4"
    >>> config.to_file("updated_config.yaml")
"""

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
    """Enumeration of available logging levels.

    Standard Python logging levels for controlling
    log verbosity throughout the application.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnvironmentType(str, Enum):
    """Enumeration of deployment environment types.

    Used to configure different behaviors and settings
    based on the deployment environment.
    """

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    """Enumeration of supported LLM provider backends.

    Defines the available LLM providers that can be configured
    for use with Calute agents.
    """

    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ExecutorConfig(BaseModel):
    """Configuration for function execution behavior.

    Controls timeout, retry, concurrency, and caching settings
    for function/tool execution within agents.

    Attributes:
        default_timeout: Default timeout in seconds for function execution.
        max_retries: Maximum number of retry attempts on failure.
        retry_delay: Delay in seconds between retry attempts.
        max_concurrent_executions: Maximum concurrent function executions.
        enable_metrics: Whether to collect execution metrics.
        enable_caching: Whether to cache function results.
        cache_ttl: Cache time-to-live in seconds.
    """

    default_timeout: float = Field(default=30.0, ge=1.0, le=600.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_concurrent_executions: int = Field(default=10, ge=1, le=100)
    enable_metrics: bool = True
    enable_caching: bool = False
    cache_ttl: int = Field(default=3600, ge=60, le=86400)


class MemoryConfig(BaseModel):
    """Configuration for the memory management system.

    Controls memory capacity, persistence, and consolidation
    settings for agent memory systems.

    Attributes:
        max_short_term: Maximum short-term memory entries.
        max_working: Maximum working memory entries.
        max_long_term: Maximum long-term memory entries.
        enable_embeddings: Whether to use embeddings for memory.
        embedding_model: Model to use for embeddings.
        enable_persistence: Whether to persist memory to disk.
        persistence_path: Path for memory persistence.
        auto_consolidate: Whether to automatically consolidate memories.
        consolidation_threshold: Threshold for memory consolidation.
    """

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
    """Security and safety configuration.

    Controls input validation, output sanitization, rate limiting,
    and authentication settings for secure operation.

    Attributes:
        enable_input_validation: Whether to validate inputs.
        enable_output_sanitization: Whether to sanitize outputs.
        max_input_length: Maximum allowed input length.
        max_output_length: Maximum allowed output length.
        allowed_functions: Whitelist of allowed function names.
        blocked_functions: Blacklist of blocked function names.
        enable_rate_limiting: Whether to enable rate limiting.
        rate_limit_per_minute: Maximum requests per minute.
        rate_limit_per_hour: Maximum requests per hour.
        enable_authentication: Whether to require authentication.
        api_key: API key for authentication.
        api_key_env_var: Environment variable for API key.
    """

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
    """Configuration for LLM provider and model settings.

    Controls LLM provider selection, model parameters, and
    generation settings for agent responses.

    Attributes:
        provider: The LLM provider to use.
        model: Model identifier (e.g., 'gpt-4', 'claude-3').
        api_key: API key for the provider.
        api_key_env_var: Environment variable for API key.
        base_url: Optional custom base URL for API.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.
        frequency_penalty: Frequency penalty for repetition.
        presence_penalty: Presence penalty for repetition.
        repetition_penalty: Repetition penalty multiplier.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        enable_streaming: Whether to enable streaming responses.
        enable_caching: Whether to cache LLM responses.
    """

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
        """Validate or load API key from environment.

        Args:
            v: The API key value.
            info: Validation context information.

        Returns:
            The validated API key, loaded from environment if not provided.
        """
        if v is None:
            env_var = info.data.get("api_key_env_var", "OPENAI_API_KEY")
            v = os.getenv(env_var)
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging behavior.

    Controls log formatting, output destinations, and rotation
    settings for application logging.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Log message format string.
        file_path: Path to log file.
        enable_console: Whether to log to console.
        enable_file: Whether to log to file.
        max_file_size: Maximum log file size in bytes.
        backup_count: Number of backup log files to keep.
        enable_json_format: Whether to use JSON log format.
    """

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str | None = None
    enable_console: bool = True
    enable_file: bool = False
    max_file_size: int = Field(default=10485760, ge=1024, le=104857600)
    backup_count: int = Field(default=5, ge=1, le=100)
    enable_json_format: bool = False


class ObservabilityConfig(BaseModel):
    """Configuration for observability and monitoring.

    Controls tracing, metrics, profiling, and request/response
    logging for system observability.

    Attributes:
        enable_tracing: Whether to enable distributed tracing.
        enable_metrics: Whether to collect metrics.
        enable_profiling: Whether to enable performance profiling.
        trace_endpoint: Endpoint for trace collection.
        metrics_endpoint: Endpoint for metrics collection.
        service_name: Name of the service for identification.
        service_version: Version of the service.
        enable_request_logging: Whether to log requests.
        enable_response_logging: Whether to log responses.
        enable_function_logging: Whether to log function calls.
    """

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
    """Main Calute configuration container.

    Root configuration object that aggregates all configuration
    sections and provides methods for loading, saving, and merging
    configurations from various sources.

    Attributes:
        environment: Deployment environment type.
        debug: Whether debug mode is enabled.
        executor: Executor configuration section.
        memory: Memory configuration section.
        security: Security configuration section.
        llm: LLM configuration section.
        logging: Logging configuration section.
        observability: Observability configuration section.
        plugins: Plugin-specific configurations.
        features: Feature flags for enabling/disabling capabilities.
    """

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
    def from_file(cls, path: str | Path) -> "CaluteConfig":
        """Load configuration from a JSON or YAML file.

        Args:
            path: Path to the configuration file.

        Returns:
            CaluteConfig instance loaded from the file.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ImportError: If YAML file is specified but PyYAML is not installed.
            ValueError: If the file format is not supported.

        Example:
            >>> config = CaluteConfig.from_file("config.yaml")
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
    def from_env(cls, prefix: str = "CALUTE_") -> "CaluteConfig":
        """Load configuration from environment variables.

        Environment variables are parsed hierarchically using underscores
        as separators. JSON values are automatically parsed.

        Args:
            prefix: Prefix for environment variables (default: "CALUTE_").

        Returns:
            CaluteConfig instance loaded from environment variables.

        Example:
            >>>
            >>> config = CaluteConfig.from_env()
            >>> print(config.llm.model)
        """
        config_dict = {}

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
        """Save configuration to a JSON or YAML file.

        Args:
            path: Path where the configuration should be saved.

        Raises:
            ValueError: If the file format is not supported.

        Note:
            If YAML format is specified but PyYAML is not installed,
            the configuration will be saved as JSON instead.
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

    def merge(self, other: "CaluteConfig") -> "CaluteConfig":
        """Merge with another configuration.

        Creates a new configuration by deep merging this configuration
        with another. The other configuration takes precedence for
        conflicting values.

        Args:
            other: Configuration to merge with.

        Returns:
            New CaluteConfig with merged values.

        Example:
            >>> base_config = CaluteConfig.from_file("base.yaml")
            >>> override_config = CaluteConfig.from_file("override.yaml")
            >>> final_config = base_config.merge(override_config)
        """
        self_dict = self.model_dump()
        other_dict = other.model_dump()

        def deep_merge(dict1: dict, dict2: dict) -> dict:
            """Deep merge two dictionaries recursively.

            Args:
                dict1: Base dictionary.
                dict2: Dictionary to merge (takes precedence).

            Returns:
                Merged dictionary.
            """
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged = deep_merge(self_dict, other_dict)
        return CaluteConfig(**merged)


_config: CaluteConfig | None = None


def get_config() -> CaluteConfig:
    """Get the global configuration instance.

    Returns:
        The global CaluteConfig instance, creating a default one if needed.

    Example:
        >>> config = get_config()
        >>> config.llm.model = "gpt-4"
    """
    global _config
    if _config is None:
        _config = CaluteConfig()
    return _config


def set_config(config: CaluteConfig) -> None:
    """Set the global configuration instance.

    Args:
        config: The configuration to set as global.

    Example:
        >>> new_config = CaluteConfig(debug=True)
        >>> set_config(new_config)
    """
    global _config
    _config = config


def load_config(path: str | Path | None = None) -> CaluteConfig:
    """Load configuration from file or environment.

    Attempts to load configuration in the following order:
    1. From specified path
    2. From CALUTE_CONFIG_FILE environment variable
    3. From default file locations (current dir, home dir)
    4. From environment variables

    Args:
        path: Optional specific path to configuration file.

    Returns:
        Loaded CaluteConfig instance (also sets as global).

    Example:
        >>> config = load_config("my_config.yaml")
        >>>
        >>> config = load_config()
    """
    if path:
        config = CaluteConfig.from_file(path)
    elif os.getenv("CALUTE_CONFIG_FILE"):
        config = CaluteConfig.from_file(os.getenv("CALUTE_CONFIG_FILE"))
    else:
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
            config = CaluteConfig.from_env()

    set_config(config)
    return config
