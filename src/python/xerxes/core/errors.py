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
"""Typed exception hierarchy used across the framework.

All Xerxes-raised errors descend from :class:`XerxesError` and carry both a
human-readable message and an optional structured ``details`` dict for
audit/log consumers. Subclasses add domain-specific fields (agent id,
function name, timeout, retry-after, ...).
"""

from typing import Any


class XerxesError(Exception):
    """Base exception carrying a ``message`` and optional structured ``details`` dict."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Store ``message`` (passed to ``Exception``) and ``details``."""
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AgentError(XerxesError):
    """Raised when an agent fails mid-run; carries the offending ``agent_id``."""

    def __init__(self, agent_id: str, message: str, details: dict[str, Any] | None = None):
        """Build the error and prefix the message with ``Agent {agent_id}:``."""
        super().__init__(f"Agent {agent_id}: {message}", details)
        self.agent_id = agent_id


class FunctionExecutionError(XerxesError):
    """Raised when a tool/function execution fails; preserves the underlying exception."""

    def __init__(
        self,
        function_name: str,
        message: str,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Build the error, prefix the message with ``Function {function_name}:``, retain ``original_error``."""
        super().__init__(f"Function {function_name}: {message}", details)
        self.function_name = function_name
        self.original_error = original_error


class XerxesTimeoutError(XerxesError):
    """Raised when an operation exceeds its allotted time budget."""

    def __init__(self, operation: str, timeout: float, details: dict[str, Any] | None = None):
        """Build the error with a friendly ``operation`` + ``timeout`` message."""
        super().__init__(f"Operation {operation} timed out after {timeout} seconds", details)
        self.operation = operation
        self.timeout = timeout


class ValidationError(XerxesError):
    """Raised when input validation rejects a value; retains the offending ``value`` for logs."""

    def __init__(self, field: str, message: str, value: Any = None, details: dict[str, Any] | None = None):
        """Build the error with ``field`` and the rejected ``value``."""
        super().__init__(f"Validation error for {field}: {message}", details)
        self.field = field
        self.value = value


class RateLimitError(XerxesError):
    """Raised when a rate limit has been hit; carries an optional ``retry_after``."""

    def __init__(
        self,
        resource: str,
        limit: int,
        window: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Build the error with a ``resource``/``limit``/``window`` summary."""
        message = f"Rate limit exceeded for {resource}: {limit} per {window}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, details)
        self.resource = resource
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


class XerxesMemoryError(XerxesError):
    """Raised on failure in memory store/retrieve operations."""

    def __init__(self, operation: str, message: str, details: dict[str, Any] | None = None):
        """Build the error with a ``Memory operation {operation}:`` prefix."""
        super().__init__(f"Memory operation {operation}: {message}", details)
        self.operation = operation


class ClientError(XerxesError):
    """Raised by an external client integration (LLM/HTTP/etc.); retains ``original_error``."""

    def __init__(
        self,
        client_type: str,
        message: str,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Build the error and store ``client_type`` and the wrapped ``original_error``."""
        super().__init__(f"Client {client_type}: {message}", details)
        self.client_type = client_type
        self.original_error = original_error


class ConfigurationError(XerxesError):
    """Raised when configuration is missing or invalid; ``config_key`` names the field."""

    def __init__(self, config_key: str, message: str, details: dict[str, Any] | None = None):
        """Build the error with a ``Configuration {config_key}:`` prefix."""
        super().__init__(f"Configuration {config_key}: {message}", details)
        self.config_key = config_key


class AgentSpecError(XerxesError):
    """Raised when an agent YAML/JSON spec fails to parse or validate."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Build the error with ``message`` and optional ``details``."""
        super().__init__(message, details)
