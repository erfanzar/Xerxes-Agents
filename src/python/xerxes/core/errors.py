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
"""Xerxes exception hierarchy.

Provides a family of typed exceptions used across the framework, all derived
from ``XerxesError``.
"""

from typing import Any


class XerxesError(Exception):
    """Base exception for all Xerxes errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initialize the error.

        Args:
            message (str): IN: human-readable error description.
            details (dict[str, Any] | None): IN: optional structured error
                metadata.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AgentError(XerxesError):
    """Error raised by an agent during execution."""

    def __init__(self, agent_id: str, message: str, details: dict[str, Any] | None = None):
        """Initialize the error.

        Args:
            agent_id (str): IN: identifier of the agent that failed.
            message (str): IN: human-readable error description.
            details (dict[str, Any] | None): IN: optional structured metadata.
        """
        super().__init__(f"Agent {agent_id}: {message}", details)
        self.agent_id = agent_id


class FunctionExecutionError(XerxesError):
    """Error raised when a tool/function execution fails."""

    def __init__(
        self,
        function_name: str,
        message: str,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize the error.

        Args:
            function_name (str): IN: name of the function that failed.
            message (str): IN: human-readable error description.
            original_error (Exception | None): IN: underlying exception.
            details (dict[str, Any] | None): IN: optional structured metadata.
        """
        super().__init__(f"Function {function_name}: {message}", details)
        self.function_name = function_name
        self.original_error = original_error


class XerxesTimeoutError(XerxesError):
    """Error raised when an operation exceeds its time budget."""

    def __init__(self, operation: str, timeout: float, details: dict[str, Any] | None = None):
        """Initialize the error.

        Args:
            operation (str): IN: description of the timed-out operation.
            timeout (float): IN: timeout duration in seconds.
            details (dict[str, Any] | None): IN: optional structured metadata.
        """
        super().__init__(f"Operation {operation} timed out after {timeout} seconds", details)
        self.operation = operation
        self.timeout = timeout


class ValidationError(XerxesError):
    """Error raised when input validation fails."""

    def __init__(self, field: str, message: str, value: Any = None, details: dict[str, Any] | None = None):
        """Initialize the error.

        Args:
            field (str): IN: name of the field that failed validation.
            message (str): IN: human-readable validation description.
            value (Any): IN: the invalid value.
            details (dict[str, Any] | None): IN: optional structured metadata.
        """
        super().__init__(f"Validation error for {field}: {message}", details)
        self.field = field
        self.value = value


class RateLimitError(XerxesError):
    """Error raised when a rate limit is exceeded."""

    def __init__(
        self,
        resource: str,
        limit: int,
        window: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize the error.

        Args:
            resource (str): IN: name of the rate-limited resource.
            limit (int): IN: maximum allowed requests.
            window (str): IN: time window for the limit.
            retry_after (float | None): IN: seconds until the limit resets.
            details (dict[str, Any] | None): IN: optional structured metadata.
        """
        message = f"Rate limit exceeded for {resource}: {limit} per {window}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, details)
        self.resource = resource
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


class XerxesMemoryError(XerxesError):
    """Error raised during memory storage or retrieval operations."""

    def __init__(self, operation: str, message: str, details: dict[str, Any] | None = None):
        """Initialize the error.

        Args:
            operation (str): IN: memory operation that failed.
            message (str): IN: human-readable error description.
            details (dict[str, Any] | None): IN: optional structured metadata.
        """
        super().__init__(f"Memory operation {operation}: {message}", details)
        self.operation = operation


class ClientError(XerxesError):
    """Error raised by an external client integration."""

    def __init__(
        self,
        client_type: str,
        message: str,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize the error.

        Args:
            client_type (str): IN: type/name of the client that failed.
            message (str): IN: human-readable error description.
            original_error (Exception | None): IN: underlying exception.
            details (dict[str, Any] | None): IN: optional structured metadata.
        """
        super().__init__(f"Client {client_type}: {message}", details)
        self.client_type = client_type
        self.original_error = original_error


class ConfigurationError(XerxesError):
    """Error raised when configuration is invalid or missing."""

    def __init__(self, config_key: str, message: str, details: dict[str, Any] | None = None):
        """Initialize the error.

        Args:
            config_key (str): IN: configuration key that is invalid.
            message (str): IN: human-readable error description.
            details (dict[str, Any] | None): IN: optional structured metadata.
        """
        super().__init__(f"Configuration {config_key}: {message}", details)
        self.config_key = config_key


class AgentSpecError(XerxesError):
    """Error raised when an agent specification is malformed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initialize the error.

        Args:
            message (str): IN: human-readable error description.
            details (dict[str, Any] | None): IN: optional structured metadata.
        """
        super().__init__(message, details)
