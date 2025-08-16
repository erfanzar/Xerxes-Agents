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

"""Custom error types for better error handling."""

from typing import Any


class CaluteError(Exception):
    """Base exception for all Calute errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AgentError(CaluteError):
    """Errors related to agent operations."""

    def __init__(self, agent_id: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"Agent {agent_id}: {message}", details)
        self.agent_id = agent_id


class FunctionExecutionError(CaluteError):
    """Errors during function execution."""

    def __init__(
        self,
        function_name: str,
        message: str,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(f"Function {function_name}: {message}", details)
        self.function_name = function_name
        self.original_error = original_error


class CaluteTimeoutError(CaluteError):
    """Function or operation timeout."""

    def __init__(self, operation: str, timeout: float, details: dict[str, Any] | None = None):
        super().__init__(f"Operation {operation} timed out after {timeout} seconds", details)
        self.operation = operation
        self.timeout = timeout


class ValidationError(CaluteError):
    """Input validation errors."""

    def __init__(self, field: str, message: str, value: Any = None, details: dict[str, Any] | None = None):
        super().__init__(f"Validation error for {field}: {message}", details)
        self.field = field
        self.value = value


class RateLimitError(CaluteError):
    """Rate limit exceeded."""

    def __init__(
        self,
        resource: str,
        limit: int,
        window: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        message = f"Rate limit exceeded for {resource}: {limit} per {window}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, details)
        self.resource = resource
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


class CaluteMemoryError(CaluteError):
    """Memory store errors."""

    def __init__(self, operation: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"Memory operation {operation}: {message}", details)
        self.operation = operation


class ClientError(CaluteError):
    """LLM client errors."""

    def __init__(
        self,
        client_type: str,
        message: str,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(f"Client {client_type}: {message}", details)
        self.client_type = client_type
        self.original_error = original_error


class ConfigurationError(CaluteError):
    """Configuration errors."""

    def __init__(self, config_key: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"Configuration {config_key}: {message}", details)
        self.config_key = config_key
