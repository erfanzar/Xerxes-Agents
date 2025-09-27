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


"""Custom error types for better error handling in Calute.

This module defines a hierarchy of exception classes for precise error
handling throughout the Calute framework. All exceptions inherit from
CaluteError and provide detailed error information including context
and debugging details.

The error hierarchy allows for:
- Specific error catching at different levels
- Detailed error messages with context
- Preservation of original exceptions when wrapping
- Structured error details for debugging

Example:
    >>> try:
    ...
    ... except FunctionExecutionError as e:
    ...     print(f"Function {e.function_name} failed: {e.message}")
    ...     if e.original_error:
    ...         print(f"Original error: {e.original_error}")
"""

from typing import Any


class CaluteError(Exception):
    """Base exception for all Calute errors.

    This is the root exception class for the Calute framework.
    All custom exceptions should inherit from this class.

    Attributes:
        message: Human-readable error message.
        details: Additional structured error details for debugging.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Initialize CaluteError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class AgentError(CaluteError):
    """Errors related to agent operations.

    Raised when agent-specific operations fail, such as agent
    initialization, switching, or execution errors.

    Attributes:
        agent_id: The ID of the agent that encountered the error.
        message: Human-readable error message.
        details: Additional structured error details.
    """

    def __init__(self, agent_id: str, message: str, details: dict[str, Any] | None = None):
        """Initialize AgentError.

        Args:
            agent_id: The ID of the agent involved in the error.
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(f"Agent {agent_id}: {message}", details)
        self.agent_id = agent_id


class FunctionExecutionError(CaluteError):
    """Errors during function execution.

    Raised when a function/tool call fails during execution.
    Preserves the original exception for debugging.

    Attributes:
        function_name: Name of the function that failed.
        message: Human-readable error message.
        original_error: The original exception that caused the failure.
        details: Additional structured error details.
    """

    def __init__(
        self,
        function_name: str,
        message: str,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize FunctionExecutionError.

        Args:
            function_name: Name of the function that failed.
            message: The error message.
            original_error: Optional original exception that was caught.
            details: Optional dictionary with additional error details.
        """
        super().__init__(f"Function {function_name}: {message}", details)
        self.function_name = function_name
        self.original_error = original_error


class CaluteTimeoutError(CaluteError):
    """Function or operation timeout.

    Raised when an operation exceeds its configured timeout duration.

    Attributes:
        operation: Name or description of the operation that timed out.
        timeout: The timeout duration in seconds that was exceeded.
        details: Additional structured error details.
    """

    def __init__(self, operation: str, timeout: float, details: dict[str, Any] | None = None):
        """Initialize CaluteTimeoutError.

        Args:
            operation: Name or description of the operation.
            timeout: The timeout value that was exceeded (in seconds).
            details: Optional dictionary with additional error details.
        """
        super().__init__(f"Operation {operation} timed out after {timeout} seconds", details)
        self.operation = operation
        self.timeout = timeout


class ValidationError(CaluteError):
    """Input validation errors.

    Raised when input validation fails for parameters, configurations,
    or user inputs.

    Attributes:
        field: The field or parameter that failed validation.
        message: Human-readable validation error message.
        value: The actual value that failed validation.
        details: Additional structured error details.
    """

    def __init__(self, field: str, message: str, value: Any = None, details: dict[str, Any] | None = None):
        """Initialize ValidationError.

        Args:
            field: Name of the field that failed validation.
            message: The validation error message.
            value: Optional value that failed validation.
            details: Optional dictionary with additional error details.
        """
        super().__init__(f"Validation error for {field}: {message}", details)
        self.field = field
        self.value = value


class RateLimitError(CaluteError):
    """Rate limit exceeded.

    Raised when a rate limit is exceeded for API calls or operations.

    Attributes:
        resource: The resource or endpoint that is rate limited.
        limit: The rate limit that was exceeded.
        window: The time window for the rate limit (e.g., 'minute', 'hour').
        retry_after: Optional time in seconds to wait before retrying.
        details: Additional structured error details.
    """

    def __init__(
        self,
        resource: str,
        limit: int,
        window: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize RateLimitError.

        Args:
            resource: Name of the rate-limited resource.
            limit: The rate limit value.
            window: Time window for the limit (e.g., 'minute', 'hour').
            retry_after: Optional seconds to wait before retry.
            details: Optional dictionary with additional error details.
        """
        message = f"Rate limit exceeded for {resource}: {limit} per {window}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, details)
        self.resource = resource
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


class CaluteMemoryError(CaluteError):
    """Memory store errors.

    Raised when memory operations fail, including storage, retrieval,
    or consolidation errors.

    Attributes:
        operation: The memory operation that failed.
        message: Human-readable error message.
        details: Additional structured error details.
    """

    def __init__(self, operation: str, message: str, details: dict[str, Any] | None = None):
        """Initialize CaluteMemoryError.

        Args:
            operation: Name of the memory operation that failed.
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(f"Memory operation {operation}: {message}", details)
        self.operation = operation


class ClientError(CaluteError):
    """LLM client errors.

    Raised when LLM client operations fail, including API errors,
    connection issues, or response parsing errors.

    Attributes:
        client_type: Type of the LLM client (e.g., 'openai', 'anthropic').
        message: Human-readable error message.
        original_error: The original exception from the client library.
        details: Additional structured error details.
    """

    def __init__(
        self,
        client_type: str,
        message: str,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize ClientError.

        Args:
            client_type: Type of the client that encountered the error.
            message: The error message.
            original_error: Optional original exception from the client.
            details: Optional dictionary with additional error details.
        """
        super().__init__(f"Client {client_type}: {message}", details)
        self.client_type = client_type
        self.original_error = original_error


class ConfigurationError(CaluteError):
    """Configuration errors.

    Raised when configuration is invalid, missing, or cannot be loaded.

    Attributes:
        config_key: The configuration key or section that has issues.
        message: Human-readable error message.
        details: Additional structured error details.
    """

    def __init__(self, config_key: str, message: str, details: dict[str, Any] | None = None):
        """Initialize ConfigurationError.

        Args:
            config_key: Configuration key or section with the error.
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(f"Configuration {config_key}: {message}", details)
        self.config_key = config_key
