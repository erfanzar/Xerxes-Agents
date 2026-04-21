# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Task definition for Cortex framework.

This module provides the CortexTask class and related data structures for defining
and executing tasks within the Cortex multi-agent orchestration framework. Tasks
represent units of work that can be assigned to agents, with support for features
like output validation, chaining, retries, and memory integration.

The module includes:
- CortexTask: Main task class for defining executable work units
- CortexTaskOutput: Rich output container with execution metadata
- ChainLink: Task chaining and conditional routing
- TaskValidationError: Custom exception for validation failures

Key features:
- Output validation with Pydantic models (JSON and XML parsing)
- Task chaining with conditional routing
- Automatic retry with configurable conditions
- Human input/feedback integration
- Memory persistence for results
- Security configurations and tool restrictions
- Context compression and priority weighting
- Streaming execution support
- Dependency management between tasks

Typical usage example:
    from xerxes.cortex.task import CortexTask
    from xerxes.cortex.agent import CortexAgent

    agent = CortexAgent(
        role="Data Analyst",
        goal="Analyze data",
        backstory="Expert analyst"
    )

    task = CortexTask(
        description="Analyze quarterly sales data",
        expected_output="Summary report with key insights",
        agent=agent,
        max_retries=3,
        importance=0.8
    )

    result = task.execute()
    print(result.output)
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from ...core.streamer_buffer import StreamerBuffer
from ...logging.console import log_error, log_retry, log_task_complete, log_task_start, log_warning
from ..core.string_utils import interpolate_inputs
from ..core.tool import CortexTool

if TYPE_CHECKING:
    from ..agents.agent import CortexAgent
    from ..agents.memory_integration import CortexMemory


class TaskValidationError(Exception):
    """Exception raised when task output validation fails.

    This exception is raised when a task's output fails to validate against
    the specified Pydantic model or JSON schema, after all retry attempts
    have been exhausted.

    Attributes:
        message: Detailed description of the validation failure, including
            which validation rules failed and any parsing errors encountered.
    """

    def __init__(self, message: str):
        """Initialize the validation error.

        Args:
            message: Description of the validation failure.
        """
        self.message = message
        super().__init__(self.message)


@dataclass
class ChainLink:
    """Represents a link in a task chain for conditional task routing.

    ChainLink enables creating conditional workflows where the next task
    to execute depends on the output of the current task. It supports
    both success and fallback paths, allowing for branching task execution.

    Attributes:
        condition: Optional callable that takes the task output string and
            returns True if the next_task should execute, False for fallback.
            If None, next_task is always executed.
        next_task: The task to execute if condition returns True or is None.
        fallback_task: The task to execute if condition returns False.

    Example:
        >>> def check_success(output: str) -> bool:
        ...     return "success" in output.lower()
        >>> chain = ChainLink(
        ...     condition=check_success,
        ...     next_task=process_results_task,
        ...     fallback_task=handle_error_task
        ... )
    """

    condition: Callable[[str], bool] | None = None
    next_task: CortexTask | None = None
    fallback_task: CortexTask | None = None


@dataclass
class CortexTaskOutput:
    """Enhanced output container from a completed task with rich metadata.

    CortexTaskOutput encapsulates the result of task execution along with
    comprehensive metadata about the execution process, including timing,
    validation results, tool usage statistics, and performance metrics.

    Attributes:
        task: Reference to the CortexTask that produced this output.
        output: The primary output string from task execution.
        agent: The CortexAgent that executed the task.
        timestamp: Unix timestamp when the output was generated.
        raw_output: Unprocessed output before any formatting or validation.
        token_usage: Dictionary tracking token consumption (e.g., prompt_tokens,
            completion_tokens, total_tokens).
        validation_results: Dictionary of validation check results. Keys are
            validation type names, values are boolean pass/fail or error strings.
        pydantic_output: Parsed and validated Pydantic model instance if
            output_json or output_pydantic was specified on the task.
        json_dict: Dictionary representation if output was valid JSON.
        execution_time: Total execution duration in seconds.
        used_tools: Count of tool invocations during execution.
        tools_errors: Count of tool invocation errors encountered.
        delegations: Number of delegations to other agents.
        retry_count: Number of retry attempts before success.
        execution_metadata: Additional metadata about execution context
            (e.g., had_human_input, security_applied, validation_applied).
        performance_metrics: Aggregated performance statistics
            (e.g., avg_execution_time, total_retries).
    """

    task: CortexTask
    output: str
    agent: CortexAgent
    timestamp: float = field(default_factory=time.time)
    raw_output: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    validation_results: dict[str, bool] = field(default_factory=dict)
    pydantic_output: object | None = None
    json_dict: dict | None = None

    execution_time: float = 0.0
    used_tools: int = 0
    tools_errors: int = 0
    delegations: int = 0
    retry_count: int = 0
    execution_metadata: dict = field(default_factory=dict)
    performance_metrics: dict = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Get a truncated summary of the output.

        Returns:
            The first 200 characters of the output with ellipsis if truncated,
            or the full output if shorter than 200 characters.
        """
        return self.output[:200] + "..." if len(self.output) > 200 else self.output

    def to_dict(self) -> dict:
        """Convert the task output to a dictionary representation.

        Returns:
            Dictionary containing all task output data including task details,
            agent information, execution metrics, and validation results.
            Suitable for serialization or logging purposes.
        """
        return {
            "task_description": self.task.description,
            "expected_output": self.task.expected_output,
            "actual_output": self.output,
            "agent_role": self.agent.role,
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage,
            "validation_results": self.validation_results,
            "used_tools": self.used_tools,
            "tools_errors": self.tools_errors,
            "delegations": self.delegations,
            "retry_count": self.retry_count,
            "execution_metadata": self.execution_metadata,
            "performance_metrics": self.performance_metrics,
        }

    def __str__(self) -> str:
        """Return a formatted string showing agent, task, output, and execution time.

        Returns:
            Multi-line string with agent role, truncated task description,
            output summary, and execution duration.
        """
        return f"""Task Output:
        Agent: {self.agent.role}
        Task: {self.task.description[:256]}...
        Output: {self.summary}
        Execution Time: {self.execution_time:.2f}s
        """

    def __repr__(self) -> str:
        """Return a concise developer-facing representation of the output.

        Returns:
            String showing agent role and total output length.
        """
        return f"CortexTaskOutput(agent={self.agent.role}, output_length={len(self.output)})"


@dataclass
class CortexTask:
    """Task definition for execution within the Cortex framework.

    CortexTask represents a unit of work to be executed by a CortexAgent.
    It provides comprehensive configuration options for task execution
    including output validation, retry logic, human interaction, security
    controls, and context management.

    Attributes:
        description: Detailed description of what the task should accomplish.
        expected_output: Description of what successful output should look like.
        agent: The CortexAgent assigned to execute this task.
        tools: List of CortexTool instances available for this specific task.
        context: List of prerequisite tasks whose outputs provide context.
        output_file: Optional file path to save the task output.
        human_feedback: Whether to request human feedback after execution.
        chain: Optional ChainLink for conditional task routing.
        max_retries: Maximum retry attempts on failure (default: 3).
        memory: CortexMemory instance for persisting task results.
        save_to_memory: Whether to automatically save results to memory.
        importance: Task importance score from 0.0 to 1.0 for prioritization.
        output_json: Pydantic model class for JSON output validation.
        output_pydantic: Pydantic model class for structured output parsing.
        create_directory: Whether to create output directory if missing.
        async_execution: Whether to execute asynchronously (reserved).
        callback: Function called after successful task completion.
        pre_execution_callback: Function called before task execution starts.
        error_callback: Function called when an error occurs.
        human_input: Whether to prompt for human input during execution.
        human_input_prompt: Custom prompt for human input requests.
        input_validator: Function to validate human input before accepting.
        security_config: Dictionary of security-related configuration options.
        tool_restrictions: List of allowed tool names (restricts agent tools).
        allow_dangerous_tools: Whether to permit dangerous tool usage.
        prompt_context: Additional context string to include in the prompt.
        context_priority: Priority weights for context sources (higher = more important).
        context_compression: Whether to compress context when exceeding limits.
        max_context_length: Maximum character length for context (default: 10000).
        dependencies: List of tasks that must complete before this task.
        conditional_execution: Function that returns whether to execute the task.
        retry_conditions: List of functions determining retry eligibility.
        timeout_behavior: Action on timeout - 'fail', 'continue', or 'return_partial'.
        timeout: Maximum execution time in seconds before timeout.

    Private Attributes:
        _output: Cached output string from last execution.
        _execution_stats: Dictionary tracking execution statistics.
        _start_time: Timestamp when execution started.
        _original_description: Original description before interpolation.
        _original_expected_output: Original expected_output before interpolation.
        _original_output_file: Original output_file before interpolation.
        _original_prompt_context: Original prompt_context before interpolation.
    """

    description: str
    expected_output: str
    agent: CortexAgent | None = None
    tools: list[CortexTool] = field(default_factory=list)
    context: list[CortexTask] | None = None
    output_file: str | None = None
    human_feedback: bool = False
    chain: ChainLink | None = None
    max_retries: int = 3
    memory: CortexMemory | None = None
    save_to_memory: bool = True
    importance: float = 0.5

    output_json: type[BaseModel] | None = None
    output_pydantic: type[BaseModel] | None = None
    create_directory: bool = True
    async_execution: bool = False

    callback: Callable | None = None
    pre_execution_callback: Callable | None = None
    error_callback: Callable | None = None

    human_input: bool = False
    human_input_prompt: str | None = None
    input_validator: Callable | None = None

    security_config: dict | None = None
    tool_restrictions: list[str] | None = None
    allow_dangerous_tools: bool = False

    prompt_context: str | None = None
    context_priority: dict[str, float] = field(default_factory=dict)
    context_compression: bool = False
    max_context_length: int = 10000

    dependencies: list[CortexTask] = field(default_factory=list)
    conditional_execution: Callable | None = None
    retry_conditions: list[Callable] = field(default_factory=list)
    timeout_behavior: Literal["fail", "continue", "return_partial"] = "fail"
    timeout: int | None = None

    _output: str | None = None
    _execution_stats: dict = field(default_factory=dict)
    _start_time: float = 0.0

    _original_description: str | None = None
    _original_expected_output: str | None = None
    _original_output_file: str | None = None
    _original_prompt_context: str | None = None

    def interpolate_inputs(self, inputs: dict[str, Any]) -> None:
        """
        Interpolate inputs into the task's description, expected output, and other fields.

        This method replaces template variables (e.g., {variable_name}) in the task's
        attributes with values from the provided inputs dictionary. Original values
        are preserved for potential re-interpolation.

        Args:
            inputs: Dictionary mapping template variables to their values.
                   Supported value types are strings, integers, floats, bools,
                   and serializable dicts/lists.

        Side Effects:
            Updates description, expected_output, output_file, and prompt_context
            with interpolated values. Stores original values if not already saved.

        Example:
            >>> task = CortexTask(
            ...     description="Analyze {topic} trends in {year}",
            ...     expected_output="Report on {topic} developments"
            ... )
            >>> task.interpolate_inputs({"topic": "AI", "year": 2025})


        """

        if self._original_description is None:
            self._original_description = self.description
        if self._original_expected_output is None:
            self._original_expected_output = self.expected_output
        if self.output_file is not None and self._original_output_file is None:
            self._original_output_file = self.output_file
        if self.prompt_context is not None and self._original_prompt_context is None:
            self._original_prompt_context = self.prompt_context

        if inputs:
            self.description = interpolate_inputs(input_string=self._original_description, inputs=inputs)
            self.expected_output = interpolate_inputs(input_string=self._original_expected_output, inputs=inputs)
            if self.output_file:
                self.output_file = interpolate_inputs(input_string=self._original_output_file, inputs=inputs)
            if self.prompt_context:
                self.prompt_context = interpolate_inputs(input_string=self._original_prompt_context, inputs=inputs)

    def _extract_json_from_output(self, output: str) -> str | None:
        """Extract JSON from output that may contain thinking process or other text.

        Attempts multiple extraction strategies to find valid JSON:
        1. JSON within markdown code blocks (```json or ```)
        2. Regex patterns for nested JSON objects
        3. Character-by-character brace matching

        Args:
            output: The raw output string potentially containing JSON.

        Returns:
            The extracted JSON string if found and valid, None otherwise.
        """
        import re

        json_patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, output, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue

        brace_level = 0
        start_idx = -1

        for i, char in enumerate(output):
            if char == "{":
                if brace_level == 0:
                    start_idx = i
                brace_level += 1
            elif char == "}":
                brace_level -= 1
                if brace_level == 0 and start_idx != -1:
                    potential_json = output[start_idx : i + 1]
                    try:
                        json.loads(potential_json)
                        return potential_json
                    except json.JSONDecodeError:
                        continue

        return None

    def _extract_xml_content(self, output: str, tag: str) -> str | None:
        """Extract content from XML tags in the output.

        Args:
            output: The raw output string containing XML.
            tag: The XML tag name to extract content from.

        Returns:
            The content between the specified tags (stripped of whitespace),
            or None if the tag is not found.
        """
        import re

        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, output, re.DOTALL)
        return match.group(1).strip() if match else None

    def _parse_xml_to_dict(self, xml_content: str) -> dict:
        """Parse XML content to dictionary for Pydantic validation.

        Recursively converts XML elements to a nested dictionary structure.
        Handles repeated elements by converting them to lists.

        Args:
            xml_content: The XML string to parse.

        Returns:
            Dictionary representation of the XML structure.
            Returns empty dict if parsing fails.
        """
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(f"<root>{xml_content}</root>")

            def xml_to_dict(element) -> dict[str, Any]:
                """Recursively convert an XML element tree to a nested dictionary.

                Args:
                    element: An xml.etree.ElementTree.Element to convert.

                Returns:
                    Dictionary representation of the element's children.
                    Repeated tags are converted to lists.
                """
                result: dict[str, Any] = {}
                for child in element:
                    if len(child) > 0:
                        if child.tag in result:
                            if not isinstance(result[child.tag], list):
                                result[child.tag] = [result[child.tag]]
                            result[child.tag].append(xml_to_dict(child))
                        else:
                            result[child.tag] = xml_to_dict(child)
                    else:
                        if child.tag in result:
                            if not isinstance(result[child.tag], list):
                                result[child.tag] = [result[child.tag]]
                            result[child.tag].append(child.text or "")
                        else:
                            result[child.tag] = child.text or ""
                return result

            return xml_to_dict(root)
        except ET.ParseError:
            return {}

    def _validate_output(self, output: str) -> tuple[bool, Any, dict[str, Any]]:
        """Validate task output against configured Pydantic models.

        Supports multiple extraction strategies for finding structured data
        within unstructured output, including JSON and XML parsing.

        Args:
            output: The raw output string to validate.

        Returns:
            Tuple of (validation_passed, pydantic_output, validation_results):
            - validation_passed: True if all validations succeeded.
            - pydantic_output: Parsed Pydantic model instance if successful.
            - validation_results: Dictionary with validation details and any errors.
        """
        validation_results: dict[str, Any] = {}
        pydantic_output = None

        if self.output_json:
            json_data = None

            try:
                json_data = json.loads(output)
                validation_results["extraction_method"] = "direct_json"
            except json.JSONDecodeError:
                extracted_json = self._extract_json_from_output(output)
                if extracted_json:
                    try:
                        json_data = json.loads(extracted_json)
                        validation_results["extraction_method"] = "extracted_json"
                        validation_results["extracted_content"] = (
                            extracted_json[:200] + "..." if len(extracted_json) > 200 else extracted_json
                        )
                    except json.JSONDecodeError:
                        pass

                if not json_data:
                    xml_content = self._extract_xml_content(output, "json") or self._extract_xml_content(
                        output, "output"
                    )
                    if xml_content:
                        try:
                            json_data = json.loads(xml_content)
                            validation_results["extraction_method"] = "xml_json"
                        except json.JSONDecodeError:
                            dict_data = self._parse_xml_to_dict(xml_content)
                            if dict_data:
                                json_data = dict_data
                                validation_results["extraction_method"] = "xml_dict"

            if json_data:
                try:
                    pydantic_output = self.output_json.parse_obj(json_data)
                    validation_results["output_json"] = True
                except ValidationError as e:
                    validation_results["output_json"] = False
                    validation_results["output_json_error"] = str(e)
            else:
                validation_results["output_json"] = False
                validation_results["output_json_error"] = "No valid JSON found in output"

        if self.output_pydantic:
            try:
                pydantic_output = self.output_pydantic.parse_raw(output)
                validation_results["output_pydantic"] = True
            except ValidationError as e:
                validation_results["output_pydantic"] = False
                validation_results["output_pydantic_error"] = str(e)

        validation_passed = all(
            result
            for key, result in validation_results.items()
            if not key.endswith("_error") and key not in ["extraction_method", "extracted_content"]
        )

        return validation_passed, pydantic_output, validation_results

    def _execute_callback(self, callback: Callable | None, *args, **kwargs):
        """Safely execute a callback function with error handling.

        If the callback raises an exception and an error_callback is configured,
        the error is passed to the error_callback. Otherwise, the error is printed.

        Args:
            callback: The callback function to execute, or None to skip.
            *args: Positional arguments to pass to the callback.
            **kwargs: Keyword arguments to pass to the callback.

        Returns:
            The return value of the callback, or None if callback is None
            or if an exception occurred.
        """
        if callback and callable(callback):
            try:
                return callback(*args, **kwargs)
            except Exception as e:
                if self.error_callback:
                    self._execute_callback(self.error_callback, e, self)
                else:
                    print(f"Callback error: {e}")

    def _check_dependencies(self) -> bool:
        """Check if all task dependencies have been completed.

        Returns:
            True if all dependencies have output (are completed),
            False if any dependency is still pending.
        """
        for dep in self.dependencies:
            if not dep.output:
                return False
        return True

    def _should_retry(self, error: Exception, retry_count: int) -> bool:
        """Determine if the task should be retried after an error.

        Checks against max_retries limit and evaluates any custom retry
        condition functions that have been configured.

        Args:
            error: The exception that caused the failure.
            retry_count: Current number of retry attempts made.

        Returns:
            True if retry should be attempted, False otherwise.
        """
        if retry_count >= self.max_retries:
            return False

        for condition in self.retry_conditions:
            try:
                if not condition(error, retry_count, self):
                    return False
            except Exception:
                continue

        return True

    def _get_human_input(self) -> str:
        """Prompt for and collect human input with optional validation.

        Displays the configured prompt and waits for user input. If an
        input_validator is configured, continues prompting until valid
        input is received.

        Returns:
            The validated user input string.
        """
        prompt = self.human_input_prompt or "Please provide input for this task: "

        while True:
            user_input = input(f"\n🤔 {prompt}")

            if self.input_validator:
                try:
                    if self.input_validator(user_input):
                        return user_input
                    else:
                        print("❌ Input validation failed. Please try again.")
                        continue
                except Exception as e:
                    print(f"❌ Input validation error: {e}. Please try again.")
                    continue

            return user_input

    def _create_output_directory(self):
        """Create the output file's parent directory if it doesn't exist.

        Only creates directories if both output_file is specified and
        create_directory is True. Creates all intermediate directories
        as needed (like mkdir -p).
        """
        if self.output_file and self.create_directory:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

    def _apply_tool_restrictions(self):
        """Apply tool restrictions to the assigned agent.

        Filters the agent's available tools to only include those
        specified in tool_restrictions. This provides security control
        over which tools can be used during task execution.

        Side Effects:
            Modifies the agent's tools list to only include allowed tools.
        """
        if not self.tool_restrictions:
            return

        if self.agent:
            allowed_tools = []
            for tool in self.agent.tools:
                tool_name = tool.__class__.__name__
                if tool_name in self.tool_restrictions:
                    allowed_tools.append(tool)

            self.agent.tools = allowed_tools

    def _build_enhanced_context(self, context: str, context_outputs: list[str] | None, human_input: str) -> str:
        """Build enhanced context with priority weighting and optional compression.

        Combines context from multiple sources (previous task outputs, base context,
        human input) and orders them by priority. Optionally compresses the result
        if it exceeds max_context_length.

        Args:
            context: Base context string from previous outputs.
            context_outputs: List of output strings from context tasks.
            human_input: Optional human input to include in context.

        Returns:
            Combined context string, sorted by priority and optionally compressed.
        """
        context_parts = []

        if context_outputs:
            for i, ctx_output in enumerate(context_outputs):
                priority = self.context_priority.get(f"context_{i}", 1.0)
                if priority > 0:
                    context_parts.append((ctx_output, priority))

        if context:
            priority = self.context_priority.get("base_context", 1.0)
            context_parts.append((context, priority))

        if human_input:
            priority = self.context_priority.get("human_input", 1.5)
            context_parts.append((f"Human Input: {human_input}", priority))

        context_parts.sort(key=lambda x: x[1], reverse=True)

        final_context = "\n\n".join([part[0] for part in context_parts])

        if self.context_compression and len(final_context) > self.max_context_length:
            compressed_parts = []
            current_length = 0

            for ctx, _priority in context_parts:
                if current_length + len(ctx) <= self.max_context_length:
                    compressed_parts.append(ctx)
                    current_length += len(ctx)
                else:
                    remaining = self.max_context_length - current_length
                    if remaining > 100:
                        compressed_parts.append(ctx[:remaining] + "...")
                    break

            final_context = "\n\n".join(compressed_parts)

        return final_context

    def _create_empty_output(self, reason: str) -> CortexTaskOutput:
        """Create an empty output for tasks that were skipped.

        Used when a task is not executed (e.g., conditional execution
        returned False or dependencies weren't met).

        Args:
            reason: Description of why the task was skipped.

        Returns:
            CortexTaskOutput with the reason as output and skipped=True
            in execution_metadata.
        """
        if self.agent is None:
            raise ValueError("Task must have an assigned agent")
        return CortexTaskOutput(
            task=self,
            output=reason,
            agent=self.agent,
            timestamp=time.time(),
            execution_metadata={"skipped": True, "reason": reason},
        )

    def execute(
        self,
        context_outputs: list[str] | None = None,
        use_streaming: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> CortexTaskOutput | tuple[StreamerBuffer, threading.Thread]:
        """Execute the task with full Cortex feature support and optional streaming.

        Main execution method for CortexTask. Handles the complete task lifecycle
        including dependency checking, tool restriction enforcement, callback
        invocation, context building, output validation, retry logic, human
        interaction, memory persistence, and file output.

        Args:
            context_outputs: Optional list of output strings from previously
                completed tasks to include as context. These are combined with
                any additional context sources (human input, prompt_context)
                and prioritized according to ``context_priority`` weights.
            use_streaming: If True, executes in a background thread with
                real-time streaming output. Returns immediately with a
                buffer/thread tuple. Defaults to False (blocking execution).
            stream_callback: Optional callback function invoked with each
                streaming chunk. Only used when ``use_streaming=True``.

        Returns:
            If ``use_streaming=False``: A CortexTaskOutput containing the result
            string, execution metrics, validation results, and metadata.
            If ``use_streaming=True``: A tuple of (StreamerBuffer, Thread) for
            asynchronous consumption of the streaming response.

        Raises:
            ValueError: If no agent is assigned or dependencies are not satisfied.
            TimeoutError: If execution exceeds the configured ``timeout`` and
                ``timeout_behavior`` is "fail".
            TaskValidationError: If output validation fails after all retries.
            Exception: If task fails after all retry attempts and
                ``timeout_behavior`` is not "return_partial".

        Side Effects:
            - Invokes ``pre_execution_callback`` before execution.
            - Invokes ``callback`` after successful completion.
            - Invokes ``error_callback`` on failure.
            - Saves results to memory if ``save_to_memory`` is True.
            - Writes output to ``output_file`` if configured.
            - Creates output directories if ``create_directory`` is True.
            - Applies tool restrictions to the agent if configured.
            - Requests human feedback if ``human_feedback`` is enabled.
        """
        if not self.agent:
            raise ValueError("Task must have an assigned agent")

        if self.dependencies and not self._check_dependencies():
            raise ValueError("Task dependencies not satisfied")

        log_task_start(
            self.description[:50] + "..." if len(self.description) > 50 else self.description, self.agent.role
        )

        self._apply_tool_restrictions()

        self._execute_callback(self.pre_execution_callback, self)

        self._create_output_directory()

        for tool in self.tools:
            if tool not in self.agent.tools:
                self.agent.tools.append(tool)

        context = ""
        if context_outputs:
            context = "\n\n".join(context_outputs)

        retries = 0
        last_error = None
        start_time = time.time()
        self._start_time = start_time

        self._execution_stats = {
            "used_tools": 0,
            "tools_errors": 0,
            "delegations": 0,
            "retry_count": 0,
        }

        while retries <= self.max_retries:
            try:
                if self.timeout and (time.time() - start_time) > self.timeout:
                    if self.timeout_behavior == "fail":
                        raise TimeoutError(f"Task timeout after {self.timeout}s")
                    elif self.timeout_behavior == "continue":
                        break

                if self.conditional_execution and not self.conditional_execution(self):
                    return self._create_empty_output("Conditional execution failed")

                human_input_text = ""
                if self.human_input:
                    human_input_text = self._get_human_input()

                enhanced_context = self._build_enhanced_context(context, context_outputs, human_input_text)

                task_prompt = f"{self.description}\n\nExpected Output: {self.expected_output}"
                if self.prompt_context:
                    task_prompt += f"\n\nAdditional Context: {self.prompt_context}"

                if hasattr(self.agent, "_generate_format_guidance") and (self.output_json or self.output_pydantic):
                    output_model = self.output_json or self.output_pydantic
                    format_guidance = self.agent._generate_format_guidance(output_model)
                    if format_guidance:
                        task_prompt += format_guidance

                initial_delegations = getattr(self.agent, "_delegation_count", 0)

                if use_streaming:
                    exec_result = self.agent.execute(
                        task_description=task_prompt,
                        context=enhanced_context,
                        use_thread=True,
                    )
                    if isinstance(exec_result, str):
                        from xerxes.types import StreamChunk

                        buffer = StreamerBuffer()
                        thread = threading.Thread(target=lambda: None, daemon=True)
                        buffer.put(
                            StreamChunk(
                                chunk=None,
                                agent_id="cortex",
                                content=exec_result,
                                buffered_content=exec_result,
                                function_calls_detected=False,
                                reinvoked=False,
                            )
                        )
                    else:
                        buffer, thread = exec_result

                    if stream_callback:

                        def process_stream(buffer: StreamerBuffer = buffer) -> None:
                            """Consume streaming chunks from the buffer and forward each to the callback."""
                            for chunk in buffer.stream():
                                stream_callback(chunk)

                        callback_thread = threading.Thread(target=process_stream, daemon=True)
                        callback_thread.start()

                    setattr(buffer, "task", self)
                    setattr(buffer, "agent", self.agent)

                    return buffer, thread

                if self.agent.allow_delegation:
                    delegated = self.agent.execute_with_delegation(
                        task_description=task_prompt, context=enhanced_context
                    )
                    result = delegated
                else:
                    executed = self.agent.execute(task_description=task_prompt, context=enhanced_context)
                    result = (
                        executed
                        if isinstance(executed, str)
                        else executed[0].get_result(1.0)
                        if (executed[0].get_result is not None)
                        else str(executed[0])
                    )

                final_delegations = getattr(self.agent, "_delegation_count", 0)
                self._execution_stats["delegations"] = final_delegations - initial_delegations

                validation_passed = True
                pydantic_output = None
                validation_results: dict[str, Any] = {}

                if self.output_json or self.output_pydantic:
                    validation_passed, pydantic_output, validation_results = self._validate_output(result)
                    if not validation_passed:
                        if retries < self.max_retries:
                            retries += 1
                            self._execution_stats["retry_count"] = retries

                            error_details = []
                            for key, value in validation_results.items():
                                if key.endswith("_error"):
                                    error_details.append(f"{key}: {value}")

                            error_msg = f"Output validation failed (attempt {retries}/{self.max_retries}): {'; '.join(error_details)}"
                            log_retry(retries, self.max_retries, error_msg)

                            continue
                        else:
                            raise TaskValidationError(f"Output validation failed: {validation_results}")

                self._output = result

                if self.output_file:
                    with open(self.output_file, "w") as f:
                        f.write(result)

                if self.human_feedback and os.getenv("ALLOW_HUMAN_FEEDBACK", "0") == "1":
                    feedback = input("\n💭 Please provide feedback on this output (or press Enter to accept): ")
                    if feedback:
                        revised = self.agent.execute(
                            task_description=(
                                f"Revise the following based on feedback:\n{result}\n\nFeedback: {feedback}"
                            ),
                            context=enhanced_context,
                        )
                        if isinstance(revised, str):
                            result = revised
                        self._output = result

                if self.save_to_memory and self.memory:
                    self.memory.save_task_result(
                        task_description=self.description,
                        result=result,
                        agent_role=self.agent.role,
                        importance=self.importance,
                        task_metadata={
                            "expected_output": self.expected_output[:100] if self.expected_output else "",
                            "tools_used": [tool.__class__.__name__ for tool in self.tools],
                            "had_context": bool(context_outputs),
                            "had_human_input": self.human_input,
                            "validation_applied": bool(self.output_json or self.output_pydantic),
                        },
                    )

                execution_time = time.time() - start_time

                task_output = CortexTaskOutput(
                    task=self,
                    output=result,
                    agent=self.agent,
                    timestamp=start_time,
                    raw_output=result,
                    execution_time=execution_time,
                    used_tools=self._execution_stats.get("used_tools", 0),
                    tools_errors=self._execution_stats.get("tools_errors", 0),
                    delegations=self._execution_stats.get("delegations", 0),
                    retry_count=self._execution_stats.get("retry_count", 0),
                    validation_results=validation_results,
                    pydantic_output=pydantic_output,
                    execution_metadata={
                        "had_human_input": self.human_input,
                        "had_dependencies": bool(self.dependencies),
                        "security_applied": bool(self.tool_restrictions),
                        "validation_applied": bool(self.output_json or self.output_pydantic),
                    },
                    performance_metrics={"avg_execution_time": execution_time, "total_retries": retries},
                )

                try:
                    task_output.json_dict = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    pass

                self._execute_callback(self.callback, task_output, self)

                log_task_complete(
                    self.description[:50] + "..." if len(self.description) > 50 else self.description, execution_time
                )

                return task_output

            except Exception as e:
                last_error = e
                self._execution_stats["tools_errors"] += 1

                self._execute_callback(self.error_callback, e, self)

                if self._should_retry(e, retries):
                    retries += 1
                    self._execution_stats["retry_count"] = retries
                    log_retry(retries, self.max_retries, str(e))
                    if retries < self.max_retries:
                        log_warning("Retrying in 5 seconds...")

                    time.sleep(5)
                else:
                    log_error(f"Task failed after all retries: {e!s}")
                    break

        execution_time = time.time() - start_time
        failure_output = CortexTaskOutput(
            task=self,
            output=f"Task failed after {retries} retries: {last_error}",
            agent=self.agent,
            timestamp=start_time,
            execution_time=execution_time,
            retry_count=retries,
            execution_metadata={"failed": True, "last_error": str(last_error)},
        )

        if self.timeout_behavior == "return_partial":
            return failure_output
        else:
            raise Exception(f"Task failed after {retries} retries: {last_error}")

    def get_execution_stats(self) -> dict:
        """Get a copy of execution statistics from the last run.

        Returns:
            Dictionary containing used_tools, tools_errors, delegations,
            and retry_count from the most recent execution.
        """
        return self._execution_stats.copy()

    def reset_stats(self):
        """Reset execution statistics to initial values.

        Clears all counters for tool usage, errors, delegations,
        and retries. Should be called before re-executing a task
        if fresh statistics are needed.
        """
        self._execution_stats = {
            "used_tools": 0,
            "tools_errors": 0,
            "delegations": 0,
            "retry_count": 0,
        }

    @property
    def output(self) -> str | None:
        """Get the cached output from the last task execution.

        Returns:
            The output string from the most recent execute() call,
            or None if the task has not been executed.
        """
        return self._output

    def add_dependency(self, task: CortexTask):
        """Add a task dependency that must complete before this task.

        Args:
            task: The CortexTask that must complete before this task executes.

        Note:
            Duplicate dependencies are ignored.
        """
        if task not in self.dependencies:
            self.dependencies.append(task)

    def remove_dependency(self, task: CortexTask):
        """Remove a task from the dependencies list.

        Args:
            task: The CortexTask to remove from dependencies.

        Note:
            Silently ignores if task is not in dependencies.
        """
        if task in self.dependencies:
            self.dependencies.remove(task)

    def add_context(self, tasks: list[CortexTask] | CortexTask):
        """Add context tasks whose outputs will be available during execution.

        Args:
            tasks: A single CortexTask or list of CortexTasks to add as context.
                The outputs of these tasks will be included when building
                the execution prompt.
        """
        if not isinstance(tasks, list):
            tasks = [tasks]
        if self.context is None:
            self.context = []
        for task in tasks:
            self.context.append(task)

    def set_callback(
        self,
        callback_type: Literal["pre_execution", "post_execution", "error"],
        callback: Callable,
    ) -> None:
        """Set a callback function for task lifecycle events.

        Args:
            callback_type: One of 'pre_execution', 'post_execution', or 'error'.
            callback: The function to call at the specified lifecycle point.

        Raises:
            ValueError: If callback_type is not a recognized type.
        """
        if callback_type == "pre_execution":
            self.pre_execution_callback = callback
        elif callback_type == "post_execution":
            self.callback = callback
        elif callback_type == "error":
            self.error_callback = callback
        else:
            raise ValueError(f"Unknown callback type: {callback_type}")

    def add_retry_condition(self, condition: Callable):
        """Add a custom retry condition function.

        Args:
            condition: A callable that takes (error, retry_count, task) and
                returns True if retry should be attempted, False otherwise.

        Note:
            Duplicate conditions are ignored.
        """
        if condition not in self.retry_conditions:
            self.retry_conditions.append(condition)

    def set_security_config(self, **config):
        """Update security configuration with provided options.

        Args:
            **config: Key-value pairs to merge into security_config.
        """
        if self.security_config is None:
            self.security_config = {}
        self.security_config.update(config)

    def validate_output_with_model(self, model: type[BaseModel]):
        """Set a Pydantic model for output validation.

        Args:
            model: A Pydantic BaseModel subclass for validating output.
        """
        self.output_pydantic = model

    def set_json_output_model(self, model: type[BaseModel]):
        """Set a Pydantic model for JSON output extraction and validation.

        Args:
            model: A Pydantic BaseModel subclass. Output will be parsed
                as JSON and validated against this model.
        """
        self.output_json = model
