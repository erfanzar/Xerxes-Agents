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
"""Task definitions, output containers, and execution logic for Cortex workflows."""

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
    """Raised when a task output fails Pydantic or JSON schema validation."""

    def __init__(self, message: str):
        """Initialize the validation error.

        Args:
            message (str): Human-readable description of the validation failure.
                IN: Explains why the output was rejected.
                OUT: Stored in ``self.message`` and passed to the base ``Exception``.
        """

        self.message = message
        super().__init__(self.message)


@dataclass
class ChainLink:
    """Conditional linking between tasks in a workflow chain.

    Attributes:
        condition (Callable[[str], bool] | None): Evaluated on task output;
            if ``True``, proceed to ``next_task``.
        next_task (CortexTask | None): The task to execute when the condition passes.
        fallback_task (CortexTask | None): The task to execute when the condition fails.
    """

    condition: Callable[[str], bool] | None = None
    next_task: CortexTask | None = None
    fallback_task: CortexTask | None = None


@dataclass
class CortexTaskOutput:
    """Container for the result of executing a ``CortexTask``.

    Attributes:
        task (CortexTask): The task that was executed.
        output (str): The primary textual output.
        agent (CortexAgent): The agent that produced the output.
        timestamp (float): Unix timestamp of execution start.
        raw_output (str | None): Unprocessed output if different from *output*.
        token_usage (dict[str, int]): Token consumption metrics.
        validation_results (dict[str, bool]): Output validation outcomes.
        pydantic_output (object | None): Parsed Pydantic model instance.
        json_dict (dict | None): Parsed JSON dictionary.
        execution_time (float): Duration of execution in seconds.
        used_tools (int): Number of tools invoked.
        tools_errors (int): Number of tool invocation errors.
        delegations (int): Number of delegations performed.
        retry_count (int): Number of retries attempted.
        execution_metadata (dict): Arbitrary execution metadata.
        performance_metrics (dict): Performance-related metrics.
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
        """Return a truncated summary of the output.

        Returns:
            str: First 200 characters of *output* with an ellipsis if truncated.
                OUT: Suitable for logging and display.
        """

        return self.output[:200] + "..." if len(self.output) > 200 else self.output

    def to_dict(self) -> dict:
        """Serialize the task output to a dictionary.

        Returns:
            dict: A dictionary with task description, output, agent role,
                timestamp, execution stats, and metadata.
                OUT: Suitable for JSON serialization.
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
        """Return a human-readable string representation.

        Returns:
            str: Multi-line summary of agent, task, output, and execution time.
                OUT: Suitable for console display.
        """

        return f"""Task Output:
        Agent: {self.agent.role}
        Task: {self.task.description[:256]}...
        Output: {self.summary}
        Execution Time: {self.execution_time:.2f}s
        """

    def __repr__(self) -> str:
        """Return a compact representation for debugging.

        Returns:
            str: Contains agent role and output length.
                OUT: Used in logs and REPL output.
        """

        return f"CortexTaskOutput(agent={self.agent.role}, output_length={len(self.output)})"


@dataclass
class CortexTask:
    """A unit of work assigned to an agent within a Cortex workflow.

    Attributes:
        description (str): What the task should accomplish.
        expected_output (str): Description of the desired result.
        agent (CortexAgent | None): The agent responsible for execution.
        tools (list[CortexTool]): Additional tools available for this task.
        context (list[CortexTask] | None): Prior tasks whose outputs provide context.
        output_file (str | None): File path to write the output to.
        human_feedback (bool): Whether human feedback is required after execution.
        chain (ChainLink | None): Conditional next/fallback task linkage.
        max_retries (int): Maximum retry attempts on failure.
        memory (CortexMemory | None): Memory subsystem for persisting results.
        save_to_memory (bool): Whether to persist the result to memory.
        importance (float): Importance score affecting memory persistence.
        output_json (type[BaseModel] | None): Expected JSON output schema.
        output_pydantic (type[BaseModel] | None): Expected Pydantic output model.
        create_directory (bool): Whether to create parent dirs for *output_file*.
        async_execution (bool): Whether the task should run asynchronously.
        callback (Callable | None): Post-execution callback.
        pre_execution_callback (Callable | None): Callback before execution.
        error_callback (Callable | None): Callback on execution error.
        human_input (bool): Whether to prompt for human input before execution.
        human_input_prompt (str | None): Custom prompt for human input.
        input_validator (Callable | None): Validator for human input.
        security_config (dict | None): Security configuration dictionary.
        tool_restrictions (list[str] | None): Whitelist of allowed tool names.
        allow_dangerous_tools (bool): Whether dangerous tools are permitted.
        prompt_context (str | None): Additional context appended to the prompt.
        context_priority (dict[str, float]): Priority weights for context sources.
        context_compression (bool): Whether to compress long context strings.
        max_context_length (int): Maximum length for the combined context.
        dependencies (list[CortexTask]): Explicit task dependencies.
        conditional_execution (Callable | None): Predicate to skip execution.
        retry_conditions (list[Callable]): Extra predicates controlling retries.
        timeout_behavior (Literal["fail", "continue", "return_partial"]): Behavior on timeout.
        timeout (int | None): Timeout in seconds.
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
        """Substitute template variables into task fields.

        Args:
            inputs (dict): Mapping of template variable names to values.
                IN: Keys should match ``{var}`` placeholders in description,
                expected output, output file, and prompt context.
                OUT: Values are interpolated into the corresponding fields.
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
        """Attempt to extract a valid JSON object from arbitrary text.

        Args:
            output (str): Text that may contain JSON.
                IN: Scanned for JSON blocks, inline objects, and brace-balanced substrings.
                OUT: Parsed to find the first valid JSON string.

        Returns:
            str | None: The extracted JSON string, or ``None`` if none found.
                OUT: Suitable for passing to ``json.loads``.
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
        """Extract the text content of an XML tag.

        Args:
            output (str): Text containing XML.
                IN: Searched for the specified tag.
                OUT: Parsed with regex to extract inner text.
            tag (str): The XML tag name (without brackets).
                IN: Determines which element to extract.
                OUT: Used in the regex pattern.

        Returns:
            str | None: The inner text of the first matching tag, or ``None``.
                OUT: Stripped of surrounding whitespace.
        """

        import re

        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, output, re.DOTALL)
        return match.group(1).strip() if match else None

    def _parse_xml_to_dict(self, xml_content: str) -> dict:
        """Convert XML content into a nested dictionary.

        Args:
            xml_content (str): Raw XML string (without a root element).
                IN: Wrapped in a ``<root>`` tag for parsing.
                OUT: Transformed into a dictionary by recursive traversal.

        Returns:
            dict: Nested dictionary representation of the XML.
                OUT: Empty dict if parsing fails.
        """

        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(f"<root>{xml_content}</root>")

            def xml_to_dict(element) -> dict[str, Any]:
                """Recursively convert an XML element to a dictionary.

                Args:
                    element: An ``xml.etree.ElementTree.Element``.
                        IN: The element whose children will be converted.
                        OUT: Transformed into nested dicts and lists.

                Returns:
                    dict[str, Any]: Dictionary with tag names as keys.
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
        """Validate task output against JSON and/or Pydantic schemas.

        Args:
            output (str): The raw output to validate.
                IN: Parsed and checked against ``self.output_json`` and
                ``self.output_pydantic``.
                OUT: Used to produce validation results and a Pydantic instance.

        Returns:
            tuple[bool, Any, dict[str, Any]]: A boolean indicating overall
                validity, the parsed Pydantic object (or ``None``), and a
                dictionary of validation details.
                OUT: ``validation_results`` includes extraction method and error keys.
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
        """Invoke a callback safely, routing errors to the error callback.

        Args:
            callback (Callable | None): The callback to execute.
                IN: Called with ``*args`` and ``**kwargs``.
                OUT: Executed in a try/except block.
            *args: Positional arguments for the callback.
                IN: Forwarded to *callback*.
                OUT: Passed through to the callable.
            **kwargs: Keyword arguments for the callback.
                IN: Forwarded to *callback*.
                OUT: Passed through to the callable.
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
        """Check whether all explicit dependencies have produced output.

        Returns:
            bool: ``True`` if all dependencies have non-empty output.
                OUT: Used to gate task execution.
        """

        for dep in self.dependencies:
            if not dep.output:
                return False
        return True

    def _should_retry(self, error: Exception, retry_count: int) -> bool:
        """Determine whether the task should be retried after an error.

        Args:
            error (Exception): The exception that triggered the retry check.
                IN: Passed to each retry condition.
                OUT: Used to evaluate custom retry logic.
            retry_count (int): The number of retries already attempted.
                IN: Compared against ``self.max_retries``.
                OUT: Determines if the retry budget is exhausted.

        Returns:
            bool: ``True`` if retrying is allowed.
                OUT: Considers ``max_retries`` and custom ``retry_conditions``.
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
        """Prompt the user for input and optionally validate it.

        Returns:
            str: The validated user input.
                OUT: Loops until validation passes or no validator is configured.
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
        """Create the parent directory for ``self.output_file`` if needed."""

        if self.output_file and self.create_directory:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

    def _apply_tool_restrictions(self):
        """Filter the agent's tools to only those in ``self.tool_restrictions``."""

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
        """Assemble and optionally compress the full execution context.

        Args:
            context (str): Base context string.
                IN: Combined with other context sources.
                OUT: Included in the final assembled context.
            context_outputs (list[str] | None): Outputs from prior tasks.
                IN: Sorted by priority and appended to the context.
                OUT: Each entry is tagged with a priority weight.
            human_input (str): Direct human input for the task.
                IN: Included with an elevated default priority.
                OUT: Prepended with ``"Human Input:"`` label.

        Returns:
            str: The assembled and optionally compressed context string.
                OUT: Truncated to ``self.max_context_length`` when compression is enabled.
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
        """Create a placeholder output when execution is skipped.

        Args:
            reason (str): Explanation for why execution was skipped.
                IN: Stored in ``execution_metadata``.
                OUT: Becomes the ``output`` field of the returned object.

        Returns:
            CortexTaskOutput: A skipped-task output container.
                OUT: Contains the skip reason and metadata.

        Raises:
            ValueError: If the task has no assigned agent.
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
        """Execute the task through the assigned agent.

        Args:
            context_outputs (list[str] | None): Outputs from previous tasks.
                IN: Combined into the execution context.
                OUT: Passed to ``_build_enhanced_context``.
            use_streaming (bool): Whether to execute in streaming mode.
                IN: If ``True``, returns a ``(StreamerBuffer, Thread)`` tuple.
                OUT: Determines the execution path and return type.
            stream_callback (Callable | None): Callback for streaming chunks.
                IN: Invoked for each chunk when streaming is enabled.
                OUT: Wired into the stream processing thread.

        Returns:
            CortexTaskOutput | tuple[StreamerBuffer, threading.Thread]: The task
                result or streaming handles. OUT: ``CortexTaskOutput`` on normal
                completion; tuple when *use_streaming* is ``True``.

        Raises:
            ValueError: If no agent is assigned or dependencies are unsatisfied.
            TaskValidationError: If output validation fails after all retries.
            Exception: If the task fails after exhausting retries.
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
                            """Consume the stream and forward chunks to the callback.

                            Args:
                                buffer (StreamerBuffer): The buffer to stream from.
                                    IN: Provides chunks via ``buffer.stream()``.
                                    OUT: Iterated to invoke *stream_callback* for each chunk.
                            """
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
        """Return a copy of the current execution statistics.

        Returns:
            dict: Snapshot of ``self._execution_stats``.
                OUT: Contains keys such as ``used_tools``, ``tools_errors``,
                ``delegations``, and ``retry_count``.
        """

        return self._execution_stats.copy()

    def reset_stats(self):
        """Reset execution statistics to their initial values."""

        self._execution_stats = {
            "used_tools": 0,
            "tools_errors": 0,
            "delegations": 0,
            "retry_count": 0,
        }

    @property
    def output(self) -> str | None:
        """Return the stored output of the last execution.

        Returns:
            str | None: The output string, or ``None`` if not yet executed.
                OUT: Reads from ``self._output``.
        """

        return self._output

    def add_dependency(self, task: CortexTask):
        """Add a task dependency.

        Args:
            task (CortexTask): The task to depend on.
                IN: Appended to ``self.dependencies`` if not already present.
                OUT: Gated by ``_check_dependencies`` during ``execute``.
        """

        if task not in self.dependencies:
            self.dependencies.append(task)

    def remove_dependency(self, task: CortexTask):
        """Remove a task dependency.

        Args:
            task (CortexTask): The task to remove from dependencies.
                IN: Removed from ``self.dependencies`` if present.
                OUT: No longer blocks execution of this task.
        """

        if task in self.dependencies:
            self.dependencies.remove(task)

    def add_context(self, tasks: list[CortexTask] | CortexTask):
        """Add prior tasks as context providers.

        Args:
            tasks (list[CortexTask] | CortexTask): One or more tasks whose outputs
                provide context.
                IN: Appended to ``self.context``.
                OUT: Their outputs are included in the execution prompt.
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
        """Register a callback for a specific lifecycle event.

        Args:
            callback_type (Literal): The event type to hook.
                IN: Must be one of ``"pre_execution"``, ``"post_execution"``, or ``"error"``.
                OUT: Determines which callback attribute is updated.
            callback (Callable): The function to invoke.
                IN: Stored in the corresponding callback attribute.
                OUT: Called at the appropriate lifecycle stage.

        Raises:
            ValueError: If *callback_type* is not recognized.
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
        """Add a custom predicate for deciding whether to retry on failure.

        Args:
            condition (Callable): A function accepting ``(error, retry_count, task)``.
                IN: Should return ``True`` to allow retry, ``False`` to stop.
                OUT: Appended to ``self.retry_conditions``.
        """

        if condition not in self.retry_conditions:
            self.retry_conditions.append(condition)

    def set_security_config(self, **config):
        """Update the security configuration dictionary.

        Args:
            **config: Arbitrary key-value security settings.
                IN: Merged into ``self.security_config``.
                OUT: Overwrites existing keys and adds new ones.
        """

        if self.security_config is None:
            self.security_config = {}
        self.security_config.update(config)

    def validate_output_with_model(self, model: type[BaseModel]):
        """Set a Pydantic model for raw output validation.

        Args:
            model (type[BaseModel]): A Pydantic ``BaseModel`` subclass.
                IN: Assigned to ``self.output_pydantic``.
                OUT: Used in ``_validate_output`` to parse and validate results.
        """

        self.output_pydantic = model

    def set_json_output_model(self, model: type[BaseModel]):
        """Set a Pydantic model for JSON output validation.

        Args:
            model (type[BaseModel]): A Pydantic ``BaseModel`` subclass.
                IN: Assigned to ``self.output_json``.
                OUT: Used in ``_validate_output`` to extract and validate JSON.
        """

        self.output_json = model
