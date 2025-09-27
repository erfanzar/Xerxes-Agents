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


"""Enhanced task definition for Cortex framework with Cortex-like features"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ..loggings import log_error, log_retry, log_task_complete, log_task_start, log_warning
from ..streamer_buffer import StreamerBuffer
from .tool import CortexTool

if TYPE_CHECKING:
    from .agent import CortexAgent
    from .memory_integration import CortexMemory


class TaskValidationError(Exception):
    """Exception raised when task output validation fails"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


@dataclass
class ChainLink:
    """Represents a link in a task chain"""

    condition: Callable[[str], bool] | None = None
    next_task: CortexTask | None = None
    fallback_task: CortexTask | None = None


@dataclass
class CortexTaskOutput:
    """Enhanced output from a completed task with rich formatting support"""

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
        """Get a summary of the output (first 200 characters)"""
        return self.output[:200] + "..." if len(self.output) > 200 else self.output

    def to_dict(self) -> dict:
        """Convert output to dictionary format"""
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
        """String representation of the task output"""
        return f"""Task Output:
        Agent: {self.agent.role}
        Task: {self.task.description[:256]}...
        Output: {self.summary}
        Execution Time: {self.execution_time:.2f}s
        """

    def __repr__(self) -> str:
        """Representation of the task output"""
        return f"CortexTaskOutput(agent={self.agent.role}, output_length={len(self.output)})"


@dataclass
class CortexTask:
    """Enhanced task"""

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
    timeout_behavior: str = "fail"
    timeout: int | None = None

    _output: str | None = None
    _execution_stats: dict = field(default_factory=dict)
    _start_time: float = 0.0

    def _extract_json_from_output(self, output: str) -> str | None:
        """Extract JSON from output that may contain thinking process or other text"""
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
        """Extract content from XML tags"""
        import re

        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, output, re.DOTALL)
        return match.group(1).strip() if match else None

    def _parse_xml_to_dict(self, xml_content: str) -> dict:
        """Parse XML content to dictionary for Pydantic validation"""
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(f"<root>{xml_content}</root>")

            def xml_to_dict(element):
                result = {}
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

    def _validate_output(self, output: str) -> tuple[bool, Any, dict]:
        """Enhanced validation supporting JSON extraction and XML parsing"""
        validation_results = {}
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
        """Safely execute a callback function"""
        if callback and callable(callback):
            try:
                return callback(*args, **kwargs)
            except Exception as e:
                if self.error_callback:
                    self._execute_callback(self.error_callback, e, self)
                else:
                    print(f"Callback error: {e}")

    def _check_dependencies(self) -> bool:
        """Check if all dependencies are completed"""
        for dep in self.dependencies:
            if not dep.output:
                return False
        return True

    def _should_retry(self, error: Exception, retry_count: int) -> bool:
        """Check if task should be retried based on conditions"""
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
        """Get human input with validation"""
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
        """Create output directory if specified"""
        if self.output_file and self.create_directory:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

    def _apply_tool_restrictions(self):
        """Apply tool restrictions based on security config"""
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
        """Build enhanced context with priority weighting and compression"""
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
        """Create an empty output for skipped tasks"""
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
        """Execute the task with enhanced Cortex-like features and optional streaming

        Args:
            context_outputs: Optional list of context outputs from previous tasks
            use_streaming: If True, executes in background thread with streaming
            stream_callback: Optional callback for streaming chunks (only used if use_streaming=True)

        Returns:
            If use_streaming=False: CortexTaskOutput with the task result
            If use_streaming=True: tuple of (StreamerBuffer, Thread) for async streaming
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
                    buffer, thread = self.agent.execute(
                        task_description=task_prompt,
                        context=enhanced_context,
                        use_thread=True,
                    )

                    if stream_callback:

                        def process_stream():
                            for chunk in buffer.stream():  # noqa
                                stream_callback(chunk)

                        callback_thread = threading.Thread(target=process_stream, daemon=True)
                        callback_thread.start()

                    buffer.task = self  # type: ignore
                    buffer.agent = self.agent  # type: ignore

                    return buffer, thread

                if self.agent.allow_delegation:
                    result = self.agent.execute_with_delegation(task_description=task_prompt, context=enhanced_context)
                else:
                    result = self.agent.execute(task_description=task_prompt, context=enhanced_context)

                final_delegations = getattr(self.agent, "_delegation_count", 0)
                self._execution_stats["delegations"] = final_delegations - initial_delegations

                validation_passed = True
                pydantic_output = None
                validation_results = {}

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
                        result = self.agent.execute(
                            task_description=(
                                f"Revise the following based on feedback:\n{result}\n\nFeedback: {feedback}"
                            ),
                            context=enhanced_context,
                        )
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

    async def execute_async(self, context_outputs: list[str] | None = None) -> CortexTaskOutput:
        """Execute the task asynchronously"""
        if not self.async_execution:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.execute, context_outputs)

        return self.execute(context_outputs)

    def get_execution_stats(self) -> dict:
        """Get execution statistics"""
        return self._execution_stats.copy()

    def reset_stats(self):
        """Reset execution statistics"""
        self._execution_stats = {
            "used_tools": 0,
            "tools_errors": 0,
            "delegations": 0,
            "retry_count": 0,
        }

    @property
    def output(self) -> str | None:
        """Get the task output"""
        return self._output

    def add_dependency(self, task: CortexTask):
        """Add a task dependency"""
        if task not in self.dependencies:
            self.dependencies.append(task)

    def remove_dependency(self, task: CortexTask):
        """Remove a task dependency"""
        if task in self.dependencies:
            self.dependencies.remove(task)

    def add_context(self, tasks: list[CortexTask] | CortexTask):
        if not isinstance(tasks, list):
            tasks = [tasks]
        if self.context is None:
            self.context = []
        for task in tasks:
            self.context.append(task)

    def set_callback(self, callback_type: str, callback: Callable):
        """Set a callback function"""
        if callback_type == "pre_execution":
            self.pre_execution_callback = callback
        elif callback_type == "post_execution":
            self.callback = callback
        elif callback_type == "error":
            self.error_callback = callback
        else:
            raise ValueError(f"Unknown callback type: {callback_type}")

    def add_retry_condition(self, condition: Callable):
        """Add a retry condition function"""
        if condition not in self.retry_conditions:
            self.retry_conditions.append(condition)

    def set_security_config(self, **config):
        """Set security configuration"""
        if self.security_config is None:
            self.security_config = {}
        self.security_config.update(config)

    def validate_output_with_model(self, model: type[BaseModel]):
        """Set output validation model"""
        self.output_pydantic = model

    def set_json_output_model(self, model: type[BaseModel]):
        """Set JSON output validation model"""
        self.output_json = model
