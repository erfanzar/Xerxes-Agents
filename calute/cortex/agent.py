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


"""Agent definition for Cortex framework.

This module provides the CortexAgent class, which represents an intelligent agent
within the Cortex framework for AI agent orchestration. CortexAgent enables
autonomous task execution, delegation, tool usage, and memory integration for
building complex multi-agent systems.

The agent supports features like:
- Task execution with context and memory
- Automatic delegation to other agents
- Rate limiting and execution timeouts
- Knowledge management and tool integration
- Step-by-step callback mechanisms
- Pydantic model output formatting

Typical usage example:
    agent = CortexAgent(
        role="Data Analyst",
        goal="Analyze data and provide insights",
        backstory="Expert in statistical analysis",
        model="gpt-4",
        tools=[analysis_tool],
        allow_delegation=True
    )
    result = agent.execute("Analyze sales trends")
"""

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from calute.llms.base import BaseLLM
from calute.streamer_buffer import StreamerBuffer
from calute.types.function_execution_types import Completion, ResponseResult, StreamingResponseType
from calute.types.tool_calls import Tool

from ..loggings import get_logger, log_delegation, log_error, log_thinking, log_warning
from ..types import Agent as CaluteAgent
from ..types import AgentCapability
from .templates import PromptTemplate
from .tool import CortexTool

if TYPE_CHECKING:
    from ..calute import Calute
    from .cortex import Cortex
    from .memory_integration import CortexMemory


@dataclass
class CortexAgent:
    """Agent with specific role, goal, and capabilities.

    CortexAgent represents an autonomous AI agent that can execute tasks,
    use tools, delegate to other agents, and maintain context through memory.
    It serves as a wrapper around Calute's Agent class with additional
    orchestration capabilities for the Cortex framework.

    Attributes:
        role: The agent's role or title defining its specialization.
        goal: The primary objective or purpose of the agent.
        backstory: Background information providing context for the agent's expertise.
        model: Optional model identifier for the LLM to use (e.g., 'gpt-4').
        instructions: Optional custom system instructions. Auto-generated if not provided.
        tools: List of CortexTool instances the agent can use.
        max_iterations: Maximum number of retry attempts for failed executions.
        verbose: Whether to output detailed logging information.
        allow_delegation: Whether the agent can delegate tasks to other agents.
        temperature: LLM temperature parameter for response randomness (0.0-1.0).
        max_tokens: Maximum tokens for LLM responses.
        memory_enabled: Whether to use memory for context building.
        capabilities: List of special capabilities the agent possesses.
        calute_instance: Reference to the Calute instance for LLM operations.
        memory: Optional CortexMemory instance for persistent context.
        llm: Optional BaseLLM instance for direct LLM access.
        reinvoke_after_function: Whether to reinvoke LLM after tool execution.
        cortex_instance: Reference to parent Cortex instance for multi-agent coordination.
        max_execution_time: Optional timeout in seconds for task execution.
        max_rpm: Optional rate limit for requests per minute.
        step_callback: Optional callback function invoked at each execution step.
        config: Dictionary for custom configuration parameters.
        knowledge: Dictionary storing agent-specific knowledge.
        knowledge_sources: List of knowledge source identifiers.
        auto_format_guidance: Whether to auto-generate format instructions for Pydantic models.
        output_format_preference: Preferred output format ('xml' or 'json').

    Private Attributes:
        _internal_agent: Internal Calute Agent instance.
        _logger: Logger instance for verbose output.
        _template_engine: PromptTemplate engine for generating prompts.
        _delegation_count: Current delegation depth to prevent infinite recursion.
        _times_executed: Total number of task executions.
        _execution_times: List of execution durations for statistics.
        _last_rpm_window: Timestamp for rate limiting window.
        _rpm_requests: List of request timestamps for rate limiting.
    """

    role: str
    goal: str
    backstory: str
    model: str | None = None
    instructions: str | None = None
    tools: list[CortexTool | Callable | Tool] = field(default_factory=list)
    max_iterations: int = 10
    verbose: bool = True
    allow_delegation: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048
    memory_enabled: bool = True
    capabilities: list[AgentCapability] = field(default_factory=list)
    calute_instance: "Calute | None" = None
    memory: Optional["CortexMemory"] = None
    llm: BaseLLM | None = None
    reinvoke_after_function: bool = True
    cortex_instance: Optional["Cortex"] = None

    max_execution_time: int | None = None
    max_rpm: int | None = None
    step_callback: Callable | None = None
    config: dict = field(default_factory=dict)
    knowledge: dict = field(default_factory=dict)
    knowledge_sources: list = field(default_factory=list)

    auto_format_guidance: bool = True
    output_format_preference: str = "xml"

    _internal_agent: CaluteAgent | None = None
    _logger = None
    _template_engine: PromptTemplate | None = None
    _delegation_count: int = 0
    _times_executed: int = 0
    _execution_times: list = field(default_factory=list)
    _last_rpm_window: float = 0
    _rpm_requests: list = field(default_factory=list)

    def __post_init__(self):
        """Initialize the internal Calute agent and supporting components.

        This method is automatically called after dataclass initialization.
        It sets up the logger, template engine, generates default instructions
        if needed, processes tools into functions, and creates the internal
        Calute Agent instance that handles actual LLM interactions.

        Side Effects:
            - Initializes _logger based on verbose setting
            - Creates _template_engine instance
            - Generates default instructions if not provided
            - Processes tools list into callable functions
            - Creates Calute instance if needed
            - Initializes _internal_agent

        Raises:
            ImportError: If Calute module cannot be imported when creating instance.
        """
        self._logger = get_logger() if self.verbose else None
        self._template_engine = PromptTemplate()

        if not self.instructions:
            self.instructions = self._template_engine.render_agent_prompt(
                role=self.role,
                goal=self.goal,
                backstory=self.backstory,
                tools=self.tools if self.tools else None,
            )

        functions = []
        for tool in self.tools:
            if callable(tool) and not isinstance(tool, CortexTool):
                functions.append(tool)
            elif hasattr(tool, "function") and callable(tool.function):
                functions.append(tool.function)
            else:
                functions.append(tool)

        if self.calute_instance is None and self.llm is not None:
            from calute.calute import Calute

            self.calute_instance = Calute(llm=self.llm)
        self._internal_agent = CaluteAgent(
            name=self.role,
            instructions=self.instructions,
            model=self.model,
            functions=functions,
            capabilities=self.capabilities if self.capabilities else [],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            parallel_tool_calls=True,
        )

    def _check_rate_limit(self) -> bool:
        """Check if agent is within rate limit.

        Verifies whether the agent can make a new request based on the
        configured max_rpm (requests per minute) limit. Maintains a sliding
        window of request timestamps to enforce the rate limit.

        Returns:
            bool: True if within rate limit or no limit configured,
                  False if rate limit would be exceeded.

        Note:
            This method cleans up old request timestamps outside the
            60-second window before checking the limit.
        """
        if not self.max_rpm:
            return True

        current_time = time.time()

        self._rpm_requests = [req_time for req_time in self._rpm_requests if current_time - req_time < 60]

        if len(self._rpm_requests) >= self.max_rpm:
            if self.verbose:
                log_warning(f"Rate limit reached ({self.max_rpm} RPM)")
            return False

        return True

    def _record_request(self):
        """Record a request for rate limiting.

        Adds the current timestamp to the list of request times used for
        rate limiting. Only records if max_rpm is configured.

        Side Effects:
            Appends current timestamp to _rpm_requests list.
        """
        if self.max_rpm:
            import time

            self._rpm_requests.append(time.time())

    def _check_execution_timeout(self, start_time: float) -> bool:
        """Check if execution has exceeded timeout.

        Determines whether the elapsed time since start_time exceeds the
        configured max_execution_time limit.

        Args:
            start_time: Unix timestamp marking the start of execution.

        Returns:
            bool: True if timeout exceeded, False otherwise or if no
                  timeout is configured.

        Side Effects:
            Logs error message if timeout is exceeded and verbose is True.
        """
        if not self.max_execution_time:
            return False

        elapsed = time.time() - start_time
        if elapsed > self.max_execution_time:
            if self.verbose:
                log_error(f"Execution timeout after {elapsed:.2f}s")
            return True
        return False

    def _execute_step_callback(self, step_info: dict):
        """Execute step callback if provided.

        Safely invokes the configured step_callback function with information
        about the current execution step. Catches and logs any exceptions to
        prevent callback failures from disrupting agent execution.

        Args:
            step_info: Dictionary containing step details such as:
                - step: Step type (e.g., 'execution_start', 'retry', 'execution_complete')
                - agent: Agent role identifier
                - Additional context-specific fields

        Side Effects:
            Invokes step_callback if configured.
            Logs errors if callback fails and verbose is True.
        """
        if self.step_callback and callable(self.step_callback):
            try:
                self.step_callback(step_info)
            except Exception as e:
                if self.verbose:
                    log_error(f"Step callback failed: {e}")

    def _build_knowledge_context(self, task_description: str) -> str:
        """Build context from knowledge sources.

        Constructs a formatted context string from the agent's knowledge
        dictionary and knowledge_sources list to provide additional context
        for task execution.

        Args:
            task_description: The task being executed (currently unused but
                            available for future context filtering).

        Returns:
            str: Formatted context string containing knowledge items and
                 sources, or empty string if no knowledge is configured.

        Example:
            Returns a string like:
            "Available Knowledge:\n- domain: healthcare\n- expertise: diagnostics\n\nKnowledge Sources:\n- medical_db\n\n"
        """
        if not self.knowledge and not self.knowledge_sources:
            return ""

        context_parts = []

        if self.knowledge:
            context_parts.append("Available Knowledge:")
            for key, value in self.knowledge.items():
                context_parts.append(f"- {key}: {value}")

        if self.knowledge_sources:
            context_parts.append("Knowledge Sources:")
            for source in self.knowledge_sources:
                context_parts.append(f"- {source}")

        return "\n".join(context_parts) + "\n\n" if context_parts else ""

    def _generate_format_guidance(self, output_model) -> str:
        """Generate format guidance from Pydantic model.

        Creates detailed formatting instructions for LLM responses based on
        a Pydantic model schema. Supports both XML and JSON output formats
        and handles nested structures with proper example generation.

        Args:
            output_model: A Pydantic BaseModel class to generate guidance for.

        Returns:
            str: Formatted instruction string with examples and validation rules,
                 or empty string if model is invalid or auto_format_guidance is False.

        Note:
            - Automatically detects Pydantic v1 vs v2 schema methods
            - Generates realistic example data based on field names and types
            - Provides special handling for nested arrays of objects
            - Includes critical formatting rules to ensure LLM compliance
        """
        if not output_model or not self.auto_format_guidance:
            return ""

        try:
            if not issubclass(output_model, BaseModel):
                return ""

            if hasattr(output_model, "model_json_schema"):
                schema = output_model.model_json_schema()
            else:
                schema = output_model.schema()  # type: ignore[attr-defined]
            model_name = schema.get("title", "Output")

            example_data = self._create_example_from_schema(schema)

            if self.output_format_preference == "xml":
                nested_arrays = self._count_nested_structures(schema)

                format_instruction = f"""

OUTPUT FORMAT REQUIREMENT:
Please provide your response in the following XML format to ensure proper parsing:

<json>
{self._format_json_example(example_data)}
</json>

CRITICAL FORMATTING RULES:
1. Follow the EXACT structure shown above - do not simplify or modify it
2. {"Arrays of objects must contain FULL objects with all required fields - do not use simple strings" if nested_arrays > 0 else ""}
3. Each nested object must have ALL the fields shown in the example
4. Use realistic values but maintain the exact JSON structure
5. Numbers must be within the specified ranges (check min/max constraints)

This format is required for validation against the {model_name} schema.
FAILURE TO FOLLOW THE EXACT STRUCTURE WILL RESULT IN VALIDATION ERRORS.
"""
            else:
                format_instruction = f"""

OUTPUT FORMAT REQUIREMENT:
Please provide your response as valid JSON matching this schema for {model_name}:

{self._format_json_example(example_data)}

Ensure your response is valid JSON that can be parsed directly.
"""

            return format_instruction

        except Exception:
            return ""

    def _resolve_schema_refs(self, schema: dict, definitions: dict | None = None) -> dict:
        """Resolve $ref references in schema (supports both Pydantic v1 and v2).

        Recursively resolves JSON Schema $ref references to their actual definitions,
        supporting both Pydantic v1 (

        Args:
            schema: JSON schema dictionary potentially containing $ref references.
            definitions: Optional dictionary of schema definitions. If not provided,
                        extracted from schema['definitions'] or schema['$defs'].

        Returns:
            dict: Schema with all $ref references resolved to their actual definitions.

        Example:
            Input schema with $ref:
            {"$ref": "#/definitions/User"}

            Returns resolved schema:
            {"type": "object", "properties": {"name": {"type": "string"}}}
        """
        if definitions is None:
            definitions = schema.get("definitions", schema.get("$defs", {}))

        if "$ref" in schema:
            ref_path = schema["$ref"]

            if ref_path.startswith("#/definitions/") or ref_path.startswith("#/$defs/"):
                ref_name = ref_path.split("/")[-1]
                if ref_name in definitions:
                    return self._resolve_schema_refs(definitions[ref_name], definitions)

        resolved = schema.copy()
        if "properties" in resolved:
            resolved["properties"] = {
                k: self._resolve_schema_refs(v, definitions) for k, v in resolved["properties"].items()
            }

        if "items" in resolved:
            resolved["items"] = self._resolve_schema_refs(resolved["items"], definitions)

        return resolved

    def _create_example_from_schema(self, schema: dict) -> dict:
        """Create realistic example data structure from JSON schema with full resolution.

        Generates example data that conforms to a JSON schema, creating realistic
        values based on field names and types. Handles nested objects, arrays,
        and various primitive types with appropriate example values.

        Args:
            schema: JSON schema dictionary defining the data structure.

        Returns:
            dict: Example data conforming to the schema with realistic values.

        Note:
            - Field names influence example values (e.g., 'name' fields get name-like values)
            - Respects minimum/maximum constraints for numeric types
            - Generates multiple items for array fields based on minItems
            - Recursively handles nested objects and arrays of objects
            - Creates industry-specific examples for recognized field names
        """

        resolved_schema = self._resolve_schema_refs(schema)

        properties = resolved_schema.get("properties", {})
        definitions = schema.get("definitions", {})
        example = {}

        for field_name, field_info in properties.items():
            field_type = field_info.get("type")
            field_description = field_info.get("description", "")

            if field_type == "string":
                if "name" in field_name.lower() or "title" in field_name.lower():
                    example[field_name] = f"Example {field_name.replace('_', ' ').title()}"
                else:
                    example[field_name] = f"Example {field_description or field_name.replace('_', ' ')}"
            elif field_type == "integer":
                min_val = field_info.get("minimum", 1)
                max_val = field_info.get("maximum", 10)
                example[field_name] = min(max_val, max(min_val, 5))
            elif field_type == "number":
                min_val = field_info.get("minimum", 1.0)
                max_val = field_info.get("maximum", 10.0)
                example[field_name] = min(max_val, max(min_val, 7.5))
            elif field_type == "boolean":
                example[field_name] = True
            elif field_type == "array":
                items = field_info.get("items", {})
                min_items = field_info.get("minItems", 2)

                if items.get("type") == "string":
                    if "industries" in field_name.lower():
                        example[field_name] = ["technology", "healthcare", "finance"][:min_items]
                    elif "tags" in field_name.lower():
                        example[field_name] = ["tag1", "tag2", "tag3"][:min_items]
                    else:
                        example[field_name] = [f"item{i + 1}" for i in range(min_items)]

                elif items.get("type") == "object" or "$ref" in items:
                    resolved_items = self._resolve_schema_refs(items, definitions)

                    if "properties" in resolved_items:
                        nested_examples = []
                        for i in range(max(min_items, 3)):
                            nested_example = self._create_nested_example(resolved_items, i + 1)
                            nested_examples.append(nested_example)
                        example[field_name] = nested_examples
                    else:
                        nested_example = self._create_example_from_schema(resolved_items)
                        example[field_name] = [nested_example] * max(min_items, 3)
                else:
                    example[field_name] = [f"item{i + 1}" for i in range(min_items)]

            elif field_type == "object":
                if "properties" in field_info:
                    example[field_name] = self._create_example_from_schema(field_info)
                else:
                    example[field_name] = {}
            else:
                example[field_name] = f"Example {field_description or field_name.replace('_', ' ')}"

        return example

    def _create_nested_example(self, schema: dict, index: int = 1) -> dict:
        """Create a realistic nested object example with variation.

        Generates varied example data for nested objects in arrays, ensuring
        each object has different but realistic values based on its index.
        Particularly useful for demonstrating arrays of complex objects.

        Args:
            schema: JSON schema for the nested object structure.
            index: Index of this object in its parent array, used to create variation.

        Returns:
            dict: Example nested object with varied realistic values.

        Example:
            For a trend object schema, generates different trend names and
            descriptions for each index:
            - Index 1: "Generative AI Enterprise Integration"
            - Index 2: "Edge AI and IoT Convergence"
            - Index 3: "AI-Powered Cybersecurity"
        """
        properties = schema.get("properties", {})

        example = {}
        for field_name, field_info in properties.items():
            field_type = field_info.get("type")
            field_description = field_info.get("description", "")

            if field_type == "string":
                if "name" in field_name.lower() or "title" in field_name.lower():
                    trend_names = [
                        "Generative AI Enterprise Integration",
                        "Edge AI and IoT Convergence",
                        "AI-Powered Cybersecurity",
                    ]
                    example[field_name] = trend_names[(index - 1) % len(trend_names)]
                elif "description" in field_name.lower():
                    descriptions = [
                        "Large enterprises are integrating generative AI into core business processes",
                        "AI processing moving closer to data sources for real-time decision making",
                        "Advanced threat detection and response using machine learning algorithms",
                    ]
                    example[field_name] = descriptions[(index - 1) % len(descriptions)]
                else:
                    example[field_name] = f"Example {field_description or field_name} {index}"
            elif field_type == "number":
                min_val = field_info.get("minimum", 1.0)
                max_val = field_info.get("maximum", 10.0)
                base_val = (min_val + max_val) / 2
                example[field_name] = round(base_val + (index - 1) * 0.5, 1)
            elif field_type == "integer":
                min_val = field_info.get("minimum", 1)
                max_val = field_info.get("maximum", 10)
                example[field_name] = min(max_val, max(min_val, index + 4))
            elif field_type == "boolean":
                example[field_name] = index % 2 == 1
            elif field_type == "array" and field_info.get("items", {}).get("type") == "string":
                if "industries" in field_name.lower():
                    industry_groups = [
                        ["technology", "finance", "retail"],
                        ["manufacturing", "automotive", "healthcare"],
                        ["cybersecurity", "finance", "government"],
                    ]
                    example[field_name] = industry_groups[(index - 1) % len(industry_groups)]
                else:
                    example[field_name] = [f"item{index}a", f"item{index}b"]
            else:
                example[field_name] = f"value{index}"

        return example

    def _count_nested_structures(self, schema: dict) -> int:
        """Count nested arrays of objects to provide better guidance.

        Analyzes a schema to count how many fields contain arrays of objects,
        which require special formatting guidance to ensure LLM compliance.

        Args:
            schema: JSON schema to analyze.

        Returns:
            int: Number of array fields containing objects or $ref references.

        Note:
            This count is used to determine whether to include additional
            formatting warnings about properly structuring nested arrays.
        """
        count = 0
        properties = schema.get("properties", {})

        for field_info in properties.values():
            if field_info.get("type") == "array":
                items = field_info.get("items", {})
                if items.get("type") == "object" or "$ref" in items:
                    count += 1

        return count

    def _format_json_example(self, data: dict) -> str:
        """Format JSON data as a readable example.

        Converts a dictionary to a formatted JSON string for use in
        LLM prompts and formatting instructions.

        Args:
            data: Dictionary to format as JSON.

        Returns:
            str: Formatted JSON string with 2-space indentation,
                 or string representation if JSON serialization fails.
        """
        import json

        try:
            return json.dumps(data, indent=2)
        except Exception:
            return str(data)

    def execute(
        self,
        task_description: str,
        context: str | None = None,
        streamer_buffer: StreamerBuffer | None = None,
        stream_callback: Callable[[Any], None] | None = None,
        use_thread: bool = False,
    ) -> str | tuple[StreamerBuffer, threading.Thread]:
        """Execute a task using the agent with advanced controls.

        Main method for task execution. Handles rate limiting, timeouts,
        memory integration, knowledge context, retries, and step callbacks.
        Streams responses from the LLM and processes function calls if configured.

        Args:
            task_description: Natural language description of the task to execute.
            context: Optional additional context to include in the prompt.
            streamer_buffer: Optional StreamerBuffer to receive streaming chunks.
            stream_callback: Optional callback function for streaming responses.
            use_thread: If True, executes in background thread and returns immediately
                       with (buffer, thread) tuple. If False, blocks until complete.

        Returns:
            If use_thread=False: str containing the agent's response.
            If use_thread=True: tuple of (StreamerBuffer, Thread) for async consumption.

        Raises:
            ValueError: If the agent is not connected to a Cortex instance.
            TimeoutError: If execution exceeds max_execution_time.
            RuntimeError: If no response is received after max_iterations.
            Exception: Re-raises any exceptions from the underlying LLM execution.

        Side Effects:
            - Records request for rate limiting
            - Updates execution statistics
            - Invokes step callbacks
            - Saves interaction to memory if enabled

        Example:
            result = agent.execute(
                "Analyze the sales data",
                context="Focus on Q4 2024 trends"
            )
        """
        if not self.calute_instance:
            raise ValueError(f"Agent {self.role} not connected to Cortex")

        if use_thread:
            return self._execute_threaded(task_description, context, streamer_buffer, stream_callback)

        if not self._check_rate_limit():
            sleep_time = 60 / self.max_rpm if self.max_rpm else 1
            if self.verbose:
                if self._logger:
                    self._logger.info(f"Sleeping {sleep_time:.1f}s for rate limit")
            time.sleep(sleep_time)

        self._record_request()

        start_time = time.time()
        self._times_executed += 1

        self._execute_step_callback(
            {
                "step": "execution_start",
                "agent": self.role,
                "task": task_description,
                "execution_count": self._times_executed,
            }
        )

        try:
            knowledge_context = self._build_knowledge_context(task_description)
            memory_context = ""
            if self.memory_enabled and self.memory:
                memory_context = self.memory.build_context_for_task(
                    task_description=task_description,
                    agent_role=self.role,
                    additional_context=context,
                    max_items=10,
                )

            full_context = ""
            contexts = [knowledge_context, memory_context, context]
            contexts = [ctx for ctx in contexts if ctx]
            if contexts:
                full_context = "\n\n".join(contexts)

            prompt = self._template_engine.render_task_prompt(
                description=task_description,
                expected_output="",
                context=full_context,
            )

            if self.verbose:
                log_thinking(self.role)

            response = None
            iteration = 0
            while iteration < self.max_iterations:
                if self._check_execution_timeout(start_time):
                    raise TimeoutError(f"Agent execution timed out after {self.max_execution_time}s")

                try:
                    if streamer_buffer is not None or stream_callback is not None:
                        buffer_was_none = streamer_buffer is None
                        if streamer_buffer is None:
                            streamer_buffer = StreamerBuffer()

                        response_gen = self.calute_instance.run(
                            prompt=prompt,
                            agent_id=self._internal_agent,
                            stream=True,
                            apply_functions=True,
                            reinvoke_after_function=self.reinvoke_after_function,
                            streamer_buffer=streamer_buffer,
                        )

                        if stream_callback:
                            collected_content = []
                            final_response = None
                            for chunk in response_gen:
                                stream_callback(chunk)
                                if hasattr(chunk, "content") and chunk.content:
                                    collected_content.append(chunk.content)
                                final_response = chunk

                            response = ResponseResult(
                                content="".join(collected_content),
                                response=final_response,
                                completion=getattr(final_response, "completion", None),
                            )
                        else:
                            final_response = None
                            for chunk in response_gen:
                                final_response = chunk

                            response = ResponseResult(
                                content=getattr(final_response, "final_content", "") if final_response else "",
                                response=final_response,
                                completion=getattr(final_response, "completion", final_response),
                            )

                        if buffer_was_none:
                            streamer_buffer.close()
                    else:
                        response = self.calute_instance.run(
                            prompt=prompt,
                            agent_id=self._internal_agent,
                            stream=False,
                            apply_functions=True,
                            reinvoke_after_function=self.reinvoke_after_function,
                            streamer_buffer=None,
                        )

                    break

                except Exception as e:
                    iteration += 1
                    if iteration >= self.max_iterations:
                        raise e

                    self._execute_step_callback(
                        {
                            "step": "retry",
                            "agent": self.role,
                            "iteration": iteration,
                            "error": str(e),
                        }
                    )

            if not response:
                raise RuntimeError("Failed to get response after maximum iterations")

            if isinstance(response, ResponseResult):
                output = response.completion
                if isinstance(output, Completion):
                    result = output.final_content if output.final_content is not None else response.final_content
                else:
                    result = response.content if hasattr(response, "content") else ""
            elif hasattr(response, "content"):
                result = response.content
            elif hasattr(response, "completion"):
                result = response.completion.content
            else:
                result = str(response)

            execution_time = time.time() - start_time
            self._execution_times.append(execution_time)

            self._execute_step_callback(
                {
                    "step": "execution_complete",
                    "agent": self.role,
                    "execution_time": execution_time,
                    "result_length": len(result),
                }
            )

            if self.memory_enabled and self.memory:
                self.memory.save_agent_interaction(
                    agent_role=self.role,
                    action="execute_task",
                    content=f"Task: {task_description[:512]} - Result: {result}",
                    importance=0.5,
                )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._execution_times.append(execution_time)
            self._execute_step_callback(
                {
                    "step": "execution_error",
                    "agent": self.role,
                    "execution_time": execution_time,
                    "error": str(e),
                }
            )

            if self.verbose:
                log_error(f"Agent {self.role}: {e!s}")

            raise

    def _execute_threaded(
        self,
        task_description: str,
        context: str | None = None,
        streamer_buffer: StreamerBuffer | None = None,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> tuple[StreamerBuffer, threading.Thread]:
        """Execute task in background thread with streaming.

        Internal method that wraps the main execute logic in a thread for
        non-blocking execution with real-time streaming capabilities.

        Args:
            task_description: Natural language description of the task to execute.
            context: Optional additional context to include in the prompt.
            streamer_buffer: Optional StreamerBuffer to use, creates one if not provided.
            stream_callback: Optional callback function for streaming responses.

        Returns:
            tuple: (StreamerBuffer, Thread) for async consumption of results.

        Note:
            This method returns immediately, allowing the caller to consume
            streaming chunks while the agent executes in the background.
        """
        if not self.calute_instance:
            raise ValueError(f"Agent {self.role} not connected to Cortex")

        knowledge_context = self._build_knowledge_context(task_description)
        memory_context = ""
        if self.memory_enabled and self.memory:
            memory_context = self.memory.build_context_for_task(
                task_description=task_description,
                agent_role=self.role,
                additional_context=context,
                max_items=10,
            )

        full_context = ""
        contexts = [knowledge_context, memory_context, context]
        contexts = [ctx for ctx in contexts if ctx]
        if contexts:
            full_context = "\n\n".join(contexts)

        prompt = self._template_engine.render_task_prompt(
            description=task_description,
            expected_output="",
            context=full_context,
        )

        buffer, thread = self.calute_instance.thread_run(
            prompt=prompt,
            agent_id=self._internal_agent,
            apply_functions=True,
            reinvoke_after_function=self.reinvoke_after_function,
            streamer_buffer=streamer_buffer,
        )

        buffer.task_description = task_description  # type: ignore
        buffer.agent_role = self.role  # type: ignore

        if stream_callback:

            def consume_and_callback():
                """Consume from buffer and invoke callback."""
                for chunk in buffer.stream():
                    stream_callback(chunk)

            callback_thread = threading.Thread(target=consume_and_callback, daemon=True)
            callback_thread.start()
            buffer.callback_thread = callback_thread

        return buffer, thread

    def _check_delegation_needed(self, task_description: str, initial_response: str) -> tuple[bool, str]:
        """Check if delegation is needed based on task and initial response.

        Analyzes the initial response and task complexity to determine whether
        the agent should delegate to another agent. Checks for indicators of
        uncertainty or need for assistance, and considers task complexity.

        Args:
            task_description: The original task description.
            initial_response: The agent's initial response to analyze.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: Whether delegation is needed
                - str: Reason for delegation (empty if not needed)

        Note:
            - Delegation is prevented if already at depth 3 to avoid infinite recursion
            - Looks for phrases like "I need help", "I cannot", "beyond my expertise"
            - Also considers task length as complexity indicator (>50 words)
        """
        if not self.allow_delegation or not self.cortex_instance or self._delegation_count >= 3:
            return False, ""

        delegation_indicators = [
            "i need help with",
            "i'm not sure",
            "i cannot",
            "beyond my expertise",
            "would require",
            "need assistance",
            "delegate",
            "ask another agent",
        ]

        response_lower = initial_response.lower()
        for indicator in delegation_indicators:
            if indicator in response_lower:
                return True, "Agent indicated need for assistance"

        if len(task_description.split()) > 50:
            return True, "Task complexity suggests delegation might help"

        return False, ""

    def _select_delegate_agent(self, task_description: str, reason: str) -> Optional["CortexAgent"]:
        """Select the best agent to delegate to.

        Uses the LLM to intelligently select the most appropriate agent
        from available agents based on the task requirements and each
        agent's role and goal.

        Args:
            task_description: Description of the task requiring delegation.
            reason: Reason why delegation is needed.

        Returns:
            Optional[CortexAgent]: The selected agent for delegation,
                                  or None if no suitable agent is found.

        Note:
            - Excludes the current agent from selection
            - Falls back to first available agent if selection fails
            - Uses fuzzy matching to find agent by role name
        """
        if not self.cortex_instance:
            return None

        available_agents = [agent for agent in self.cortex_instance.agents if agent.role != self.role]

        if not available_agents:
            return None

        selection_prompt = f"""
        Current agent: {self.role}
        Task requiring delegation: {task_description}
        Reason for delegation: {reason}

        Available agents to delegate to:
        {chr(10).join([f"- {agent.role}: {agent.goal}" for agent in available_agents])}

        Which agent would be best suited for this task?
        Respond with ONLY the role name of the selected agent, nothing else.
        """

        try:
            response_generator = self.calute_instance.run(
                prompt=selection_prompt,
                agent_id=self._internal_agent,
                stream=True,
                apply_functions=False,
                reinvoke_after_function=False,
            )

            response_content = []
            for chunk in response_generator:
                if hasattr(chunk, "content") and chunk.content is not None:
                    response_content.append(chunk.content)

            class StreamedResponse:
                def __init__(self, content):
                    self.content = content

            response = StreamedResponse("".join(response_content))

            selected_role = response.content.strip() if hasattr(response, "content") else str(response).strip()

            for agent in available_agents:
                if agent.role.lower() == selected_role.lower() or selected_role.lower() in agent.role.lower():
                    return agent
        except Exception as e:
            if self.verbose:
                log_error(f"Agent {self.role} - Failed to select delegate: {e}")

        return available_agents[0] if available_agents else None

    def execute_stream(
        self,
        task_description: str,
        context: str | None = None,
        callback: Callable[[StreamingResponseType], None] | None = None,
    ) -> str:
        """Execute task with real-time streaming and optional callback.

        Convenience method that executes a task in a background thread and
        processes streaming chunks through an optional callback while returning
        the final result.

        Args:
            task_description: Natural language description of the task to execute.
            context: Optional additional context to include in the prompt.
            callback: Optional function to call with each streaming chunk.
                     If not provided, chunks are silently consumed.

        Returns:
            str: The final agent response after all streaming is complete.

        Example:
            def print_chunk(chunk):
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end='', flush=True)

            result = agent.execute_stream(
                "Write a poem",
                callback=print_chunk
            )
        """
        from calute.types import StreamChunk

        buffer, thread = self.execute(task_description=task_description, context=context, use_thread=True)

        for chunk in buffer.stream():
            if callback:
                callback(chunk)

            elif self.verbose and isinstance(chunk, StreamChunk) and chunk.content:
                if self._logger:
                    self._logger.info(f"[{self.role}]: {chunk.content}")

        thread.join(timeout=1.0)
        result = buffer.get_result()

        if hasattr(result, "content"):
            return result.content
        return str(result)

    def delegate_task(self, task_description: str, context: str | None = None) -> str:
        """Delegate a task to another agent.

        Delegates a task to another agent in the Cortex system. Selects the
        most appropriate agent and passes the task with additional context
        about the delegation.

        Args:
            task_description: Description of the task to delegate.
            context: Optional additional context for the delegated task.

        Returns:
            str: Result from the delegate agent, or error message if delegation fails.

        Side Effects:
            - Increments delegation count
            - Logs delegation activity
            - Saves delegation to memory if enabled
            - Passes delegation count to delegate to maintain depth tracking

        Note:
            Returns "Delegation not available" if delegation is disabled or
            no Cortex instance is available.
        """
        if not self.allow_delegation or not self.cortex_instance:
            return "Delegation not available"

        self._delegation_count += 1

        delegate = self._select_delegate_agent(task_description, "Delegation requested")

        if not delegate:
            self._delegation_count -= 1
            return "No suitable agent found for delegation"

        if self.verbose:
            log_delegation(self.role, delegate.role)

        delegation_context = f"""
        This task has been delegated from {self.role}.
        Original context: {context or "No additional context"}
        Please complete this task to the best of your ability.
        """

        delegate._delegation_count = self._delegation_count
        result = delegate.execute(task_description, delegation_context)

        if self.verbose:
            if self._logger:
                self._logger.info(f"✅ Delegation from {self.role} to {delegate.role} complete")

        if self.memory_enabled and self.memory:
            self.memory.save_agent_interaction(
                agent_role=self.role,
                action="delegated_task",
                content=f"Delegated to {delegate.role}: {task_description[:100]}",
                importance=0.6,
            )

        self._delegation_count -= 1
        return result

    def execute_with_delegation(self, task_description: str, context: str | None = None) -> str:
        """Execute task with automatic delegation if needed and allowed.

        Executes a task with intelligent delegation support. First attempts
        to execute the task directly, then checks if delegation would be
        beneficial. If delegation occurs, combines both responses into a
        comprehensive final answer.

        Args:
            task_description: Description of the task to execute.
            context: Optional additional context.

        Returns:
            str: Final response, potentially combining insights from multiple agents.

        Note:
            - Automatically determines if delegation is needed based on initial response
            - Combines delegated results with initial response for comprehensive answer
            - Falls back to delegated result if combination fails
            - Returns initial result if delegation is not needed or not available

        Example:

            result = agent.execute_with_delegation(
                "Create a marketing strategy with technical implementation details"
            )
        """

        initial_result = self.execute(task_description, context)

        needs_delegation, reason = self._check_delegation_needed(task_description, initial_result)

        if needs_delegation and self.allow_delegation and self.cortex_instance:
            if self.verbose:
                if self._logger:
                    self._logger.info(f"Agent {self.role} considering delegation: {reason}")

            delegated_result = self.delegate_task(task_description, context)

            final_prompt = f"""
            You initially responded with: {initial_result}

            After delegating to another agent, they provided: {delegated_result}

            Please provide a final, comprehensive response combining both insights.
            """

            try:
                response_generator = self.calute_instance.run(
                    prompt=final_prompt,
                    agent_id=self._internal_agent,
                    stream=True,
                    apply_functions=False,
                    reinvoke_after_function=False,
                )

                response_content = []
                for chunk in response_generator:
                    if hasattr(chunk, "content") and chunk.content is not None:
                        response_content.append(chunk.content)

                class StreamedResponse:
                    def __init__(self, content):
                        self.content = content

                final_response = StreamedResponse("".join(response_content))

                result = final_response.content if hasattr(final_response, "content") else str(final_response)
                return result
            except Exception:
                return delegated_result

        return initial_result

    def get_execution_stats(self) -> dict:
        """Get execution statistics.

        Compiles comprehensive statistics about the agent's execution history,
        including timing metrics and execution counts.

        Returns:
            dict: Statistics dictionary containing:
                - times_executed: Total number of executions
                - avg_execution_time: Average execution duration in seconds
                - total_execution_time: Sum of all execution times
                - min_execution_time: Fastest execution time
                - max_execution_time: Slowest execution time
                - recent_execution_times: Last 5 execution durations (if available)

        Note:
            Returns zero values for timing metrics if no executions have occurred.
        """
        if not self._execution_times:
            return {
                "times_executed": self._times_executed,
                "avg_execution_time": 0,
                "total_execution_time": 0,
                "min_execution_time": 0,
                "max_execution_time": 0,
            }

        return {
            "times_executed": self._times_executed,
            "avg_execution_time": sum(self._execution_times) / len(self._execution_times),
            "total_execution_time": sum(self._execution_times),
            "min_execution_time": min(self._execution_times),
            "max_execution_time": max(self._execution_times),
            "recent_execution_times": self._execution_times[-5:],
        }

    def reset_stats(self):
        """Reset execution statistics.

        Clears all execution statistics and counters, returning the agent
        to a fresh state for metric tracking.

        Side Effects:
            - Resets _times_executed to 0
            - Clears _execution_times list
            - Clears _rpm_requests list
            - Resets _delegation_count to 0
        """
        self._times_executed = 0
        self._execution_times.clear()
        self._rpm_requests.clear()
        self._delegation_count = 0

    def add_knowledge(self, key: str, value: str):
        """Add knowledge to the agent's knowledge base.

        Adds or updates a knowledge entry that will be included in the
        agent's context during task execution.

        Args:
            key: Knowledge identifier or category.
            value: Knowledge content or description.

        Side Effects:
            Updates the knowledge dictionary.

        Example:
            agent.add_knowledge("company_policy", "Always prioritize customer satisfaction")
            agent.add_knowledge("domain_expertise", "Specialized in healthcare analytics")
        """
        self.knowledge[key] = value

    def add_knowledge_source(self, source: str):
        """Add a knowledge source.

        Registers a knowledge source identifier that represents an external
        source of information available to the agent.

        Args:
            source: Knowledge source identifier (e.g., database name, API endpoint).

        Side Effects:
            Appends to knowledge_sources list if not already present.

        Note:
            Duplicate sources are automatically prevented.
        """
        if source not in self.knowledge_sources:
            self.knowledge_sources.append(source)

    def update_config(self, key: str, value):
        """Update agent configuration.

        Sets or updates a configuration parameter for the agent.
        Configuration can be used to store agent-specific settings
        or runtime parameters.

        Args:
            key: Configuration parameter name.
            value: Configuration parameter value (any type).

        Side Effects:
            Updates the config dictionary.
        """
        self.config[key] = value

    def get_config(self, key: str, default=None):
        """Get configuration value.

        Retrieves a configuration parameter value with optional default.

        Args:
            key: Configuration parameter name to retrieve.
            default: Default value if key is not found.

        Returns:
            The configuration value for the key, or default if not found.
        """
        return self.config.get(key, default)

    def set_step_callback(self, callback: Callable):
        """Set step callback function.

        Registers a callback function that will be invoked at each step
        of task execution, useful for monitoring and debugging.

        Args:
            callback: Callable that accepts a dict parameter containing step information.
                     The dict includes fields like 'step', 'agent', 'task', etc.

        Example:
            def my_callback(step_info):
                print(f"Step: {step_info['step']} for agent: {step_info['agent']}")

            agent.set_step_callback(my_callback)
        """
        self.step_callback = callback

    def is_rate_limited(self) -> bool:
        """Check if agent is currently rate limited.

        Determines whether the agent would be rate limited if it attempted
        to make a request right now.

        Returns:
            bool: True if currently rate limited, False otherwise.

        Note:
            This is a read-only check that doesn't modify rate limit state.
        """
        return not self._check_rate_limit()

    def get_rate_limit_status(self) -> dict:
        """Get current rate limiting status.

        Provides detailed information about the current rate limiting state,
        including requests made and remaining capacity.

        Returns:
            dict: Rate limit status containing:
                - rate_limited: Whether currently rate limited
                - max_rpm: Maximum requests per minute allowed (None if no limit)
                - current_requests: Number of requests in current 60-second window
                - requests_remaining: Available requests before hitting limit

        Example:
            status = agent.get_rate_limit_status()
            if status['requests_remaining'] < 5:
                print("Approaching rate limit")
        """
        if not self.max_rpm:
            return {"rate_limited": False, "max_rpm": None, "current_requests": 0}

        current_time = time.time()
        recent_requests = [req for req in self._rpm_requests if current_time - req < 60]

        return {
            "rate_limited": len(recent_requests) >= self.max_rpm,
            "max_rpm": self.max_rpm,
            "current_requests": len(recent_requests),
            "requests_remaining": max(0, self.max_rpm - len(recent_requests)),
        }

    def create_ui(self):
        from calute.ui import create_application

        return create_application(executor=self)

    def __eq__(self, other: object) -> bool:
        """Check equality between two CortexAgent instances.

        Two agents are considered equal if they have the same role, goal,
        backstory, and model configuration.

        Args:
            other: Another object to compare with.

        Returns:
            bool: True if agents are equal, False otherwise.
        """
        if not isinstance(other, CortexAgent):
            return False
        return (
            self.role == other.role
            and self.goal == other.goal
            and self.backstory == other.backstory
            and self.model == other.model
        )

    def __hash__(self) -> int:
        """Generate a hash value for the CortexAgent using SHA256.

        The hash is computed from the agent's role, goal, backstory, and model
        to ensure consistent hashing based on the agent's core identity.

        Returns:
            int: Hash value derived from SHA256 digest of agent attributes.

        Example:
            agent1 = CortexAgent(role="Analyst", goal="Analyze data", ...)
            agent2 = CortexAgent(role="Analyst", goal="Analyze data", ...)
            assert hash(agent1) == hash(agent2)
        """

        identity_str = f"{self.role}|{self.goal}|{self.backstory}|{self.model or 'default'}"

        sha256_hash = hashlib.sha256(identity_str.encode("utf-8")).digest()

        return int.from_bytes(sha256_hash[:8], byteorder="big", signed=False)
