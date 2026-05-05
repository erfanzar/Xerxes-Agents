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
"""Core agent implementation for the Cortex orchestration framework."""

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel

from xerxes.core.streamer_buffer import StreamerBuffer
from xerxes.llms.base import BaseLLM
from xerxes.types.function_execution_types import Completion, ResponseResult, StreamingResponseType
from xerxes.types.tool_calls import Tool

from ...agents.auto_compact_agent import AutoCompactAgent
from ...logging.console import get_logger, log_delegation, log_error, log_success, log_thinking, log_warning
from ...types import Agent as XerxesAgent
from ...types import AgentCapability
from ...types.function_execution_types import CompactionStrategy
from ..core.string_utils import interpolate_inputs
from ..core.templates import PromptTemplate
from ..core.tool import CortexTool

if TYPE_CHECKING:
    from ...xerxes import Xerxes
    from ..orchestration.cortex import Cortex
    from .memory_integration import CortexMemory


@dataclass
class CortexAgent:
    """An agent within the Cortex framework that executes tasks via an LLM.

    Wraps a ``XerxesAgent`` with additional capabilities such as memory
    integration, rate limiting, delegation, auto-compaction, format guidance,
    and MCP tool attachment.

    Attributes:
        role (str): The agent's role name.
        goal (str): The agent's primary objective.
        backstory (str): Narrative context shaping the agent's behavior.
        model (str | None): LLM model identifier.
        instructions (str | None): System instructions for the agent.
        tools (list): Available tools (callables, ``CortexTool``, or ``Tool``).
        max_iterations (int): Maximum retry iterations on failure.
        verbose (bool): Whether to log agent activity.
        allow_delegation (bool): Whether the agent may delegate to peers.
        temperature (float): LLM sampling temperature.
        max_tokens (int): Maximum tokens per LLM response.
        memory_enabled (bool): Whether to use memory context.
        capabilities (list[AgentCapability]): Declared agent capabilities.
        xerxes_instance (Xerxes | None): The parent Xerxes LLM runner.
        memory (CortexMemory | None): Memory subsystem for context and recall.
        llm (BaseLLM | None): Direct LLM backend (used to build a Xerxes instance).
        reinvoke_after_function (bool): Whether to reinvoke after tool execution.
        cortex_instance (Cortex | None): The parent Cortex orchestrator.
        max_execution_time (int | None): Execution timeout in seconds.
        max_rpm (int | None): Maximum requests per minute.
        step_callback (Callable | None): Callback for execution step events.
        config (dict): Arbitrary configuration dictionary.
        knowledge (dict): Key-value knowledge entries.
        knowledge_sources (list): Source references for knowledge.
        auto_format_guidance (bool): Whether to inject format guidance.
        output_format_preference (Literal["xml", "json"]): Preferred output format.
        auto_compact (bool): Whether to auto-compact conversation history.
        compact_threshold (float): Token ratio threshold for compaction.
        compact_target (float): Target token ratio after compaction.
        max_context_tokens (int | None): Maximum context token budget.
        compaction_strategy (CompactionStrategy): Strategy for compaction.
        preserve_system_prompt (bool): Whether to preserve the system prompt.
        preserve_recent_messages (int): Number of recent messages to preserve.
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
    xerxes_instance: "Xerxes | None" = None
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
    output_format_preference: Literal["xml", "json"] = "xml"

    auto_compact: bool = False
    compact_threshold: float = 0.8
    compact_target: float = 0.5
    max_context_tokens: int | None = None
    compaction_strategy: CompactionStrategy = CompactionStrategy.SUMMARIZE
    preserve_system_prompt: bool = True
    preserve_recent_messages: int = 5

    _internal_agent: XerxesAgent | None = None
    _auto_compact_agent: AutoCompactAgent | None = None
    _conversation_history: list[dict[str, str]] = field(default_factory=list)
    _messages_history: Any = None
    _logger: Any | None = None
    _template_engine: PromptTemplate | None = None
    _delegation_count: int = 0
    _times_executed: int = 0
    _execution_times: list = field(default_factory=list)
    _rpm_requests: list = field(default_factory=list)
    _last_rpm_window: float = 0

    _original_role: str | None = None
    _original_goal: str | None = None
    _original_backstory: str | None = None
    _original_instructions: str | None = None

    def __post_init__(self):
        """Finalize initialization after dataclass field assignment.

        Sets up the logger, template engine, internal XerxesAgent, and
        optional auto-compaction agent.
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

        if self.xerxes_instance is None and self.llm is not None:
            from xerxes.xerxes import Xerxes

            self.xerxes_instance = Xerxes(llm=self.llm)
        self._internal_agent = XerxesAgent(
            id=self.role.lower().replace(" ", "_")[:32],
            name=self.role,
            instructions=self.instructions,
            model=self.model,
            functions=functions,
            capabilities=self.capabilities if self.capabilities else [],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            parallel_tool_calls=True,
        )

        if self.auto_compact:
            llm_for_compaction = None
            if hasattr(self, "xerxes_instance") and self.xerxes_instance:
                llm_for_compaction = self.xerxes_instance
            elif hasattr(self, "llm") and self.llm:
                llm_for_compaction = self.llm

            self._auto_compact_agent = AutoCompactAgent(
                model=self.model,
                auto_compact=True,
                compact_threshold=self.compact_threshold,
                compact_target=self.compact_target,
                max_context_tokens=self.max_context_tokens,
                compaction_strategy=self.compaction_strategy,
                preserve_system_prompt=self.preserve_system_prompt,
                preserve_recent_messages=self.preserve_recent_messages,
                llm_client=llm_for_compaction,
                verbose=self.verbose,
            )

    @property
    def functions(self) -> list[Callable]:
        """Return the internal agent's function list.

        Returns:
            list[Callable]: The list of callable tools/functions.
                OUT: Empty if the internal agent is not initialized.
        """

        if self._internal_agent is None:
            return []
        return self._internal_agent.functions

    @functions.setter
    def functions(self, value: list[Callable] | None) -> None:
        """Set the internal agent's function list.

        Args:
            value (list[Callable] | None): The new function list.
                IN: Replaces the existing functions.
                OUT: Assigned to ``self._internal_agent.functions``.
        """

        if self._internal_agent is None:
            return
        self._internal_agent.functions = value or []

    def get_compaction_stats(self) -> dict[str, Any] | None:
        """Return statistics from the auto-compaction agent.

        Returns:
            dict[str, Any] | None: Compaction metrics, or ``None`` if disabled.
                OUT: Delegated to ``AutoCompactAgent.get_statistics``.
        """

        if self._auto_compact_agent:
            return self._auto_compact_agent.get_statistics()
        return None

    def check_context_usage(self) -> dict[str, Any] | None:
        """Return context usage metrics from the auto-compaction agent.

        Returns:
            dict[str, Any] | None: Usage metrics, or ``None`` if disabled.
                OUT: Delegated to ``AutoCompactAgent.check_usage``.
        """

        if self._auto_compact_agent:
            return self._auto_compact_agent.check_usage()
        return None

    def _check_rate_limit(self) -> bool:
        """Check whether the agent is within its rate limit.

        Returns:
            bool: ``True`` if the request may proceed, ``False`` if rate limited.
                OUT: Compares recent requests against ``max_rpm``.
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
        """Record the current request timestamp for rate limiting."""

        if self.max_rpm:
            self._rpm_requests.append(time.time())

    def interpolate_inputs(self, inputs: dict[str, Any]) -> None:
        """Substitute template variables into role, goal, backstory, and instructions.

        Args:
            inputs (dict): Mapping of template variable names to values.
                IN: Applied to the original (uninterpolated) versions of text fields.
                OUT: Updates the agent's descriptive fields and internal agent instructions.
        """

        if self._original_role is None:
            self._original_role = self.role
        if self._original_goal is None:
            self._original_goal = self.goal
        if self._original_backstory is None:
            self._original_backstory = self.backstory
        if self._original_instructions is None:
            self._original_instructions = self.instructions

        if inputs:
            self.role = interpolate_inputs(input_string=self._original_role, inputs=inputs)
            self.goal = interpolate_inputs(input_string=self._original_goal, inputs=inputs)
            self.backstory = interpolate_inputs(input_string=self._original_backstory, inputs=inputs)
            if self.instructions:
                self.instructions = interpolate_inputs(input_string=self._original_instructions, inputs=inputs)

            if self._internal_agent:
                self._internal_agent.instructions = self.instructions

    def attach_mcp(self, mcp_servers: Any, server_names: list[str] | None = None) -> None:
        """Attach MCP (Model Context Protocol) servers to the agent.

        Args:
            mcp_servers (Any): An ``MCPManager``, ``MCPServerConfig``, or list of configs.
                IN: Used to set up MCP tool access.
                OUT: Tools are added to the internal agent via ``add_mcp_tools_to_agent``.
            server_names (list[str] | None): Optional subset of server names to include.
                IN: Filters which MCP servers contribute tools.
                OUT: Passed to ``add_mcp_tools_to_agent``.

        Raises:
            TypeError: If *mcp_servers* is not a recognized type.
        """

        from ...core.utils import run_sync
        from ..mcp import MCPManager, MCPServerConfig
        from ..mcp.integration import add_mcp_tools_to_agent

        if isinstance(mcp_servers, MCPManager):
            manager = mcp_servers
        elif isinstance(mcp_servers, MCPServerConfig):
            manager = MCPManager()
            run_sync(manager.add_server(mcp_servers))
        elif isinstance(mcp_servers, list):
            manager = MCPManager()
            for config in mcp_servers:
                if isinstance(config, MCPServerConfig):
                    run_sync(manager.add_server(config))
                else:
                    raise TypeError(f"Expected MCPServerConfig in list, got {type(config)}")
        else:
            raise TypeError(f"Expected MCPManager, MCPServerConfig, or list, got {type(mcp_servers)}")

        if self._internal_agent:
            run_sync(add_mcp_tools_to_agent(self._internal_agent, manager, server_names))
        else:
            if self.verbose and self._logger:
                self._logger.warning(
                    "Internal agent not yet initialized. MCP tools will be added during initialization."
                )

        if not hasattr(self, "_mcp_managers"):
            self._mcp_managers = []
        self._mcp_managers.append(manager)

    def _check_execution_timeout(self, start_time: float) -> bool:
        """Check whether execution has exceeded ``max_execution_time``.

        Args:
            start_time (float): The execution start timestamp.
                IN: Compared against the current time.
                OUT: Used to compute elapsed duration.

        Returns:
            bool: ``True`` if the timeout has been exceeded.
                OUT: Triggers a ``TimeoutError`` in the caller.
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
        """Invoke the user-registered step callback safely.

        Args:
            step_info (dict): Information about the current execution step.
                IN: Passed to ``self.step_callback``.
                OUT: Enables external observation of agent execution progress.
        """

        if self.step_callback and callable(self.step_callback):
            try:
                self.step_callback(step_info)
            except Exception as e:
                if self.verbose:
                    log_error(f"Step callback failed: {e}")

    def _build_knowledge_context(self, task_description: str) -> str:
        """Assemble knowledge entries and sources into a context string.

        Args:
            task_description (str): The current task (unused but kept for API consistency).
                IN: Reserved for future relevance filtering.
                OUT: Not currently used in context assembly.

        Returns:
            str: A formatted knowledge context string.
                OUT: Empty if no knowledge or sources are configured.
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
        """Generate format instructions based on a Pydantic model schema.

        Args:
            output_model: A ``BaseModel`` subclass or ``None``.
                IN: Its JSON schema drives the generated guidance.
                OUT: Used to create an example and formatting rules.

        Returns:
            str: Format instructions for the LLM, or an empty string.
                OUT: XML or JSON guidance depending on ``output_format_preference``.
        """

        if not output_model or not self.auto_format_guidance:
            return ""

        try:
            if not issubclass(output_model, BaseModel):
                return ""

            if hasattr(output_model, "model_json_schema"):
                schema = output_model.model_json_schema()
            else:
                schema = output_model.schema()
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
        """Recursively resolve ``$ref`` references in a JSON schema.

        Args:
            schema (dict): The schema dict that may contain ``$ref``.
                IN: Walked recursively to resolve references.
                OUT: Returns a copy with all ``$ref`` values inlined.
            definitions (dict | None): The schema's definitions or ``$defs``.
                IN: Used to look up referenced types.
                OUT: Passed down during recursive resolution.

        Returns:
            dict: A schema with all references resolved.
                OUT: Safe for example generation and validation.
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
        """Generate an example dictionary from a JSON schema.

        Args:
            schema (dict): A JSON schema dict with ``properties``.
                IN: Used to infer field types and generate realistic examples.
                OUT: Resolved for ``$ref`` before traversal.

        Returns:
            dict: A dictionary with example values for each property.
                OUT: Suitable for embedding in LLM prompts.
        """

        resolved_schema = self._resolve_schema_refs(schema)

        properties = resolved_schema.get("properties", {})
        definitions = schema.get("definitions", {})
        example: dict[str, Any] = {}

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
        """Generate an example for a nested object schema.

        Args:
            schema (dict): The nested object's JSON schema.
                IN: Should have a ``properties`` key.
                OUT: Used to generate example values per field.
            index (int): A numeric index to vary example values.
                IN: Used to rotate through example data sets.
                OUT: Produces distinct examples for array items.

        Returns:
            dict: A dictionary with example values for the nested object.
                OUT: Suitable for embedding in LLM prompts.
        """

        properties = schema.get("properties", {})

        example: dict[str, Any] = {}
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
        """Count the number of array-of-object properties in a schema.

        Args:
            schema (dict): A JSON schema dict.
                IN: Inspected for nested array-of-object structures.
                OUT: Used to decide whether to add extra formatting guidance.

        Returns:
            int: The number of nested array-of-object fields.
                OUT: A positive count triggers stronger formatting rules.
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
        """Format a dictionary as an indented JSON string.

        Args:
            data (dict): The dictionary to serialize.
                IN: Typically an example generated from a schema.
                OUT: Rendered as a pretty-printed JSON string.

        Returns:
            str: Indented JSON or ``str(data)`` on failure.
                OUT: Suitable for embedding in LLM prompts.
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
        """Execute a task through the agent.

        Args:
            task_description (str): The task to execute.
                IN: Rendered into a prompt and sent to the LLM.
                OUT: Drives the agent's response generation.
            context (str | None): Additional context for the task.
                IN: Combined with knowledge and memory context.
                OUT: Appended to the task prompt.
            streamer_buffer (StreamerBuffer | None): Optional buffer for streaming.
                IN: If provided, the agent streams its response.
                OUT: Receives chunks during execution.
            stream_callback (Callable | None): Callback for streamed chunks.
                IN: Invoked for each chunk during streaming.
                OUT: Enables real-time observation.
            use_thread (bool): Whether to execute in a background thread.
                IN: If ``True``, returns ``(StreamerBuffer, Thread)``.
                OUT: Delegates to ``_execute_threaded``.

        Returns:
            str | tuple[StreamerBuffer, threading.Thread]: The result string, or
                streaming handles when *use_thread* is ``True``.
                OUT: ``str`` on normal completion; tuple for threaded execution.

        Raises:
            ValueError: If the agent is not connected to a Xerxes instance.
            TimeoutError: If execution exceeds ``max_execution_time``.
            RuntimeError: If the template engine is not initialized or no response is received.
        """

        if not self.xerxes_instance:
            raise ValueError(f"Agent {self.role} not connected to Xerxes instance")

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
            contexts: list[str] = [ctx for ctx in [knowledge_context, memory_context, context] if ctx is not None]
            if contexts:
                full_context = "\n\n".join(contexts)

            if self._template_engine is None:
                raise RuntimeError("Template engine not initialized")
            prompt = self._template_engine.render_task_prompt(
                description=task_description,
                expected_output="",
                context=full_context,
            )

            if self._auto_compact_agent and self._messages_history is None:
                from xerxes.types.messages import MessagesHistory

                self._messages_history = MessagesHistory(messages=[])

            if self._auto_compact_agent and self._messages_history:
                from xerxes.types.messages import UserMessage

                self._messages_history.messages.append(UserMessage(content=prompt))

            if self._auto_compact_agent and self._messages_history:
                messages = []
                for msg in self._messages_history.messages:
                    messages.append({"role": msg.role, "content": msg.content or ""})

                conversation_tokens = self._auto_compact_agent.token_counter.count_tokens(messages)

                if conversation_tokens >= self._auto_compact_agent.threshold_tokens:
                    if self.verbose:
                        log_warning(
                            f" Conversation history: {conversation_tokens} tokens - compacting to fit {self._auto_compact_agent.max_context_tokens} limit"
                        )

                    compacted_messages, _stats = self._auto_compact_agent.compact(messages)

                    if compacted_messages:
                        from xerxes.types.messages import (
                            AssistantMessage,
                            MessagesHistory,
                            SystemMessage,
                            UserMessage,
                        )

                        new_messages: list[SystemMessage | UserMessage | AssistantMessage] = []
                        for msg in compacted_messages:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")

                            if role == "system":
                                new_messages.append(SystemMessage(content=content))
                            elif role == "assistant":
                                new_messages.append(AssistantMessage(content=content))
                            else:
                                new_messages.append(UserMessage(content=content))

                        self._messages_history = MessagesHistory(messages=list(new_messages))

                        new_conversation_tokens = self._auto_compact_agent.token_counter.count_tokens(compacted_messages)
                        if self.verbose:
                            log_success(
                                f"Conversation compacted: {conversation_tokens} → {new_conversation_tokens} tokens "
                                f"(saved {conversation_tokens - new_conversation_tokens} tokens, {((conversation_tokens - new_conversation_tokens) / conversation_tokens * 100):.1f}% reduction)"
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

                        from typing import cast

                        response_gen = self.xerxes_instance.run(
                            prompt=prompt,
                            messages=self._messages_history,
                            agent_id=self._internal_agent,
                            stream=True,
                            apply_functions=True,
                            reinvoke_after_function=self.reinvoke_after_function,
                            streamer_buffer=streamer_buffer,
                        )

                        if isinstance(response_gen, ResponseResult):
                            response = response_gen
                        elif stream_callback:
                            collected_content = []
                            final_response: Any = None
                            for chunk in response_gen:
                                stream_callback(chunk)
                                if hasattr(chunk, "content") and chunk.content:
                                    collected_content.append(chunk.content)
                                final_response = chunk

                            response = ResponseResult(
                                content="".join(collected_content),
                                response=cast(Any, final_response),
                                completion=cast(Any, getattr(final_response, "completion", None)),
                            )
                        else:
                            final_response = None
                            for chunk in response_gen:
                                final_response = chunk

                            response = ResponseResult(
                                content=getattr(final_response, "final_content", "") if final_response else "",
                                response=cast(Any, final_response),
                                completion=cast(Any, getattr(final_response, "completion", final_response)),
                            )

                        if buffer_was_none:
                            streamer_buffer.close()
                    else:
                        response = cast(
                            ResponseResult,
                            self.xerxes_instance.run(
                                prompt=prompt,
                                messages=self._messages_history,
                                agent_id=self._internal_agent,
                                stream=False,
                                apply_functions=True,
                                reinvoke_after_function=self.reinvoke_after_function,
                                streamer_buffer=None,
                            ),
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

            if self._auto_compact_agent and result:
                from xerxes.types.messages import AssistantMessage

                self._messages_history.messages.append(AssistantMessage(content=result))

                response_message = {"role": "assistant", "content": result}
                self._conversation_history.append(response_message)

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
        """Execute a task in a background thread with optional streaming.

        Args:
            task_description (str): The task to execute.
                IN: Rendered into a prompt and sent to the LLM in the thread.
                OUT: Drives the agent's response generation.
            context (str | None): Additional context for the task.
                IN: Combined with knowledge and memory context.
                OUT: Appended to the task prompt.
            streamer_buffer (StreamerBuffer | None): Optional pre-existing buffer.
                IN: Used for streaming if provided.
                OUT: Passed to ``xerxes_instance.thread_run``.
            stream_callback (Callable | None): Callback for streamed chunks.
                IN: If provided, a consumer thread is started to forward chunks.
                OUT: Invoked for each chunk from the buffer.

        Returns:
            tuple[StreamerBuffer, threading.Thread]: The streaming buffer and
                the execution thread. OUT: The buffer yields chunks; the thread
                runs the agent.

        Raises:
            ValueError: If the agent is not connected to a Xerxes instance.
            RuntimeError: If the template engine is not initialized.
        """

        if not self.xerxes_instance:
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
        contexts: list[str] = [ctx for ctx in [knowledge_context, memory_context, context] if ctx is not None]
        if contexts:
            full_context = "\n\n".join(contexts)

        if self._template_engine is None:
            raise RuntimeError("Template engine not initialized")
        prompt = self._template_engine.render_task_prompt(
            description=task_description,
            expected_output="",
            context=full_context,
        )

        buffer, thread = self.xerxes_instance.thread_run(
            prompt=prompt,
            agent_id=self._internal_agent,
            apply_functions=True,
            reinvoke_after_function=self.reinvoke_after_function,
            streamer_buffer=streamer_buffer,
        )

        setattr(buffer, "task_description", task_description)
        setattr(buffer, "agent_role", self.role)

        if stream_callback:

            def consume_and_callback() -> None:
                """Consume the buffer stream and forward chunks to the callback.

                Args:
                    None: Closure over *buffer* and *stream_callback*.
                """
                for chunk in buffer.stream():
                    stream_callback(chunk)

            callback_thread = threading.Thread(target=consume_and_callback, daemon=True)
            callback_thread.start()
            setattr(buffer, "callback_thread", callback_thread)

        return buffer, thread

    def _check_delegation_needed(self, task_description: str, initial_response: str) -> tuple[bool, str]:
        """Determine whether the agent's response indicates a need for delegation.

        Args:
            task_description (str): The original task.
                IN: Used as a complexity heuristic (word count).
                OUT: Tasks longer than 50 words may trigger delegation.
            initial_response (str): The agent's initial output.
                IN: Scanned for delegation indicators.
                OUT: Matched against a list of submissive phrases.

        Returns:
            tuple[bool, str]: Whether delegation is recommended and a reason string.
                OUT: ``(False, "")`` when delegation is unnecessary.
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
        """Select the most suitable peer agent for delegation.

        Args:
            task_description (str): The task requiring delegation.
                IN: Included in the selection prompt sent to the LLM.
                OUT: Used to evaluate which peer is best suited.
            reason (str): Why delegation was triggered.
                IN: Included in the selection prompt.
                OUT: Provides context for the LLM's selection.

        Returns:
            CortexAgent | None: The selected peer agent, or ``None`` if unavailable.
                OUT: Falls back to the first available agent if LLM selection fails.
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
            if self.xerxes_instance is None:
                raise RuntimeError("Xerxes instance not available")
            response_generator = self.xerxes_instance.run(
                prompt=selection_prompt,
                agent_id=self._internal_agent,
                stream=True,
                apply_functions=False,
                reinvoke_after_function=False,
            )

            if isinstance(response_generator, ResponseResult):
                selected_role = (
                    response_generator.content if hasattr(response_generator, "content") else str(response_generator)
                )
            else:
                response_content = []
                for chunk in response_generator:
                    if hasattr(chunk, "content") and chunk.content is not None:
                        response_content.append(chunk.content)

                selected_role = "".join(response_content).strip()

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
        """Execute a task with streaming and return the final result.

        Args:
            task_description (str): The task to execute.
                IN: Passed to ``execute`` with ``use_thread=True``.
                OUT: Drives the agent's response generation.
            context (str | None): Additional context for the task.
                IN: Passed to ``execute``.
                OUT: Included in the agent's prompt.
            callback (Callable | None): Callback for each streamed chunk.
                IN: Invoked for every chunk received from the stream.
                OUT: Enables real-time observation.

        Returns:
            str: The concatenated final output.
                OUT: Extracted from the buffer after the thread completes.
        """

        from xerxes.types import StreamChunk

        exec_result = self.execute(task_description=task_description, context=context, use_thread=True)
        if isinstance(exec_result, str):
            return exec_result
        buffer, thread = exec_result

        for chunk in buffer.stream():
            if callback:
                callback(chunk)

            elif self.verbose and isinstance(chunk, StreamChunk) and chunk.content:
                if self._logger:
                    self._logger.info(f"[{self.role}]: {chunk.content}")

        thread.join(timeout=1.0)
        if buffer.get_result is not None:
            result = buffer.get_result(1.0)
        else:
            result = ""

        if hasattr(result, "content"):
            return result.content
        return str(result)

    def delegate_task(self, task_description: str, context: str | None = None) -> str:
        """Delegate a task to a peer agent within the same Cortex.

        Args:
            task_description (str): The task to delegate.
                IN: Passed to the selected delegate agent's ``execute`` method.
                OUT: Drives the delegate's response generation.
            context (str | None): Additional context for the delegate.
                IN: Included in the delegate's prompt.
                OUT: Prepended with delegation metadata.

        Returns:
            str: The result from the delegate agent.
                OUT: Empty or error message if delegation is unavailable.
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
        _delegated = delegate.execute(task_description, delegation_context)
        result = (
            _delegated
            if isinstance(_delegated, str)
            else _delegated[0].get_result(1.0)
            if (_delegated[0].get_result is not None)
            else str(_delegated[0])
        )

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
        """Execute a task and automatically delegate if the response suggests it.

        Args:
            task_description (str): The task to execute.
                IN: First executed directly; then evaluated for delegation need.
                OUT: May be forwarded to a delegate agent.
            context (str | None): Additional context.
                IN: Passed to both initial and delegated execution.
                OUT: Included in all prompts.

        Returns:
            str: The final result, potentially from a delegate agent.
                OUT: Combines initial and delegated insights when delegation occurs.
        """

        initial_result = self.execute(task_description, context)
        if not isinstance(initial_result, str):
            initial_result = (
                initial_result[0].get_result(1.0)
                if (initial_result[0].get_result is not None)
                else str(initial_result[0])
            )

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
                if self.xerxes_instance is None:
                    raise RuntimeError("Xerxes instance not available")
                response_generator = self.xerxes_instance.run(
                    prompt=final_prompt,
                    agent_id=self._internal_agent,
                    stream=True,
                    apply_functions=False,
                    reinvoke_after_function=False,
                )

                if isinstance(response_generator, ResponseResult):
                    return (
                        response_generator.content if hasattr(response_generator, "content") else str(response_generator)
                    )

                response_content = []
                for chunk in response_generator:
                    if hasattr(chunk, "content") and chunk.content is not None:
                        response_content.append(chunk.content)

                return "".join(response_content)
            except Exception:
                return delegated_result

        return initial_result

    def get_execution_stats(self) -> dict:
        """Return aggregated execution statistics for the agent.

        Returns:
            dict: Contains ``times_executed``, average/min/max execution times,
                and recent execution times. OUT: Empty averages if no executions yet.
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
        """Reset all execution and rate-limit statistics."""

        self._times_executed = 0
        self._execution_times.clear()
        self._rpm_requests.clear()
        self._delegation_count = 0

    def add_knowledge(self, key: str, value: str):
        """Add a key-value entry to the agent's knowledge base.

        Args:
            key (str): The knowledge entry key.
                IN: Used as the dictionary key.
                OUT: Retrievable via ``self.knowledge``.
            value (str): The knowledge entry value.
                IN: Stored as the dictionary value.
                OUT: Included in ``_build_knowledge_context``.
        """

        self.knowledge[key] = value

    def add_knowledge_source(self, source: str):
        """Add a source reference to the agent's knowledge.

        Args:
            source (str): A knowledge source string.
                IN: Appended to ``self.knowledge_sources`` if unique.
                OUT: Included in ``_build_knowledge_context``.
        """

        if source not in self.knowledge_sources:
            self.knowledge_sources.append(source)

    def update_config(self, key: str, value: Any) -> None:
        """Update the agent's configuration dictionary.

        Args:
            key (str): The configuration key.
                IN: Used as the dictionary key.
                OUT: Stored in ``self.config``.
            value (Any): The configuration value.
                IN: Stored as the dictionary value.
                OUT: Retrievable via ``get_config``.
        """

        self.config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the agent's configuration.

        Args:
            key (str): The configuration key to look up.
                IN: Used for dictionary lookup.
                OUT: Matched against ``self.config`` keys.
            default (Any): Value to return if the key is absent.
                IN: Fallback when *key* is not found.
                OUT: Returned instead of raising ``KeyError``.

        Returns:
            Any: The stored configuration value, or *default*.
                OUT: Direct result from ``self.config.get``.
        """

        return self.config.get(key, default)

    def set_step_callback(self, callback: Callable):
        """Register a callback for execution step events.

        Args:
            callback (Callable): A function accepting a step info dict.
                IN: Stored in ``self.step_callback``.
                OUT: Invoked during ``execute`` at key lifecycle points.
        """

        self.step_callback = callback

    def is_rate_limited(self) -> bool:
        """Check whether the agent is currently rate limited.

        Returns:
            bool: ``True`` if the rate limit is active.
                OUT: Negated result of ``_check_rate_limit``.
        """

        return not self._check_rate_limit()

    def get_rate_limit_status(self) -> dict:
        """Return detailed rate limit status.

        Returns:
            dict: Contains ``rate_limited``, ``max_rpm``, ``current_requests``,
                and ``requests_remaining``. OUT: Empty status if ``max_rpm`` is ``None``.
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

    def __eq__(self, other: object) -> bool:
        """Compare two ``CortexAgent`` instances by identity fields.

        Args:
            other (object): The object to compare against.
                IN: Checked for type and field equality.
                OUT: Compared on ``role``, ``goal``, ``backstory``, and ``model``.

        Returns:
            bool: ``True`` if the agents are equivalent.
                OUT: ``False`` for non-``CortexAgent`` objects.
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
        """Compute a hash from the agent's identity fields.

        Returns:
            int: A stable hash derived from ``role``, ``goal``, ``backstory``,
                and ``model``. OUT: Used for set membership and deduplication.
        """

        identity_str = f"{self.role}|{self.goal}|{self.backstory}|{self.model or 'default'}"

        sha256_hash = hashlib.sha256(identity_str.encode("utf-8")).digest()

        return int.from_bytes(sha256_hash[:8], byteorder="big", signed=False)
