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


"""Core Calute module implementing the main agent orchestration framework.

This module contains the primary Calute class that provides sophisticated
agent management capabilities including:
- Multi-agent orchestration and switching
- Function/tool execution with retry logic
- Memory system integration
- Streaming response handling
- Prompt template management
- Asynchronous and synchronous execution modes

The module also includes prompt templating utilities and helper functions
for formatting and parsing agent responses.
"""

import asyncio
import json
import pprint
import queue
import re
import textwrap
import threading
import typing as tp
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass
from enum import Enum

from calute.types.function_execution_types import ReinvokeSignal
from calute.types.messages import ChatMessage, MessagesHistory, SystemMessage, UserMessage

from .executors import AgentOrchestrator, FunctionExecutor
from .llms import BaseLLM
from .memory import MemoryStore, MemoryType
from .streamer_buffer import StreamerBuffer
from .types import (
    Agent,
    AgentFunction,
    AgentSwitch,
    AgentSwitchTrigger,
    AssistantMessage,
    Completion,
    ExecutionResult,
    ExecutionStatus,
    FunctionCallInfo,
    FunctionCallsExtracted,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    RequestFunctionCall,
    ResponseResult,
    StreamChunk,
    StreamingResponseType,
    SwitchContext,
    ToolCall,
    ToolCallStreamChunk,
    ToolMessage,
)
from .types.oai_protocols import ToolDefinition
from .types.tool_calls import FunctionCall
from .utils import debug_print, function_to_json

SEP = "  "
add_depth = lambda x, ep=False: SEP + x.replace("\n", f"\n{SEP}") if ep else x.replace("\n", f"\n{SEP}")  # noqa


class PromptSection(Enum):
    """Enumeration of different sections in a structured prompt.

    This enum defines the standard sections that can be included in a
    structured prompt template, allowing for consistent prompt organization
    across different agents and use cases.
    """

    SYSTEM = "system"
    PERSONA = "persona"
    RULES = "rules"
    FUNCTIONS = "functions"
    TOOLS = "tools"
    EXAMPLES = "examples"
    CONTEXT = "context"
    HISTORY = "history"
    PROMPT = "prompt"


@dataclass
class PromptTemplate:
    """Configurable template for structuring agent prompts.

    This class provides a flexible way to structure prompts with different
    sections that can be customized or reordered based on requirements.

    Attributes:
        sections: Dictionary mapping PromptSection enums to their header strings.
        section_order: List defining the order in which sections appear in the prompt.

    Example:
        >>> template = PromptTemplate(
        ...     sections={PromptSection.SYSTEM: "INSTRUCTIONS:"},
        ...     section_order=[PromptSection.SYSTEM, PromptSection.PROMPT]
        ... )
    """

    sections: dict[PromptSection, str] | None = None
    section_order: list[PromptSection] | None = None

    def __post_init__(self):
        """Initialize default sections and ordering if not provided.

        Sets up standard prompt sections with appropriate headers and
        establishes a default ordering that works well for most use cases.
        """
        self.sections = self.sections or {
            PromptSection.SYSTEM: "SYSTEM:",
            PromptSection.RULES: "RULES:",
            PromptSection.FUNCTIONS: "FUNCTIONS:",
            PromptSection.TOOLS: f"TOOLS:\n{SEP}When using tools, follow this format:",
            PromptSection.EXAMPLES: f"EXAMPLES:\n{SEP}",
            PromptSection.CONTEXT: "CONTEXT:\n",
            PromptSection.HISTORY: f"HISTORY:\n{SEP}Conversation so far:\n",
            PromptSection.PROMPT: "PROMPT:\n",
        }

        self.section_order = self.section_order or [
            PromptSection.SYSTEM,
            PromptSection.RULES,
            PromptSection.FUNCTIONS,
            PromptSection.TOOLS,
            PromptSection.EXAMPLES,
            PromptSection.CONTEXT,
            PromptSection.HISTORY,
            PromptSection.PROMPT,
        ]


class Calute:
    """Main Calute orchestration class for managing AI agents.

    This is the primary interface for interacting with the Calute framework.
    It manages agent registration, prompt formatting, function execution,
    memory integration, and response generation with support for both
    streaming and non-streaming modes.

    Attributes:
        SEP: Class variable defining the separator used for indentation.
        llm_client: The LLM backend client for generating completions.
        template: Prompt template for structuring agent prompts.
        orchestrator: Agent orchestrator for managing multi-agent workflows.
        executor: Function executor for handling tool calls.
        enable_memory: Whether memory system is enabled.
        memory_store: Memory storage system (if enabled).

    Example:
        >>> from calute import Calute, OpenAILLM
        >>> llm = OpenAILLM(api_key="your-key")
        >>> calute = Calute(llm=llm, enable_memory=True)
        >>> response = calute.run(prompt="Hello!")
    """

    SEP: tp.ClassVar[str] = SEP

    def __init__(
        self,
        llm: BaseLLM | None = None,
        template: PromptTemplate | None = None,
        enable_memory: bool = False,
        memory_config: dict[str, tp.Any] | None = None,
    ):
        """Initialize Calute with an LLM.

        Args:
            llm: A BaseLLM instance for generating completions.
            template: Optional prompt template for structuring prompts.
            enable_memory: Whether to enable the memory system.
            memory_config: Configuration for MemoryStore with keys:
                - max_short_term: Maximum short-term memory entries (default: 100)
                - max_working: Maximum working memory entries (default: 10)
                - max_long_term: Maximum long-term memory entries (default: 10000)
                - enable_vector_search: Enable vector similarity search (default: False)
                - embedding_dimension: Dimension for embeddings (default: 768)
                - enable_persistence: Enable persistent storage (default: False)
                - persistence_path: Path for persistent storage
                - cache_size: Size of memory cache (default: 100)

        Example:
            >>> llm = OpenAILLM(api_key="key")
            >>> calute = Calute(
            ...     llm=llm,
            ...     enable_memory=True,
            ...     memory_config={"max_short_term": 50}
            ... )
        """

        self.llm_client: BaseLLM = llm

        self.template = template or PromptTemplate()
        self.orchestrator = AgentOrchestrator()
        self.executor = FunctionExecutor(self.orchestrator)
        self.enable_memory = enable_memory
        if enable_memory:
            memory_config = memory_config or {}
            self.memory_store = MemoryStore(
                max_short_term=memory_config.get("max_short_term", 100),
                max_working=memory_config.get("max_working", 10),
                max_long_term=memory_config.get("max_long_term", 10000),
                enable_vector_search=memory_config.get("enable_vector_search", False),
                embedding_dimension=memory_config.get("embedding_dimension", 768),
                enable_persistence=memory_config.get("enable_persistence", False),
                persistence_path=memory_config.get("persistence_path"),
                cache_size=memory_config.get("cache_size", 100),
            )
        self._setup_default_triggers()

    def _setup_default_triggers(self):
        """Setup default agent switching triggers.

        Registers default trigger functions for agent switching including:
        - Capability-based switching: Switch to agent with required capabilities
        - Error recovery switching: Switch to fallback agent on errors
        """

        def capability_based_switch(context, agents, current_agent_id):  # type:ignore
            """Switch agent based on required capabilities"""
            required_capability = context.get("required_capability")
            if not required_capability:
                return None

            best_agent = None
            best_score = 0

            for agent_id, agent in agents.items():
                if agent.has_capability(required_capability):
                    for cap in agent.capabilities:
                        if cap.name == required_capability and cap.performance_score > best_score:
                            best_agent = agent_id
                            best_score = cap.performance_score

            return best_agent

        def error_recovery_switch(context, agents, current_agent_id):
            """Switch agent on function execution errors"""
            if context.get("execution_error") and current_agent_id:
                current_agent = agents[current_agent_id]
                if current_agent.fallback_agent_id:
                    return current_agent.fallback_agent_id
            return None

        self.orchestrator.register_switch_trigger(AgentSwitchTrigger.CAPABILITY_BASED, capability_based_switch)
        self.orchestrator.register_switch_trigger(AgentSwitchTrigger.ERROR_RECOVERY, error_recovery_switch)

    def register_agent(self, agent: Agent):
        """Register an agent with the orchestrator.

        Args:
            agent: The Agent instance to register for orchestration.
        """
        self.orchestrator.register_agent(agent)

    def _update_memory_from_response(
        self,
        content: str,
        agent_id: str,
        context_variables: dict | None = None,
        function_calls: list[RequestFunctionCall] | None = None,
    ):
        """Update memory system based on agent response.

        Args:
            content: The response content from the agent.
            agent_id: ID of the agent that generated the response.
            context_variables: Optional context variables to store with memory.
            function_calls: Optional list of function calls made in the response.
        """
        if not self.enable_memory:
            return

        self.memory_store.add_memory(
            content=f"Assistant response: {content[:200]}...",
            memory_type=MemoryType.SHORT_TERM,
            agent_id=agent_id,
            context=context_variables or {},
            importance_score=0.6,
        )

        if function_calls:
            for call in function_calls:
                self.memory_store.add_memory(
                    content=f"Function called: {call.name} with args: {call.arguments}",
                    memory_type=MemoryType.WORKING,
                    agent_id=agent_id,
                    context={"function_id": call.id, "status": call.status.value},
                    importance_score=0.7,
                    tags=["function_call", call.name],
                )

    def _update_memory_from_prompt(self, prompt: str, agent_id: str):
        """Update memory system from user prompt.

        Args:
            prompt: The user's input prompt.
            agent_id: ID of the agent receiving the prompt.
        """
        if not self.enable_memory:
            return

        self.memory_store.add_memory(
            content=f"User prompt: {prompt}",
            memory_type=MemoryType.SHORT_TERM,
            agent_id=agent_id,
            importance_score=0.8,
            tags=["user_input"],
        )

    def _format_section(
        self,
        header: str,
        content: str | list[str] | None,
        item_prefix: str | None = "- ",
    ) -> str | None:
        """
        Formats a section of the prompt with a header and indented content.
        Returns None if the content is empty.
        """
        if not content:
            return None

        if isinstance(content, list):
            content_str = "\n".join(f"{item_prefix or ''}{str(line).strip()}" for line in content)
        else:
            content_str = str(content).strip()

        if not content_str:
            return None

        indented_content = textwrap.indent(content_str, SEP)

        return f"{header}\n{indented_content}"

    def _extract_from_markdown(self, content: str, field: str) -> list[RequestFunctionCall]:
        """Extract function calls from markdown code blocks.

        Args:
            content: The response content to search.
            field: The markdown field identifier (e.g., 'tool_call').

        Returns:
            List of extracted function call strings from markdown blocks.
        """

        pattern = rf"```{field}\s*\n(.*?)\n```"
        return re.findall(pattern, content, re.DOTALL)

    def manage_messages(
        self,
        agent: Agent | None,
        prompt: str | None = None,
        context_variables: dict | None = None,
        messages: MessagesHistory | None = None,
        include_memory: bool = True,
        use_instructed_prompt: bool = False,
        use_chain_of_thought: bool = False,
        require_reflection: bool = False,
    ) -> MessagesHistory:
        """Generate a structured list of ChatMessage objects for the LLM.

        Constructs a properly formatted message history including system prompts,
        rules, functions, examples, context, and user messages based on the
        agent's configuration and provided parameters.

        Args:
            agent: The agent to generate messages for.
            prompt: Optional user prompt to include.
            context_variables: Optional context variables to include.
            messages: Optional existing message history.
            include_memory: Whether to include memory context.
            use_instructed_prompt: Whether to use instructed prompt format.
            use_chain_of_thought: Whether to add chain-of-thought instructions.
            require_reflection: Whether to request reflection in response.

        Returns:
            MessagesHistory containing the formatted messages.

        Example:
            >>> messages = calute.manage_messages(
            ...     agent=my_agent,
            ...     prompt="Hello",
            ...     use_chain_of_thought=True
            ... )
        """
        if not agent:
            return MessagesHistory(messages=[UserMessage(content=prompt or "You are a helpful assistant.")])

        system_parts = []

        assert self.template.sections is not None
        persona_header = self.template.sections.get(PromptSection.SYSTEM, "SYSTEM:") if use_instructed_prompt else ""
        instructions = str((agent.instructions() if callable(agent.instructions) else agent.instructions) or "")
        if use_chain_of_thought:
            instructions += (
                "\n\nApproach every task systematically:\n"
                "- Understand the request fully.\n"
                "- Break down complex problems.\n"
                "- If functions are available, determine if they are needed.\n"
                "- Formulate your response or function call.\n"
                "- Verify your output addresses the request completely."
            )
        system_parts.append(self._format_section(persona_header, instructions, item_prefix=None))
        rules_header = self.template.sections.get(PromptSection.RULES, "RULES:")
        rules: list[str] = (
            agent.rules
            if isinstance(agent.rules, list)
            else (agent.rules() if callable(agent.rules) else ([str(agent.rules)] if agent.rules else []))
        )
        if agent.functions and use_instructed_prompt:
            rules.append(
                "If a function can satisfy the user request, you MUST respond only with a valid tool call in the"
                " specified format. Do not add any conversational text before or after the tool call."
            )
        if self.enable_memory and include_memory:
            rules.extend(
                [
                    "Consider previous context and conversation history.",
                    "Build upon earlier interactions when appropriate.",
                ]
            )
        system_parts.append(self._format_section(rules_header, rules))

        if agent.functions and use_instructed_prompt:
            functions_header = self.template.sections.get(PromptSection.FUNCTIONS, "FUNCTIONS:")

            tool_format_instruction = textwrap.dedent(
                """
                When calling a function, you must use the following XML format.
                The tag name is the function name and parameters are a JSON object within <arguments> tags.

                Example:
                    <my_function_name><arguments>{"param1": "value1"}</arguments></my_function_name>

                The available functions are listed with their schemas:
                """
            ).strip()

            fn_docs_raw = self.generate_function_section(agent.functions)
            indented_fn_docs = textwrap.indent(fn_docs_raw, SEP)
            full_function_content = f"{tool_format_instruction}\n\n{indented_fn_docs}"
            system_parts.append(self._format_section(functions_header, full_function_content, item_prefix=None))

        if agent.examples:
            examples_header = self.template.sections.get(PromptSection.EXAMPLES, "EXAMPLES:")
            example_content = "\n\n".join(ex.strip() for ex in agent.examples)
            system_parts.append(self._format_section(examples_header, example_content, item_prefix=None))

        context_header = self.template.sections.get(PromptSection.CONTEXT, "CONTEXT:")
        context_content_list = []
        if self.enable_memory and include_memory:
            memory_context = self.memory_store.consolidate_memories(agent.id or "default")
            if memory_context:
                context_content_list.append(f"Relevant information from memory:\n{memory_context}")
        if context_variables:
            ctx_vars_formatted = self.format_context_variables(context_variables)
            if ctx_vars_formatted:
                context_content_list.append(f"Current variables:\n{ctx_vars_formatted}")

        if context_content_list:
            system_parts.append(
                self._format_section(context_header, "\n\n".join(context_content_list), item_prefix=None)
            )

        instructed_messages: list[ChatMessage] = []

        final_system_content = "\n\n".join(part for part in system_parts if part)
        instructed_messages.append(SystemMessage(content=final_system_content))

        if messages and messages.messages:
            instructed_messages.extend(messages.messages)

        if prompt is not None:
            final_prompt_content = prompt
            if require_reflection:
                final_prompt_content += (
                    f"\n\nAfter your primary response, add a reflection section in `<reflection>` tags:\n"
                    f"{self.SEP}- Assumptions made.\n"
                    f"{self.SEP}- Potential limitations of your response."
                )
            instructed_messages.append(UserMessage(content=final_prompt_content))

        return MessagesHistory(messages=instructed_messages)

    def _build_reinvoke_messages(
        self,
        original_messages: MessagesHistory,
        assistant_content: str,
        function_calls: list[RequestFunctionCall],
        results: list[RequestFunctionCall],
    ) -> MessagesHistory:
        """Build message history for reinvocation including function results.

        Constructs a new message history that includes the original messages,
        the assistant's response with tool calls, and the tool execution results.

        Args:
            original_messages: The original message history.
            assistant_content: The assistant's response content.
            function_calls: List of function calls made by the assistant.
            results: List of function execution results.

        Returns:
            Updated MessagesHistory with function calls and results included.
        """
        messages = original_messages.messages.copy()

        tool_calls = []
        for fc in function_calls:
            tool_call = ToolCall(
                id=fc.id,
                function=FunctionCall(
                    name=fc.name,
                    arguments=json.dumps(fc.arguments) if isinstance(fc.arguments, dict) else fc.arguments,
                ),
            )
            tool_calls.append(tool_call)

        clean_content = self._remove_function_calls_from_content(assistant_content)
        assistant_msg = AssistantMessage(
            content=clean_content if clean_content.strip() else None,
            tool_calls=tool_calls if tool_calls else None,
        )
        messages.append(assistant_msg)

        for fc, result in zip(function_calls, results, strict=False):
            if result.status == ExecutionStatus.SUCCESS:
                tool_content = json.dumps(result.result) if not isinstance(result.result, str) else result.result
            else:
                tool_content = f"Error: {result.error}"

            tool_msg = ToolMessage(content=tool_content, tool_call_id=fc.id)
            messages.append(tool_msg)

        return MessagesHistory(messages=messages)

    @staticmethod
    def extract_md_block(input_string: str) -> list[tuple[str, str]]:
        """Extract Markdown code blocks from a string.

        This function finds all Markdown code blocks (delimited by triple backticks)
        in the input string and returns their content along with the optional language
        specifier (if present).

        Args:
            input_string: The input string containing one or more Markdown code blocks.

        Returns:
            List of tuples, where each tuple contains:
                - The language specifier (e.g., 'xml', 'python', or '' if not specified).
                - The content of the code block.

        Example:
            >>> text = '''```xml
            ... <web_research>
            ...   <arguments>
            ...     {"query": "quantum computing breakthroughs 2024"}
            ...   </arguments>
            ... </web_research>
            ... ```'''
            >>> Calute.extract_md_block(text)
            [('xml', '<web_research>\n  <arguments>\n    {"query": "quantum computing breakthroughs 2024"}\n  </arguments>\n</web_research>')]
        """
        pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(pattern, input_string, re.DOTALL)
        return [(lang, content.strip()) for lang, content in matches]

    def _remove_function_calls_from_content(self, content: str) -> str:
        """Remove function call XML blocks from content.

        Cleans the response content by removing XML-formatted function calls
        and markdown tool_call blocks, leaving only the conversational text.

        Args:
            content: The content to clean.

        Returns:
            Content with function call blocks removed.
        """
        pattern = r"<(\w+)>\s*<arguments>.*?</arguments>\s*</\w+>"
        cleaned = re.sub(pattern, "", content, flags=re.DOTALL)
        pattern = r"```tool_call.*?```"
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

        return cleaned.strip()

    def _extract_function_calls_from_xml(self, content: str, agent: Agent) -> list[RequestFunctionCall]:
        """Extract function calls from response content using XML tags.

        Parses XML-formatted function calls from the response content.
        Expected format: <function_name><arguments>{...}</arguments></function_name>

        Args:
            content: The response content to parse.
            agent: The agent context for timeout and retry settings.

        Returns:
            List of RequestFunctionCall objects extracted from XML.
        """
        function_calls = []
        pattern = r"<(\w+)>\s*<arguments>(.*?)</arguments>\s*</\w+>"
        matches = re.findall(pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            name = match[0]
            arguments_str = match[1].strip()
            try:
                arguments = json.loads(arguments_str)
                function_call = RequestFunctionCall(
                    name=name,
                    arguments=arguments,
                    id=f"call_{i}_{hash(match)}",
                    timeout=agent.function_timeout,
                    max_retries=agent.max_function_retries,
                )
                function_calls.append(function_call)
            except json.JSONDecodeError:
                continue

        return function_calls

    def _convert_function_calls(
        self,
        function_calls_data: list[dict[str, tp.Any]],
        agent: Agent,
    ) -> list[RequestFunctionCall]:
        """Convert function call data from LLM streaming to RequestFunctionCall objects.

        Args:
            function_calls_data: Raw function call data from LLM response.
            agent: The agent context for timeout and retry settings.

        Returns:
            List of RequestFunctionCall objects.
        """
        function_calls = []
        for call_data in function_calls_data:
            try:
                arguments = call_data.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass

                function_calls.append(
                    RequestFunctionCall(
                        name=call_data.get("name"),
                        arguments=arguments,
                        id=call_data.get("id", f"call_{len(function_calls)}"),
                        timeout=agent.function_timeout,
                        max_retries=agent.max_function_retries,
                    )
                )
            except Exception:
                continue
        return function_calls

    def _extract_function_calls(
        self,
        content: str,
        agent: Agent,
        tool_calls: None | list[tp.Any] = None,
    ) -> list[RequestFunctionCall]:
        """Extract function calls from response content.

        Attempts multiple extraction methods including tool_calls from LLM,
        XML format, and markdown blocks.

        Args:
            content: The response content to parse.
            agent: The agent context for timeout and retry settings.
            tool_calls: Optional pre-parsed tool calls from LLM.

        Returns:
            List of RequestFunctionCall objects.
        """

        if tool_calls is not None:
            function_calls = []
            for call_ in tool_calls:
                try:
                    arguments = call_.function.arguments
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            try:
                                arguments = json.loads(arguments + "}")
                            except json.JSONDecodeError:
                                pass

                    function_calls.append(
                        RequestFunctionCall(
                            name=call_.function.name,
                            arguments=arguments,
                            id=call_.id,
                            timeout=agent.function_timeout,
                            max_retries=agent.max_function_retries,
                        )
                    )
                except Exception as e:
                    debug_print(True, f"Error processing tool call: {e}")
                    continue
            return function_calls
        function_calls = self._extract_function_calls_from_xml(content, agent)
        if function_calls:
            return function_calls

        function_calls = []
        matches = self._extract_from_markdown(content=content, field="tool_call")

        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match)
                function_call = RequestFunctionCall(
                    name=call_data.get("name"),
                    arguments=call_data.get("content", {}),
                    id=f"call_{i}_{hash(match)}",
                    timeout=agent.function_timeout,
                    max_retries=agent.max_function_retries,
                )
                function_calls.append(function_call)
            except json.JSONDecodeError:
                continue

        return function_calls

    @staticmethod
    def extract_from_markdown(format: str, string: str) -> str | None | dict:  # noqa:A002
        """Extract content from a markdown code block with specific format.

        Args:
            format: The markdown format identifier to search for.
            string: The string containing the markdown block.

        Returns:
            Parsed JSON dictionary if valid JSON, raw string if not JSON,
            or None if format not found.

        Example:
            >>> content = '```json\n{"key": "value"}\n```'
            >>> Calute.extract_from_markdown("json", content)
            {'key': 'value'}
        """
        search_mour = f"```{format}"
        index = string.find(search_mour)

        if index != -1:
            choosen = string[index + len(search_mour) :]
            if choosen.endswith("```"):
                choosen = choosen[:-3]
            try:
                return json.loads(choosen)
            except Exception:
                return choosen
        return None

    def _detect_function_calls(self, content: str, agent: Agent) -> bool:
        """Detect if content contains valid function calls.

        Quick check to determine if the response contains function calls
        without fully parsing them.

        Args:
            content: The response content to check.
            agent: The agent with available functions.

        Returns:
            True if function calls are detected, False otherwise.
        """
        if not agent.functions:
            return False
        function_names = [func.__name__ for func in agent.functions]
        for func_name in function_names:
            if f"<{func_name}>" in content or f"<{func_name} " in content:
                if "<arguments>" in content:
                    return True
        if "```tool_call" in content:
            return True

        return False

    def _detect_function_calls_regex(self, content: str, agent: Agent) -> bool:
        """Detect function calls using regex for more precision.

        More accurate detection using regular expressions to find
        XML-formatted function calls.

        Args:
            content: The response content to check.
            agent: The agent with available functions.

        Returns:
            True if function calls are detected via regex, False otherwise.
        """
        if not agent.functions:
            return False
        function_names = [func.__name__ for func in agent.functions]
        for func_name in function_names:
            pattern = rf"<{func_name}(?:\s[^>]*)?>.*?<arguments>"
            if re.search(pattern, content, re.DOTALL):
                return True
        return False

    @staticmethod
    def get_thoughts(response: str, tag: str = "think") -> str | None:
        """Extract thinking/reasoning content from tagged sections.

        Args:
            response: The response containing tagged thoughts.
            tag: The XML tag name to extract (default: 'think').

        Returns:
            The content within the tags, or None if not found.

        Example:
            >>> response = "Some text <think>Internal reasoning</think> more text"
            >>> Calute.get_thoughts(response)
            'Internal reasoning'
        """
        inside = None
        match = re.search(rf"<{tag}>(.*?)</{tag}>", response, flags=re.S)
        if match:
            inside = match.group(1).strip()
        return inside

    @staticmethod
    def filter_thoughts(response: str, tag: str = "think") -> str:
        """Remove all thinking tags from the response.

        Args:
            response: The response containing tagged thoughts.
            tag: The XML tag name to remove (default: 'think').

        Returns:
            The response with all tagged sections removed.

        Example:
            >>> response = "Answer <think>reasoning</think> continues"
            >>> Calute.filter_thoughts(response)
            'Answer continues'
        """
        filtered = re.sub(rf"<{tag}>.*?</{tag}>", "", response, flags=re.S)
        return filtered.strip()

    def format_function_parameters(self, parameters: dict) -> str:
        """Format function parameters in a clear, structured way.

        Args:
            parameters: Dictionary of parameter definitions from function schema.

        Returns:
            Formatted string representation of parameters with types,
            requirements, and descriptions.
        """
        if not parameters.get("properties"):
            return ""

        formatted_params = []
        required_params = parameters.get("required", [])

        for param_name, param_info in parameters["properties"].items():
            if param_name == "context_variables":
                continue

            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            required = "(required)" if param_name in required_params else "(optional)"

            param_str = f"    - {param_name}: {param_type} {required}"
            if param_desc:
                param_str += f"\n      Description: {param_desc}"
            if "enum" in param_info:
                param_str += f"\n      Allowed values: {', '.join(str(v) for v in param_info['enum'])}"

            formatted_params.append(param_str)

        return "\n".join(formatted_params)

    def generate_function_section(self, functions: list[AgentFunction]) -> str:
        """Generate detailed function documentation for agent prompts.

        Creates comprehensive documentation for available functions, organized
        by category if applicable, with full parameter schemas and examples.

        Args:
            functions: List of AgentFunction objects to document.

        Returns:
            Formatted string containing complete function documentation.
        """
        if not functions:
            return ""

        function_docs = []
        categorized_functions: dict[str, list[AgentFunction]] = {}
        uncategorized = []

        for func in functions:
            if hasattr(func, "category"):
                category = func.category  # type:ignore
                if category not in categorized_functions:
                    categorized_functions[category] = []
                categorized_functions[category].append(func)
            else:
                uncategorized.append(func)

        for category, funcs in categorized_functions.items():
            function_docs.append(f"## {category} Functions\n")
            for func in funcs:
                try:
                    schema = function_to_json(func)["function"]
                    doc = self._format_function_doc(schema)
                    function_docs.append(doc)
                except Exception as e:
                    func_name = getattr(func, "__name__", str(func))
                    function_docs.append(f"Warning: Unable to parse function {func_name}: {e!s}")
        if uncategorized:
            if categorized_functions:
                function_docs.append("## Other Functions\n")
            for func in uncategorized:
                try:
                    schema = function_to_json(func)["function"]
                    doc = self._format_function_doc(schema)
                    function_docs.append(doc)
                except Exception as e:
                    func_name = getattr(func, "__name__", str(func))
                    function_docs.append(f"Warning: Unable to parse function {func_name}: {e!s}")

        return "\n\n".join(function_docs)

    def _format_function_doc(self, schema: dict) -> str:
        """Format a single function's documentation block.

        Creates a structured documentation block for a function including
        its name, purpose, parameters, return type, and usage example.

        Args:
            schema: Function schema dictionary containing name, description,
                    parameters, returns, and optional examples.

        Returns:
            Formatted documentation string for the function.

        Note:
            The output format includes:
            - Function name and purpose
            - Parameter details with types and requirements
            - Return type
            - Call pattern example in XML format
            - Optional additional examples if provided
        """
        ind1 = SEP
        ind2 = SEP * 2
        ind3 = SEP * 3

        doc_lines: list[str] = []
        doc_lines.append(f"Function: {schema['name']}")
        if desc := schema.get("description", "").strip():
            doc_lines.append(f"{ind1}Purpose: {desc}")
        params_block = []
        params = schema.get("parameters", {})
        properties: dict = params.get("properties", {})
        required = set(params.get("required", []))

        for pname, pinfo in properties.items():
            if pname == "context_variables":
                continue

            ptype = pinfo.get("type", "any")
            req = "required" if pname in required else "optional"

            params_block.append(f"{ind2}- {pname} ({ptype}, {req})")

            if pdesc := pinfo.get("description", "").strip():
                params_block.append(f"{ind3}Description : {pdesc}")

            if enum_vals := pinfo.get("enum"):
                joined = ", ".join(map(str, enum_vals))
                params_block.append(f"{ind3}Allowed values : {joined}")

        if params_block:
            doc_lines.append(f"\n{ind1}Parameters:")
            doc_lines.extend(params_block)
        if ret := schema.get("returns"):
            doc_lines.append(f"\n{ind1}Returns : {ret}")
        call_example = textwrap.dedent(
            f'<{schema["name"]}><arguments>{{"param": "value"}}</arguments></{schema["name"]}>'.rstrip()
        )
        doc_lines.append(f"\n{ind1}Call-pattern:")
        doc_lines.append(textwrap.indent(call_example, ind2))
        if schema_examples := schema.get("examples"):
            doc_lines.append(f"\n{ind1}Examples:")
            for example in schema_examples:
                json_example = json.dumps(example, indent=2)
                doc_lines.append(textwrap.indent(f"```json\n{json_example}\n```", ind2))

        return "\n".join(doc_lines)

    def format_context_variables(self, variables: dict[str, tp.Any]) -> str:
        """Format context variables with type information.

        Args:
            variables: Dictionary of context variables to format.

        Returns:
            Formatted string representation of variables with types and values.
        """
        if not variables:
            return ""
        formatted_vars = []
        for key, value in variables.items():
            if not callable(value):
                var_type = type(value).__name__
                formatted_value = str(value)
                formatted_vars.append(f"- {key} ({var_type}): {formatted_value}")
        return "\n".join(formatted_vars)

    def format_prompt(self, prompt: str | None) -> str:
        """Format a prompt string.

        Args:
            prompt: The prompt to format.

        Returns:
            The formatted prompt or empty string if None.
        """
        if not prompt:
            return ""
        return prompt

    def format_chat_history(self, messages: MessagesHistory) -> str:
        """Format chat messages with improved readability.

        Args:
            messages: MessagesHistory object containing chat messages.

        Returns:
            Formatted string representation of the chat history.
        """
        formatted_messages = []
        for msg in messages.messages:
            formatted_messages.append(f"## {msg.role}:\n{msg.content}")
        return "\n\n".join(formatted_messages)

    async def create_response(
        self,
        prompt: str | None = None,
        context_variables: dict | None = None,
        messages: MessagesHistory | None = None,
        agent_id: str | None | Agent = None,
        stream: bool = True,
        apply_functions: bool = True,
        print_formatted_prompt: bool = False,
        use_instructed_prompt: bool = False,
        conversation_name_holder: str = "Messages",
        mention_last_turn: bool = True,
        reinvoke_after_function: bool = True,
        reinvoked_runtime: bool = False,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> ResponseResult | AsyncIterator[StreamingResponseType]:
        """Create response with enhanced function calling and agent switching.

        Main async method for generating agent responses with support for
        streaming, function execution, and multi-agent orchestration.

        Args:
            prompt: Optional user prompt to process.
            context_variables: Optional context variables for the agent.
            messages: Optional message history.
            agent_id: Optional specific agent ID or Agent instance to use.
            stream: Whether to stream the response.
            apply_functions: Whether to execute detected function calls.
            print_formatted_prompt: Whether to print the formatted prompt.
            use_instructed_prompt: Whether to use instructed prompt format.
            conversation_name_holder: Name for conversation in instructed format.
            mention_last_turn: Whether to mention last turn in instructed format.
            reinvoke_after_function: Whether to reinvoke after function execution.
            reinvoked_runtime: Internal flag indicating this is a reinvocation.
            streamer_buffer: Optional buffer for streaming chunks.

        Returns:
            ResponseResult if stream=False, AsyncIterator[StreamingResponseType] if stream=True.

        Example:
            >>> response = await calute.create_response(
            ...     prompt="Calculate 5 + 3",
            ...     stream=False
            ... )
            >>> print(response.content)
        """
        if isinstance(agent_id, Agent):
            agent = agent_id
        else:
            if agent_id:
                self.orchestrator.switch_agent(agent_id, "User specified agent")
            agent = self.orchestrator.get_current_agent()

        context_variables = context_variables or {}
        prompt_messages: MessagesHistory = self.manage_messages(
            agent=agent,
            prompt=prompt,
            context_variables=context_variables,
            use_instructed_prompt=use_instructed_prompt,
            messages=messages,
        )

        if use_instructed_prompt:
            prompt_str = prompt_messages.make_instruction_prompt(
                conversation_name_holder=conversation_name_holder,
                mention_last_turn=mention_last_turn,
            )
        else:
            prompt_str = prompt_messages.to_openai()["messages"]

        if print_formatted_prompt:
            if use_instructed_prompt:
                print(prompt_str)
            else:
                pprint.pprint(prompt_messages.to_openai())

        response = await self.llm_client.generate_completion(
            prompt=prompt_str,
            model=agent.model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            top_p=agent.top_p,
            stop=agent.stop if isinstance(agent.stop, list) else ([agent.stop] if agent.stop else None),
            top_k=agent.top_k,
            min_p=agent.min_p,
            tools=None if use_instructed_prompt else [ToolDefinition(**function_to_json(fn)) for fn in agent.functions],
            presence_penalty=agent.presence_penalty,
            frequency_penalty=agent.frequency_penalty,
            repetition_penalty=agent.repetition_penalty,
            extra_body=agent.extra_body,
            stream=True,
        )

        if not apply_functions:
            if stream:
                return self._handle_streaming(response, reinvoked_runtime, agent)
            else:
                collected = []
                async for chunk in self._handle_streaming(
                    response,
                    reinvoked_runtime,
                    agent,
                    streamer_buffer,
                ):
                    collected.append(chunk)
                return (
                    collected[-1]
                    if collected
                    else ResponseResult(
                        content="",
                        response=None,
                        function_calls=[],
                        agent_id=agent.id,
                        execution_history=[],
                        reinvoked=reinvoked_runtime,
                    )
                )

        if stream:
            return self._handle_streaming_with_functions(
                response,
                agent,
                context_variables,
                prompt_messages,
                reinvoke_after_function,
                reinvoked_runtime,
                use_instructed_prompt,
                streamer_buffer,
            )
        else:
            collected_content = []
            function_calls = []
            execution_history = []
            async for chunk in self._handle_streaming_with_functions(
                response,
                agent,
                context_variables,
                prompt_messages,
                reinvoke_after_function,
                reinvoked_runtime,
                use_instructed_prompt,
                streamer_buffer,
            ):
                if hasattr(chunk, "content") and chunk.content:
                    collected_content.append(chunk.content)
                if hasattr(chunk, "function_calls"):
                    function_calls = chunk.function_calls
                if hasattr(chunk, "result"):
                    execution_history.append(chunk)

            final_content = "".join(collected_content)
            return ResponseResult(
                content=final_content,
                response=response,
                function_calls=function_calls,
                agent_id=agent.id or "default",
                execution_history=execution_history,
                reinvoked=reinvoked_runtime,
            )

    async def _handle_streaming_with_functions(
        self,
        response: tp.Any,
        agent: Agent,
        context: dict,
        prompt_messages: MessagesHistory,
        reinvoke_after_function: bool,
        reinvoked_runtime: bool,
        use_instructed_prompt: bool,
        streamer_buffer: StreamerBuffer | None,
    ) -> AsyncIterator[StreamingResponseType]:
        """Handle streaming response with function calls and optional reinvocation.

        Processes streaming LLM responses, detects and executes function calls,
        and optionally reinvokes the agent with function results.

        Args:
            response: The LLM response stream.
            agent: The current agent.
            context: Context variables for function execution.
            prompt_messages: The original prompt messages.
            reinvoke_after_function: Whether to reinvoke after functions.
            reinvoked_runtime: Whether this is already a reinvocation.
            use_instructed_prompt: Whether using instructed prompt format.
            streamer_buffer: Optional buffer for streaming chunks.

        Yields:
            StreamingResponseType objects including chunks, function notifications, etc.
        """
        buffered_content = ""
        function_calls_detected = False
        function_calls = []

        if hasattr(response, "__aiter__"):
            stream_generator = self.llm_client.astream_completion(response, agent)
            async for chunk_data in stream_generator:
                content = chunk_data.get("content")
                buffered_content = chunk_data.get("buffered_content", buffered_content)

                streaming_tool_calls_data = chunk_data.get("streaming_tool_calls")
                tool_call_chunks = []

                if streaming_tool_calls_data:
                    for tool_idx, tool_delta in streaming_tool_calls_data.items():
                        if tool_delta:
                            tool_id = tool_delta.get("id", f"tool_{tool_idx}")

                            tool_call_chunks.append(
                                ToolCallStreamChunk(
                                    id=tool_id,
                                    type="function",
                                    function_name=tool_delta.get("name"),
                                    arguments=tool_delta.get("arguments"),
                                    index=tool_idx,
                                    is_complete=False,
                                )
                            )
                            function_calls_detected = True

                if content and not function_calls_detected:
                    function_calls_detected = self._detect_function_calls(buffered_content, agent)

                out = StreamChunk(
                    chunk=chunk_data.get("raw_chunk"),
                    agent_id=agent.id or "default",
                    content=content,
                    buffered_content=buffered_content,
                    function_calls_detected=function_calls_detected,
                    reinvoked=reinvoked_runtime,
                    tool_calls=None,
                    streaming_tool_calls=tool_call_chunks if tool_call_chunks else None,
                )

                if streamer_buffer is not None:
                    streamer_buffer.put(out)
                yield out

                if chunk_data.get("is_final") and chunk_data.get("function_calls"):
                    function_calls = self._convert_function_calls(chunk_data["function_calls"], agent)
                    function_calls_detected = True
        else:
            stream_generator = self.llm_client.stream_completion(response, agent)
            for chunk_data in stream_generator:
                content = chunk_data.get("content")
                buffered_content = chunk_data.get("buffered_content", buffered_content)

                streaming_tool_calls_data = chunk_data.get("streaming_tool_calls")
                tool_call_chunks = []

                if streaming_tool_calls_data:
                    for tool_idx, tool_delta in (
                        streaming_tool_calls_data.items()
                        if isinstance(streaming_tool_calls_data, dict)
                        else enumerate(streaming_tool_calls_data or [])
                    ):
                        if tool_delta:
                            tool_id = tool_delta.get("id", f"tool_{tool_idx}")

                            tool_call_chunks.append(
                                ToolCallStreamChunk(
                                    id=tool_id,
                                    type="function",
                                    function_name=tool_delta.get("name"),
                                    arguments=tool_delta.get("arguments"),
                                    index=tool_idx if isinstance(tool_idx, int) else 0,
                                    is_complete=False,
                                )
                            )
                            function_calls_detected = True

                if content and not function_calls_detected:
                    function_calls_detected = self._detect_function_calls(buffered_content, agent)

                out = StreamChunk(
                    chunk=chunk_data.get("raw_chunk"),
                    agent_id=agent.id or "default",
                    content=content,
                    buffered_content=buffered_content,
                    function_calls_detected=function_calls_detected,
                    reinvoked=reinvoked_runtime,
                    tool_calls=None,
                    streaming_tool_calls=tool_call_chunks if tool_call_chunks else None,
                )

                if streamer_buffer is not None:
                    streamer_buffer.put(out)
                yield out

                if chunk_data.get("is_final") and chunk_data.get("function_calls"):
                    function_calls = self._convert_function_calls(chunk_data["function_calls"], agent)
                    function_calls_detected = True

        if function_calls_detected:
            out = FunctionDetection(message="Processing function calls...", agent_id=agent.id or "default")

            if streamer_buffer is not None:
                streamer_buffer.put(out)
            yield out

            if not function_calls:
                function_calls = self._extract_function_calls(buffered_content, agent, None)

            if function_calls:
                out = FunctionCallsExtracted(
                    function_calls=[FunctionCallInfo(name=fc.name, id=fc.id) for fc in function_calls],
                    agent_id=agent.id or "default",
                )

                if streamer_buffer is not None:
                    streamer_buffer.put(out)
                yield out

                results = []
                for i, call in enumerate(function_calls):
                    out = FunctionExecutionStart(
                        function_name=call.name,
                        function_id=call.id,
                        progress=f"{i + 1}/{len(function_calls)}",
                        agent_id=agent.id or "default",
                    )

                    if streamer_buffer is not None:
                        streamer_buffer.put(out)
                    yield out

                    result = await self.executor._execute_single_call(call, context, agent)
                    results.append(result)

                    out = FunctionExecutionComplete(
                        function_name=call.name,
                        function_id=call.id,
                        status=result.status.value,
                        result=result.result if result.status == ExecutionStatus.SUCCESS else None,
                        error=result.error,
                        agent_id=agent.id or "default",
                    )

                    if streamer_buffer is not None:
                        streamer_buffer.put(out)
                    yield out

                exec_results = [
                    ExecutionResult(
                        status=r.status,
                        result=r.result if hasattr(r, "result") else None,
                        error=r.error if hasattr(r, "error") else None,
                    )
                    for r in results
                ]
                switch_context = SwitchContext(
                    function_results=exec_results,
                    execution_error=any(r.status == ExecutionStatus.FAILURE for r in results),
                    buffered_content=buffered_content,
                )

                target_agent = self.orchestrator.should_switch_agent(switch_context.__dict__)
                if target_agent:
                    old_agent = agent.id
                    self.orchestrator.switch_agent(target_agent, "Post-execution switch")

                    out = AgentSwitch(
                        from_agent=old_agent or "default",
                        to_agent=target_agent,
                        reason="Post-execution switch",
                    )

                    if streamer_buffer is not None:
                        streamer_buffer.put(out)
                    yield out

                if reinvoke_after_function and function_calls:
                    updated_messages = self._build_reinvoke_messages(
                        prompt_messages,
                        buffered_content,
                        function_calls,
                        results,
                    )
                    out = ReinvokeSignal(
                        message="Reinvoking agent with function results...",
                        agent_id=agent.id or "default",
                    )

                    if streamer_buffer is not None:
                        streamer_buffer.put(out)
                    yield out

                    reinvoke_response = await self.create_response(
                        prompt=None,
                        context_variables=context,
                        messages=updated_messages,
                        agent_id=agent,
                        stream=True,
                        apply_functions=True,
                        print_formatted_prompt=False,
                        use_instructed_prompt=use_instructed_prompt,
                        reinvoke_after_function=True,
                        reinvoked_runtime=True,
                    )

                    if isinstance(reinvoke_response, ResponseResult):
                        pass
                    else:
                        async for chunk in reinvoke_response:
                            if streamer_buffer is not None and chunk is not None:
                                streamer_buffer.put(chunk)
                            yield chunk
                    return

        out = Completion(
            final_content=buffered_content,
            function_calls_executed=len(function_calls),
            agent_id=agent.id or "default",
            execution_history=self.orchestrator.execution_history[-3:],
        )

        if streamer_buffer is not None:
            streamer_buffer.put(out)
        yield out

    async def _handle_streaming(
        self,
        response: tp.Any,
        reinvoked_runtime,
        agent: Agent,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> AsyncIterator[StreamingResponseType]:
        """Handle streaming response without function calls.

        Simple streaming handler for responses that don't require function execution.

        Args:
            response: The LLM response stream.
            reinvoked_runtime: Whether this is a reinvocation.
            agent: The current agent.
            streamer_buffer: Optional buffer for streaming chunks.

        Yields:
            StreamChunk and Completion objects.
        """
        buffered_content = ""

        if hasattr(response, "__aiter__"):
            stream_generator = self.llm_client.astream_completion(response, agent)
            async for chunk_data in stream_generator:
                content = chunk_data.get("content")
                buffered_content = chunk_data.get("buffered_content", buffered_content)

                out = StreamChunk(
                    chunk=chunk_data.get("raw_chunk"),
                    agent_id=agent.id or "default",
                    content=content,
                    buffered_content=buffered_content,
                    function_calls_detected=False,
                    reinvoked=reinvoked_runtime,
                )
                if streamer_buffer is not None:
                    streamer_buffer.put(out)
                yield out
        else:
            stream_generator = self.llm_client.stream_completion(response, agent)
            for chunk_data in stream_generator:
                content = chunk_data.get("content")
                buffered_content = chunk_data.get("buffered_content", buffered_content)

                out = StreamChunk(
                    chunk=chunk_data.get("raw_chunk"),
                    agent_id=agent.id or "default",
                    content=content,
                    buffered_content=buffered_content,
                    function_calls_detected=False,
                    reinvoked=reinvoked_runtime,
                )
                if streamer_buffer is not None:
                    streamer_buffer.put(out)
                yield out
        out = Completion(
            final_content=buffered_content,
            function_calls_executed=0,
            agent_id=agent.id or "default",
            execution_history=self.orchestrator.execution_history[-3:],
        )

        if streamer_buffer is not None:
            streamer_buffer.put(out)

        yield out

    def run(
        self,
        prompt: str | None = None,
        context_variables: dict | None = None,
        messages: MessagesHistory | None = None,
        agent_id: str | None | Agent = None,
        stream: bool = True,
        apply_functions: bool = True,
        print_formatted_prompt: bool = False,
        use_instructed_prompt: bool = False,
        conversation_name_holder: str = "Messages",
        mention_last_turn: bool = True,
        reinvoke_after_function: bool = True,
        reinvoked_runtime: bool = False,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> ResponseResult | Generator[StreamingResponseType, None, None]:
        """Synchronous wrapper for create_response.

        Main synchronous interface for generating agent responses. Handles both
        streaming and non-streaming modes, with full support for function calling
        and agent orchestration.

        Args:
            prompt: Optional user prompt to process.
            context_variables: Optional context variables for the agent.
            messages: Optional message history.
            agent_id: Optional specific agent ID or Agent instance to use.
            stream: Whether to stream the response (True) or return complete (False).
            apply_functions: Whether to execute detected function calls.
            print_formatted_prompt: Whether to print the formatted prompt.
            use_instructed_prompt: Whether to use instructed prompt format.
            conversation_name_holder: Name for conversation in instructed format.
            mention_last_turn: Whether to mention last turn in instructed format.
            reinvoke_after_function: Whether to reinvoke after function execution.
            reinvoked_runtime: Internal flag indicating this is a reinvocation.
            streamer_buffer: Optional buffer for streaming chunks.

        Returns:
            Generator[StreamingResponseType] if stream=True, ResponseResult if stream=False.

        Example:
            >>>
            >>> for chunk in calute.run(prompt="Hello", stream=True):
            ...     if chunk.content:
            ...         print(chunk.content, end="")
            >>>
            >>>
            >>> result = calute.run(prompt="Hello", stream=False)
            >>> print(result.content)
        """
        if stream:
            return self._run_stream(
                prompt=prompt,
                context_variables=context_variables,
                messages=messages,
                agent_id=agent_id,
                apply_functions=apply_functions,
                print_formatted_prompt=print_formatted_prompt,
                use_instructed_prompt=use_instructed_prompt,
                conversation_name_holder=conversation_name_holder,
                mention_last_turn=mention_last_turn,
                reinvoke_after_function=reinvoke_after_function,
                reinvoked_runtime=reinvoked_runtime,
                streamer_buffer=streamer_buffer,
            )
        else:
            stream_generator = self._run_stream(
                prompt=prompt,
                context_variables=context_variables,
                messages=messages,
                agent_id=agent_id,
                apply_functions=apply_functions,
                print_formatted_prompt=print_formatted_prompt,
                use_instructed_prompt=use_instructed_prompt,
                conversation_name_holder=conversation_name_holder,
                mention_last_turn=mention_last_turn,
                reinvoke_after_function=reinvoke_after_function,
                reinvoked_runtime=reinvoked_runtime,
                streamer_buffer=streamer_buffer,
            )

            collected_content = []
            response = None
            completion = None
            function_calls = []
            agent_id_result = "default"
            execution_history = []
            reinvoked = False

            for chunk in stream_generator:
                if hasattr(chunk, "content") and chunk.content:
                    collected_content.append(chunk.content)
                if hasattr(chunk, "agent_id"):
                    agent_id_result = chunk.agent_id
                if hasattr(chunk, "reinvoked"):
                    reinvoked = chunk.reinvoked
                if hasattr(chunk, "function_calls"):
                    function_calls = chunk.function_calls
                if hasattr(chunk, "result"):
                    execution_history.append(chunk)
                if isinstance(chunk, Completion):
                    completion = chunk
                response = chunk
            final_content = "".join(collected_content)

            return ResponseResult(
                content=final_content,
                response=response,
                completion=completion,
                function_calls=function_calls,
                agent_id=agent_id_result,
                execution_history=execution_history,
                reinvoked=reinvoked,
            )

    def _run_stream(
        self,
        prompt: str | None = None,
        context_variables: dict | None = None,
        messages: MessagesHistory | None = None,
        agent_id: str | None | Agent = None,
        apply_functions: bool = True,
        print_formatted_prompt: bool = False,
        use_instructed_prompt: bool = False,
        conversation_name_holder: str = "Messages",
        mention_last_turn: bool = True,
        reinvoke_after_function: bool = True,
        reinvoked_runtime: bool = False,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> Generator[StreamingResponseType, None, None]:
        """Internal method for streaming execution.

        Runs the async create_response method in a background thread and
        yields results through a queue for synchronous iteration.

        Args:
            Same as create_response.

        Yields:
            StreamingResponseType objects from the async response.

        Raises:
            Any exception that occurs during async execution.
        """
        output_queue = queue.Queue()
        exception_holder = [None]

        def run_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def async_runner():
                    try:
                        response = await self.create_response(
                            prompt=prompt,
                            context_variables=context_variables,
                            messages=messages,
                            agent_id=agent_id,
                            stream=True,
                            apply_functions=apply_functions,
                            print_formatted_prompt=print_formatted_prompt,
                            use_instructed_prompt=use_instructed_prompt,
                            conversation_name_holder=conversation_name_holder,
                            mention_last_turn=mention_last_turn,
                            reinvoke_after_function=reinvoke_after_function,
                            reinvoked_runtime=reinvoked_runtime,
                            streamer_buffer=streamer_buffer,
                        )

                        async for output in response:
                            if output is not None:
                                output_queue.put(output)

                    except Exception as e:
                        exception_holder[0] = e

                loop.run_until_complete(async_runner())
                loop.close()

            except Exception as e:
                exception_holder[0] = e

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()

        while True:
            try:
                output = output_queue.get(timeout=1.0)
                if output is None:
                    break
                yield output
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

        if exception_holder[0]:
            raise exception_holder[0]

        thread.join(timeout=1.0)

    def thread_run(
        self,
        prompt: str | None = None,
        context_variables: dict | None = None,
        messages: MessagesHistory | None = None,
        agent_id: str | None | Agent = None,
        apply_functions: bool = True,
        print_formatted_prompt: bool = False,
        use_instructed_prompt: bool = False,
        conversation_name_holder: str = "Messages",
        mention_last_turn: bool = True,
        reinvoke_after_function: bool = True,
        reinvoked_runtime: bool = False,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> tuple[StreamerBuffer, threading.Thread]:
        """Run Calute in a background thread with automatic buffer creation.

        Returns immediately with a StreamerBuffer and the thread handle.
        You can start consuming from the buffer while generation is happening.
        This is useful for non-blocking execution in synchronous contexts.

        Args:
            prompt: Optional user prompt to process.
            context_variables: Optional context variables for the agent.
            messages: Optional message history.
            agent_id: Optional specific agent ID or Agent instance to use.
            apply_functions: Whether to execute detected function calls.
            print_formatted_prompt: Whether to print the formatted prompt.
            use_instructed_prompt: Whether to use instructed prompt format.
            conversation_name_holder: Name for conversation in instructed format.
            mention_last_turn: Whether to mention last turn in instructed format.
            reinvoke_after_function: Whether to reinvoke after function execution.
            reinvoked_runtime: Internal flag indicating this is a reinvocation.
            streamer_buffer: Optional pre-created buffer (creates new if None).

        Returns:
            Tuple of (StreamerBuffer, Thread) where:
            - StreamerBuffer: Buffer that will receive all streaming chunks
            - Thread: The background thread handle for monitoring/joining

        Example:
            >>> buffer, thread = calute.thread_run(prompt="Hello")
            >>> for chunk in buffer.stream():
            ...     print(chunk.content, end="")
            >>> thread.join()
            >>>
            >>>
            >>> result = buffer.get_result(timeout=30)
            >>> print(result.content)
        """

        buffer_was_none = streamer_buffer is None
        if streamer_buffer is None:
            streamer_buffer = StreamerBuffer()

        result_holder = [None]
        exception_holder = [None]

        def run_in_thread():
            try:
                result = self.run(
                    prompt=prompt,
                    context_variables=context_variables,
                    messages=messages,
                    agent_id=agent_id,
                    stream=False,
                    apply_functions=apply_functions,
                    print_formatted_prompt=print_formatted_prompt,
                    use_instructed_prompt=use_instructed_prompt,
                    conversation_name_holder=conversation_name_holder,
                    mention_last_turn=mention_last_turn,
                    reinvoke_after_function=reinvoke_after_function,
                    reinvoked_runtime=reinvoked_runtime,
                    streamer_buffer=streamer_buffer,
                )
                result_holder[0] = result
            except Exception as e:
                exception_holder[0] = e
            finally:
                if buffer_was_none:
                    streamer_buffer.close()

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        streamer_buffer.thread = thread
        streamer_buffer.result_holder = result_holder
        streamer_buffer.exception_holder = exception_holder

        def get_result(timeout: float | None = None) -> ResponseResult:
            """Helper to get final result after thread completes."""
            thread.join(timeout=timeout)
            if exception_holder[0]:
                raise exception_holder[0]
            return result_holder[0]

        streamer_buffer.get_result = get_result

        return streamer_buffer, thread

    async def athread_run(
        self,
        prompt: str | None = None,
        context_variables: dict | None = None,
        messages: MessagesHistory | None = None,
        agent_id: str | None | Agent = None,
        apply_functions: bool = True,
        print_formatted_prompt: bool = False,
        use_instructed_prompt: bool = False,
        conversation_name_holder: str = "Messages",
        mention_last_turn: bool = True,
        reinvoke_after_function: bool = True,
        reinvoked_runtime: bool = False,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> tuple[StreamerBuffer, asyncio.Task]:
        """Async version of thread_run that creates a task instead of thread.

        Returns immediately with a StreamerBuffer and the task handle.

        Args:
            Same as create_response except stream (always streams internally).

        Returns:
            Tuple of (StreamerBuffer, Task) where:
            - StreamerBuffer: Buffer that will receive all streaming chunks
            - Task: The asyncio task handle for monitoring/awaiting

        Example:
            >>> buffer, task = await calute.athread_run(prompt="Hello")
            >>> async for chunk in buffer.astream():
            ...     print(chunk.content, end="")
            >>> await task
        """

        buffer_was_none = streamer_buffer is None
        if streamer_buffer is None:
            streamer_buffer = StreamerBuffer()

        result_holder = [None]
        exception_holder = [None]

        async def run_async():
            try:
                stream = await self.create_response(
                    prompt=prompt,
                    context_variables=context_variables,
                    messages=messages,
                    agent_id=agent_id,
                    stream=True,
                    apply_functions=apply_functions,
                    print_formatted_prompt=print_formatted_prompt,
                    use_instructed_prompt=use_instructed_prompt,
                    conversation_name_holder=conversation_name_holder,
                    mention_last_turn=mention_last_turn,
                    reinvoke_after_function=reinvoke_after_function,
                    reinvoked_runtime=reinvoked_runtime,
                    streamer_buffer=streamer_buffer,
                )

                collected_content = []
                final_response = None
                async for chunk in stream:
                    if hasattr(chunk, "content") and chunk.content:
                        collected_content.append(chunk.content)
                    final_response = chunk

                result = ResponseResult(
                    content="".join(collected_content),
                    response=final_response,
                    function_calls=getattr(final_response, "function_calls", []),
                    agent_id=getattr(final_response, "agent_id", "default"),
                    execution_history=getattr(final_response, "execution_history", []),
                    reinvoked=getattr(final_response, "reinvoked", False),
                )
                result_holder[0] = result

            except Exception as e:
                exception_holder[0] = e
            finally:
                if buffer_was_none:
                    streamer_buffer.close()

        task = asyncio.create_task(run_async())

        streamer_buffer.task = task
        streamer_buffer.result_holder = result_holder
        streamer_buffer.exception_holder = exception_holder

        async def aget_result() -> ResponseResult:
            """Helper to get final result after task completes."""
            await task
            if exception_holder[0]:
                raise exception_holder[0]
            return result_holder[0]

        streamer_buffer.aget_result = aget_result

        return streamer_buffer, task


__all__ = ("Calute", "PromptSection", "PromptTemplate")
