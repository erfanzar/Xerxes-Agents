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


"""Core Xerxes module implementing the main agent orchestration framework.

This module contains the primary Xerxes class that provides sophisticated
agent management capabilities including:
- Multi-agent orchestration and switching
- Function/tool execution with retry logic
- Memory system integration
- Streaming response handling
- Prompt template management
- Asynchronous and synchronous execution modes

The module also includes prompt templating utilities and helper functions
for formatting and parsing agent responses.

Key components:
- Xerxes: Main orchestration class for managing AI agents
- PromptSection: Enumeration for structured prompt sections
- PromptTemplate: Configurable template for structuring agent prompts

Typical usage example:
    from xerxes import Xerxes, Agent
    from xerxes.llms import OpenAILLM


    llm = OpenAILLM(api_key="your-api-key")
    xerxes = Xerxes(llm=llm, enable_memory=True)


    agent = Agent(
        id="assistant",
        instructions="You are a helpful assistant.",
        model="gpt-4"
    )
    xerxes.register_agent(agent)


    for chunk in xerxes.run(prompt="Hello!"):
        if chunk.content:
            print(chunk.content, end="")


    result = xerxes.run(prompt="Hello!", stream=False)
    print(result.content)
"""

import asyncio
import json
import logging
import os
import pprint
import queue
import re
import textwrap
import threading
import typing as tp
import uuid
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from xerxes.types.function_execution_types import ReinvokeSignal
from xerxes.types.messages import ChatMessage, MessagesHistory, SystemMessage, UserMessage

from .core.prompt_template import SEP, PromptSection, PromptTemplate
from .core.streamer_buffer import StreamerBuffer
from .core.utils import debug_print, function_to_json, get_callable_public_name
from .executors import EnhancedAgentOrchestrator, EnhancedFunctionExecutor
from .llms import BaseLLM
from .memory import MemoryStore, MemoryType
from .operators import OperatorRuntimeConfig
from .runtime.features import RuntimeFeaturesConfig, RuntimeFeaturesState
from .runtime.loop_detection import LoopDetector
from .runtime.session import RuntimeSession
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
    Result,
    StreamChunk,
    StreamingResponseType,
    SwitchContext,
    ToolCall,
    ToolCallStreamChunk,
    ToolMessage,
)
from .types.oai_protocols import ToolDefinition
from .types.tool_calls import FunctionCall

logger = logging.getLogger(__name__)


def add_depth(x, add_prefix=False):
    return SEP + x.replace("\n", f"\n{SEP}") if add_prefix else x.replace("\n", f"\n{SEP}")


_TOOL_PARAMETER_TAG_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.-]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class _RuntimeTurnState:
    """Tracks one user-visible turn across tool reinvocation cycles."""

    turn_id: str
    prompt: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    tool_calls: list[tp.Any] = field(default_factory=list)
    finalized: bool = False


class Xerxes:
    """Main Xerxes orchestration class for managing AI agents.

    This is the primary interface for interacting with the Xerxes framework.
    It manages agent registration, prompt formatting, function execution,
    memory integration, and response generation with support for both
    streaming and non-streaming modes.

    The Xerxes class provides:
    - Agent registration and orchestration across multiple agents
    - Automatic function/tool calling with retry logic
    - Memory system integration for context persistence
    - Both synchronous (run) and asynchronous (create_response) interfaces
    - Streaming and non-streaming response modes
    - Prompt template customization
    - Agent switching based on capabilities or error recovery

    Attributes:
        SEP: Class variable defining the separator used for indentation.
        llm_client: The BaseLLM instance for generating completions.
        template: PromptTemplate for structuring agent prompts.
        orchestrator: AgentOrchestrator for managing multi-agent workflows.
        executor: FunctionExecutor for handling tool/function calls.
        enable_memory: Whether the memory system is enabled.
        auto_add_memory_tools: Whether to auto-add memory tools to agents.
        memory_store: MemoryStore instance for persistent context (if enabled).

    Example:
        Basic usage with streaming:

        >>> from xerxes import Xerxes, Agent
        >>> from xerxes.llms import OpenAILLM
        >>> llm = OpenAILLM(api_key="your-key")
        >>> xerxes = Xerxes(llm=llm, enable_memory=True)
        >>> agent = Agent(id="helper", instructions="You are helpful.")
        >>> xerxes.register_agent(agent)
        >>> for chunk in xerxes.run(prompt="Hello!"):
        ...     print(chunk.content, end="")

        Non-streaming usage:

        >>> result = xerxes.run(prompt="Hello!", stream=False)
        >>> print(result.content)

        Async usage:

        >>> async for chunk in await xerxes.create_response(prompt="Hi"):
        ...     print(chunk.content, end="")
    """

    SEP: tp.ClassVar[str] = SEP
    REINVOKE_FOLLOWUP_INSTRUCTION: tp.ClassVar[str] = (
        "Use the function results above to continue the task. If the results already answer the user's"
        " request, respond to the user directly. Only call another function if the returned data is"
        " missing something necessary or the user explicitly asked for a fresh lookup. If a web/search tool"
        " already ran, do not claim you cannot browse or access current information. Treat search-result"
        " snippets as leads rather than verified facts; say that the search results indicate or suggest"
        " something unless you opened a source page and confirmed it."
    )

    def __init__(
        self,
        llm: BaseLLM | None = None,
        template: PromptTemplate | None = None,
        enable_memory: bool = False,
        memory_config: dict[str, tp.Any] | None = None,
        auto_add_memory_tools: bool = True,
        runtime_features: RuntimeFeaturesConfig | None = None,
    ):
        """Initialize Xerxes with an LLM.

        Args:
            llm: A BaseLLM instance for generating completions.
            template: Optional prompt template for structuring prompts.
            enable_memory: Whether to enable the memory system.
            auto_add_memory_tools: Whether to automatically add memory tools to agents when memory is enabled.
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
            >>> xerxes = Xerxes(
            ...     llm=llm,
            ...     enable_memory=True,
            ...     memory_config={"max_short_term": 50}
            ... )
        """

        self.llm_client: BaseLLM = llm

        self.template = template or PromptTemplate()
        self.orchestrator = EnhancedAgentOrchestrator()
        self.executor = EnhancedFunctionExecutor(self.orchestrator)
        self.enable_memory = enable_memory
        self.auto_add_memory_tools = auto_add_memory_tools
        self._launch_workspace_root = os.path.abspath(os.getcwd())
        self.runtime_features = self._normalize_runtime_features(runtime_features, self._launch_workspace_root)
        self._runtime_features_state: RuntimeFeaturesState | None = (
            RuntimeFeaturesState(self.runtime_features)
            if (
                self.runtime_features.enabled
                or (self.runtime_features.operator is not None and self.runtime_features.operator.enabled)
            )
            else None
        )
        if self._runtime_features_state is not None and self._runtime_features_state.operator_state is not None:
            self._runtime_features_state.operator_state.attach_runtime(self, self._runtime_features_state)
        self._session_id: str | None = None
        if self._runtime_features_state is not None and self._runtime_features_state.session_manager is not None:
            session = self._runtime_features_state.session_manager.start_session()
            self._session_id = session.session_id
            if self._runtime_features_state.audit_emitter is not None:
                self._runtime_features_state.audit_emitter._session_id = self._session_id
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

    @staticmethod
    def _normalize_runtime_features(
        runtime_features: RuntimeFeaturesConfig | None,
        workspace_root: str,
    ) -> RuntimeFeaturesConfig:
        """Ensure Xerxes has operator runtime available by default.

        When no runtime config is supplied, Xerxes starts with runtime features
        enabled and operator tooling available. When runtime features are explicitly
        enabled but no operator config is provided, operator tooling is attached with
        its default allow-by-default policy.
        """
        if runtime_features is None:
            return RuntimeFeaturesConfig(
                enabled=True,
                workspace_root=workspace_root,
                operator=OperatorRuntimeConfig(
                    enabled=True,
                    power_tools_enabled=True,
                    shell_default_workdir=workspace_root,
                ),
            )

        if runtime_features.workspace_root is None:
            runtime_features.workspace_root = workspace_root

        if runtime_features.enabled and runtime_features.operator is None:
            runtime_features.operator = OperatorRuntimeConfig(enabled=True, power_tools_enabled=True)

        if runtime_features.operator is not None and runtime_features.operator.shell_default_workdir is None:
            runtime_features.operator.shell_default_workdir = workspace_root

        return runtime_features

    def _setup_default_triggers(self) -> None:
        """Setup default agent switching triggers.

        Registers default trigger handler functions with the orchestrator for
        automatic agent switching based on context conditions. These triggers
        enable intelligent multi-agent orchestration.

        Registered Triggers:
            - CAPABILITY_BASED: Switches to an agent that has a required
              capability when 'required_capability' is present in context.
              Selects the agent with the highest performance score for that
              capability.
            - ERROR_RECOVERY: Switches to the current agent's fallback_agent_id
              when 'execution_error' is present in context, enabling graceful
              error recovery.

        Returns:
            None

        Side Effects:
            Registers two switch trigger handlers with self.orchestrator.
        """

        def capability_based_switch(context, agents, current_agent_id):
            """Switch to the highest-scoring agent that has the required capability."""
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
            """Switch to the current agent's fallback when an execution error is present."""
            if context.get("execution_error") and current_agent_id:
                current_agent = agents[current_agent_id]
                if current_agent.fallback_agent_id:
                    return current_agent.fallback_agent_id
            return None

        self.orchestrator.register_switch_trigger(AgentSwitchTrigger.CAPABILITY_BASED, capability_based_switch)
        self.orchestrator.register_switch_trigger(AgentSwitchTrigger.ERROR_RECOVERY, error_recovery_switch)

    def create_query_engine(
        self,
        model: str = "",
        system_prompt: str = "",
        **config_kwargs: tp.Any,
    ):
        """Create a fully-wired QueryEngine from this Xerxes instance.

        The QueryEngine provides a multi-turn conversation interface with
        budget control, auto-compaction, cost tracking, and history logging.

        Args:
            model: Model name override. If empty, uses the current agent's model.
            system_prompt: System prompt override.
            **config_kwargs: Additional QueryEngineConfig kwargs.

        Returns:
            A :class:`QueryEngine` instance.

        Example:
            >>> engine = xerxes.create_query_engine(model="gpt-4o")
            >>> result = engine.submit("Hello!")
            >>> print(result.output)
        """
        from .runtime.bridge import create_query_engine

        agent = self.orchestrator.get_current_agent() if self.orchestrator.current_agent_id else None
        if not model and agent:
            model = agent.model or ""
        if not system_prompt and agent:
            system_prompt = agent.instructions or ""

        return create_query_engine(
            xerxes_instance=self,
            agent=agent,
            model=model,
            system_prompt=system_prompt,
            **config_kwargs,
        )

    def create_runtime_session(self, prompt: str = "") -> RuntimeSession:
        """Create a new RuntimeSession capturing current context.

        Args:
            prompt: Initial prompt or session description.

        Returns:
            A :class:`RuntimeSession` instance.
        """
        from .runtime.session import RuntimeSession

        agent = self.orchestrator.get_current_agent() if self.orchestrator.current_agent_id else None
        model = (agent.model if agent else "") or ""
        return RuntimeSession.create(model=model, prompt=prompt)

    def bootstrap(self, extra_context: str = ""):
        """Run the bootstrap sequence for this Xerxes instance.

        Performs environment detection, git info loading, XERXES.md loading,
        tool registration, and system prompt building.

        Args:
            extra_context: Additional context for the system prompt.

        Returns:
            A :class:`BootstrapResult`.
        """
        from .runtime.bridge import bootstrap_xerxes

        agent = self.orchestrator.get_current_agent() if self.orchestrator.current_agent_id else None
        model = (agent.model if agent else "") or ""
        return bootstrap_xerxes(
            xerxes_instance=self,
            agent=agent,
            model=model,
            extra_context=extra_context,
        )

    def get_execution_registry(self):
        """Get a populated ExecutionRegistry with all Xerxes tools.

        Returns:
            An :class:`ExecutionRegistry` with all tools registered.
        """
        from .runtime.bridge import populate_registry

        return populate_registry()

    def get_tool_executor(self):
        """Get a tool executor callable for the streaming agent loop.

        Returns:
            A callable ``(tool_name: str, tool_input: dict) -> str``.
        """
        from .runtime.bridge import build_tool_executor

        agent = self.orchestrator.get_current_agent() if self.orchestrator.current_agent_id else None
        registry = self.get_execution_registry()
        return build_tool_executor(xerxes_instance=self, agent=agent, registry=registry)

    def create_subagent_manager(
        self,
        max_concurrent: int = 5,
        max_depth: int = 5,
    ):
        """Create a SubAgentManager with the streaming agent loop wired up.

        The manager uses a thread pool for concurrent sub-agent execution,
        supports git worktree isolation, named agents, and inbox queues.

        Args:
            max_concurrent: Maximum number of concurrent sub-agents.
            max_depth: Maximum nesting depth.

        Returns:
            A :class:`SubAgentManager` instance.

        Example:
            >>> mgr = xerxes.create_subagent_manager()
            >>> from xerxes.agents import get_agent_definition
            >>> task = mgr.spawn(
            ...     prompt="Review this code",
            ...     config={"model": "gpt-4o"},
            ...     system_prompt="You are helpful.",
            ...     agent_def=get_agent_definition("reviewer"),
            ...     name="code-review",
            ... )
            >>> mgr.wait(task.id)
            >>> print(task.result)
        """
        from .agents.subagent_manager import SubAgentManager
        from .runtime.bridge import build_tool_executor, populate_registry

        mgr = SubAgentManager(max_concurrent=max_concurrent, max_depth=max_depth)

        registry = populate_registry()
        tool_executor = build_tool_executor(xerxes_instance=self, registry=registry)

        def runner(prompt, config, system_prompt, depth, cancel_check):
            from .streaming.events import AgentState, TextChunk
            from .streaming.loop import run

            state = AgentState()
            output_parts = []
            eff_tool_executor = tool_executor
            eff_tool_schemas = registry.tool_schemas()
            whitelist = config.get("_tools_whitelist")
            if whitelist:
                allowed = set(whitelist)
                eff_tool_schemas = [s for s in eff_tool_schemas if s.get("name") in allowed]

                def _filtered_executor(tool_name: str, tool_input: dict[str, Any]) -> str:
                    if tool_name not in allowed:
                        return f"Error: tool '{tool_name}' is not allowed for this agent."
                    return tool_executor(tool_name, tool_input)

                eff_tool_executor = _filtered_executor

            for event in run(
                user_message=prompt,
                state=state,
                config=config,
                system_prompt=system_prompt,
                tool_executor=eff_tool_executor,
                tool_schemas=eff_tool_schemas,
                depth=depth,
                cancel_check=cancel_check,
            ):
                if isinstance(event, TextChunk):
                    output_parts.append(event.text)
            return "".join(output_parts)

        mgr.set_runner(runner)
        return mgr

    def _notify_turn_start(self, agent_id: str | None, messages: MessagesHistory | None = None) -> None:
        """Fire the ``on_turn_start`` hook if any listeners are registered."""
        runtime_state = self._runtime_features_state
        if runtime_state is None or not runtime_state.hook_runner.has_hooks("on_turn_start"):
            return
        runtime_state.hook_runner.run("on_turn_start", agent_id=agent_id, messages=messages)

    def _notify_turn_end(self, agent_id: str | None, response: str | None = None) -> None:
        """Fire the ``on_turn_end`` hook if any listeners are registered."""
        runtime_state = self._runtime_features_state
        if runtime_state is None or not runtime_state.hook_runner.has_hooks("on_turn_end"):
            return
        runtime_state.hook_runner.run("on_turn_end", agent_id=agent_id, response=response)

    def _notify_runtime_error(self, agent_id: str | None, error: Exception) -> None:
        """Fire the ``on_error`` hook if any listeners are registered."""
        runtime_state = self._runtime_features_state
        if runtime_state is None or not runtime_state.hook_runner.has_hooks("on_error"):
            return
        runtime_state.hook_runner.run("on_error", agent_id=agent_id, error=error)

    @staticmethod
    def _new_runtime_turn_id() -> str:
        """Generate a compact random identifier for a new runtime turn."""
        return uuid.uuid4().hex[:12]

    def _append_turn_tool_results(
        self,
        turn_state: _RuntimeTurnState | None,
        results: list[RequestFunctionCall],
    ) -> None:
        """Append tool-call execution records to the current turn state."""
        if turn_state is None:
            return

        from .session import ToolCallRecord

        operator_state = (
            self._runtime_features_state.operator_state if self._runtime_features_state is not None else None
        )

        for result in results:
            arguments = result.arguments if isinstance(result.arguments, dict) else {}
            persisted_result: tp.Any = result.result
            metadata: dict[str, tp.Any] = {}
            if operator_state is not None:
                persisted_result, metadata = operator_state.summarize_result(result.result)
            turn_state.tool_calls.append(
                ToolCallRecord(
                    call_id=result.id,
                    tool_name=result.name,
                    arguments=arguments,
                    result=str(persisted_result)[:500] if persisted_result is not None else None,
                    status=result.status.value,
                    error=result.error,
                    metadata=metadata,
                )
            )

    def _finalize_runtime_turn(
        self,
        agent_id: str | None,
        response_content: str,
        turn_state: _RuntimeTurnState | None = None,
    ) -> None:
        """Close out a turn: fire hooks, emit audit event, and persist the session record."""
        self._notify_turn_end(agent_id, response_content)

        runtime_state = self._runtime_features_state
        if runtime_state is None:
            return

        function_calls_count = len(turn_state.tool_calls) if turn_state is not None else 0
        if runtime_state.audit_emitter is not None:
            runtime_state.audit_emitter.emit_turn_end(
                agent_id=agent_id,
                turn_id=turn_state.turn_id if turn_state is not None else None,
                content=response_content,
                fc_count=function_calls_count,
            )

        if turn_state is None or turn_state.finalized:
            return

        if runtime_state.session_manager is not None and self._session_id is not None:
            from .session import TurnRecord

            turn = TurnRecord(
                turn_id=turn_state.turn_id,
                agent_id=agent_id,
                prompt=turn_state.prompt,
                response_content=response_content[:1000] if response_content else None,
                tool_calls=list(turn_state.tool_calls),
                started_at=turn_state.started_at,
                ended_at=datetime.now(UTC).isoformat(),
                status="success",
            )
            runtime_state.session_manager.record_turn(self._session_id, turn)

        turn_state.finalized = True

    def _record_runtime_error(
        self,
        agent_id: str | None,
        error: Exception,
        context: str,
        turn_state: _RuntimeTurnState | None = None,
    ) -> None:
        """Fire error hooks, emit an audit error event, and persist a failed turn record."""
        self._notify_runtime_error(agent_id, error)

        runtime_state = self._runtime_features_state
        if runtime_state is None:
            return

        if runtime_state.audit_emitter is not None:
            runtime_state.audit_emitter.emit_error(
                error_type=type(error).__name__,
                error_msg=str(error),
                context=context,
                agent_id=agent_id,
                turn_id=turn_state.turn_id if turn_state is not None else None,
            )

        if turn_state is None or turn_state.finalized:
            return

        if runtime_state.session_manager is not None and self._session_id is not None:
            from .session import TurnRecord

            turn = TurnRecord(
                turn_id=turn_state.turn_id,
                agent_id=agent_id,
                prompt=turn_state.prompt,
                response_content=None,
                tool_calls=list(turn_state.tool_calls),
                started_at=turn_state.started_at,
                ended_at=datetime.now(UTC).isoformat(),
                status="error",
                error=str(error),
            )
            runtime_state.session_manager.record_turn(self._session_id, turn)

        turn_state.finalized = True

    @classmethod
    def _is_reinvoke_followup_message(cls, message: ChatMessage) -> bool:
        """Return True if the message is the standard reinvoke follow-up instruction."""
        return isinstance(message, UserMessage) and (message.content or "").strip() == cls.REINVOKE_FOLLOWUP_INSTRUCTION

    @staticmethod
    def _is_operator_reinvoke_attachment(message: ChatMessage) -> bool:
        """Return True for synthetic user messages injected from operator tool results."""
        if not isinstance(message, UserMessage) or isinstance(message.content, str):
            return False
        if not message.content:
            return False
        first_chunk = message.content[0]
        return hasattr(first_chunk, "text") and str(first_chunk.text).startswith("[TOOL IMAGE RESULT]")

    @classmethod
    def _compact_reinvoke_history(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Strip trailing reinvoke cycles (follow-up instruction + tool messages + assistant call) from history."""
        compacted = messages.copy()

        while compacted and cls._is_reinvoke_followup_message(compacted[-1]):
            compacted.pop()
            while compacted and cls._is_operator_reinvoke_attachment(compacted[-1]):
                compacted.pop()
            while compacted and isinstance(compacted[-1], ToolMessage):
                compacted.pop()
            if compacted and isinstance(compacted[-1], AssistantMessage) and compacted[-1].tool_calls:
                compacted.pop()

        return compacted

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator.

        Registers the agent for multi-agent orchestration, optionally adding
        memory tools if memory is enabled and auto_add_memory_tools is True.
        The first registered agent becomes the default active agent.

        Args:
            agent: The Agent instance to register for orchestration.

        Returns:
            None

        Side Effects:
            - Adds memory tools to agent if memory is enabled
            - Updates orchestrator's agent registry
            - Sets agent as current if it's the first registered

        Example:
            >>> agent = Agent(id="helper", instructions="Be helpful")
            >>> xerxes.register_agent(agent)
        """
        if self.enable_memory and self.auto_add_memory_tools:
            self._add_memory_tools_to_agent(agent)
        if self._runtime_features_state is not None:
            self._runtime_features_state.merge_plugin_tools(agent)
            self._runtime_features_state.merge_operator_tools(agent)
        self.orchestrator.register_agent(agent)

    def _add_memory_tools_to_agent(self, agent: Agent) -> None:
        """Add memory tools to an agent if not already present.

        Imports and adds the standard memory tools (store, retrieve, etc.)
        to the agent's function list, enabling the agent to interact with
        the memory system during conversations.

        Args:
            agent: The Agent instance to add memory tools to.

        Returns:
            None

        Side Effects:
            - Initializes agent.functions to empty list if None
            - Appends memory tools not already present in agent.functions

        Note:
            This method checks for existing functions by name to avoid
            duplicate tool registrations.
        """
        from .tools.memory_tool import MEMORY_TOOLS

        if agent.functions is None:
            agent.functions = []

        current_func_names = {get_callable_public_name(func) for func in agent.functions}

        for tool in MEMORY_TOOLS:
            if get_callable_public_name(tool) not in current_func_names:
                agent.functions.append(tool)

    def _update_memory_from_response(
        self,
        content: str,
        agent_id: str,
        context_variables: dict | None = None,
        function_calls: list[RequestFunctionCall] | None = None,
    ) -> None:
        """Update memory system based on agent response.

        Stores the agent's response and any function calls in the memory
        system for future context retrieval. Response content is stored
        as short-term memory, while function calls are stored as working
        memory with higher importance.

        Args:
            content: The response content from the agent.
            agent_id: ID of the agent that generated the response.
            context_variables: Optional context variables to store with memory.
            function_calls: Optional list of function calls made in the response.

        Returns:
            None

        Side Effects:
            - Adds response content to short-term memory (importance: 0.6)
            - Adds each function call to working memory (importance: 0.7)

        Note:
            This method is a no-op if memory is not enabled.
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

    def _update_memory_from_prompt(self, prompt: str, agent_id: str) -> None:
        """Update memory system from user prompt.

        Stores the user's prompt in short-term memory with high importance
        for context retrieval in subsequent interactions.

        Args:
            prompt: The user's input prompt text.
            agent_id: ID of the agent receiving the prompt.

        Returns:
            None

        Side Effects:
            - Adds user prompt to short-term memory (importance: 0.8)
            - Tags the memory entry with "user_input"

        Note:
            This method is a no-op if memory is not enabled.
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
        """Format a section of the prompt with a header and indented content.

        Creates a formatted prompt section with proper indentation and
        optional item prefixes for list content.

        Args:
            header: The section header text (e.g., "RULES:", "CONTEXT:").
            content: The section content as a string or list of strings.
            item_prefix: Optional prefix for list items (default: "- ").
                Set to None to disable prefixing.

        Returns:
            Formatted section string with header and indented content,
            or None if content is empty or None.

        Example:
            >>> xerxes._format_section("RULES:", ["Be helpful", "Be concise"])
            'RULES:\\n  - Be helpful\\n  - Be concise'
        """
        if not content:
            return None

        if isinstance(content, list):
            content_str = "\n".join(f"{item_prefix or ''}{str(line).strip()}" for line in content)
        else:
            content_str = str(content).strip()

        if not content_str:
            return None

        if not header:
            return content_str

        indented = textwrap.indent(content_str, SEP)
        return f"{header}\n{indented}"

    def _extract_from_markdown(self, content: str, field: str) -> list[str]:
        """Extract content from markdown code blocks with a specific field identifier.

        Searches for all markdown code blocks with the specified field identifier
        and extracts their raw content as strings.

        Args:
            content: The response content to search through.
            field: The markdown field identifier (e.g., 'tool_call', 'json').
                This is matched after the opening triple backticks.

        Returns:
            List of extracted content strings from matching markdown blocks.
            Returns empty list if no matching blocks are found.

        Example:
            >>> content = '```tool_call\\n{"name": "func"}\\n```'
            >>> xerxes._extract_from_markdown(content, "tool_call")
            ['{"name": "func"}']
        """
        pattern = rf"```{field}\s*\n(.*?)\n```"
        return re.findall(pattern, content, re.DOTALL)

    @staticmethod
    def _system_message_to_text(message: SystemMessage) -> str | None:
        """Convert a system message into plain text for prompt deduplication."""
        if isinstance(message.content, str):
            content = message.content.strip()
            return content or None

        parts: list[str] = []
        for chunk in message.content:
            text = getattr(chunk, "text", None)
            if text:
                cleaned = str(text).strip()
                if cleaned:
                    parts.append(cleaned)

        if not parts:
            return None
        return "\n".join(parts)

    def _merge_system_history(
        self,
        final_system_content: str,
        messages: MessagesHistory | None,
    ) -> tuple[str, list[ChatMessage]]:
        """Collapse any prior system messages into one prompt header."""
        if not messages or not messages.messages:
            return final_system_content, []

        merged_parts: list[str] = []
        if final_system_content.strip():
            merged_parts.append(final_system_content.strip())

        remaining_messages: list[ChatMessage] = []
        for message in messages.messages:
            if isinstance(message, SystemMessage):
                system_text = self._system_message_to_text(message)
                if system_text:
                    merged_parts.append(system_text)
                continue
            remaining_messages.append(message)

        deduped_parts: list[str] = []
        seen_parts: set[str] = set()
        for part in merged_parts:
            normalized = part.strip()
            if not normalized or normalized in seen_parts:
                continue
            seen_parts.add(normalized)
            deduped_parts.append(normalized)

        return "\n\n".join(deduped_parts), remaining_messages

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
            >>> messages = xerxes.manage_messages(
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
        if self._runtime_features_state is not None:
            prompt_prefix = self._runtime_features_state.build_prompt_prefix(
                agent_id=agent.id,
                tool_names=[self._build_tool_prompt_label(func) for func in agent.functions],
            )
            if prompt_prefix:
                instructions = f"{prompt_prefix}\n\n{instructions}".strip()
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
                "Do not call a function for greetings, simple conversation, or requests you can answer directly"
                " from the current conversation and instructions. Prefer a normal response unless a function is"
                " required to get missing information or take an action."
            )
            rules.append(
                "If the user explicitly asks to search, look up, browse, or find something on the web and"
                " `web.search_query` is available, call it instead of answering from memory."
            )
            rules.append(
                "If the user gives a generic follow-up like `search the web`, `look it up`, or `find it`,"
                " infer the target topic from the latest relevant user request instead of asking the same"
                " clarification again, then call `web.search_query` if it is needed."
            )
            rules.append(
                "If web tools are available or prior tool results are present in the conversation, do not say"
                " that you cannot browse, search the web, or access current information."
            )
            rules.append(
                "Search-result snippets are not the same as verified facts. Say that search results indicate or"
                " suggest something unless you have opened the source and confirmed it."
            )
            rules.append(
                "If a function can satisfy the user request, you MUST respond only with a valid tool call in the"
                " specified format. Do not add any conversational text before or after the tool call."
            )
        elif agent.functions:
            rules.extend(
                [
                    "Do not use functions for greetings, simple conversation, or requests you can answer directly"
                    " from the current conversation and instructions. Use them only when they are needed to gather"
                    " missing information or take actions.",
                    "If the user explicitly asks to search, look up, browse, or find something on the web and"
                    " `web.search_query` is available, use it instead of answering from memory.",
                    "If the user gives a generic follow-up like `search the web`, `look it up`, or `find it`,"
                    " infer the topic from the latest relevant user request instead of asking the same"
                    " clarification again, then use `web.search_query` if needed.",
                    "If web tools are available or prior tool results are present in the conversation, do not say"
                    " that you cannot browse, search the web, or access current information.",
                    "Search-result snippets are not verified facts. Describe them as indications or leads unless"
                    " you opened the source and confirmed the claim.",
                    "After a function returns a result, use that result to continue the task and answer the user.",
                    "Do not repeat the same function call with the same arguments if the available result already"
                    " answers the request unless the user asks for refreshed data or the result is incomplete.",
                ]
            )
        if self.enable_memory and include_memory:
            rules.extend(
                [
                    "Consider previous context and conversation history.",
                    "Build upon earlier interactions when appropriate.",
                ]
            )
        system_parts.append(self._format_section(rules_header, rules))

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
        final_system_content, history_messages = self._merge_system_history(final_system_content, messages)
        instructed_messages.append(SystemMessage(content=final_system_content))

        if history_messages:
            instructed_messages.extend(history_messages)

        if prompt is not None:
            final_prompt_content = prompt
            if require_reflection:
                final_prompt_content += (
                    f"\n\nAfter your primary response, add a reflection section in `<reflection>` tags:\n"
                    f"{self.SEP}- Assumptions made.\n"
                    f"{self.SEP}- Potential limitations of your response."
                )
            instructed_messages.append(UserMessage(content=final_prompt_content))

        message_out = MessagesHistory(messages=instructed_messages)

        return message_out

    def _build_reinvoke_messages(
        self,
        original_messages: MessagesHistory,
        assistant_content: str,
        function_calls: list[RequestFunctionCall],
        results: list[RequestFunctionCall],
        agent_id: str | None = None,
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
        messages = self._compact_reinvoke_history(original_messages.messages)

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

        runtime_state = self._runtime_features_state
        for fc, result in zip(function_calls, results, strict=False):
            if result.status == ExecutionStatus.SUCCESS:
                tool_result: tp.Any = result.result
            else:
                tool_result = f"Error: {result.error}"

            if runtime_state is not None and runtime_state.hook_runner.has_hooks("tool_result_persist"):
                tool_result = runtime_state.hook_runner.run(
                    "tool_result_persist",
                    tool_name=fc.name,
                    result=tool_result,
                    agent_id=agent_id,
                )

            tool_content = json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result

            tool_msg = ToolMessage(content=tool_content, tool_call_id=fc.id)
            messages.append(tool_msg)
            if runtime_state is not None and runtime_state.operator_state is not None:
                operator_message = runtime_state.operator_state.create_reinvoke_message(result.result)
                if operator_message is not None:
                    messages.append(operator_message)

        messages.append(UserMessage(content=self.REINVOKE_FOLLOWUP_INSTRUCTION))

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
            >>> Xerxes.extract_md_block(text)
            [('xml', '<web_research>\n  <arguments>\n    {"query": "quantum computing breakthroughs 2024"}\n  </arguments>\n</web_research>')]
        """
        pattern = r"```(\w*)\n(.*?)\n```"
        matches = re.findall(pattern, input_string, re.DOTALL)
        return [(lang, content.strip()) for lang, content in matches]

    def _remove_function_calls_from_content(self, content: str) -> str:
        """Remove function call XML blocks from content.

        Cleans the response content by removing XML-formatted function calls,
        tagged function-call blocks, and markdown function-call blocks,
        leaving only the conversational text.

        Args:
            content: The content to clean.

        Returns:
            Content with function call blocks removed.
        """
        pattern = r"<(\w+)>\s*<arguments>.*?</arguments>\s*</\w+>"
        cleaned = re.sub(pattern, "", content, flags=re.DOTALL)
        pattern = r"<function=[A-Za-z0-9_.:-]+>\s*.*?</function>"
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
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
        valid_function_names = set(agent.get_available_functions())
        pattern = r"<(\w+)>\s*<arguments>(.*?)</arguments>\s*</\w+>"
        matches = re.findall(pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            name = match[0]
            if name not in valid_function_names:
                logger.debug("Ignoring XML function call for unknown tool '%s'", name)
                continue
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

    def _extract_function_calls_from_tagged_markup(self, content: str, agent: Agent) -> list[RequestFunctionCall]:
        """Extract pseudo-XML tool calls such as ``<function=name>`` blocks."""
        function_calls = []
        functions_by_name = {get_callable_public_name(func): func for func in agent.functions}
        valid_function_names = set(functions_by_name)
        pattern = r"<function=([A-Za-z0-9_.:-]+)>\s*(.*?)\s*</function>"
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

        for i, (name, body) in enumerate(matches):
            if name not in valid_function_names:
                logger.debug("Ignoring tagged function call for unknown tool '%s'", name)
                continue

            arguments: dict[str, tp.Any] = {}
            for param_name, raw_value in _TOOL_PARAMETER_TAG_RE.findall(body):
                value = raw_value.strip()
                if not value:
                    continue
                if value.startswith(("'", '"', "{", "[", "-")) or value in {"true", "false", "null"} or value.isdigit():
                    try:
                        arguments[param_name] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                arguments[param_name] = value

            try:
                required_fields = set(
                    function_to_json(functions_by_name[name])["function"]["parameters"].get("required", [])
                )
            except Exception:
                required_fields = set()
            if required_fields and not required_fields.issubset(arguments):
                logger.debug("Ignoring tagged function call for '%s' because required arguments are missing", name)
                continue
            if not arguments and required_fields:
                continue

            function_calls.append(
                RequestFunctionCall(
                    name=name,
                    arguments=arguments,
                    id=f"call_{i}_{hash((name, body))}",
                    timeout=agent.function_timeout,
                    max_retries=agent.max_function_retries,
                )
            )

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
        valid_function_names = set(agent.get_available_functions())
        for call_data in function_calls_data:
            try:
                name = call_data.get("name")
                if name not in valid_function_names:
                    logger.debug("Ignoring provider function call for unknown tool '%s'", name)
                    continue
                arguments = call_data.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass

                function_calls.append(
                    RequestFunctionCall(
                        name=name,
                        arguments=arguments,
                        id=call_data.get("id", f"call_{len(function_calls)}"),
                        timeout=agent.function_timeout,
                        max_retries=agent.max_function_retries,
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                logger.debug("Skipping malformed function call data: %s", e)
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
            valid_function_names = set(agent.get_available_functions())
            for call_ in tool_calls:
                try:
                    name = call_.function.name
                    if name not in valid_function_names:
                        logger.debug("Ignoring provider tool call for unknown tool '%s'", name)
                        continue
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
                            name=name,
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
        function_calls = self._extract_function_calls_from_tagged_markup(content, agent)
        if function_calls:
            return function_calls

        function_calls = []
        valid_function_names = set(agent.get_available_functions())
        matches = self._extract_from_markdown(content=content, field="tool_call")

        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match)
                name = call_data.get("name")
                if name not in valid_function_names:
                    logger.debug("Ignoring markdown function call for unknown tool '%s'", name)
                    continue
                function_call = RequestFunctionCall(
                    name=name,
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
    def extract_from_markdown(format: str, string: str) -> str | None | dict:  # noqa: A002
        """Extract content from a markdown code block with specific format.

        Searches for a markdown code block with the specified format identifier
        and extracts its content. If the content is valid JSON, it is parsed
        and returned as a dictionary.

        Args:
            format: The markdown format identifier to search for (e.g., 'json',
                'python', 'xml'). This is matched after the opening triple backticks.
            string: The string containing the markdown block to search.

        Returns:
            - dict: If the block content is valid JSON
            - str: If the block content is not valid JSON
            - None: If no matching format block is found

        Example:
            >>> content = '```json\\n{"key": "value"}\\n```'
            >>> Xerxes.extract_from_markdown("json", content)
            {'key': 'value'}

            >>> content = '```python\\nprint("hello")\\n```'
            >>> Xerxes.extract_from_markdown("python", content)
            'print("hello")'

        Note:
            Only the first matching block is extracted if multiple exist.
        """
        pattern = rf"```{re.escape(format)}\s*\n(.*?)\n```"
        m = re.search(pattern, string, re.DOTALL)
        if not m:
            return None
        block = m.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return block

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
        function_names = [get_callable_public_name(func) for func in agent.functions]
        for func_name in function_names:
            if f"<{func_name}>" in content or f"<{func_name} " in content:
                if "<arguments>" in content:
                    return True
            if f"<function={func_name}>" in content and "<parameter=" in content:
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
        function_names = [get_callable_public_name(func) for func in agent.functions]
        for func_name in function_names:
            pattern = rf"<{func_name}(?:\s[^>]*)?>.*?<arguments>"
            if re.search(pattern, content, re.DOTALL):
                return True
            tagged_pattern = rf"<function={re.escape(func_name)}>.*?<parameter="
            if re.search(tagged_pattern, content):
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
            >>> Xerxes.get_thoughts(response)
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
            >>> Xerxes.filter_thoughts(response)
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
                category = func.category
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
                    func_name = get_callable_public_name(func)
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
                    func_name = get_callable_public_name(func)
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

    def _build_tool_prompt_label(self, func: AgentFunction) -> str:
        """Build a short tool label for visible system-prompt tool summaries.

        Native tool schemas are still passed to provider APIs. This helper only
        improves the human-readable prompt prefix so models that are weak at
        native tool use still see what a tool is for.
        """
        name = get_callable_public_name(func)
        try:
            schema = function_to_json(func)["function"]
        except Exception:
            return name

        description = str(schema.get("description") or "").strip()
        if not description:
            return name

        first_paragraph = description.split("\n\n", 1)[0].strip()
        first_line = first_paragraph.splitlines()[0].strip()
        summary = re.sub(r"\s+", " ", first_line).strip()
        if not summary:
            return name
        if len(summary) > 140:
            summary = summary[:137].rstrip() + "..."
        return f"{name}: {summary}"

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
        _runtime_loop_detector: LoopDetector | None = None,
        _runtime_turn_state: _RuntimeTurnState | None = None,
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
            >>> response = await xerxes.create_response(
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
        runtime_state = self._runtime_features_state
        if runtime_state is not None and not reinvoked_runtime and _runtime_turn_state is None:
            _runtime_turn_state = _RuntimeTurnState(
                turn_id=self._new_runtime_turn_id(),
                prompt=prompt or "",
            )
            self._notify_turn_start(agent.id or "default", messages)
            if runtime_state.audit_emitter is not None:
                runtime_state.audit_emitter.emit_turn_start(
                    agent_id=agent.id or "default",
                    turn_id=_runtime_turn_state.turn_id,
                    prompt=_runtime_turn_state.prompt,
                )

        if runtime_state is not None and _runtime_loop_detector is None:
            _runtime_loop_detector = runtime_state.create_loop_detector(agent.id or "default")

        try:
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
            with open("debug_prompt.json", "a") as f:
                json.dump({"key": prompt_str}, f, indent=2)
                f.write("\n")
            response = await self.llm_client.generate_completion(
                prompt=prompt_str,
                model=agent.model,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                top_p=agent.top_p,
                stop=agent.stop if isinstance(agent.stop, list) else ([agent.stop] if agent.stop else None),
                top_k=agent.top_k,
                min_p=agent.min_p,
                tools=(
                    None if use_instructed_prompt else [ToolDefinition(**function_to_json(fn)) for fn in agent.functions]
                ),
                presence_penalty=agent.presence_penalty,
                frequency_penalty=agent.frequency_penalty,
                repetition_penalty=agent.repetition_penalty,
                extra_body=agent.extra_body,
                stream=True,
            )
        except Exception as e:
            self._record_runtime_error(
                agent.id or "default", e, context="create_response_setup", turn_state=_runtime_turn_state
            )
            raise

        if not apply_functions:
            if stream:
                return self._handle_streaming(response, reinvoked_runtime, agent, streamer_buffer, _runtime_turn_state)
            else:
                collected_content = []
                collected_reasoning = ""
                completion = None
                async for chunk in self._handle_streaming(
                    response,
                    reinvoked_runtime,
                    agent,
                    streamer_buffer,
                    _runtime_turn_state,
                ):
                    if hasattr(chunk, "content") and chunk.content:
                        collected_content.append(chunk.content)
                    if hasattr(chunk, "buffered_reasoning_content") and chunk.buffered_reasoning_content:
                        collected_reasoning = chunk.buffered_reasoning_content
                    if hasattr(chunk, "reasoning_content") and chunk.reasoning_content and not collected_reasoning:
                        collected_reasoning = chunk.reasoning_content
                    if isinstance(chunk, Completion):
                        completion = chunk

                return ResponseResult(
                    content="".join(collected_content),
                    reasoning_content=collected_reasoning,
                    response=response,
                    completion=completion,
                    function_calls=[],
                    agent_id=agent.id,
                    execution_history=[],
                    reinvoked=reinvoked_runtime,
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
                _runtime_loop_detector,
                _runtime_turn_state,
            )
        else:
            collected_content = []
            collected_reasoning = ""
            function_calls = []
            execution_history = []
            completion = None
            async for chunk in self._handle_streaming_with_functions(
                response,
                agent,
                context_variables,
                prompt_messages,
                reinvoke_after_function,
                reinvoked_runtime,
                use_instructed_prompt,
                streamer_buffer,
                _runtime_loop_detector,
                _runtime_turn_state,
            ):
                if hasattr(chunk, "content") and chunk.content:
                    collected_content.append(chunk.content)
                if hasattr(chunk, "buffered_reasoning_content") and chunk.buffered_reasoning_content:
                    collected_reasoning = chunk.buffered_reasoning_content
                if hasattr(chunk, "reasoning_content") and chunk.reasoning_content and not collected_reasoning:
                    collected_reasoning = chunk.reasoning_content
                if hasattr(chunk, "function_calls"):
                    function_calls = chunk.function_calls
                if hasattr(chunk, "result"):
                    execution_history.append(chunk)
                if isinstance(chunk, Completion):
                    completion = chunk

            final_content = "".join(collected_content)
            return ResponseResult(
                content=final_content,
                reasoning_content=collected_reasoning,
                response=response,
                completion=completion,
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
        runtime_loop_detector: LoopDetector | None = None,
        runtime_turn_state: _RuntimeTurnState | None = None,
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
        buffered_reasoning_content = ""
        function_calls_detected = False
        function_calls = []
        tool_id_by_index: dict[int, str] = {}

        try:
            if hasattr(response, "__aiter__"):
                stream_generator = self.llm_client.astream_completion(response, agent)
                async for chunk_data in stream_generator:
                    content = chunk_data.get("content")
                    buffered_content = chunk_data.get("buffered_content", buffered_content)
                    buffered_reasoning_content = chunk_data.get("buffered_reasoning_content", buffered_reasoning_content)

                    streaming_tool_calls_data = chunk_data.get("streaming_tool_calls")
                    tool_call_chunks = []

                    if streaming_tool_calls_data:
                        for tool_idx, tool_delta in streaming_tool_calls_data.items():
                            if tool_delta:
                                if tool_delta.get("id"):
                                    tool_id_by_index[tool_idx] = tool_delta["id"]
                                tool_id = tool_id_by_index.get(tool_idx, f"tool_{tool_idx}")

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
                        reasoning_content=chunk_data.get("reasoning_content"),
                        buffered_reasoning_content=buffered_reasoning_content or None,
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
                        function_calls_detected = bool(function_calls) or self._detect_function_calls(
                            buffered_content, agent
                        )
            else:
                stream_generator = self.llm_client.stream_completion(response, agent)
                for chunk_data in stream_generator:
                    content = chunk_data.get("content")
                    buffered_content = chunk_data.get("buffered_content", buffered_content)
                    buffered_reasoning_content = chunk_data.get("buffered_reasoning_content", buffered_reasoning_content)

                    streaming_tool_calls_data = chunk_data.get("streaming_tool_calls")
                    tool_call_chunks = []

                    if streaming_tool_calls_data:
                        for tool_idx, tool_delta in (
                            streaming_tool_calls_data.items()
                            if isinstance(streaming_tool_calls_data, dict)
                            else enumerate(streaming_tool_calls_data or [])
                        ):
                            if tool_delta:
                                idx = tool_idx if isinstance(tool_idx, int) else 0
                                if tool_delta.get("id"):
                                    tool_id_by_index[idx] = tool_delta["id"]
                                tool_id = tool_id_by_index.get(idx, f"tool_{idx}")

                                tool_call_chunks.append(
                                    ToolCallStreamChunk(
                                        id=tool_id,
                                        type="function",
                                        function_name=tool_delta.get("name"),
                                        arguments=tool_delta.get("arguments"),
                                        index=idx,
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
                        reasoning_content=chunk_data.get("reasoning_content"),
                        buffered_reasoning_content=buffered_reasoning_content or None,
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
                        function_calls_detected = bool(function_calls) or self._detect_function_calls(
                            buffered_content, agent
                        )

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

                        enhanced_context = context.copy()
                        if self.enable_memory:
                            enhanced_context["memory_store"] = self.memory_store
                        enhanced_context["agent_id"] = agent.id or "default"

                        result = await self.executor._execute_single_call(
                            call,
                            enhanced_context,
                            agent,
                            runtime_features_state=self._runtime_features_state,
                            loop_detector=runtime_loop_detector,
                            audit_turn_id=runtime_turn_state.turn_id if runtime_turn_state is not None else None,
                        )
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

                    for r in results:
                        if isinstance(r.result, Result) and r.result.agent is not None:
                            handoff = r.result.agent
                            if handoff.id and handoff.id != agent.id:
                                if handoff.id not in self.orchestrator.agents:
                                    self.orchestrator.register_agent(handoff)
                                self.orchestrator.switch_agent(handoff.id, f"Tool handoff from {agent.id}")
                                agent = handoff
                                break

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
                    self._append_turn_tool_results(runtime_turn_state, results)

                    target_agent = self.orchestrator.should_switch_agent(switch_context.__dict__)
                    if target_agent:
                        old_agent = agent.id
                        self.orchestrator.switch_agent(target_agent, "Post-execution switch")

                        out = AgentSwitch(
                            from_agent=old_agent or "default",
                            to_agent=target_agent,
                            reason="Post-execution switch",
                        )
                        if (
                            self._runtime_features_state is not None
                            and self._runtime_features_state.session_manager is not None
                            and self._session_id is not None
                        ):
                            from datetime import datetime

                            from .session import AgentTransitionRecord

                            self._runtime_features_state.session_manager.record_agent_transition(
                                self._session_id,
                                AgentTransitionRecord(
                                    from_agent=old_agent or "default",
                                    to_agent=target_agent,
                                    reason="Post-execution switch",
                                    turn_id=runtime_turn_state.turn_id if runtime_turn_state is not None else "",
                                    timestamp=datetime.now(UTC).isoformat(),
                                ),
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
                            agent_id=agent.id or "default",
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
                            _runtime_loop_detector=runtime_loop_detector,
                            _runtime_turn_state=runtime_turn_state,
                        )

                        if not isinstance(reinvoke_response, ResponseResult):
                            async for chunk in reinvoke_response:
                                if streamer_buffer is not None and chunk is not None:
                                    streamer_buffer.put(chunk)
                                yield chunk
                        return

            self._finalize_runtime_turn(agent.id or "default", buffered_content, runtime_turn_state)
            out = Completion(
                final_content=buffered_content,
                reasoning_content=buffered_reasoning_content,
                function_calls_executed=len(function_calls),
                agent_id=agent.id or "default",
                execution_history=self.orchestrator.execution_history[-3:],
            )

            if streamer_buffer is not None:
                streamer_buffer.put(out)
            yield out
        except Exception as e:
            self._record_runtime_error(
                agent.id or "default",
                e,
                context="handle_streaming_with_functions",
                turn_state=runtime_turn_state,
            )
            raise

    async def _handle_streaming(
        self,
        response: tp.Any,
        reinvoked_runtime,
        agent: Agent,
        streamer_buffer: StreamerBuffer | None = None,
        runtime_turn_state: _RuntimeTurnState | None = None,
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
        buffered_reasoning_content = ""

        try:
            if hasattr(response, "__aiter__"):
                stream_generator = self.llm_client.astream_completion(response, agent)
                async for chunk_data in stream_generator:
                    content = chunk_data.get("content")
                    buffered_content = chunk_data.get("buffered_content", buffered_content)
                    buffered_reasoning_content = chunk_data.get("buffered_reasoning_content", buffered_reasoning_content)

                    out = StreamChunk(
                        chunk=chunk_data.get("raw_chunk"),
                        agent_id=agent.id or "default",
                        content=content,
                        buffered_content=buffered_content,
                        reasoning_content=chunk_data.get("reasoning_content"),
                        buffered_reasoning_content=buffered_reasoning_content or None,
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
                    buffered_reasoning_content = chunk_data.get("buffered_reasoning_content", buffered_reasoning_content)

                    out = StreamChunk(
                        chunk=chunk_data.get("raw_chunk"),
                        agent_id=agent.id or "default",
                        content=content,
                        buffered_content=buffered_content,
                        reasoning_content=chunk_data.get("reasoning_content"),
                        buffered_reasoning_content=buffered_reasoning_content or None,
                        function_calls_detected=False,
                        reinvoked=reinvoked_runtime,
                    )
                    if streamer_buffer is not None:
                        streamer_buffer.put(out)
                    yield out

            self._finalize_runtime_turn(agent.id or "default", buffered_content, runtime_turn_state)
            out = Completion(
                final_content=buffered_content,
                reasoning_content=buffered_reasoning_content,
                function_calls_executed=0,
                agent_id=agent.id or "default",
                execution_history=self.orchestrator.execution_history[-3:],
            )

            if streamer_buffer is not None:
                streamer_buffer.put(out)

            yield out
        except Exception as e:
            self._record_runtime_error(
                agent.id or "default",
                e,
                context="handle_streaming",
                turn_state=runtime_turn_state,
            )
            raise

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
            >>> for chunk in xerxes.run(prompt="Hello", stream=True):
            ...     if chunk.content:
            ...         print(chunk.content, end="")
            >>>
            >>>
            >>> result = xerxes.run(prompt="Hello", stream=False)
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

        def run_async() -> None:
            """Run the async response pipeline on a dedicated event loop in a background thread."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def async_runner() -> None:
                    """Await create_response and forward each chunk to the output queue."""
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
        """Run Xerxes in a background thread with automatic buffer creation.

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
            >>> buffer, thread = xerxes.thread_run(prompt="Hello")
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

        def run_in_thread() -> None:
            """Execute the synchronous run loop and store the result or exception."""
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
            >>> buffer, task = await xerxes.athread_run(prompt="Hello")
            >>> async for chunk in buffer.astream():
            ...     print(chunk.content, end="")
            >>> await task
        """

        buffer_was_none = streamer_buffer is None
        if streamer_buffer is None:
            streamer_buffer = StreamerBuffer()

        result_holder = [None]
        exception_holder = [None]

        async def run_async() -> None:
            """Stream the response into the buffer and build the final ResponseResult."""
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
                    completion=final_response if isinstance(final_response, Completion) else None,
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


__all__ = ("PromptSection", "PromptTemplate", "Xerxes")
