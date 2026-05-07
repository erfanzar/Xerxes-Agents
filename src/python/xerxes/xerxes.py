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
"""Xerxes module for Xerxes.

Exports:
    - logger
    - add_depth
    - Xerxes"""

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
    """Add depth.

    Args:
        x (Any): IN: x. OUT: Consumed during execution.
        add_prefix (Any, optional): IN: add prefix. Defaults to False. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    return SEP + x.replace("\n", f"\n{SEP}") if add_prefix else x.replace("\n", f"\n{SEP}")


_TOOL_PARAMETER_TAG_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.-]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class _RuntimeTurnState:
    """Runtime turn state.

    Attributes:
        turn_id (str): turn id.
        prompt (str): prompt.
        started_at (str): started at.
        tool_calls (list[tp.Any]): tool calls.
        finalized (bool): finalized."""

    turn_id: str
    prompt: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    tool_calls: list[tp.Any] = field(default_factory=list)
    finalized: bool = False


class Xerxes:
    """Xerxes.

    Attributes:
        SEP (tp.ClassVar[str]): sep.
        REINVOKE_FOLLOWUP_INSTRUCTION (tp.ClassVar[str]): reinvoke followup instruction."""

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
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            llm (BaseLLM | None, optional): IN: llm. Defaults to None. OUT: Consumed during execution.
            template (PromptTemplate | None, optional): IN: template. Defaults to None. OUT: Consumed during execution.
            enable_memory (bool, optional): IN: enable memory. Defaults to False. OUT: Consumed during execution.
            memory_config (dict[str, tp.Any] | None, optional): IN: memory config. Defaults to None. OUT: Consumed during execution.
            auto_add_memory_tools (bool, optional): IN: auto add memory tools. Defaults to True. OUT: Consumed during execution.
            runtime_features (RuntimeFeaturesConfig | None, optional): IN: runtime features. Defaults to None. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

        self.llm_client: BaseLLM | None = llm

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
        """Internal helper to normalize runtime features.

        Args:
            runtime_features (RuntimeFeaturesConfig | None): IN: runtime features. OUT: Consumed during execution.
            workspace_root (str): IN: workspace root. OUT: Consumed during execution.
        Returns:
            RuntimeFeaturesConfig: OUT: Result of the operation."""

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
        """Internal helper to setup default triggers.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        def capability_based_switch(context, agents, current_agent_id):
            """Capability based switch.

            Args:
                context (Any): IN: context. OUT: Consumed during execution.
                agents (Any): IN: agents. OUT: Consumed during execution.
                current_agent_id (Any): IN: current agent id. OUT: Consumed during execution.
            Returns:
                Any: OUT: Result of the operation."""

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
            """Error recovery switch.

            Args:
                context (Any): IN: context. OUT: Consumed during execution.
                agents (Any): IN: agents. OUT: Consumed during execution.
                current_agent_id (Any): IN: current agent id. OUT: Consumed during execution.
            Returns:
                Any: OUT: Result of the operation."""

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
        """Create query engine.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            model (str, optional): IN: model. Defaults to ''. OUT: Consumed during execution.
            system_prompt (str, optional): IN: system prompt. Defaults to ''. OUT: Consumed during execution.
            **config_kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

        from .runtime.bridge import create_query_engine

        agent = self.orchestrator.get_current_agent() if self.orchestrator.current_agent_id else None
        if not model and agent:
            model = agent.model or ""
        if not system_prompt and agent:
            system_prompt = (agent.instructions() if callable(agent.instructions) else agent.instructions) or ""

        return create_query_engine(
            xerxes_instance=self,
            agent=agent,
            model=model,
            system_prompt=system_prompt,
            **config_kwargs,
        )

    def create_runtime_session(self, prompt: str = "") -> RuntimeSession:
        """Create runtime session.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str, optional): IN: prompt. Defaults to ''. OUT: Consumed during execution.
        Returns:
            RuntimeSession: OUT: Result of the operation."""

        from .runtime.session import RuntimeSession

        agent = self.orchestrator.get_current_agent() if self.orchestrator.current_agent_id else None
        model = (agent.model if agent else "") or ""
        return RuntimeSession.create(model=model, prompt=prompt)

    def bootstrap(self, extra_context: str = ""):
        """Bootstrap.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            extra_context (str, optional): IN: extra context. Defaults to ''. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

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
        """Retrieve the execution registry.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            Any: OUT: Result of the operation."""

        from .runtime.bridge import populate_registry

        return populate_registry()

    def get_tool_executor(self):
        """Retrieve the tool executor.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            Any: OUT: Result of the operation."""

        from .runtime.bridge import build_tool_executor

        agent = self.orchestrator.get_current_agent() if self.orchestrator.current_agent_id else None
        registry = self.get_execution_registry()
        return build_tool_executor(xerxes_instance=self, agent=agent, registry=registry)

    def create_subagent_manager(
        self,
        max_concurrent: int = 5,
        max_depth: int = 5,
    ):
        """Create subagent manager.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            max_concurrent (int, optional): IN: max concurrent. Defaults to 5. OUT: Consumed during execution.
            max_depth (int, optional): IN: max depth. Defaults to 5. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

        from .agents.subagent_manager import SubAgentManager
        from .runtime.bridge import build_tool_executor, populate_registry

        mgr = SubAgentManager(max_concurrent=max_concurrent, max_depth=max_depth)

        registry = populate_registry()
        tool_executor = build_tool_executor(xerxes_instance=self, registry=registry)

        def runner(prompt, config, system_prompt, depth, cancel_check):
            """Runner.

            Args:
                prompt (Any): IN: prompt. OUT: Consumed during execution.
                config (Any): IN: config. OUT: Consumed during execution.
                system_prompt (Any): IN: system prompt. OUT: Consumed during execution.
                depth (Any): IN: depth. OUT: Consumed during execution.
                cancel_check (Any): IN: cancel check. OUT: Consumed during execution.
            Returns:
                Any: OUT: Result of the operation."""
            from .agents.subagent_manager import _filter_subagent_tools
            from .streaming.events import AgentState, TextChunk
            from .streaming.loop import run

            state = AgentState()
            output_parts = []
            eff_tool_schemas, eff_tool_executor = _filter_subagent_tools(
                tool_schemas=registry.tool_schemas(),
                tool_executor=tool_executor,
                config=config,
                is_subagent=depth > 0,
            )

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
        """Internal helper to notify turn start.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
            messages (MessagesHistory | None, optional): IN: messages. Defaults to None. OUT: Consumed during execution."""

        runtime_state = self._runtime_features_state
        if runtime_state is None or not runtime_state.hook_runner.has_hooks("on_turn_start"):
            return
        runtime_state.hook_runner.run("on_turn_start", agent_id=agent_id, messages=messages)

    def _notify_turn_end(self, agent_id: str | None, response: str | None = None) -> None:
        """Internal helper to notify turn end.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
            response (str | None, optional): IN: response. Defaults to None. OUT: Consumed during execution."""

        runtime_state = self._runtime_features_state
        if runtime_state is None or not runtime_state.hook_runner.has_hooks("on_turn_end"):
            return
        runtime_state.hook_runner.run("on_turn_end", agent_id=agent_id, response=response)

    def _notify_runtime_error(self, agent_id: str | None, error: Exception) -> None:
        """Internal helper to notify runtime error.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
            error (Exception): IN: error. OUT: Consumed during execution."""

        runtime_state = self._runtime_features_state
        if runtime_state is None or not runtime_state.hook_runner.has_hooks("on_error"):
            return
        runtime_state.hook_runner.run("on_error", agent_id=agent_id, error=error)

    @staticmethod
    def _new_runtime_turn_id() -> str:
        """Internal helper to new runtime turn id.

        Returns:
            str: OUT: Result of the operation."""

        return uuid.uuid4().hex[:12]

    def _append_turn_tool_results(
        self,
        turn_state: _RuntimeTurnState | None,
        results: list[RequestFunctionCall],
    ) -> None:
        """Internal helper to append turn tool results.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            turn_state (_RuntimeTurnState | None): IN: turn state. OUT: Consumed during execution.
            results (list[RequestFunctionCall]): IN: results. OUT: Consumed during execution."""

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
        """Internal helper to finalize runtime turn.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
            response_content (str): IN: response content. OUT: Consumed during execution.
            turn_state (_RuntimeTurnState | None, optional): IN: turn state. Defaults to None. OUT: Consumed during execution."""

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
        """Internal helper to record runtime error.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str | None): IN: agent id. OUT: Consumed during execution.
            error (Exception): IN: error. OUT: Consumed during execution.
            context (str): IN: context. OUT: Consumed during execution.
            turn_state (_RuntimeTurnState | None, optional): IN: turn state. Defaults to None. OUT: Consumed during execution."""

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
        """Internal helper to is reinvoke followup message.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            message (ChatMessage): IN: message. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        if not isinstance(message, UserMessage) or not isinstance(message.content, str):
            return False
        return message.content.strip() == cls.REINVOKE_FOLLOWUP_INSTRUCTION

    @staticmethod
    def _is_operator_reinvoke_attachment(message: ChatMessage) -> bool:
        """Internal helper to is operator reinvoke attachment.

        Args:
            message (ChatMessage): IN: message. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        if not isinstance(message, UserMessage) or isinstance(message.content, str):
            return False
        if not message.content:
            return False
        first_chunk = message.content[0]
        return hasattr(first_chunk, "text") and str(first_chunk.text).startswith("[TOOL IMAGE RESULT]")

    @classmethod
    def _compact_reinvoke_history(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Internal helper to compact reinvoke history.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            messages (list[ChatMessage]): IN: messages. OUT: Consumed during execution.
        Returns:
            list[ChatMessage]: OUT: Result of the operation."""

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
        """Register agent.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent (Agent): IN: agent. OUT: Consumed during execution."""

        if self.enable_memory and self.auto_add_memory_tools:
            self._add_memory_tools_to_agent(agent)
        if self._runtime_features_state is not None:
            self._runtime_features_state.merge_plugin_tools(agent)
            self._runtime_features_state.merge_operator_tools(agent)
        self.orchestrator.register_agent(agent)

    def _add_memory_tools_to_agent(self, agent: Agent) -> None:
        """Internal helper to add memory tools to agent.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent (Agent): IN: agent. OUT: Consumed during execution."""

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
        """Internal helper to update memory from response.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            agent_id (str): IN: agent id. OUT: Consumed during execution.
            context_variables (dict | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
            function_calls (list[RequestFunctionCall] | None, optional): IN: function calls. Defaults to None. OUT: Consumed during execution."""

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
        """Internal helper to update memory from prompt.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str): IN: prompt. OUT: Consumed during execution.
            agent_id (str): IN: agent id. OUT: Consumed during execution."""

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
        """Internal helper to format section.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            header (str): IN: header. OUT: Consumed during execution.
            content (str | list[str] | None): IN: content. OUT: Consumed during execution.
            item_prefix (str | None, optional): IN: item prefix. Defaults to '- '. OUT: Consumed during execution.
        Returns:
            str | None: OUT: Result of the operation."""

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
        """Internal helper to extract from markdown.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            field (str): IN: field. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        pattern = rf"```{field}\s*\n(.*?)\n```"
        return re.findall(pattern, content, re.DOTALL)

    @staticmethod
    def _system_message_to_text(message: SystemMessage) -> str | None:
        """Internal helper to system message to text.

        Args:
            message (SystemMessage): IN: message. OUT: Consumed during execution.
        Returns:
            str | None: OUT: Result of the operation."""

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
        """Internal helper to merge system history.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            final_system_content (str): IN: final system content. OUT: Consumed during execution.
            messages (MessagesHistory | None): IN: messages. OUT: Consumed during execution.
        Returns:
            tuple[str, list[ChatMessage]]: OUT: Result of the operation."""

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
        """Manage messages.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent (Agent | None): IN: agent. OUT: Consumed during execution.
            prompt (str | None, optional): IN: prompt. Defaults to None. OUT: Consumed during execution.
            context_variables (dict | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
            messages (MessagesHistory | None, optional): IN: messages. Defaults to None. OUT: Consumed during execution.
            include_memory (bool, optional): IN: include memory. Defaults to True. OUT: Consumed during execution.
            use_instructed_prompt (bool, optional): IN: use instructed prompt. Defaults to False. OUT: Consumed during execution.
            use_chain_of_thought (bool, optional): IN: use chain of thought. Defaults to False. OUT: Consumed during execution.
            require_reflection (bool, optional): IN: require reflection. Defaults to False. OUT: Consumed during execution.
        Returns:
            MessagesHistory: OUT: Result of the operation."""

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
        """Internal helper to build reinvoke messages.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            original_messages (MessagesHistory): IN: original messages. OUT: Consumed during execution.
            assistant_content (str): IN: assistant content. OUT: Consumed during execution.
            function_calls (list[RequestFunctionCall]): IN: function calls. OUT: Consumed during execution.
            results (list[RequestFunctionCall]): IN: results. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
        Returns:
            MessagesHistory: OUT: Result of the operation."""

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
        """Extract md block.

        Args:
            input_string (str): IN: input string. OUT: Consumed during execution.
        Returns:
            list[tuple[str, str]]: OUT: Result of the operation."""

        pattern = r"```(\w*)\n(.*?)\n```"
        matches = re.findall(pattern, input_string, re.DOTALL)
        return [(lang, content.strip()) for lang, content in matches]

    def _remove_function_calls_from_content(self, content: str) -> str:
        """Internal helper to remove function calls from content.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        pattern = r"<(\w+)>\s*<arguments>.*?</arguments>\s*</\w+>"
        cleaned = re.sub(pattern, "", content, flags=re.DOTALL)
        pattern = r"<function=[A-Za-z0-9_.:-]+>\s*.*?</function>"
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        pattern = r"```tool_call.*?```"
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

        return cleaned.strip()

    def _extract_function_calls_from_xml(self, content: str, agent: Agent) -> list[RequestFunctionCall]:
        """Internal helper to extract function calls from xml.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            agent (Agent): IN: agent. OUT: Consumed during execution.
        Returns:
            list[RequestFunctionCall]: OUT: Result of the operation."""

        function_calls: list[RequestFunctionCall] = []
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
        """Internal helper to extract function calls from tagged markup.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            agent (Agent): IN: agent. OUT: Consumed during execution.
        Returns:
            list[RequestFunctionCall]: OUT: Result of the operation."""

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
        """Internal helper to convert function calls.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            function_calls_data (list[dict[str, tp.Any]]): IN: function calls data. OUT: Consumed during execution.
            agent (Agent): IN: agent. OUT: Consumed during execution.
        Returns:
            list[RequestFunctionCall]: OUT: Result of the operation."""

        function_calls: list[RequestFunctionCall] = []
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
        """Internal helper to extract function calls.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            agent (Agent): IN: agent. OUT: Consumed during execution.
            tool_calls (None | list[tp.Any], optional): IN: tool calls. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[RequestFunctionCall]: OUT: Result of the operation."""

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
    def extract_from_markdown(fmt: str, string: str) -> str | None | dict:
        """Extract from markdown.

        Args:
            fmt (str): IN: format. OUT: Consumed during execution.
            string (str): IN: string. OUT: Consumed during execution.
        Returns:
            str | None | dict: OUT: Result of the operation."""

        pattern = rf"```{re.escape(fmt)}\s*\n(.*?)\n```"
        m = re.search(pattern, string, re.DOTALL)
        if not m:
            return None
        block = m.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return block

    def _detect_function_calls(self, content: str, agent: Agent) -> bool:
        """Internal helper to detect function calls.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            agent (Agent): IN: agent. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Internal helper to detect function calls regex.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            content (str): IN: content. OUT: Consumed during execution.
            agent (Agent): IN: agent. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Retrieve the thoughts.

        Args:
            response (str): IN: response. OUT: Consumed during execution.
            tag (str, optional): IN: tag. Defaults to 'think'. OUT: Consumed during execution.
        Returns:
            str | None: OUT: Result of the operation."""

        inside = None
        match = re.search(rf"<{tag}>(.*?)</{tag}>", response, flags=re.S)
        if match:
            inside = match.group(1).strip()
        return inside

    @staticmethod
    def filter_thoughts(response: str, tag: str = "think") -> str:
        """Filter thoughts.

        Args:
            response (str): IN: response. OUT: Consumed during execution.
            tag (str, optional): IN: tag. Defaults to 'think'. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        filtered = re.sub(rf"<{tag}>.*?</{tag}>", "", response, flags=re.S)
        return filtered.strip()

    def format_function_parameters(self, parameters: dict) -> str:
        """Format function parameters.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            parameters (dict): IN: parameters. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Generate function section.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            functions (list[AgentFunction]): IN: functions. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Internal helper to format function doc.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            schema (dict): IN: schema. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Internal helper to build tool prompt label.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            func (AgentFunction): IN: func. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Format context variables.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            variables (dict[str, tp.Any]): IN: variables. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Format prompt.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str | None): IN: prompt. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        if not prompt:
            return ""
        return prompt

    def format_chat_history(self, messages: MessagesHistory) -> str:
        """Format chat history.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            messages (MessagesHistory): IN: messages. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Asynchronously Create response.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str | None, optional): IN: prompt. Defaults to None. OUT: Consumed during execution.
            context_variables (dict | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
            messages (MessagesHistory | None, optional): IN: messages. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None | Agent, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            stream (bool, optional): IN: stream. Defaults to True. OUT: Consumed during execution.
            apply_functions (bool, optional): IN: apply functions. Defaults to True. OUT: Consumed during execution.
            print_formatted_prompt (bool, optional): IN: print formatted prompt. Defaults to False. OUT: Consumed during execution.
            use_instructed_prompt (bool, optional): IN: use instructed prompt. Defaults to False. OUT: Consumed during execution.
            conversation_name_holder (str, optional): IN: conversation name holder. Defaults to 'Messages'. OUT: Consumed during execution.
            mention_last_turn (bool, optional): IN: mention last turn. Defaults to True. OUT: Consumed during execution.
            reinvoke_after_function (bool, optional): IN: reinvoke after function. Defaults to True. OUT: Consumed during execution.
            reinvoked_runtime (bool, optional): IN: reinvoked runtime. Defaults to False. OUT: Consumed during execution.
            streamer_buffer (StreamerBuffer | None, optional): IN: streamer buffer. Defaults to None. OUT: Consumed during execution.
            _runtime_loop_detector (LoopDetector | None, optional): IN: runtime loop detector. Defaults to None. OUT: Consumed during execution.
            _runtime_turn_state (_RuntimeTurnState | None, optional): IN: runtime turn state. Defaults to None. OUT: Consumed during execution.
        Returns:
            ResponseResult | AsyncIterator[StreamingResponseType]: OUT: Result of the operation."""

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

            prompt_str: str | list[dict[str, str]]
            if use_instructed_prompt:
                prompt_str = prompt_messages.make_instruction_prompt(
                    conversation_name_holder=conversation_name_holder,
                    mention_last_turn=mention_last_turn,
                )
            else:
                prompt_str = tp.cast(list[dict[str, str]], prompt_messages.to_openai()["messages"])

            if print_formatted_prompt:
                if use_instructed_prompt:
                    print(prompt_str)
                else:
                    pprint.pprint(prompt_messages.to_openai())
            with open("debug_prompt.json", "a") as f:
                json.dump({"key": prompt_str}, f, indent=2)
                f.write("\n")
            assert self.llm_client is not None
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
                    agent_id=agent.id or "default",
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
            function_calls: list[RequestFunctionCall] = []
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
                    function_calls = tp.cast(list[RequestFunctionCall], chunk.function_calls)
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
        """Asynchronously Internal helper to handle streaming with functions.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (tp.Any): IN: response. OUT: Consumed during execution.
            agent (Agent): IN: agent. OUT: Consumed during execution.
            context (dict): IN: context. OUT: Consumed during execution.
            prompt_messages (MessagesHistory): IN: prompt messages. OUT: Consumed during execution.
            reinvoke_after_function (bool): IN: reinvoke after function. OUT: Consumed during execution.
            reinvoked_runtime (bool): IN: reinvoked runtime. OUT: Consumed during execution.
            use_instructed_prompt (bool): IN: use instructed prompt. OUT: Consumed during execution.
            streamer_buffer (StreamerBuffer | None): IN: streamer buffer. OUT: Consumed during execution.
            runtime_loop_detector (LoopDetector | None, optional): IN: runtime loop detector. Defaults to None. OUT: Consumed during execution.
            runtime_turn_state (_RuntimeTurnState | None, optional): IN: runtime turn state. Defaults to None. OUT: Consumed during execution.
        Returns:
            AsyncIterator[StreamingResponseType]: OUT: Result of the operation."""

        buffered_content = ""
        buffered_reasoning_content = ""
        function_calls_detected = False
        function_calls = []
        tool_id_by_index: dict[int, str] = {}
        out: StreamingResponseType

        assert self.llm_client is not None
        try:
            if hasattr(response, "__aiter__"):
                async_stream = self.llm_client.astream_completion(response, agent)
                async for chunk_data in async_stream:
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
                sync_stream = self.llm_client.stream_completion(response, agent)
                for chunk_data in sync_stream:
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
        """Asynchronously Internal helper to handle streaming.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            response (tp.Any): IN: response. OUT: Consumed during execution.
            reinvoked_runtime (Any): IN: reinvoked runtime. OUT: Consumed during execution.
            agent (Agent): IN: agent. OUT: Consumed during execution.
            streamer_buffer (StreamerBuffer | None, optional): IN: streamer buffer. Defaults to None. OUT: Consumed during execution.
            runtime_turn_state (_RuntimeTurnState | None, optional): IN: runtime turn state. Defaults to None. OUT: Consumed during execution.
        Returns:
            AsyncIterator[StreamingResponseType]: OUT: Result of the operation."""

        buffered_content = ""
        buffered_reasoning_content = ""
        out: StreamingResponseType

        assert self.llm_client is not None
        try:
            if hasattr(response, "__aiter__"):
                async_stream = self.llm_client.astream_completion(response, agent)
                async for chunk_data in async_stream:
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
                sync_stream = self.llm_client.stream_completion(response, agent)
                for chunk_data in sync_stream:
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
        """Run.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str | None, optional): IN: prompt. Defaults to None. OUT: Consumed during execution.
            context_variables (dict | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
            messages (MessagesHistory | None, optional): IN: messages. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None | Agent, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            stream (bool, optional): IN: stream. Defaults to True. OUT: Consumed during execution.
            apply_functions (bool, optional): IN: apply functions. Defaults to True. OUT: Consumed during execution.
            print_formatted_prompt (bool, optional): IN: print formatted prompt. Defaults to False. OUT: Consumed during execution.
            use_instructed_prompt (bool, optional): IN: use instructed prompt. Defaults to False. OUT: Consumed during execution.
            conversation_name_holder (str, optional): IN: conversation name holder. Defaults to 'Messages'. OUT: Consumed during execution.
            mention_last_turn (bool, optional): IN: mention last turn. Defaults to True. OUT: Consumed during execution.
            reinvoke_after_function (bool, optional): IN: reinvoke after function. Defaults to True. OUT: Consumed during execution.
            reinvoked_runtime (bool, optional): IN: reinvoked runtime. Defaults to False. OUT: Consumed during execution.
            streamer_buffer (StreamerBuffer | None, optional): IN: streamer buffer. Defaults to None. OUT: Consumed during execution.
        Returns:
            ResponseResult | Generator[StreamingResponseType, None, None]: OUT: Result of the operation."""

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
            function_calls: list[RequestFunctionCall] = []
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
                    function_calls = tp.cast(list[RequestFunctionCall], chunk.function_calls)
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
        """Internal helper to run stream.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str | None, optional): IN: prompt. Defaults to None. OUT: Consumed during execution.
            context_variables (dict | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
            messages (MessagesHistory | None, optional): IN: messages. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None | Agent, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            apply_functions (bool, optional): IN: apply functions. Defaults to True. OUT: Consumed during execution.
            print_formatted_prompt (bool, optional): IN: print formatted prompt. Defaults to False. OUT: Consumed during execution.
            use_instructed_prompt (bool, optional): IN: use instructed prompt. Defaults to False. OUT: Consumed during execution.
            conversation_name_holder (str, optional): IN: conversation name holder. Defaults to 'Messages'. OUT: Consumed during execution.
            mention_last_turn (bool, optional): IN: mention last turn. Defaults to True. OUT: Consumed during execution.
            reinvoke_after_function (bool, optional): IN: reinvoke after function. Defaults to True. OUT: Consumed during execution.
            reinvoked_runtime (bool, optional): IN: reinvoked runtime. Defaults to False. OUT: Consumed during execution.
            streamer_buffer (StreamerBuffer | None, optional): IN: streamer buffer. Defaults to None. OUT: Consumed during execution.
        Returns:
            Generator[StreamingResponseType, None, None]: OUT: Result of the operation."""

        output_queue: queue.Queue[StreamingResponseType | None] = queue.Queue()
        exception_holder: list[Exception | None] = [None]

        def run_async() -> None:
            """Run async."""

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def async_runner() -> None:
                    """Asynchronously Async runner."""

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

                        assert not isinstance(response, ResponseResult)
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
        """Thread run.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str | None, optional): IN: prompt. Defaults to None. OUT: Consumed during execution.
            context_variables (dict | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
            messages (MessagesHistory | None, optional): IN: messages. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None | Agent, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            apply_functions (bool, optional): IN: apply functions. Defaults to True. OUT: Consumed during execution.
            print_formatted_prompt (bool, optional): IN: print formatted prompt. Defaults to False. OUT: Consumed during execution.
            use_instructed_prompt (bool, optional): IN: use instructed prompt. Defaults to False. OUT: Consumed during execution.
            conversation_name_holder (str, optional): IN: conversation name holder. Defaults to 'Messages'. OUT: Consumed during execution.
            mention_last_turn (bool, optional): IN: mention last turn. Defaults to True. OUT: Consumed during execution.
            reinvoke_after_function (bool, optional): IN: reinvoke after function. Defaults to True. OUT: Consumed during execution.
            reinvoked_runtime (bool, optional): IN: reinvoked runtime. Defaults to False. OUT: Consumed during execution.
            streamer_buffer (StreamerBuffer | None, optional): IN: streamer buffer. Defaults to None. OUT: Consumed during execution.
        Returns:
            tuple[StreamerBuffer, threading.Thread]: OUT: Result of the operation."""

        buffer_was_none = streamer_buffer is None
        if streamer_buffer is None:
            streamer_buffer = StreamerBuffer()

        result_holder: list[ResponseResult | None] = [None]
        exception_holder: list[Exception | None] = [None]

        def run_in_thread() -> None:
            """Run in thread."""

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
                assert isinstance(result, ResponseResult)
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
            """Retrieve the result.

            Args:
                timeout (float | None, optional): IN: timeout. Defaults to None. OUT: Consumed during execution.
            Returns:
                ResponseResult: OUT: Result of the operation."""

            thread.join(timeout=timeout)
            if exception_holder[0]:
                raise exception_holder[0]
            assert result_holder[0] is not None
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
        """Asynchronously Athread run.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str | None, optional): IN: prompt. Defaults to None. OUT: Consumed during execution.
            context_variables (dict | None, optional): IN: context variables. Defaults to None. OUT: Consumed during execution.
            messages (MessagesHistory | None, optional): IN: messages. Defaults to None. OUT: Consumed during execution.
            agent_id (str | None | Agent, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            apply_functions (bool, optional): IN: apply functions. Defaults to True. OUT: Consumed during execution.
            print_formatted_prompt (bool, optional): IN: print formatted prompt. Defaults to False. OUT: Consumed during execution.
            use_instructed_prompt (bool, optional): IN: use instructed prompt. Defaults to False. OUT: Consumed during execution.
            conversation_name_holder (str, optional): IN: conversation name holder. Defaults to 'Messages'. OUT: Consumed during execution.
            mention_last_turn (bool, optional): IN: mention last turn. Defaults to True. OUT: Consumed during execution.
            reinvoke_after_function (bool, optional): IN: reinvoke after function. Defaults to True. OUT: Consumed during execution.
            reinvoked_runtime (bool, optional): IN: reinvoked runtime. Defaults to False. OUT: Consumed during execution.
            streamer_buffer (StreamerBuffer | None, optional): IN: streamer buffer. Defaults to None. OUT: Consumed during execution.
        Returns:
            tuple[StreamerBuffer, asyncio.Task]: OUT: Result of the operation."""

        buffer_was_none = streamer_buffer is None
        if streamer_buffer is None:
            streamer_buffer = StreamerBuffer()

        result_holder: list[ResponseResult | None] = [None]
        exception_holder: list[Exception | None] = [None]

        async def run_async() -> None:
            """Asynchronously Run async."""

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
                assert not isinstance(stream, ResponseResult)
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
            """Asynchronously Aget result.

            Returns:
                ResponseResult: OUT: Result of the operation."""

            await task
            if exception_holder[0]:
                raise exception_holder[0]
            assert result_holder[0] is not None
            return result_holder[0]

        streamer_buffer.aget_result = aget_result

        return streamer_buffer, task


__all__ = ("PromptSection", "PromptTemplate", "Xerxes")
