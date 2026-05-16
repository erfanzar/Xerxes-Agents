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
"""Pydantic dataclasses describing an Agent's configuration and outputs.

Defines the :class:`Agent` record (identity, instructions, tools, sampling,
compaction, switch triggers, MCP attachments), the :class:`AgentBaseFn`
metaclass for declaring tools as classes, plus :class:`Response` and
:class:`Result` envelopes for agent outputs.
"""

from __future__ import annotations

import functools
import typing as tp
from abc import ABCMeta, abstractmethod

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.utils import get_callable_public_name
from .function_execution_types import AgentCapability, AgentSwitchTrigger, CompactionStrategy, FunctionCallStrategy

if tp.TYPE_CHECKING:
    from xerxes.mcp.manager import MCPManager
    from xerxes.mcp.types import MCPServerConfig


class AgentBaseFn(ABCMeta):
    """Abstract base class for agent-defined static function calls.

    Subclass this to define a named tool that can be registered with an agent.
    The subclass must implement ``static_call``, which is the entry point invoked
    when the tool is selected by the LLM.

    Inherits from:
        ABCMeta: Enables ``@abstractmethod`` decorator enforcement.
    """

    @staticmethod
    @abstractmethod
    def static_call(*args, **kwargs) -> tp.Any:
        """Entry point invoked when this tool is selected by the LLM.

        Implement this method in a subclass to define the behavior of the tool.
        It is called with the arguments extracted from the LLM's function call request.

        Args:
            *args: Positional arguments provided by the LLM.
            **kwargs: Keyword arguments provided by the LLM.

        Returns:
            tp.Any: The result of the tool execution, which is passed back to the LLM.
        """
        ...


_WRAPPED_MARKER = "__xerxes_wrapped_static_call__"


def _wrap_static_call(cls: type[AgentBaseFn]) -> tp.Callable:
    """Wrap an AgentBaseFn subclass so it can be used as a plain callable.

    Transforms ``cls.static_call`` into a standalone callable (the ``__call__``
    proxy) while preserving the function's name, docstring, and module metadata.

    Args:
        cls: The ``AgentBaseFn`` subclass to wrap.

    Returns:
        tp.Callable: A callable that delegates to ``cls.static_call``.
    """

    if getattr(cls, _WRAPPED_MARKER, False):
        return cls

    static_fn = cls.static_call

    @functools.wraps(static_fn)
    def _proxy(*args, **kwargs):
        """Proxy callable that delegates to ``cls.static_call``."""
        return static_fn(*args, **kwargs)

    _proxy.__name__ = cls.__name__
    _proxy.__qualname__ = f"{cls.__qualname__}.static_call"
    _proxy.__doc__ = static_fn.__doc__
    _proxy.__module__ = cls.__module__
    setattr(_proxy, _WRAPPED_MARKER, True)
    return _proxy


AgentFunction = tp.Callable[[], tp.Union[str, "Agent", dict]] | AgentBaseFn


class Agent(BaseModel):
    """Configuration for an agent, including its identity, tools, and sampling parameters.

    An ``Agent`` wraps the system prompt, tools, capabilities, and LLM sampling
    settings needed to run an autonomous agent. It can be registered with a
    ``Xerxes`` instance and executed with a user prompt.

    Inherits from:
        BaseModel: Pydantic base model for serialization and validation.

    Attributes:
        model: Identifier for the LLM (e.g., ``"gpt-4"``).
        id: Unique identifier for the agent.
        name: Human-readable display name.
        instructions: System prompt text or a callable that returns one.
        rules: Behavioral rules or a callable returning rule strings.
        examples: Example conversations embedded in the prompt.
        functions: Callable tools or ``AgentBaseFn`` subclasses available to this agent.
        capabilities: Declared capabilities for routing and introspection.
        function_call_strategy: Strategy for selecting and ordering tool calls.
        tool_choice: Force a specific tool or set of tools.
        parallel_tool_calls: Whether to issue multiple tool calls concurrently.
        function_timeout: Seconds before a tool call is considered stalled.
        max_function_retries: Number of retries on tool failure.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum tokens in the LLM response.
        temperature: Sampling temperature controlling randomness.
        top_k: Top-k sampling cutoff.
        min_p: Minimum probability threshold for token sampling.
        presence_penalty: Penalty for repeating tokens already in the prompt.
        frequency_penalty: Penalty proportional to token frequency in prior output.
        repetition_penalty: General repetition penalty multiplier.
        extra_body: Additional provider-specific parameters.
        stop: Stop sequences that halt generation.
        auto_compact: Whether to auto-compact conversation history.
        compact_threshold: Token ratio that triggers compaction.
        compact_target: Target token ratio after compaction.
        max_context_tokens: Maximum context token budget for compaction.
        compaction_strategy: Strategy for compaction (summarize, truncate, etc.).
        preserve_system_prompt: Whether to keep the system prompt during compaction.
        preserve_recent_messages: Number of recent messages to preserve during compaction.
        switch_triggers: Conditions that cause this agent to hand off to another.
        fallback_agent_id: Agent to switch to when triggers fire.
        model_config: Pydantic model configuration.
    """

    model: str | None = None
    id: str | None = None
    name: str | None = None
    instructions: str | tp.Callable[[], str] | None = None
    rules: list[str] | tp.Callable[[], list[str]] | None = None
    examples: list[str] | None = None
    functions: list[tp.Callable | AgentBaseFn] = Field(default_factory=list)
    capabilities: list[AgentCapability] = Field(default_factory=list)

    function_call_strategy: FunctionCallStrategy = FunctionCallStrategy.SEQUENTIAL
    tool_choice: str | list[str] | None = None
    parallel_tool_calls: bool = True
    function_timeout: float | None = 30.0
    max_function_retries: int = 3

    top_p: float = 0.95
    max_tokens: int = 2048
    temperature: float = 0.7
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    extra_body: dict | None = None

    stop: str | list[str] | None = None

    auto_compact: bool = False
    compact_threshold: float = 0.8
    compact_target: float = 0.5
    max_context_tokens: int | None = None
    compaction_strategy: CompactionStrategy = CompactionStrategy.SUMMARIZE
    preserve_system_prompt: bool = True
    preserve_recent_messages: int = 5

    switch_triggers: list[AgentSwitchTrigger] = Field(default_factory=list)
    fallback_agent_id: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_model(self, model_id: str) -> None:
        """Set the model identifier for this agent.

        Args:
            model_id: The model identifier to assign (e.g., ``"gpt-4"``)."""

        self.model = model_id

    def set_sampling_params(
        self,
        top_p: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        """Set LLM sampling parameters for this agent.

        Only provided arguments are updated; omitted parameters retain their
        current values.

        Args:
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens in the LLM response.
            temperature: Sampling temperature.
            top_k: Top-k sampling cutoff.
            min_p: Minimum probability threshold.
            presence_penalty: Penalty for tokens already in the prompt.
            frequency_penalty: Penalty proportional to prior token frequency.
            repetition_penalty: General repetition penalty multiplier.
        """

        if top_p is not None:
            self.top_p = top_p
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
        if top_k is not None:
            self.top_k = top_k
        if min_p is not None:
            self.min_p = min_p
        if presence_penalty is not None:
            self.presence_penalty = presence_penalty
        if frequency_penalty is not None:
            self.frequency_penalty = frequency_penalty
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty

    @field_validator("functions")
    def _resolve_static_calls(cls, v: list) -> list[tp.Callable]:
        """Validate and normalize the functions list.

        Wraps any ``AgentBaseFn`` subclasses in a callable proxy and checks
        for duplicate function names.

        Args:
            v: The raw list of function items from the model field.

        Returns:
            A list of validated, deduplicated callables.

        Raises:
            ValueError: If a function is not callable or a duplicate name is detected.
        """

        processed: list[tp.Callable] = []
        seen_names: set[str] = set()

        for fn in v or []:
            if not callable(fn):
                raise ValueError(f"Agent.functions must contain callables, got {type(fn).__name__}")
            if isinstance(fn, type) and issubclass(fn, AgentBaseFn):
                fn = _wrap_static_call(fn)
            public_name = get_callable_public_name(fn)
            if public_name in seen_names:
                raise ValueError(f"Duplicate function name '{public_name}' detected in Agent.functions")
            seen_names.add(public_name)
            processed.append(fn)

        return processed

    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to this agent.

        Args:
            capability: The capability to register with this agent.
        """

        self.capabilities.append(capability)

    def has_capability(self, capability_name: str) -> bool:
        """Check whether this agent declares a given capability.

        Args:
            capability_name: The name of the capability to check.

        Returns:
            True if a matching capability is found, False otherwise.
        """

        return any(cap.name == capability_name for cap in self.capabilities)

    def get_available_functions(self) -> list[str]:
        """Return the public names of all available functions.

        Returns:
            A list of function names as strings.
        """

        return [get_callable_public_name(func) for func in self.functions]

    def get_functions_mapping(self) -> dict[str, tp.Callable]:
        """Return a mapping from function names to callables.

        Returns:
            A dictionary mapping each function's public name to the callable itself.
        """

        return {get_callable_public_name(func): func for func in self.functions}

    def attach_mcp(
        self,
        mcp_servers: MCPManager | MCPServerConfig | list,
        server_names: list[str] | None = None,
    ) -> None:
        """Attach MCP servers and expose their tools to this agent.

        Args:
            mcp_servers: An ``MCPManager``, a single ``MCPServerConfig``, or a list
                of configurations.
            server_names: Optional subset of server names to include. All servers
                are included if not specified.
        """

        from xerxes.core.utils import run_sync
        from xerxes.mcp import MCPManager, MCPServerConfig
        from xerxes.mcp.integration import add_mcp_tools_to_agent

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

        run_sync(add_mcp_tools_to_agent(self, manager, server_names))

        if not hasattr(self, "_mcp_managers"):
            self._mcp_managers = []
        self._mcp_managers.append(manager)


class Response(BaseModel):
    """The result of an agent execution, including messages and context.

    Inherits from:
        BaseModel: Pydantic base model for serialization and validation.

    Attributes:
        messages: The list of messages exchanged during the execution.
        agent: The agent that produced this response.
        context_variables: Arbitrary key-value context carried through execution.
    """

    messages: list = Field(default_factory=list)
    agent: Agent | None = None
    context_variables: dict = Field(default_factory=dict)


class Result(BaseModel):
    """A simple result wrapper with a value, agent reference, and context.

    Inherits from:
        BaseModel: Pydantic base model for serialization and validation.

    Attributes:
        value: The primary result value as a string.
        agent: The agent that produced this result.
        context_variables: Arbitrary key-value context carried through execution.
    """

    value: str = ""
    agent: Agent | None = None
    context_variables: dict = Field(default_factory=dict)


__all__ = "Agent", "AgentFunction", "Result"
