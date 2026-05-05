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
"""Agent types module for Xerxes.

Exports:
    - AgentBaseFn
    - AgentFunction
    - Agent
    - Response
    - Result"""

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
    """Agent base fn.

    Inherits from: ABCMeta
    """

    @staticmethod
    @abstractmethod
    def static_call(*args, **kwargs) -> tp.Any:
        """Static call.

        Args:
            *args: IN: Additional positional arguments. OUT: Passed through to downstream calls.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            tp.Any: OUT: Result of the operation."""
        ...


_WRAPPED_MARKER = "__xerxes_wrapped_static_call__"


def _wrap_static_call(cls: type[AgentBaseFn]) -> tp.Callable:
    """Internal helper to wrap static call.

    Args:
        cls: IN: The class. OUT: Used for class-level operations.
    Returns:
        tp.Callable: OUT: Result of the operation."""

    if getattr(cls, _WRAPPED_MARKER, False):
        return cls

    static_fn = cls.static_call

    @functools.wraps(static_fn)
    def _proxy(*args, **kwargs):
        """Internal helper to proxy.

        Args:
            *args: IN: Additional positional arguments. OUT: Passed through to downstream calls.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""
        return static_fn(*args, **kwargs)

    _proxy.__name__ = cls.__name__
    _proxy.__qualname__ = f"{cls.__qualname__}.static_call"
    _proxy.__doc__ = static_fn.__doc__
    _proxy.__module__ = cls.__module__
    setattr(_proxy, _WRAPPED_MARKER, True)
    return _proxy


AgentFunction = tp.Callable[[], tp.Union[str, "Agent", dict]] | AgentBaseFn


class Agent(BaseModel):
    """Agent.

    Inherits from: BaseModel

    Attributes:
        model (str | None): model.
        id (str | None): id.
        name (str | None): name.
        instructions (str | tp.Callable[[], str] | None): instructions.
        rules (list[str] | tp.Callable[[], list[str]] | None): rules.
        examples (list[str] | None): examples.
        functions (list[tp.Callable | AgentBaseFn]): functions.
        capabilities (list[AgentCapability]): capabilities.
        function_call_strategy (FunctionCallStrategy): function call strategy.
        tool_choice (str | list[str] | None): tool choice.
        parallel_tool_calls (bool): parallel tool calls.
        function_timeout (float | None): function timeout.
        max_function_retries (int): max function retries.
        top_p (float): top p.
        max_tokens (int): max tokens.
        temperature (float): temperature.
        top_k (int): top k.
        min_p (float): min p.
        presence_penalty (float): presence penalty.
        frequency_penalty (float): frequency penalty.
        repetition_penalty (float): repetition penalty.
        extra_body (dict | None): extra body.
        stop (str | list[str] | None): stop.
        auto_compact (bool): auto compact.
        compact_threshold (float): compact threshold.
        compact_target (float): compact target.
        max_context_tokens (int | None): max context tokens.
        compaction_strategy (CompactionStrategy): compaction strategy.
        preserve_system_prompt (bool): preserve system prompt.
        preserve_recent_messages (int): preserve recent messages.
        switch_triggers (list[AgentSwitchTrigger]): switch triggers.
        fallback_agent_id (str | None): fallback agent id.
        model_config (Any): model config."""

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
        """Set the model.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            model_id (str): IN: model id. OUT: Consumed during execution."""

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
        """Set the sampling params.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            top_p (float | None, optional): IN: top p. Defaults to None. OUT: Consumed during execution.
            max_tokens (int | None, optional): IN: max tokens. Defaults to None. OUT: Consumed during execution.
            temperature (float | None, optional): IN: temperature. Defaults to None. OUT: Consumed during execution.
            top_k (int | None, optional): IN: top k. Defaults to None. OUT: Consumed during execution.
            min_p (float | None, optional): IN: min p. Defaults to None. OUT: Consumed during execution.
            presence_penalty (float | None, optional): IN: presence penalty. Defaults to None. OUT: Consumed during execution.
            frequency_penalty (float | None, optional): IN: frequency penalty. Defaults to None. OUT: Consumed during execution.
            repetition_penalty (float | None, optional): IN: repetition penalty. Defaults to None. OUT: Consumed during execution."""

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
        """Internal helper to resolve static calls.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            v (list): IN: v. OUT: Consumed during execution.
        Returns:
            list[tp.Callable]: OUT: Result of the operation."""

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
        """Add capability.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            capability (AgentCapability): IN: capability. OUT: Consumed during execution."""

        self.capabilities.append(capability)

    def has_capability(self, capability_name: str) -> bool:
        """Check whether capability.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            capability_name (str): IN: capability name. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        return any(cap.name == capability_name for cap in self.capabilities)

    def get_available_functions(self) -> list[str]:
        """Retrieve the available functions.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[str]: OUT: Result of the operation."""

        return [get_callable_public_name(func) for func in self.functions]

    def get_functions_mapping(self) -> dict[str, tp.Callable]:
        """Retrieve the functions mapping.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Callable]: OUT: Result of the operation."""

        return {get_callable_public_name(func): func for func in self.functions}

    def attach_mcp(
        self,
        mcp_servers: MCPManager | MCPServerConfig | list,
        server_names: list[str] | None = None,
    ) -> None:
        """Attach mcp.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            mcp_servers (MCPManager | MCPServerConfig | list): IN: mcp servers. OUT: Consumed during execution.
            server_names (list[str] | None, optional): IN: server names. Defaults to None. OUT: Consumed during execution."""

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
    """Response.

    Inherits from: BaseModel

    Attributes:
        messages (list): messages.
        agent (Agent | None): agent.
        context_variables (dict): context variables."""

    messages: list = Field(default_factory=list)
    agent: Agent | None = None
    context_variables: dict = Field(default_factory=dict)


class Result(BaseModel):
    """Result.

    Inherits from: BaseModel

    Attributes:
        value (str): value.
        agent (Agent | None): agent.
        context_variables (dict): context variables."""

    value: str = ""
    agent: Agent | None = None
    context_variables: dict = Field(default_factory=dict)


__all__ = "Agent", "AgentFunction", "Result"
