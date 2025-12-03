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


"""Defines Pydantic models for the vSurge API, mimicking OpenAI's structure."""

from __future__ import annotations

import functools
import typing as tp
from abc import ABCMeta, abstractmethod

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .function_execution_types import AgentCapability, AgentSwitchTrigger, CompactionStrategy, FunctionCallStrategy

if tp.TYPE_CHECKING:
    from calute.mcp.manager import MCPManager
    from calute.mcp.types import MCPServerConfig


class AgentBaseFn(ABCMeta):
    @staticmethod
    @abstractmethod
    def static_call(*args, **kwargs) -> tp.Any:
        """place-holder for static-call method"""


def _wrap_static_call(cls: type[AgentBaseFn]) -> tp.Callable:
    """
    Return a new function that forwards to `cls.static_call`
    but is *named* after the class (so the LLM sees a unique tool).
    """
    static_fn = cls.static_call

    @functools.wraps(static_fn)
    def _proxy(*args, **kwargs):
        return static_fn(*args, **kwargs)

    _proxy.__name__ = cls.__name__
    _proxy.__qualname__ = f"{cls.__qualname__}.static_call"
    _proxy.__doc__ = static_fn.__doc__
    _proxy.__module__ = cls.__module__
    return _proxy


AgentFunction = tp.Callable[[], tp.Union[str, "Agent", dict]] | AgentBaseFn  # type:ignore


class Agent(BaseModel):
    """Agent with function calling and switching capabilities"""

    model: str | None = None
    id: str | None = None
    name: str | None = None
    instructions: str | tp.Callable[[], str] | None = None
    rules: list[str] | tp.Callable[[], list[str]] | None = None
    examples: list[str] | None = None
    functions: list[tp.Callable | AgentBaseFn] = Field(default_factory=list)
    capabilities: list[AgentCapability] = Field(default_factory=list)

    function_call_strategy: FunctionCallStrategy = FunctionCallStrategy.SEQUENTIAL
    tool_choice: str | list[str] = None
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
    compaction_strategy: CompactionStrategy = CompactionStrategy.SMART
    preserve_system_prompt: bool = True
    preserve_recent_messages: int = 5

    switch_triggers: list[AgentSwitchTrigger] = Field(default_factory=list)
    fallback_agent_id: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_model(self, model_id: str):
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
    ):
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
    def _resolve_static_calls(cls, v):
        processed: list[tp.Callable] = []
        seen_names: set[str] = set()

        for fn in v or []:
            if isinstance(fn, type) and issubclass(fn, AgentBaseFn):
                fn = _wrap_static_call(fn)
            if fn.__name__ in seen_names:
                raise ValueError(f"Duplicate function name '{fn.__name__}' detected in Agent.functions")
            seen_names.add(fn.__name__)
            processed.append(fn)

        return processed

    def add_capability(self, capability: AgentCapability):
        """Add a capability to the agent"""
        self.capabilities.append(capability)

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        return any(cap.name == capability_name for cap in self.capabilities)

    def get_available_functions(self) -> list[str]:
        """Get list of available function names"""
        return [func.__name__ for func in self.functions]

    def get_functions_mapping(self) -> dict[str, tp.Callable]:
        """Get list of available function names"""
        return {func.__name__: func for func in self.functions}

    def attach_mcp(
        self,
        mcp_servers: MCPManager | MCPServerConfig | list,
        server_names: list[str] | None = None,
    ) -> None:
        """Attach MCP servers to this agent, connecting and adding their tools.

        This method provides a convenient way to connect MCP servers and automatically
        add their tools to the agent's function list.

        Args:
            mcp_servers: Can be one of:
                - MCPManager: An existing MCP manager instance
                - MCPServerConfig: A single server config (will create manager and connect)
                - list[MCPServerConfig]: Multiple server configs (will create manager and connect all)
            server_names: Optional list of server names to filter tools from.
                         If None, adds tools from all servers in the manager.

        Example:
            >>>
            >>> agent.attach_mcp(MCPServerConfig(
            ...     name="filesystem",
            ...     command="npx",
            ...     args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            ... ))
            >>>
            >>>
            >>> agent.attach_mcp([
            ...     MCPServerConfig(name="filesystem", ...),
            ...     MCPServerConfig(name="sqlite", ...)
            ... ])
            >>>
            >>>
            >>> manager = MCPManager()
            >>> await manager.add_server(config1)
            >>> await manager.add_server(config2)
            >>> agent.attach_mcp(manager, server_names=["filesystem"])
        """
        from calute.mcp import MCPManager, MCPServerConfig
        from calute.mcp.integration import add_mcp_tools_to_agent
        from calute.utils import run_sync

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
    """
    Represents a response from an agent or a snapshot of the interaction state,
    containing messages and contextual information.

        Attributes:
            messages (tp.List):
                A list of messages representing the conversation history up to this response.
                Each message is typically a dictionary or a dedicated message object
                (e.g., `{'role': 'user', 'content': '...'}`,
                `{'role': 'assistant', 'content': '...', 'tool_calls': [...]}`).
                Defaults to an empty list.
            agent (tp.Optional[Agent]):
                An optional reference to the `Agent` instance that generated this response
                or is associated with this state. This can be useful for understanding
                the configuration that led to the response. Defaults to None.
            context_variables (dict):
                A dictionary for storing arbitrary contextual variables related to this
                response or the ongoing conversation state. Useful for carrying state
                between turns, logging, or passing information to downstream processes.
                Defaults to an empty dictionary.
    """

    messages: list = Field(default_factory=list)
    agent: Agent | None = None
    context_variables: dict = Field(default_factory=dict)


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Agent | None = None
    context_variables: dict = Field(default_factory=dict)


__all__ = "Agent", "AgentFunction", "Result"
