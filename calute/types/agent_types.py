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


"""Agent type definitions for the Calute framework.

This module provides Pydantic-based data models for defining AI agents
and their associated types within the Calute framework. It includes:

- Agent: Core agent model with function calling, sampling parameters, and capabilities
- AgentBaseFn: Abstract base class for defining class-based agent functions
- AgentFunction: Type alias for callable agent functions
- Response: Container for agent response data and conversation history
- Result: Encapsulation of agent function return values

The agent types support features like:
- Function calling with configurable strategies and timeouts
- LLM sampling parameter configuration (temperature, top_p, etc.)
- Automatic context compaction for long conversations
- Agent switching triggers for multi-agent workflows
- MCP (Model Context Protocol) server integration
- Capability-based agent specialization

Typical usage example:
    >>> from calute.types.agent_types import Agent
    >>> agent = Agent(
    ...     name="Assistant",
    ...     model="gpt-4",
    ...     instructions="You are a helpful assistant.",
    ...     temperature=0.7,
    ...     functions=[my_tool_function],
    ... )
    >>> agent.add_capability(AgentCapability(name="code_execution"))
"""

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
    """Abstract base metaclass for class-based agent functions.

    AgentBaseFn provides a pattern for defining agent tools/functions as classes
    rather than plain functions. This enables more complex tool implementations
    with state, inheritance, and better organization while still being callable
    by the agent's function execution system.

    Classes using this metaclass must implement the static_call method, which
    serves as the entry point when the agent invokes the tool. The class name
    becomes the function name visible to the LLM.

    Example:
        >>> class MyTool(metaclass=AgentBaseFn):
        ...     @staticmethod
        ...     def static_call(query: str) -> str:
        ...         '''Search for information.'''
        ...         return f"Results for: {query}"
        >>>
        >>> agent = Agent(functions=[MyTool])
    """

    @staticmethod
    @abstractmethod
    def static_call(*args, **kwargs) -> tp.Any:
        """Execute the tool's main functionality.

        This abstract method must be implemented by all classes using AgentBaseFn
        as their metaclass. It defines the actual behavior when the agent calls
        this tool.

        Args:
            *args: Positional arguments passed from the agent.
            **kwargs: Keyword arguments passed from the agent.

        Returns:
            The result of the tool execution, typically a string, dict, or Agent.

        Raises:
            NotImplementedError: If not overridden in subclass.
        """


def _wrap_static_call(cls: type[AgentBaseFn]) -> tp.Callable:
    """Wrap a class-based function into a callable with the class name.

    Creates a proxy function that forwards calls to the class's static_call
    method while preserving the class name as the function name. This allows
    the LLM to see unique, descriptive tool names based on class names rather
    than generic 'static_call' identifiers.

    Args:
        cls: A class that uses AgentBaseFn as its metaclass and implements
            the static_call method.

    Returns:
        A callable that wraps the static_call method with the class name,
        preserving the original docstring and module information.

    Example:
        >>> class SearchTool(metaclass=AgentBaseFn):
        ...     @staticmethod
        ...     def static_call(query: str) -> str:
        ...         return f"Searching: {query}"
        >>> wrapped = _wrap_static_call(SearchTool)
        >>> wrapped.__name__
        'SearchTool'
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
    """Core agent model with function calling and switching capabilities.

    Agent is the primary Pydantic model for defining AI agents in the Calute
    framework. It encapsulates all configuration needed for an agent including
    its identity, instructions, available functions/tools, LLM sampling parameters,
    and advanced features like context compaction and agent switching.

    The Agent class supports both simple single-agent use cases and complex
    multi-agent workflows with dynamic agent switching based on triggers.

    Attributes:
        model: LLM model identifier (e.g., 'gpt-4', 'claude-3'). If None,
            uses the default model from the Calute instance.
        id: Unique identifier for the agent, used in multi-agent routing.
        name: Human-readable name for the agent.
        instructions: System prompt or callable returning system prompt text.
        rules: List of rules or callable returning rules the agent must follow.
        examples: List of example interactions for few-shot learning.
        functions: List of callable functions/tools available to the agent.
        capabilities: List of AgentCapability objects defining special abilities.
        function_call_strategy: Strategy for executing multiple function calls
            (SEQUENTIAL, PARALLEL, etc.).
        tool_choice: Specific tool(s) the agent should prefer using.
        parallel_tool_calls: Whether to allow parallel tool execution.
        function_timeout: Timeout in seconds for individual function calls.
        max_function_retries: Maximum retry attempts for failed function calls.
        top_p: Nucleus sampling parameter (0.0-1.0).
        max_tokens: Maximum tokens in generated responses.
        temperature: Sampling temperature controlling randomness (0.0-2.0).
        top_k: Top-k sampling parameter (0 disables).
        min_p: Minimum probability threshold for sampling.
        presence_penalty: Penalty for token presence (-2.0 to 2.0).
        frequency_penalty: Penalty for token frequency (-2.0 to 2.0).
        repetition_penalty: Multiplicative penalty for repetition.
        extra_body: Additional parameters passed to the LLM API.
        stop: Stop sequences that halt generation.
        auto_compact: Whether to automatically compact context when near limit.
        compact_threshold: Context usage ratio triggering compaction (0.0-1.0).
        compact_target: Target context usage ratio after compaction (0.0-1.0).
        max_context_tokens: Maximum context tokens before compaction.
        compaction_strategy: Strategy for compacting conversation history.
        preserve_system_prompt: Whether to preserve system prompt during compaction.
        preserve_recent_messages: Number of recent messages to preserve.
        switch_triggers: List of triggers for switching to other agents.
        fallback_agent_id: Agent ID to switch to when no triggers match.

    Example:
        >>> agent = Agent(
        ...     name="CodeAssistant",
        ...     model="gpt-4",
        ...     instructions="You are a helpful coding assistant.",
        ...     temperature=0.3,
        ...     functions=[search_docs, run_code],
        ...     max_tokens=4096,
        ... )
        >>> agent.set_sampling_params(top_p=0.9)
        >>> print(agent.get_available_functions())
        ['search_docs', 'run_code']
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

    def set_model(self, model_id: str) -> None:
        """Set the LLM model identifier for this agent.

        Updates the agent's model configuration to use the specified model.
        This is useful for dynamically switching models at runtime.

        Args:
            model_id: The model identifier string (e.g., 'gpt-4', 'claude-3-opus').
        """
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
        """Update LLM sampling parameters for text generation.

        Allows selective updating of sampling parameters. Only parameters
        that are explicitly provided (not None) will be updated; others
        retain their current values.

        Args:
            top_p: Nucleus sampling probability threshold (0.0-1.0).
                Higher values include more tokens in sampling.
            max_tokens: Maximum number of tokens to generate.
            temperature: Controls randomness in generation (0.0-2.0).
                Lower values make output more deterministic.
            top_k: Number of highest probability tokens to consider.
                0 disables top-k sampling.
            min_p: Minimum probability threshold for token consideration.
            presence_penalty: Penalty for tokens already present (-2.0 to 2.0).
                Positive values encourage new topics.
            frequency_penalty: Penalty based on token frequency (-2.0 to 2.0).
                Positive values reduce repetition.
            repetition_penalty: Multiplicative penalty for repeated tokens.
                Values > 1.0 reduce repetition.

        Example:
            >>> agent.set_sampling_params(
            ...     temperature=0.5,
            ...     top_p=0.9,
            ...     max_tokens=2048
            ... )
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
        """Validate and process the functions list during model initialization.

        This Pydantic field validator performs two key operations:
        1. Converts class-based functions (AgentBaseFn subclasses) into wrapped
           callables with proper naming for LLM visibility.
        2. Validates that all function names are unique to prevent conflicts.

        Args:
            v: The list of functions/tools provided to the agent.

        Returns:
            A processed list of callable functions ready for agent use.

        Raises:
            ValueError: If duplicate function names are detected in the list.
        """
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

    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent's capability list.

        Capabilities define special abilities or permissions for the agent,
        such as code execution, file access, or web browsing.

        Args:
            capability: An AgentCapability object defining the capability
                to add to this agent.

        Example:
            >>> agent.add_capability(AgentCapability(
            ...     name="code_execution",
            ...     description="Can execute Python code"
            ... ))
        """
        self.capabilities.append(capability)

    def has_capability(self, capability_name: str) -> bool:
        """Check if the agent has a specific capability by name.

        Useful for conditional logic based on agent capabilities in
        multi-agent systems or capability-gated operations.

        Args:
            capability_name: The name of the capability to check for.

        Returns:
            True if the agent has a capability with the specified name,
            False otherwise.

        Example:
            >>> if agent.has_capability("code_execution"):
            ...     result = execute_code(code_snippet)
        """
        return any(cap.name == capability_name for cap in self.capabilities)

    def get_available_functions(self) -> list[str]:
        """Get a list of all available function/tool names.

        Returns the names of all functions registered with this agent,
        useful for introspection and debugging.

        Returns:
            A list of function name strings.

        Example:
            >>> print(agent.get_available_functions())
            ['search_docs', 'execute_code', 'send_email']
        """
        return [func.__name__ for func in self.functions]

    def get_functions_mapping(self) -> dict[str, tp.Callable]:
        """Get a mapping of function names to their callable objects.

        Creates a dictionary for quick lookup of function objects by name,
        useful for dynamic function invocation and introspection.

        Returns:
            A dictionary mapping function names (str) to their callable
            implementations.

        Example:
            >>> mapping = agent.get_functions_mapping()
            >>> search_fn = mapping.get('search_docs')
            >>> result = search_fn(query="python tutorials")
        """
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
    """Container for agent response data and conversation state.

    Response encapsulates the result of an agent interaction, including the
    full conversation history, the responding agent reference, and any context
    variables that should persist across conversation turns.

    This model is typically returned by Calute's run methods and provides
    a complete snapshot of the interaction state for downstream processing.

    Attributes:
        messages: List of message dictionaries representing the conversation
            history. Each message typically contains 'role' (user/assistant/system)
            and 'content' keys, with optional 'tool_calls' for function invocations.
        agent: Optional reference to the Agent instance that generated this
            response, useful for multi-agent workflows and debugging.
        context_variables: Dictionary for storing arbitrary state that persists
            across conversation turns, such as extracted entities, user preferences,
            or accumulated data from tool calls.

    Example:
        >>> response = calute.run(agent, messages)
        >>> print(response.messages[-1]['content'])  # Last assistant message
        >>> updated_context = response.context_variables
    """

    messages: list = Field(default_factory=list)
    agent: Agent | None = None
    context_variables: dict = Field(default_factory=dict)


class Result(BaseModel):
    """Encapsulates return values from agent function/tool calls.

    Result provides a structured way for agent functions to return not just
    a value, but also control flow information (like switching agents) and
    context updates. This enables sophisticated multi-agent workflows where
    tools can influence agent behavior.

    Functions can return a Result to:
    - Provide a string value to be included in the conversation
    - Trigger an agent switch by specifying a new agent
    - Update context variables for subsequent processing

    Attributes:
        value: The string result to be included in the conversation as
            the function's output. Defaults to empty string.
        agent: Optional Agent instance to switch to after this function
            completes. Used for dynamic agent handoffs.
        context_variables: Dictionary of variables to merge into the
            conversation context, persisting across turns.

    Example:
        >>> def transfer_to_specialist(context: dict) -> Result:
        ...     specialist = Agent(name="Specialist", ...)
        ...     return Result(
        ...         value="Transferring you to our specialist.",
        ...         agent=specialist,
        ...         context_variables={"transferred": True}
        ...     )
    """

    value: str = ""
    agent: Agent | None = None
    context_variables: dict = Field(default_factory=dict)


__all__ = "Agent", "AgentFunction", "Result"
