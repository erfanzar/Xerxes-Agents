# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Integration helpers for MCP with Xerxes agents.

This module provides utilities for integrating Model Context Protocol (MCP)
tools with Xerxes agents, enabling seamless use of external MCP servers
and their tools within the Xerxes framework.

Key features:
- Convert MCP tools to Xerxes-compatible functions
- Add MCP tools to existing agents dynamically
- Create MCP-enabled agents with automatic tool integration
- Handle synchronous/asynchronous execution transparently

The integration preserves the MCP tool's input schema, allowing Xerxes's
function introspection to properly extract parameter signatures for
LLM function calling.
"""

from collections.abc import Callable
from typing import Any

from ..core.utils import run_sync
from ..logging.console import get_logger
from .manager import MCPManager
from .types import MCPTool

logger = get_logger()


def mcp_tool_to_xerxes_function(tool: MCPTool, manager: MCPManager) -> Callable:
    """Convert an MCP tool to a Xerxes-compatible function.

    Creates a synchronous wrapper function with explicit parameters based on
    the MCP tool's input schema, enabling Xerxes's function_to_json to properly
    extract the signature for LLM function calling.

    The generated function:
    - Has a dynamic signature matching the MCP tool's input schema
    - Includes proper type annotations for all parameters
    - Contains a docstring with parameter descriptions and server info
    - Handles async-to-sync conversion transparently

    Args:
        tool: The MCP tool to convert, containing name, description,
            and input schema.
        manager: The MCP manager instance responsible for executing
            the tool on its server.

    Returns:
        A callable function compatible with Xerxes agents that wraps
        the MCP tool execution.

    Raises:
        Exception: Re-raised from MCP tool execution failures after logging.
    """
    import inspect

    properties = tool.input_schema.get("properties", {}) if tool.input_schema else {}
    required_params = set(tool.input_schema.get("required", [])) if tool.input_schema else set()

    params = []
    annotations = {}
    param_docs = []

    for param_name, param_info in properties.items():
        param_type = _map_schema_type(param_info.get("type", "string"))
        annotations[param_name] = param_type

        if param_name in required_params:
            params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=param_type,
                )
            )
        else:
            params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=param_type,
                )
            )

        if "description" in param_info:
            param_docs.append(f"{param_name}: {param_info['description']}")

    def sync_wrapper(**kwargs) -> Any:
        """Synchronous wrapper for MCP tool execution.

        Executes the MCP tool call synchronously by wrapping the async
        manager.call_tool method with run_sync.

        Args:
            **kwargs: Arguments to pass to the MCP tool.

        Returns:
            The result from the MCP tool execution.

        Raises:
            Exception: Re-raised after logging if tool execution fails.
        """
        try:
            return run_sync(manager.call_tool(tool.name, kwargs))
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool.name}: {e}")
            raise

    func_name = tool.name.replace("-", "_").replace(".", "_")
    sync_wrapper.__name__ = func_name

    docstring_parts = [tool.description]
    if param_docs:
        docstring_parts.append("\nParameters:")
        docstring_parts.extend([f"    {doc}" for doc in param_docs])
    docstring_parts.append(f"\n\nMCP Server: {tool.server_name}")
    sync_wrapper.__doc__ = "\n".join(docstring_parts)

    sync_wrapper.__annotations__ = annotations

    sync_wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=dict)  # type: ignore

    return sync_wrapper


def _map_schema_type(json_type: str) -> type:
    """Map JSON schema type to Python type.

    Converts JSON Schema type strings to their corresponding Python type
    objects for use in function signatures and type annotations.

    Args:
        json_type: JSON Schema type string (e.g., "string", "number",
            "integer", "boolean", "array", "object").

    Returns:
        The corresponding Python type. Defaults to str for unknown types.
    """
    type_mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_mapping.get(json_type, str)


async def add_mcp_tools_to_agent(agent: Any, manager: MCPManager, server_names: list[str] | None = None) -> None:
    """Add MCP tools to a Xerxes agent.

    Dynamically extends an agent's function list with tools from MCP servers.
    Supports both direct Agent instances and CortexAgent wrappers that use
    an internal agent.

    The function converts each MCP tool to a Xerxes-compatible function and
    appends it to the agent's functions list.

    Args:
        agent: A CortexAgent or Xerxes Agent instance. Must have either
            a 'functions' attribute or an '_internal_agent' with 'functions'.
        manager: The MCP manager instance containing connected servers
            and their available tools.
        server_names: Optional list of server names to filter tools from.
            If None, adds tools from all connected servers.

    Returns:
        None. The agent is modified in-place.

    Note:
        If the agent does not support adding functions (missing expected
        attributes), a warning is logged and no tools are added.
    """
    logger = get_logger()

    all_tools = manager.get_all_tools()

    if server_names:
        tools = [t for t in all_tools if t.server_name in server_names]
    else:
        tools = all_tools

    functions = [mcp_tool_to_xerxes_function(tool, manager) for tool in tools]

    if hasattr(agent, "functions"):
        if agent.functions is None:
            agent.functions = []
        agent.functions.extend(functions)
        logger.info(
            f"Added {len(functions)} MCP tools to agent {getattr(agent, 'role', getattr(agent, 'name', 'unknown'))}"
        )
    elif hasattr(agent, "_internal_agent") and hasattr(agent._internal_agent, "functions"):
        if agent._internal_agent.functions is None:
            agent._internal_agent.functions = []
        agent._internal_agent.functions.extend(functions)
        logger.info(
            f"Added {len(functions)} MCP tools to agent {getattr(agent, 'role', getattr(agent, 'name', 'unknown'))}"
        )
    else:
        logger.warning("Agent does not support adding functions")


def create_mcp_enabled_agent(
    agent_class: type,
    manager: MCPManager,
    server_names: list[str] | None = None,
    **agent_kwargs,
) -> Any:
    """Create an agent with MCP tools automatically added.

    Factory function that instantiates an agent and automatically adds
    MCP tools from the specified servers. This provides a convenient
    one-step method for creating MCP-enabled agents.

    Args:
        agent_class: The agent class to instantiate. Should be either
            CortexAgent or Agent, or any compatible class.
        manager: The MCP manager instance containing connected servers
            and their available tools.
        server_names: Optional list of server names to filter tools from.
            If None, adds tools from all connected servers.
        **agent_kwargs: Additional keyword arguments passed directly to
            the agent class constructor.

    Returns:
        An instantiated agent with MCP tools added to its functions list.

    Example:
        >>> manager = MCPManager()
        >>> await manager.connect_server("my-server", server_config)
        >>> agent = create_mcp_enabled_agent(
        ...     Agent,
        ...     manager,
        ...     server_names=["my-server"],
        ...     name="MyAgent",
        ...     model="gpt-4",
        ... )
    """

    agent = agent_class(**agent_kwargs)

    run_sync(add_mcp_tools_to_agent(agent, manager, server_names))

    return agent
