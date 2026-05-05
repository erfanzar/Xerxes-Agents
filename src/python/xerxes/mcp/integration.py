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
"""Integration module for Xerxes.

Exports:
    - logger
    - mcp_tool_to_xerxes_function
    - add_mcp_tools_to_agent
    - create_mcp_enabled_agent"""

from collections.abc import Callable
from typing import Any, cast

from ..core.utils import run_sync
from ..logging.console import get_logger
from .manager import MCPManager
from .types import MCPTool

logger = get_logger()


def mcp_tool_to_xerxes_function(tool: MCPTool, manager: MCPManager) -> Callable:
    """Mcp tool to xerxes function.

    Args:
        tool (MCPTool): IN: tool. OUT: Consumed during execution.
        manager (MCPManager): IN: manager. OUT: Consumed during execution.
    Returns:
        Callable: OUT: Result of the operation."""

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
        """Sync wrapper.

        Args:
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

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

    _wrapper = cast(Any, sync_wrapper)
    _wrapper.__annotations__ = annotations
    _wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=dict)

    return sync_wrapper


def _map_schema_type(json_type: str) -> type:
    """Internal helper to map schema type.

    Args:
        json_type (str): IN: json type. OUT: Consumed during execution.
    Returns:
        type: OUT: Result of the operation."""

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
    """Asynchronously Add mcp tools to agent.

    Args:
        agent (Any): IN: agent. OUT: Consumed during execution.
        manager (MCPManager): IN: manager. OUT: Consumed during execution.
        server_names (list[str] | None, optional): IN: server names. Defaults to None. OUT: Consumed during execution."""

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
    """Create mcp enabled agent.

    Args:
        agent_class (type): IN: agent class. OUT: Consumed during execution.
        manager (MCPManager): IN: manager. OUT: Consumed during execution.
        server_names (list[str] | None, optional): IN: server names. Defaults to None. OUT: Consumed during execution.
        **agent_kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
    Returns:
        Any: OUT: Result of the operation."""

    agent = agent_class(**agent_kwargs)

    run_sync(add_mcp_tools_to_agent(agent, manager, server_names))

    return agent
