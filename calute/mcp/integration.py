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


"""Integration helpers for MCP with Calute agents."""

from collections.abc import Callable
from typing import Any

from ..loggings import get_logger
from ..utils import run_sync
from .manager import MCPManager
from .types import MCPTool

logger = get_logger()


def mcp_tool_to_calute_function(tool: MCPTool, manager: MCPManager) -> Callable:
    """Convert an MCP tool to a Calute-compatible function.

    Creates a function with explicit parameters based on the MCP tool's input schema,
    so that Calute's function_to_json can properly extract the signature.

    Args:
        tool: MCP tool to convert
        manager: MCP manager instance for executing the tool

    Returns:
        Callable function compatible with Calute agents
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
        """Synchronous wrapper for MCP tool execution."""
        try:
            return run_sync(manager.call_tool(tool.name, kwargs))
        except Exception as e:
            logger.error(f"Error executing MCP tool {tool.name}: {e}")
            raise  # Re-raise instead of returning error dict

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

    Args:
        json_type: JSON schema type string

    Returns:
        Python type
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
    """Add MCP tools to a Calute agent.

    Args:
        agent: CortexAgent or Calute Agent instance
        manager: MCP manager instance
        server_names: Optional list of server names to filter tools from.
                     If None, adds tools from all servers.
    """
    logger = get_logger()

    all_tools = manager.get_all_tools()

    if server_names:
        tools = [t for t in all_tools if t.server_name in server_names]
    else:
        tools = all_tools

    functions = [mcp_tool_to_calute_function(tool, manager) for tool in tools]

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

    Args:
        agent_class: Agent class to instantiate (CortexAgent or Agent)
        manager: MCP manager instance
        server_names: Optional list of server names to get tools from
        **agent_kwargs: Additional arguments for agent initialization

    Returns:
        Agent instance with MCP tools added
    """

    agent = agent_class(**agent_kwargs)

    run_sync(add_mcp_tools_to_agent(agent, manager, server_names))

    return agent
