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
"""MCP (Model Context Protocol) tool operations."""

from __future__ import annotations

from ...types import AgentBaseFn


class MCPTool(AgentBaseFn):
    """Interface for Model Context Protocol tools.

    Allows invoking tools from configured MCP servers.

    Example:
        >>> MCPTool.static_call(server_name="filesystem", tool_name="read_file")
    """

    @staticmethod
    def static_call(
        server_name: str,
        tool_name: str,
        arguments: str | dict | None = None,
        **context_variables,
    ) -> str:
        """Call an MCP tool.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to call.
            arguments: Tool arguments as dict or JSON string.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Tool result or informational message.
        """
        import importlib.util

        if importlib.util.find_spec("xerxes.mcp") is not None:
            return (
                f"[MCP] server={server_name} tool={tool_name}\n"
                "Use xerxes.mcp.MCPManager for async MCP tool invocation. "
                "This tool is a placeholder for the synchronous tool interface."
            )
        return "Error: xerxes.mcp module not available. Install xerxes[mcp]."


class ListMcpResourcesTool(AgentBaseFn):
    """List available resources from MCP servers.

    Example:
        >>> ListMcpResourcesTool.static_call(server_name="database")
    """

    @staticmethod
    def static_call(server_name: str = "", **context_variables) -> str:
        """List MCP resources.

        Args:
            server_name: Filter by server name, or empty for all servers.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            List of available resources.
        """
        return (
            f"[MCP Resources] server={server_name or '(all)'}\n"
            "Use xerxes.mcp.MCPManager.list_resources() for async MCP resource listing."
        )


class ReadMcpResourceTool(AgentBaseFn):
    """Read a specific resource from an MCP server.

    Example:
        >>> ReadMcpResourceTool.static_call(server_name="config", uri="file:///settings.json")
    """

    @staticmethod
    def static_call(
        server_name: str,
        uri: str,
        **context_variables,
    ) -> str:
        """Read an MCP resource.

        Args:
            server_name: MCP server name.
            uri: Resource URI to read.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Resource content or error.
        """
        return (
            f"[MCP Read] server={server_name} uri={uri}\n"
            "Use xerxes.mcp.MCPManager.read_resource() for async MCP resource reading."
        )
