# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""MCP Manager for managing multiple MCP server connections.

This module provides the MCPManager class for orchestrating connections
to multiple MCP servers, including:
- Connection lifecycle management (add, remove, disconnect)
- Unified access to tools, resources, and prompts across all servers
- Cross-server tool invocation and resource reading
- Capability discovery and summarization

The manager abstracts away the complexity of dealing with multiple
MCP servers, providing a single interface for tool and resource access.
"""

from typing import Any

from ..logging.console import get_logger
from .client import MCPClient
from .types import MCPPrompt, MCPResource, MCPServerConfig, MCPTool


class MCPManager:
    """Manager for multiple MCP server connections.

    Manages connections to multiple MCP servers, provides unified access
    to tools and resources, and converts MCP tools to Xerxes functions.

    Attributes:
        servers: Dictionary of server name to MCPClient
        logger: Logger instance
    """

    def __init__(self):
        """Initialize the MCP manager.

        Creates a new MCPManager instance with an empty server registry.
        Use ``add_server`` to register and connect to MCP servers after
        initialization.

        Example:
            >>> manager = MCPManager()
            >>> await manager.add_server(MCPServerConfig(name="my-server", command="npx", args=[...]))
            >>> tools = manager.get_all_tools()
        """
        self.servers: dict[str, MCPClient] = {}
        self.logger = get_logger()

    async def add_server(self, config: MCPServerConfig) -> bool:
        """Add and connect to an MCP server.

        Creates a new MCPClient instance, attempts to connect, and registers
        it with the manager if successful. Skips servers that are disabled
        in their configuration.

        Args:
            config: Server configuration specifying connection details.

        Returns:
            True if server added successfully, False otherwise.
        """
        if config.name in self.servers:
            self.logger.warning(f"MCP server {config.name} already exists")
            return False

        if not config.enabled:
            self.logger.info(f"MCP server {config.name} is disabled, skipping")
            return False

        client = MCPClient(config)
        success = await client.connect()

        if success:
            self.servers[config.name] = client
            self.logger.info(f"Added MCP server: {config.name}")
            return True
        else:
            self.logger.error(f"Failed to add MCP server: {config.name}")
            return False

    async def remove_server(self, name: str) -> None:
        """Remove and disconnect from an MCP server.

        Disconnects from the specified server and removes it from the manager.
        If the server is not found, this method does nothing.

        Args:
            name: Name of the server to remove.
        """
        if name in self.servers:
            await self.servers[name].disconnect()
            del self.servers[name]
            self.logger.info(f"Removed MCP server: {name}")

    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from all connected servers.

        Aggregates tools from all connected MCP servers into a single list.

        Returns:
            List of all available MCP tools across all servers.
        """
        tools = []
        for client in self.servers.values():
            tools.extend(client.tools)
        return tools

    def get_all_resources(self) -> list[MCPResource]:
        """Get all resources from all connected servers.

        Aggregates resources from all connected MCP servers into a single list.

        Returns:
            List of all available MCP resources across all servers.
        """
        resources = []
        for client in self.servers.values():
            resources.extend(client.resources)
        return resources

    def get_all_prompts(self) -> list[MCPPrompt]:
        """Get all prompts from all connected servers.

        Aggregates prompts from all connected MCP servers into a single list.

        Returns:
            List of all available MCP prompts across all servers.
        """
        prompts = []
        for client in self.servers.values():
            prompts.extend(client.prompts)
        return prompts

    def get_server(self, name: str) -> MCPClient | None:
        """Get MCP server client by name.

        Retrieves the MCPClient instance for a specific server, allowing
        direct access to the server's capabilities and methods.

        Args:
            name: Name of the server to retrieve.

        Returns:
            MCPClient instance or None if not found.
        """
        return self.servers.get(name)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool by name.

        Finds the tool across all connected servers and executes it on the
        server that provides it. The first matching tool found is used.

        Args:
            tool_name: Name of the tool to call.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            Tool execution result from the MCP server.

        Raises:
            ValueError: If the tool is not found in any connected server.
        """
        for client in self.servers.values():
            for tool in client.tools:
                if tool.name == tool_name:
                    return await client.call_tool(tool_name, arguments)

        raise ValueError(f"Tool {tool_name} not found in any connected MCP server")

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI.

        Searches all connected servers for a resource matching the given URI
        and reads its content from the server that provides it.

        Args:
            uri: Resource URI to read.

        Returns:
            Resource content as returned by the MCP server.

        Raises:
            ValueError: If the resource is not found in any connected server.
        """
        for client in self.servers.values():
            for resource in client.resources:
                if resource.uri == uri:
                    return await client.read_resource(uri)

        raise ValueError(f"Resource {uri} not found in any connected MCP server")

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Get a prompt by name.

        Searches all connected servers for a prompt matching the given name
        and retrieves it from the server that provides it.

        Args:
            name: Name of the prompt to retrieve.
            arguments: Optional dictionary of arguments for prompt rendering.

        Returns:
            Rendered prompt text as a string.

        Raises:
            ValueError: If the prompt is not found in any connected server.
        """
        for client in self.servers.values():
            for prompt in client.prompts:
                if prompt.name == name:
                    return await client.get_prompt(name, arguments)

        raise ValueError(f"Prompt {name} not found in any connected MCP server")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers.

        Gracefully closes all active server connections and clears the
        internal server registry. This method is safe to call even if
        no servers are connected.
        """
        for client in list(self.servers.values()):
            await client.disconnect()
        self.servers.clear()
        self.logger.info("Disconnected from all MCP servers")

    def list_servers(self) -> list[str]:
        """Get list of connected server names.

        Returns:
            List of server names currently connected to the manager.
        """
        return list(self.servers.keys())

    def get_capabilities_summary(self) -> dict[str, Any]:
        """Get summary of all capabilities across servers.

        Provides an overview of the tools, resources, and prompts available
        from each connected server, useful for debugging and monitoring.

        Returns:
            Dictionary mapping server names to capability counts, with keys
            'tools', 'resources', and 'prompts' for each server.
        """
        summary = {}
        for name, client in self.servers.items():
            summary[name] = {
                "tools": len(client.tools),
                "resources": len(client.resources),
                "prompts": len(client.prompts),
            }
        return summary
