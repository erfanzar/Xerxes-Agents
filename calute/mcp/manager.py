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


"""MCP Manager for managing multiple MCP server connections."""

from typing import Any

from ..loggings import get_logger
from .client import MCPClient
from .types import MCPPrompt, MCPResource, MCPServerConfig, MCPTool


class MCPManager:
    """Manager for multiple MCP server connections.

    Manages connections to multiple MCP servers, provides unified access
    to tools and resources, and converts MCP tools to Calute functions.

    Attributes:
        servers: Dictionary of server name to MCPClient
        logger: Logger instance
    """

    def __init__(self):
        """Initialize the MCP manager."""
        self.servers: dict[str, MCPClient] = {}
        self.logger = get_logger()

    async def add_server(self, config: MCPServerConfig) -> bool:
        """Add and connect to an MCP server.

        Args:
            config: Server configuration

        Returns:
            True if server added successfully, False otherwise
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

        Args:
            name: Server name
        """
        if name in self.servers:
            await self.servers[name].disconnect()
            del self.servers[name]
            self.logger.info(f"Removed MCP server: {name}")

    def get_all_tools(self) -> list[MCPTool]:
        """Get all tools from all connected servers.

        Returns:
            List of all available MCP tools
        """
        tools = []
        for client in self.servers.values():
            tools.extend(client.tools)
        return tools

    def get_all_resources(self) -> list[MCPResource]:
        """Get all resources from all connected servers.

        Returns:
            List of all available MCP resources
        """
        resources = []
        for client in self.servers.values():
            resources.extend(client.resources)
        return resources

    def get_all_prompts(self) -> list[MCPPrompt]:
        """Get all prompts from all connected servers.

        Returns:
            List of all available MCP prompts
        """
        prompts = []
        for client in self.servers.values():
            prompts.extend(client.prompts)
        return prompts

    def get_server(self, name: str) -> MCPClient | None:
        """Get MCP server by name.

        Args:
            name: Server name

        Returns:
            MCPClient instance or None if not found
        """
        return self.servers.get(name)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool by name.

        Finds the tool across all servers and executes it.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        for client in self.servers.values():
            for tool in client.tools:
                if tool.name == tool_name:
                    return await client.call_tool(tool_name, arguments)

        raise ValueError(f"Tool {tool_name} not found in any connected MCP server")

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        for client in self.servers.values():
            for resource in client.resources:
                if resource.uri == uri:
                    return await client.read_resource(uri)

        raise ValueError(f"Resource {uri} not found in any connected MCP server")

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Get a prompt by name.

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            Rendered prompt text
        """
        for client in self.servers.values():
            for prompt in client.prompts:
                if prompt.name == name:
                    return await client.get_prompt(name, arguments)

        raise ValueError(f"Prompt {name} not found in any connected MCP server")

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for client in list(self.servers.values()):
            await client.disconnect()
        self.servers.clear()
        self.logger.info("Disconnected from all MCP servers")

    def list_servers(self) -> list[str]:
        """Get list of connected server names.

        Returns:
            List of server names
        """
        return list(self.servers.keys())

    def get_capabilities_summary(self) -> dict[str, Any]:
        """Get summary of all capabilities across servers.

        Returns:
            Dictionary with counts of tools, resources, and prompts per server
        """
        summary = {}
        for name, client in self.servers.items():
            summary[name] = {
                "tools": len(client.tools),
                "resources": len(client.resources),
                "prompts": len(client.prompts),
            }
        return summary
