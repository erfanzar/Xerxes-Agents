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
"""Manager module for Xerxes.

Exports:
    - MCPManager"""

from typing import Any

from ..logging.console import get_logger
from .client import MCPClient
from .types import MCPPrompt, MCPResource, MCPServerConfig, MCPTool


class MCPManager:
    """Mcpmanager."""

    def __init__(self):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            Any: OUT: Result of the operation."""

        self.servers: dict[str, MCPClient] = {}
        self.logger = get_logger()

    async def add_server(self, config: MCPServerConfig) -> bool:
        """Asynchronously Add server.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (MCPServerConfig): IN: config. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

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
        """Asynchronously Remove server.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution."""

        if name in self.servers:
            await self.servers[name].disconnect()
            del self.servers[name]
            self.logger.info(f"Removed MCP server: {name}")

    def get_all_tools(self) -> list[MCPTool]:
        """Retrieve the all tools.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[MCPTool]: OUT: Result of the operation."""

        tools = []
        for client in self.servers.values():
            tools.extend(client.tools)
        return tools

    def get_all_resources(self) -> list[MCPResource]:
        """Retrieve the all resources.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[MCPResource]: OUT: Result of the operation."""

        resources = []
        for client in self.servers.values():
            resources.extend(client.resources)
        return resources

    def get_all_prompts(self) -> list[MCPPrompt]:
        """Retrieve the all prompts.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[MCPPrompt]: OUT: Result of the operation."""

        prompts = []
        for client in self.servers.values():
            prompts.extend(client.prompts)
        return prompts

    def get_server(self, name: str) -> MCPClient | None:
        """Retrieve the server.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
        Returns:
            MCPClient | None: OUT: Result of the operation."""

        return self.servers.get(name)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Asynchronously Call tool.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            arguments (dict[str, Any]): IN: arguments. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

        for client in self.servers.values():
            for tool in client.tools:
                if tool.name == tool_name:
                    return await client.call_tool(tool_name, arguments)

        raise ValueError(f"Tool {tool_name} not found in any connected MCP server")

    async def read_resource(self, uri: str) -> Any:
        """Asynchronously Read resource.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            uri (str): IN: uri. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""

        for client in self.servers.values():
            for resource in client.resources:
                if resource.uri == uri:
                    return await client.read_resource(uri)

        raise ValueError(f"Resource {uri} not found in any connected MCP server")

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Asynchronously Retrieve the prompt.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            arguments (dict[str, Any] | None, optional): IN: arguments. Defaults to None. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        for client in self.servers.values():
            for prompt in client.prompts:
                if prompt.name == name:
                    return await client.get_prompt(name, arguments)

        raise ValueError(f"Prompt {name} not found in any connected MCP server")

    async def disconnect_all(self) -> None:
        """Asynchronously Disconnect all.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        for client in list(self.servers.values()):
            await client.disconnect()
        self.servers.clear()
        self.logger.info("Disconnected from all MCP servers")

    def list_servers(self) -> list[str]:
        """List servers.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[str]: OUT: Result of the operation."""

        return list(self.servers.keys())

    def get_capabilities_summary(self) -> dict[str, Any]:
        """Retrieve the capabilities summary.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        summary = {}
        for name, client in self.servers.items():
            summary[name] = {
                "tools": len(client.tools),
                "resources": len(client.resources),
                "prompts": len(client.prompts),
            }
        return summary
