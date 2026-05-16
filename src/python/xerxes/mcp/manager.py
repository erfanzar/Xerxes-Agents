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
"""Registry of live MCP server connections.

:class:`MCPManager` holds one :class:`MCPClient` per server, exposes union
views over their tools / resources / prompts, and routes ``call_tool`` /
``read_resource`` / ``get_prompt`` to whichever client published the
requested capability.
"""

from typing import Any

from ..logging.console import get_logger
from .client import MCPClient
from .types import MCPPrompt, MCPResource, MCPServerConfig, MCPTool


class MCPManager:
    """Owns one :class:`MCPClient` per connected MCP server."""

    def __init__(self):
        """Build an empty registry."""

        self.servers: dict[str, MCPClient] = {}
        self.logger = get_logger()

    async def add_server(self, config: MCPServerConfig) -> bool:
        """Build, connect, and register a new server; return ``True`` on success.

        Already-registered or disabled configs are skipped (returning ``False``).
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
        """Disconnect and drop the named server (no-op if absent)."""

        if name in self.servers:
            await self.servers[name].disconnect()
            del self.servers[name]
            self.logger.info(f"Removed MCP server: {name}")

    def get_all_tools(self) -> list[MCPTool]:
        """Return every connected server's :attr:`MCPClient.tools` flattened together."""

        tools = []
        for client in self.servers.values():
            tools.extend(client.tools)
        return tools

    def get_all_resources(self) -> list[MCPResource]:
        """Return every connected server's :attr:`MCPClient.resources` flattened together."""

        resources = []
        for client in self.servers.values():
            resources.extend(client.resources)
        return resources

    def get_all_prompts(self) -> list[MCPPrompt]:
        """Return every connected server's :attr:`MCPClient.prompts` flattened together."""

        prompts = []
        for client in self.servers.values():
            prompts.extend(client.prompts)
        return prompts

    def get_server(self, name: str) -> MCPClient | None:
        """Return the registered client for ``name``, or ``None``."""

        return self.servers.get(name)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Route ``tool_name`` to the first server that published it; raises ``ValueError`` if none."""

        for client in self.servers.values():
            for tool in client.tools:
                if tool.name == tool_name:
                    return await client.call_tool(tool_name, arguments)

        raise ValueError(f"Tool {tool_name} not found in any connected MCP server")

    async def read_resource(self, uri: str) -> Any:
        """Route ``uri`` to the server that publishes it; raises ``ValueError`` if none."""

        for client in self.servers.values():
            for resource in client.resources:
                if resource.uri == uri:
                    return await client.read_resource(uri)

        raise ValueError(f"Resource {uri} not found in any connected MCP server")

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Route prompt ``name`` to the server that exposes it; raises ``ValueError`` if none."""

        for client in self.servers.values():
            for prompt in client.prompts:
                if prompt.name == name:
                    return await client.get_prompt(name, arguments)

        raise ValueError(f"Prompt {name} not found in any connected MCP server")

    async def disconnect_all(self) -> None:
        """Disconnect every registered client and clear the registry."""

        for client in list(self.servers.values()):
            await client.disconnect()
        self.servers.clear()
        self.logger.info("Disconnected from all MCP servers")

    def list_servers(self) -> list[str]:
        """Return the names of every currently-registered server."""

        return list(self.servers.keys())

    def get_capabilities_summary(self) -> dict[str, Any]:
        """Return ``{server_name: {tools, resources, prompts}}`` counts for each registered server."""

        summary = {}
        for name, client in self.servers.items():
            summary[name] = {
                "tools": len(client.tools),
                "resources": len(client.resources),
                "prompts": len(client.prompts),
            }
        return summary
