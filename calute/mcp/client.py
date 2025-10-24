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


"""MCP client implementation for connecting to MCP servers."""

import asyncio
import json
import subprocess
from typing import Any

from ..loggings import get_logger
from .types import MCPPrompt, MCPResource, MCPServerConfig, MCPTool, MCPTransportType


class MCPClient:
    """Client for connecting to and interacting with MCP servers.

    This client supports stdio, HTTP, and WebSocket transports for
    communicating with MCP servers according to the Model Context Protocol.

    Attributes:
        config: MCP server configuration
        process: Subprocess for stdio transport (if applicable)
        session_id: Session identifier for this connection
        tools: Available tools from the server
        resources: Available resources from the server
        prompts: Available prompts from the server
    """

    def __init__(self, config: MCPServerConfig):
        """Initialize MCP client with server configuration.

        Args:
            config: Configuration for the MCP server
        """
        self.config = config
        self.process: subprocess.Popen | None = None
        self.session_id: str | None = None
        self.connected = False
        self.logger = get_logger()

        self.tools: list[MCPTool] = []
        self.resources: list[MCPResource] = []
        self.prompts: list[MCPPrompt] = []

    async def connect(self) -> bool:
        """Connect to the MCP server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.config.transport == MCPTransportType.STDIO:
                return await self._connect_stdio()
            elif self.config.transport == MCPTransportType.HTTP:
                return await self._connect_http()
            elif self.config.transport == MCPTransportType.WEBSOCKET:
                return await self._connect_websocket()
            else:
                self.logger.error(f"Unsupported transport type: {self.config.transport}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server {self.config.name}: {e}")
            return False

    async def _connect_stdio(self) -> bool:
        """Connect using stdio transport."""
        if not self.config.command:
            self.logger.error(f"No command specified for stdio MCP server {self.config.name}")
            return False

        try:
            env = None
            if self.config.env:
                import os

                env = os.environ.copy()
                env.update(self.config.env)

            self.logger.info(f"Starting MCP server: {self.config.command} {' '.join(self.config.args)}")

            self.process = subprocess.Popen(
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )

            await asyncio.sleep(0.5)

            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read() if self.process.stderr else ""
                self.logger.error(f"MCP server process failed to start. Exit code: {self.process.returncode}")
                self.logger.error(f"stderr: {stderr_output}")
                return False

            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "Calute", "version": "0.1.2"},
                },
                "id": 1,
            }

            self._write_message(init_request)
            response = await self._read_message()

            if response and response.get("result"):
                initialized_notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
                self._write_message(initialized_notification)

                self.session_id = str(id(self))
                self.connected = True
                self.logger.debug(f"Connected to MCP server {self.config.name}")
                await self._discover_capabilities()
                return True
            elif response and response.get("error"):
                self.logger.error(f"MCP server returned error: {response['error']}")
                return False
            else:
                self.logger.error(f"No valid response from MCP server {self.config.name}")
                return False

        except FileNotFoundError:
            self.logger.error(f"Command not found: {self.config.command}. Make sure it's installed and in PATH.")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect via stdio: {e}")
            if self.process:
                self.logger.error(f"Process poll: {self.process.poll()}")
            return False

    async def _connect_http(self) -> bool:
        """Connect using HTTP transport."""

        self.logger.warning("HTTP transport not yet implemented")
        return False

    async def _connect_websocket(self) -> bool:
        """Connect using WebSocket transport."""

        self.logger.warning("WebSocket transport not yet implemented")
        return False

    def _write_message(self, message: dict[str, Any]) -> None:
        """Write a message to the MCP server (stdio)."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server process not available")

        json_str = json.dumps(message)
        self.process.stdin.write(json_str + "\n")
        self.process.stdin.flush()

    async def _read_message(self) -> dict[str, Any] | None:
        """Read a message from the MCP server (stdio)."""
        if not self.process or not self.process.stdout:
            return None

        try:
            line = await asyncio.wait_for(
                asyncio.create_task(asyncio.to_thread(self.process.stdout.readline)), timeout=10.0
            )

            if line:
                line_str = line.strip()
                if line_str:
                    return json.loads(line_str)
            return None
        except TimeoutError:
            self.logger.error(
                f"Timeout reading from MCP server {self.config.name}. Server may not be running or configured correctly."
            )

            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read() if self.process.stderr else ""
                self.logger.error(f"MCP server process exited. stderr: {stderr_output}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse MCP response: {e}")
            return None

    async def _discover_capabilities(self) -> None:
        """Discover tools, resources, and prompts from the server."""

        tools_request = {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2}
        self._write_message(tools_request)
        tools_response = await self._read_message()

        if tools_response and tools_response.get("result"):
            tools_data = tools_response["result"].get("tools", [])
            self.tools = [
                MCPTool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    server_name=self.config.name,
                )
                for tool in tools_data
            ]
            self.logger.info(f"Discovered {len(self.tools)} tools from {self.config.name}")

        resources_request = {"jsonrpc": "2.0", "method": "resources/list", "params": {}, "id": 3}
        self._write_message(resources_request)
        resources_response = await self._read_message()

        if resources_response and resources_response.get("result"):
            resources_data = resources_response["result"].get("resources", [])
            self.resources = [
                MCPResource(
                    uri=resource["uri"],
                    name=resource.get("name", ""),
                    description=resource.get("description", ""),
                    mime_type=resource.get("mimeType"),
                    server_name=self.config.name,
                )
                for resource in resources_data
            ]
            self.logger.info(f"Discovered {len(self.resources)} resources from {self.config.name}")

        prompts_request = {"jsonrpc": "2.0", "method": "prompts/list", "params": {}, "id": 4}
        self._write_message(prompts_request)
        prompts_response = await self._read_message()

        if prompts_response and prompts_response.get("result"):
            prompts_data = prompts_response["result"].get("prompts", [])
            self.prompts = [
                MCPPrompt(
                    name=prompt["name"],
                    description=prompt.get("description", ""),
                    arguments=prompt.get("arguments", []),
                    server_name=self.config.name,
                )
                for prompt in prompts_data
            ]
            self.logger.info(f"Discovered {len(self.prompts)} prompts from {self.config.name}")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self.connected:
            raise RuntimeError(f"Not connected to MCP server {self.config.name}")

        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": 100,
        }

        self._write_message(request)
        response = await self._read_message()

        if response and response.get("result"):
            return response["result"].get("content", [])
        elif response and response.get("error"):
            raise RuntimeError(f"MCP tool call error: {response['error']}")
        else:
            raise RuntimeError("Invalid response from MCP server")

    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the MCP server.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        if not self.connected:
            raise RuntimeError(f"Not connected to MCP server {self.config.name}")

        request = {"jsonrpc": "2.0", "method": "resources/read", "params": {"uri": uri}, "id": 101}

        self._write_message(request)
        response = await self._read_message()

        if response and response.get("result"):
            return response["result"].get("contents", [])
        elif response and response.get("error"):
            raise RuntimeError(f"MCP resource read error: {response['error']}")
        else:
            raise RuntimeError("Invalid response from MCP server")

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Get a prompt from the MCP server.

        Args:
            name: Prompt name
            arguments: Prompt arguments

        Returns:
            Rendered prompt text
        """
        if not self.connected:
            raise RuntimeError(f"Not connected to MCP server {self.config.name}")

        request = {
            "jsonrpc": "2.0",
            "method": "prompts/get",
            "params": {"name": name, "arguments": arguments or {}},
            "id": 102,
        }

        self._write_message(request)
        response = await self._read_message()

        if response and response.get("result"):
            messages = response["result"].get("messages", [])
            if messages:
                return messages[0].get("content", {}).get("text", "")
            return ""
        elif response and response.get("error"):
            raise RuntimeError(f"MCP prompt get error: {response['error']}")
        else:
            raise RuntimeError("Invalid response from MCP server")

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                self.logger.error(f"Error disconnecting from MCP server: {e}")

        self.connected = False
        self.session_id = None
        self.logger.info(f"Disconnected from MCP server {self.config.name}")

    def __del__(self):
        """Cleanup on deletion."""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass
