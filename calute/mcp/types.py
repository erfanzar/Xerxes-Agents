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


"""Type definitions for MCP integration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPTransportType(Enum):
    """MCP transport types."""

    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection.

    Attributes:
        name: Unique name for this MCP server
        command: Command to start the server (for stdio transport)
        args: Arguments for the server command
        env: Environment variables for the server
        transport: Transport type (stdio, http, websocket)
        url: Server URL (for http/websocket transport)
        headers: HTTP headers for authentication
        enabled: Whether this server is enabled
    """

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: MCPTransportType = MCPTransportType.STDIO
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class MCPTool:
    """Represents an MCP tool that can be called by agents.

    Attributes:
        name: Tool name
        description: Tool description
        input_schema: JSON schema for tool input
        server_name: Name of the MCP server providing this tool
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPResource:
    """Represents an MCP resource (data) accessible to agents.

    Attributes:
        uri: Resource URI
        name: Resource name
        description: Resource description
        mime_type: MIME type of the resource
        server_name: Name of the MCP server providing this resource
    """

    uri: str
    name: str
    description: str
    mime_type: str | None = None
    server_name: str = ""


@dataclass
class MCPPrompt:
    """Represents an MCP prompt template.

    Attributes:
        name: Prompt name
        description: Prompt description
        arguments: Expected prompt arguments
        server_name: Name of the MCP server providing this prompt
    """

    name: str
    description: str
    arguments: list[dict[str, Any]] = field(default_factory=list)
    server_name: str = ""
