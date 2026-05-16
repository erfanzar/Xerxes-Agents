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
"""Plain-old-data records for the Model Context Protocol.

These dataclasses mirror the JSON shapes the MCP spec defines: a server
config tells the client how to launch (stdio) or reach (SSE / streamable
HTTP) the server, while :class:`MCPTool`, :class:`MCPResource`, and
:class:`MCPPrompt` carry the capabilities the client learns at connect
time.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPTransportType(Enum):
    """MCP transport variants understood by :class:`xerxes.mcp.MCPClient`.

    ``HTTP`` and ``WEBSOCKET`` are legacy aliases that map onto ``SSE`` and
    ``STREAMABLE_HTTP`` respectively.
    """

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"

    HTTP = "sse"
    WEBSOCKET = "streamable_http"


@dataclass
class MCPServerConfig:
    """Connection settings for one MCP server.

    Attributes:
        name: Logical name (used in the manager registry).
        command: Executable for the stdio transport.
        args: Extra command-line args for the stdio transport.
        env: Environment overrides applied to the spawned process.
        transport: Wire transport to use.
        url: Endpoint URL for SSE / streamable HTTP transports.
        headers: Extra HTTP headers for SSE / streamable HTTP.
        enabled: When false, the manager skips this entry on load.
        timeout: Connect / request timeout in seconds.
        sse_read_timeout: SSE long-poll read timeout in seconds.
    """

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: MCPTransportType = MCPTransportType.STDIO
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout: float = 30.0
    sse_read_timeout: float = 300.0


@dataclass
class MCPTool:
    """One callable exposed by an MCP server.

    Attributes:
        name: Tool name as the server publishes it.
        description: Human-readable description.
        input_schema: JSON-Schema for ``call_tool`` arguments.
        server_name: Owning server's :attr:`MCPServerConfig.name`.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPResource:
    """One readable resource exposed by an MCP server.

    Attributes:
        uri: Stable URI used to fetch the resource via ``read_resource``.
        name: Display name.
        description: Human-readable description.
        mime_type: Optional MIME type if the server declares one.
        server_name: Owning server's :attr:`MCPServerConfig.name`.
    """

    uri: str
    name: str
    description: str
    mime_type: str | None = None
    server_name: str = ""


@dataclass
class MCPPrompt:
    """One templated prompt exposed by an MCP server.

    Attributes:
        name: Prompt name as the server publishes it.
        description: Human-readable description.
        arguments: Schema fragments for each templated parameter.
        server_name: Owning server's :attr:`MCPServerConfig.name`.
    """

    name: str
    description: str
    arguments: list[dict[str, Any]] = field(default_factory=list)
    server_name: str = ""
