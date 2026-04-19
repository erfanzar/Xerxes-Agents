# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Type definitions for Model Context Protocol (MCP) integration.

This module provides data structures and type definitions for MCP
(Model Context Protocol) integration within the Xerxes framework. It includes:
- Transport type enumeration for different communication protocols
- Server configuration dataclass for MCP server connections
- Tool, resource, and prompt dataclasses for MCP primitives

MCP enables standardized communication between AI agents and external
tools/services. These types form the foundation for the MCP client
implementation in Xerxes.

Example:
    >>> from xerxes.mcp.types import MCPServerConfig, MCPTransportType
    >>> config = MCPServerConfig(
    ...     name="filesystem",
    ...     command="npx",
    ...     args=["-y", "@modelcontextprotocol/server-filesystem"],
    ...     transport=MCPTransportType.STDIO
    ... )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPTransportType(Enum):
    """Enumeration of available MCP transport types.

    Defines the communication protocols supported for connecting
    to MCP servers. Each transport type has different use cases
    and performance characteristics.

    Transport Types:
        STDIO: Local subprocess communication via standard input/output.
            Best for local tools like npx or uvx style servers.
        SSE: Server-Sent Events over HTTP (legacy 2024-11-05 protocol).
            Suitable for remote servers with one-way streaming.
        STREAMABLE_HTTP: Streamable HTTP transport (recommended for 2025+).
            Modern bidirectional streaming protocol for remote servers.

    Note:
        HTTP and WEBSOCKET are deprecated aliases maintained for
        backwards compatibility. Use SSE or STREAMABLE_HTTP instead.
    """

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"

    HTTP = "sse"
    WEBSOCKET = "streamable_http"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection.

    Defines all settings required to establish and manage a connection
    to an MCP server. Supports both local subprocess (STDIO) and remote
    HTTP-based transports with configurable timeouts and authentication.

    Attributes:
        name: Unique identifier name for this MCP server.
        command: Command to start the server (required for STDIO transport).
        args: Command-line arguments for the server process.
        env: Environment variables to set for the server process.
        transport: Communication protocol type (STDIO, SSE, or STREAMABLE_HTTP).
        url: Server URL (required for SSE/Streamable HTTP transports).
        headers: HTTP headers for authentication and custom metadata.
        enabled: Whether this server connection is active.
        timeout: Timeout in seconds for HTTP operations (default: 30.0).
        sse_read_timeout: Timeout in seconds for SSE event stream reads (default: 300.0).

    Example:
        >>> config = MCPServerConfig(
        ...     name="github",
        ...     url="https://mcp.github.com/sse",
        ...     transport=MCPTransportType.SSE,
        ...     headers={"Authorization": "Bearer token"}
        ... )
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
    """Represents an MCP tool that can be called by agents.

    Encapsulates the metadata and schema for an MCP tool, enabling
    agents to discover and invoke external capabilities exposed by
    MCP servers. Tools are the primary mechanism for agent actions.

    Attributes:
        name: Unique identifier for the tool within its server.
        description: Human-readable description of tool functionality.
        input_schema: JSON Schema defining required and optional parameters.
        server_name: Name of the MCP server that provides this tool.

    Example:
        >>> tool = MCPTool(
        ...     name="read_file",
        ...     description="Read contents of a file",
        ...     input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ...     server_name="filesystem"
        ... )
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPResource:
    """Represents an MCP resource (data) accessible to agents.

    Encapsulates metadata for data resources exposed by MCP servers.
    Resources provide read-only access to contextual information that
    agents can use to inform their responses and decisions.

    Attributes:
        uri: Unique resource identifier URI for accessing the resource.
        name: Human-readable display name for the resource.
        description: Detailed description of the resource content.
        mime_type: MIME type indicating the resource format (e.g., 'text/plain').
        server_name: Name of the MCP server that provides this resource.

    Example:
        >>> resource = MCPResource(
        ...     uri="file:///path/to/document.txt",
        ...     name="Document",
        ...     description="Project documentation file",
        ...     mime_type="text/plain",
        ...     server_name="filesystem"
        ... )
    """

    uri: str
    name: str
    description: str
    mime_type: str | None = None
    server_name: str = ""


@dataclass
class MCPPrompt:
    """Represents an MCP prompt template.

    Encapsulates reusable prompt templates exposed by MCP servers.
    Prompts allow servers to provide pre-defined interaction patterns
    that agents can invoke with specific arguments to generate responses.

    Attributes:
        name: Unique identifier for the prompt template.
        description: Human-readable description of the prompt purpose.
        arguments: List of argument definitions with name, type, and description.
        server_name: Name of the MCP server that provides this prompt.

    Example:
        >>> prompt = MCPPrompt(
        ...     name="summarize",
        ...     description="Summarize the given text",
        ...     arguments=[{"name": "text", "type": "string", "required": True}],
        ...     server_name="text-tools"
        ... )
    """

    name: str
    description: str
    arguments: list[dict[str, Any]] = field(default_factory=list)
    server_name: str = ""
