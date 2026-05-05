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
"""Types module for Xerxes.

Exports:
    - MCPTransportType
    - MCPServerConfig
    - MCPTool
    - MCPResource
    - MCPPrompt"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPTransportType(Enum):
    """Mcptransport type.

    Inherits from: Enum
    """

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"

    HTTP = "sse"
    WEBSOCKET = "streamable_http"


@dataclass
class MCPServerConfig:
    """Mcpserver config.

    Attributes:
        name (str): name.
        command (str | None): command.
        args (list[str]): args.
        env (dict[str, str]): env.
        transport (MCPTransportType): transport.
        url (str | None): url.
        headers (dict[str, str]): headers.
        enabled (bool): enabled.
        timeout (float): timeout.
        sse_read_timeout (float): sse read timeout."""

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
    """Mcptool.

    Attributes:
        name (str): name.
        description (str): description.
        input_schema (dict[str, Any]): input schema.
        server_name (str): server name."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPResource:
    """Mcpresource.

    Attributes:
        uri (str): uri.
        name (str): name.
        description (str): description.
        mime_type (str | None): mime type.
        server_name (str): server name."""

    uri: str
    name: str
    description: str
    mime_type: str | None = None
    server_name: str = ""


@dataclass
class MCPPrompt:
    """Mcpprompt.

    Attributes:
        name (str): name.
        description (str): description.
        arguments (list[dict[str, Any]]): arguments.
        server_name (str): server name."""

    name: str
    description: str
    arguments: list[dict[str, Any]] = field(default_factory=list)
    server_name: str = ""
