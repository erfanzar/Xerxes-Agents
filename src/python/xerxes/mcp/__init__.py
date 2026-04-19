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


"""MCP (Model Context Protocol) integration for Xerxes.

This module provides integration with MCP servers, allowing Xerxes agents
to access external resources, tools, and prompts through the standardized
Model Context Protocol. The MCP integration enables agents to connect to
external services, databases, and APIs through a unified protocol.

Key Features:
    - Multiple transport support (STDIO, SSE, Streamable HTTP)
    - Multi-server management with unified tool access
    - Automatic capability discovery (tools, resources, prompts)
    - Asynchronous connection handling
    - Server configuration with environment variables and headers

Example:
    >>> from xerxes.mcp import MCPManager, MCPServerConfig
    >>>
    >>> manager = MCPManager()
    >>> config = MCPServerConfig(
    ...     name="my-server",
    ...     command="npx",
    ...     args=["-y", "@modelcontextprotocol/server-example"]
    ... )
    >>>
    >>>
"""

from .client import MCPClient
from .manager import MCPManager
from .types import MCPResource, MCPServerConfig, MCPTool

__all__ = [
    "MCPClient",
    "MCPManager",
    "MCPResource",
    "MCPServerConfig",
    "MCPTool",
]
