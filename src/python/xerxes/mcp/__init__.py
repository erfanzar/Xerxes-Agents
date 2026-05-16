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
"""Model Context Protocol client and types.

Re-exports :class:`MCPClient` (one connection to an MCP server),
:class:`MCPManager` (a registry of clients), and the protocol record types
(:class:`MCPTool`, :class:`MCPResource`, :class:`MCPServerConfig`). The
companion modules :mod:`xerxes.mcp.server`, :mod:`xerxes.mcp.oauth`,
:mod:`xerxes.mcp.reconnect`, and :mod:`xerxes.mcp.osv` cover the server-side
facade, OAuth flows, reconnect policy, and dependency vulnerability gating.
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
