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


"""MCP (Model Context Protocol) integration for Calute.

This module provides integration with MCP servers, allowing Calute agents
to access external resources, tools, and prompts through the standardized
Model Context Protocol.
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
