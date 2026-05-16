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
"""Agent Client Protocol (ACP) adapter package.

Maps Xerxes's streaming/loop surface to the ACP interface that IDE
clients (Claude Code, Cursor, Cline) speak. ACP is a young
standard; the registry metadata + session adapter live here so the
``xerxes-acp`` entry point can register Xerxes as an ACP server."""

from .events import AcpEvent, AcpEventKind, to_acp_event
from .permissions import AcpPermissionRequest, route_permission
from .registry import REGISTRY_METADATA, write_registry_file
from .server import AcpServer, ServerCapabilities
from .session import AcpSession, AcpSessionStore

__all__ = [
    "REGISTRY_METADATA",
    "AcpEvent",
    "AcpEventKind",
    "AcpPermissionRequest",
    "AcpServer",
    "AcpSession",
    "AcpSessionStore",
    "ServerCapabilities",
    "route_permission",
    "to_acp_event",
    "write_registry_file",
]
