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
"""Session persistence, replay, and workspace identity.

The session subsystem records every turn, tool call, and agent transition into
durable :class:`SessionRecord` objects, then offers replay, search, and
forward-only schema migrations on top of them. Stores come in two flavours:
:class:`InMemorySessionStore` (tests, ephemeral runs) and
:class:`FileSessionStore` (the daemon's default, with atomic writes and
inline migrations on load).

Re-exports the public surface used by the daemon, the TUI history view, and
the ``/replay`` slash command.
"""

from .migrations import MIGRATIONS, migrate_record, register
from .models import (
    CURRENT_SCHEMA_VERSION,
    AgentTransitionRecord,
    SessionId,
    SessionRecord,
    ToolCallRecord,
    TurnRecord,
    WorkspaceId,
)
from .replay import ReplayView, SessionReplay, TimelineEvent
from .store import (
    FileSessionStore,
    InMemorySessionStore,
    SessionManager,
    SessionStore,
)
from .workspace import WorkspaceIdentity, WorkspaceManager

__all__ = (
    "CURRENT_SCHEMA_VERSION",
    "MIGRATIONS",
    "AgentTransitionRecord",
    "FileSessionStore",
    "InMemorySessionStore",
    "ReplayView",
    "SessionId",
    "SessionManager",
    "SessionRecord",
    "SessionReplay",
    "SessionStore",
    "TimelineEvent",
    "ToolCallRecord",
    "TurnRecord",
    "WorkspaceId",
    "WorkspaceIdentity",
    "WorkspaceManager",
    "migrate_record",
    "register",
)
