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


"""Session persistence, replay, and workspace management for Xerxes.

Public API:
    Models:   SessionRecord, TurnRecord, ToolCallRecord, AgentTransitionRecord,
              SessionId, WorkspaceId
    Stores:   SessionStore, InMemorySessionStore, FileSessionStore, SessionManager
    Replay:   SessionReplay, ReplayView, TimelineEvent
    Workspace: WorkspaceIdentity, WorkspaceManager
"""

from .models import (
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
)
