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
"""Workspace identity records and an in-process registry.

A workspace is the long-lived container sessions are filed under (project
root, telegram chat, etc.). :class:`WorkspaceIdentity` is the serialisable
record; :class:`WorkspaceManager` is a thread-safe in-memory registry used
by the daemon to dedupe workspace creation during a process lifetime.
"""

from __future__ import annotations

import threading
import typing as tp
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class WorkspaceIdentity:
    """Stable identity record describing one workspace.

    Attributes:
        workspace_id: Unique opaque id (typically a UUID hex).
        name: Human-readable label.
        root_path: Filesystem root, when the workspace is a project on disk.
        created_at: ISO-8601 creation timestamp.
        metadata: Arbitrary JSON-serialisable metadata.
    """

    workspace_id: str
    name: str
    root_path: str | None = None
    created_at: str = ""
    metadata: dict[str, tp.Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, tp.Any]:
        """Return a JSON-ready shallow copy."""

        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "root_path": self.root_path,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> WorkspaceIdentity:
        """Reconstruct an identity from a JSON-decoded dict."""

        return cls(
            workspace_id=data["workspace_id"],
            name=data["name"],
            root_path=data.get("root_path"),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )


class WorkspaceManager:
    """Thread-safe in-memory registry of :class:`WorkspaceIdentity` records.

    The registry is process-local — restarting the daemon empties it. The
    file-backed :class:`FileSessionStore` derives its directory layout from
    ``workspace_id``, so persistent identity ultimately lives on disk via
    the sessions themselves.
    """

    def __init__(self) -> None:
        """Create an empty registry."""

        self._workspaces: dict[str, WorkspaceIdentity] = {}
        self._lock = threading.Lock()

    def create_workspace(
        self,
        name: str,
        root_path: str | None = None,
        *,
        workspace_id: str | None = None,
        metadata: dict[str, tp.Any] | None = None,
    ) -> WorkspaceIdentity:
        """Register a new workspace and return its identity.

        Args:
            name: Human-readable label.
            root_path: Filesystem root associated with the workspace.
            workspace_id: Override the generated UUID hex (mostly for tests).
            metadata: Arbitrary extra fields stored on the identity.
        """

        ws = WorkspaceIdentity(
            workspace_id=workspace_id or uuid.uuid4().hex,
            name=name,
            root_path=root_path,
            created_at=datetime.now(UTC).isoformat(),
            metadata=metadata or {},
        )
        with self._lock:
            self._workspaces[ws.workspace_id] = ws
        return ws

    def get_workspace(self, workspace_id: str) -> WorkspaceIdentity | None:
        """Return the registered identity or ``None`` if it does not exist."""

        with self._lock:
            return self._workspaces.get(workspace_id)

    def list_workspaces(self) -> list[WorkspaceIdentity]:
        """Return a snapshot of every registered workspace."""

        with self._lock:
            return list(self._workspaces.values())
