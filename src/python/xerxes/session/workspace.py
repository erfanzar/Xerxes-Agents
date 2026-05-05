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
"""Workspace module for Xerxes.

Exports:
    - WorkspaceIdentity
    - WorkspaceManager"""

from __future__ import annotations

import threading
import typing as tp
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class WorkspaceIdentity:
    """Workspace identity.

    Attributes:
        workspace_id (str): workspace id.
        name (str): name.
        root_path (str | None): root path.
        created_at (str): created at.
        metadata (dict[str, tp.Any]): metadata."""

    workspace_id: str
    name: str
    root_path: str | None = None
    created_at: str = ""
    metadata: dict[str, tp.Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, tp.Any]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "root_path": self.root_path,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> WorkspaceIdentity:
        """From dict.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            data (dict[str, tp.Any]): IN: data. OUT: Consumed during execution.
        Returns:
            WorkspaceIdentity: OUT: Result of the operation."""

        return cls(
            workspace_id=data["workspace_id"],
            name=data["name"],
            root_path=data.get("root_path"),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )


class WorkspaceManager:
    """Workspace manager."""

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Create workspace.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            root_path (str | None, optional): IN: root path. Defaults to None. OUT: Consumed during execution.
            workspace_id (str | None, optional): IN: workspace id. Defaults to None. OUT: Consumed during execution.
            metadata (dict[str, tp.Any] | None, optional): IN: metadata. Defaults to None. OUT: Consumed during execution.
        Returns:
            WorkspaceIdentity: OUT: Result of the operation."""

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
        """Retrieve the workspace.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            workspace_id (str): IN: workspace id. OUT: Consumed during execution.
        Returns:
            WorkspaceIdentity | None: OUT: Result of the operation."""

        with self._lock:
            return self._workspaces.get(workspace_id)

    def list_workspaces(self) -> list[WorkspaceIdentity]:
        """List workspaces.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[WorkspaceIdentity]: OUT: Result of the operation."""

        with self._lock:
            return list(self._workspaces.values())
