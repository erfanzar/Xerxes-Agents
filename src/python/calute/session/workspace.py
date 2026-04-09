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

"""Workspace identity and management.

Provides WorkspaceIdentity for describing a workspace and
WorkspaceManager for CRUD operations on workspaces.
"""

from __future__ import annotations

import threading
import typing as tp
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class WorkspaceIdentity:
    """Describes a workspace and its associated metadata.

    A workspace groups related sessions together under a shared identity,
    optionally tied to a filesystem path.

    Attributes:
        workspace_id: Unique identifier for the workspace.
        name: Human-readable workspace name.
        root_path: Optional filesystem path associated with the workspace.
        created_at: ISO 8601 timestamp of workspace creation.
        metadata: Arbitrary metadata for extensibility.

    Example:
        >>> ws = WorkspaceIdentity(workspace_id="ws-1", name="My Project")
        >>> ws.name
        'My Project'
    """

    workspace_id: str
    name: str
    root_path: str | None = None
    created_at: str = ""
    metadata: dict[str, tp.Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialize the workspace identity to a JSON-compatible dictionary.

        Mutable containers (e.g., ``metadata``) are shallow-copied to
        prevent unintended mutation of the original instance.

        Returns:
            A dictionary containing all workspace identity fields with
            JSON-serializable values.

        Example:
            >>> ws = WorkspaceIdentity(workspace_id="ws-1", name="demo")
            >>> ws.to_dict()["name"]
            'demo'
        """
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "root_path": self.root_path,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> WorkspaceIdentity:
        """Deserialize a WorkspaceIdentity from a plain dictionary.

        Reconstructs a ``WorkspaceIdentity`` instance from a dictionary
        previously produced by :meth:`to_dict` or any compatible mapping.
        Missing optional keys fall back to sensible defaults.

        Args:
            data: A dictionary containing workspace identity fields. Must
                include ``workspace_id`` and ``name`` at minimum.

        Returns:
            A new ``WorkspaceIdentity`` instance populated from *data*.

        Raises:
            KeyError: If required keys (``workspace_id``, ``name``) are missing.

        Example:
            >>> data = {"workspace_id": "ws-1", "name": "demo"}
            >>> ws = WorkspaceIdentity.from_dict(data)
            >>> ws.workspace_id
            'ws-1'
        """
        return cls(
            workspace_id=data["workspace_id"],
            name=data["name"],
            root_path=data.get("root_path"),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )


class WorkspaceManager:
    """In-memory workspace manager for CRUD operations.

    Provides thread-safe creation, retrieval, and listing of workspaces.
    Workspaces are stored in a dictionary keyed by ``workspace_id`` and
    protected by a :class:`threading.Lock`.

    Attributes:
        _workspaces: Internal dictionary mapping workspace IDs to identities.
        _lock: Threading lock for safe concurrent access.

    Example:
        >>> manager = WorkspaceManager()
        >>> ws = manager.create_workspace(name="test")
        >>> manager.get_workspace(ws.workspace_id) is ws
        True
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory workspace manager with a threading lock."""
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
        """Create and register a new workspace.

        Args:
            name: Human-readable workspace name.
            root_path: Optional filesystem path.
            workspace_id: Explicit ID; auto-generated if omitted.
            metadata: Optional metadata dict.

        Returns:
            The newly created WorkspaceIdentity.
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
        """Retrieve a workspace by ID.

        Args:
            workspace_id: The workspace ID to look up.

        Returns:
            The WorkspaceIdentity, or None if not found.
        """
        with self._lock:
            return self._workspaces.get(workspace_id)

    def list_workspaces(self) -> list[WorkspaceIdentity]:
        """List all registered workspaces.

        Returns:
            List of WorkspaceIdentity objects.
        """
        with self._lock:
            return list(self._workspaces.values())
