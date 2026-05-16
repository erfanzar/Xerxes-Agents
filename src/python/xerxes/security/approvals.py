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
"""Per-session approval registry.

When the user approves a tool call in MANUAL mode, they can opt to
remember the decision for the rest of the session (``once`` /
``session``) or forever (``always``). This module stores those
decisions in memory and, for ``always`` scope, on disk so subsequent
runs reload them — preventing repeated prompting fatigue while
still letting the user revoke approvals later by editing the file."""

from __future__ import annotations

import enum
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ApprovalScope(enum.Enum):
    """How long a remembered approval remains in effect."""

    ONCE = "once"
    SESSION = "session"
    ALWAYS = "always"


@dataclass
class ApprovalRecord:
    """A single remembered approval/denial decision.

    Attributes:
        tool_name: tool the decision applies to.
        scope: lifetime (once/session/always).
        granted: True if approved, False if denied.
        session_id: session this decision is bound to (empty for ALWAYS).
        args_hash: opaque hash of the specific call args; only used for ONCE.
        created_at: ISO-8601 UTC timestamp at decision time.
    """

    tool_name: str
    scope: ApprovalScope
    granted: bool
    session_id: str = ""
    args_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialise as a JSON-friendly dict with ``scope`` as its string value."""
        d = asdict(self)
        d["scope"] = self.scope.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalRecord:
        """Rebuild a record from the dict produced by :meth:`to_dict`."""
        return cls(
            tool_name=data["tool_name"],
            scope=ApprovalScope(data.get("scope", "once")),
            granted=bool(data.get("granted", False)),
            session_id=data.get("session_id", ""),
            args_hash=data.get("args_hash", ""),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
        )


class ApprovalStore:
    """Track per-session and persistent tool approvals.

    A ``check`` call returns:
        * ``True`` — pre-approved (don't ask).
        * ``False`` — pre-denied (don't ask, deny).
        * ``None`` — no decision yet (ask the user)."""

    def __init__(self, persist_path: Path | None = None) -> None:
        """Load ALWAYS approvals from ``persist_path`` if present.

        A corrupt or unreadable persistence file is silently treated as
        empty rather than blocking startup."""
        self._persist_path = persist_path
        self._lock = threading.Lock()
        self._records: list[ApprovalRecord] = []
        if persist_path is not None and persist_path.exists():
            try:
                data = json.loads(persist_path.read_text(encoding="utf-8"))
                self._records = [ApprovalRecord.from_dict(r) for r in data]
            except (OSError, json.JSONDecodeError):
                self._records = []

    def add(self, record: ApprovalRecord) -> None:
        """Append ``record``; persist if its scope is ALWAYS."""
        with self._lock:
            self._records.append(record)
            if record.scope is ApprovalScope.ALWAYS and self._persist_path is not None:
                self._flush()

    def check(self, *, tool_name: str, session_id: str, args_hash: str = "") -> bool | None:
        """Look up the most recent applicable decision.

        Returns True if pre-approved, False if pre-denied, and None when
        no remembered decision matches and the caller must prompt the
        user. Iterates newest-first so later overrides win."""
        with self._lock:
            for r in reversed(self._records):
                if r.tool_name != tool_name:
                    continue
                if r.scope is ApprovalScope.ALWAYS:
                    return r.granted
                if r.scope is ApprovalScope.SESSION and r.session_id == session_id:
                    return r.granted
                if r.scope is ApprovalScope.ONCE and r.session_id == session_id and r.args_hash == args_hash:
                    return r.granted
            return None

    def list(self) -> list[ApprovalRecord]:
        """Snapshot of every record currently in memory."""
        with self._lock:
            return list(self._records)

    def clear_session(self, session_id: str) -> int:
        """Drop all records bound to ``session_id``; return how many were removed."""
        with self._lock:
            before = len(self._records)
            self._records = [r for r in self._records if r.session_id != session_id]
            return before - len(self._records)

    def _flush(self) -> None:
        """Write the ALWAYS-scope records to ``persist_path`` (creating dirs)."""
        if self._persist_path is None:
            return
        always = [r for r in self._records if r.scope is ApprovalScope.ALWAYS]
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps([r.to_dict() for r in always], indent=2), encoding="utf-8")


__all__ = ["ApprovalRecord", "ApprovalScope", "ApprovalStore"]
