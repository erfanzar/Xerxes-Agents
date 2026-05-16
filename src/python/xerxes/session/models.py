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
"""On-disk session record schema and versioning.

Dataclasses here describe the wire format the session store reads and writes.
Every shape change must bump :data:`CURRENT_SCHEMA_VERSION` and register a
migration in :mod:`xerxes.session.migrations` — old records are upgraded on
load, never downgraded.

The :class:`SessionRecord` is the top of the tree: it owns an ordered list of
:class:`TurnRecord` (user prompt + agent response + tool calls) and
:class:`AgentTransitionRecord` (handoffs between agents during a session).
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

SessionId = tp.NewType("SessionId", str)

WorkspaceId = tp.NewType("WorkspaceId", str)

# Bump when the on-disk shape of SessionRecord changes. Each bump must add a
# migration in xerxes.session.migrations. Forward-only — no downgrades.
CURRENT_SCHEMA_VERSION = 1


@dataclass
class ToolCallRecord:
    """One tool invocation captured inside a turn.

    Attributes:
        call_id: Provider-supplied call id (matches the ``ToolStart`` event).
        tool_name: Registered tool name (e.g. ``"Bash"``, ``"Read"``).
        arguments: JSON-serialisable input the tool was called with.
        result: Tool result; may be any JSON value or ``None``.
        status: ``"success"``, ``"error"``, ``"denied"`` etc.
        error: Error message when ``status`` is not success.
        duration_ms: Wall-clock duration spent executing the tool.
        sandbox_context: Sandbox label that gated the call, if any.
        metadata: Free-form ancillary data (audit ids, permission decisions).
    """

    call_id: str
    tool_name: str
    arguments: dict[str, tp.Any]
    result: tp.Any = None
    status: str = "success"
    error: str | None = None
    duration_ms: float | None = None
    sandbox_context: str | None = None
    metadata: dict[str, tp.Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, tp.Any]:
        """Return a JSON-ready shallow copy."""

        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "status": self.status,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "sandbox_context": self.sandbox_context,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> ToolCallRecord:
        """Reconstruct a record from a JSON-decoded dict."""

        return cls(
            call_id=data["call_id"],
            tool_name=data["tool_name"],
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            status=data.get("status", "success"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms"),
            sandbox_context=data.get("sandbox_context"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TurnRecord:
    """One user-prompt → agent-response cycle.

    Attributes:
        turn_id: Unique id for this turn within a session.
        agent_id: Agent that ran the turn (``None`` for system turns).
        prompt: Raw text the user (or parent agent) sent.
        response_content: Final assistant text; ``None`` if the turn ended
            before a textual response was produced.
        tool_calls: Tool invocations made while answering, in order.
        started_at: ISO-8601 timestamp when the turn began.
        ended_at: ISO-8601 timestamp when the turn finished, if it did.
        status: ``"success"``, ``"error"``, ``"cancelled"``, etc.
        error: Error message when ``status`` is not success.
        audit_event_ids: Linked audit log event ids for correlation.
        metadata: Free-form per-turn metadata (model name, token counts).
    """

    turn_id: str
    agent_id: str | None = None
    prompt: str = ""
    response_content: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    started_at: str = ""
    ended_at: str | None = None
    status: str = "success"
    error: str | None = None
    audit_event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, tp.Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, tp.Any]:
        """Return a JSON-ready shallow copy of the turn."""

        return {
            "turn_id": self.turn_id,
            "agent_id": self.agent_id,
            "prompt": self.prompt,
            "response_content": self.response_content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "error": self.error,
            "audit_event_ids": list(self.audit_event_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> TurnRecord:
        """Reconstruct a turn from a JSON-decoded dict."""

        return cls(
            turn_id=data["turn_id"],
            agent_id=data.get("agent_id"),
            prompt=data.get("prompt", ""),
            response_content=data.get("response_content"),
            tool_calls=[ToolCallRecord.from_dict(tc) for tc in data.get("tool_calls", [])],
            started_at=data.get("started_at", ""),
            ended_at=data.get("ended_at"),
            status=data.get("status", "success"),
            error=data.get("error"),
            audit_event_ids=data.get("audit_event_ids", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentTransitionRecord:
    """Marker recorded when one agent hands control to another mid-session.

    Attributes:
        from_agent: Agent id we are leaving (``None`` for the initial entry).
        to_agent: Agent id taking over.
        reason: Optional human-readable reason for the switch.
        turn_id: Turn during which the transition occurred.
        timestamp: ISO-8601 timestamp of the transition.
    """

    from_agent: str | None
    to_agent: str
    reason: str | None = None
    turn_id: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, tp.Any]:
        """Return a JSON-ready shallow copy."""

        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "reason": self.reason,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> AgentTransitionRecord:
        """Reconstruct from a JSON-decoded dict."""

        return cls(
            from_agent=data.get("from_agent"),
            to_agent=data["to_agent"],
            reason=data.get("reason"),
            turn_id=data.get("turn_id", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class SessionRecord:
    """Root persistence object — one entry per conversation.

    The session is the unit of replay, search, and branching. Forks set
    :attr:`parent_session_id` so lineage can be walked, and
    :attr:`schema_version` lets the loader run forward-only migrations on
    older payloads without manual intervention.

    Attributes:
        session_id: Unique id (typically a UUID hex).
        workspace_id: Workspace this conversation lives in, or ``None``.
        created_at: ISO-8601 timestamp set at creation.
        updated_at: ISO-8601 timestamp refreshed on every mutation.
        agent_id: Initial agent that owns the session.
        turns: All recorded turns in order.
        agent_transitions: Agent handoff markers across the session.
        metadata: Free-form metadata (title, ``ended`` flag, etc.).
        parent_session_id: If this session was branched, points at its source.
        schema_version: On-disk schema version of this record.
    """

    session_id: str
    workspace_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    agent_id: str | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    agent_transitions: list[AgentTransitionRecord] = field(default_factory=list)
    metadata: dict[str, tp.Any] = field(default_factory=dict)
    parent_session_id: str | None = None
    schema_version: int = CURRENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, tp.Any]:
        """Return a JSON-ready snapshot including nested records."""

        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "agent_id": self.agent_id,
            "turns": [t.to_dict() for t in self.turns],
            "agent_transitions": [at.to_dict() for at in self.agent_transitions],
            "metadata": dict(self.metadata),
            "parent_session_id": self.parent_session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> SessionRecord:
        """Reconstruct a session from a (possibly migrated) JSON dict."""

        return cls(
            session_id=data["session_id"],
            workspace_id=data.get("workspace_id"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            agent_id=data.get("agent_id"),
            turns=[TurnRecord.from_dict(t) for t in data.get("turns", [])],
            agent_transitions=[AgentTransitionRecord.from_dict(at) for at in data.get("agent_transitions", [])],
            metadata=data.get("metadata", {}),
            parent_session_id=data.get("parent_session_id"),
            schema_version=data.get("schema_version", CURRENT_SCHEMA_VERSION),
        )
