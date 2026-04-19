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


"""Data models for session persistence and replay.

Provides serializable dataclasses for recording session turns,
tool calls, agent transitions, and full session records.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

SessionId = tp.NewType("SessionId", str)
"""A distinct string type representing a unique session identifier."""

WorkspaceId = tp.NewType("WorkspaceId", str)
"""A distinct string type representing a unique workspace identifier."""


@dataclass
class ToolCallRecord:
    """Record of a single tool/function call within a turn.

    Attributes:
        call_id: Unique identifier for this tool call.
        tool_name: Name of the tool/function invoked.
        arguments: Arguments passed to the tool.
        result: Result returned by the tool.
        status: Execution status (success, error, timeout, skipped).
        error: Error message if the call failed.
        duration_ms: Execution duration in milliseconds.
        sandbox_context: Sandbox context identifier, if applicable.
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
        """Serialize the tool call record to a JSON-compatible dictionary.

        Converts all fields of this record into a plain dictionary suitable
        for JSON serialization. Mutable containers (e.g., ``metadata``) are
        shallow-copied to prevent unintended mutation of the original record.

        Returns:
            A dictionary containing all tool call record fields with
            JSON-serializable values.

        Example:
            >>> record = ToolCallRecord(
            ...     call_id="tc-1", tool_name="search", arguments={"q": "hello"}
            ... )
            >>> data = record.to_dict()
            >>> data["tool_name"]
            'search'
        """
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
        """Deserialize a ToolCallRecord from a plain dictionary.

        Reconstructs a ``ToolCallRecord`` instance from a dictionary previously
        produced by :meth:`to_dict` or any compatible mapping. Missing optional
        keys fall back to sensible defaults.

        Args:
            data: A dictionary containing tool call record fields. Must include
                ``call_id`` and ``tool_name`` at minimum.

        Returns:
            A new ``ToolCallRecord`` instance populated from *data*.

        Raises:
            KeyError: If required keys (``call_id``, ``tool_name``) are missing.

        Example:
            >>> data = {"call_id": "tc-1", "tool_name": "search", "arguments": {}}
            >>> record = ToolCallRecord.from_dict(data)
            >>> record.tool_name
            'search'
        """
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
    """Record of a single conversational turn.

    Attributes:
        turn_id: Unique identifier for this turn.
        agent_id: Identifier of the agent handling the turn.
        prompt: The user prompt for this turn.
        response_content: The agent's text response content.
        tool_calls: List of tool calls made during this turn.
        started_at: ISO 8601 timestamp when the turn started.
        ended_at: ISO 8601 timestamp when the turn ended.
        status: Turn outcome (success, error, cancelled).
        error: Error message if the turn failed.
        audit_event_ids: References to audit/logging events.
        metadata: Arbitrary metadata for extensibility.
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
        """Serialize the turn record to a JSON-compatible dictionary.

        Nested ``ToolCallRecord`` instances are recursively serialized via
        their own ``to_dict`` methods. Mutable containers are shallow-copied.

        Returns:
            A dictionary containing all turn record fields with
            JSON-serializable values.

        Example:
            >>> turn = TurnRecord(turn_id="t-1", prompt="Hello")
            >>> data = turn.to_dict()
            >>> data["prompt"]
            'Hello'
        """
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
        """Deserialize a TurnRecord from a plain dictionary.

        Reconstructs a ``TurnRecord`` instance, including nested
        ``ToolCallRecord`` objects, from a dictionary previously produced
        by :meth:`to_dict` or any compatible mapping.

        Args:
            data: A dictionary containing turn record fields. Must include
                ``turn_id`` at minimum.

        Returns:
            A new ``TurnRecord`` instance populated from *data*.

        Raises:
            KeyError: If the required key ``turn_id`` is missing.

        Example:
            >>> data = {"turn_id": "t-1", "prompt": "Hello"}
            >>> turn = TurnRecord.from_dict(data)
            >>> turn.prompt
            'Hello'
        """
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
    """Record of a transition between agents.

    Attributes:
        from_agent: ID of the agent being switched away from.
        to_agent: ID of the agent being switched to.
        reason: Human-readable reason for the transition.
        turn_id: ID of the turn during which the transition occurred.
        timestamp: ISO 8601 timestamp of the transition.
    """

    from_agent: str | None
    to_agent: str
    reason: str | None = None
    turn_id: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialize the agent transition record to a JSON-compatible dictionary.

        Returns:
            A dictionary containing all agent transition record fields with
            JSON-serializable values.

        Example:
            >>> transition = AgentTransitionRecord(
            ...     from_agent="agent-a", to_agent="agent-b", reason="escalation"
            ... )
            >>> data = transition.to_dict()
            >>> data["to_agent"]
            'agent-b'
        """
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "reason": self.reason,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> AgentTransitionRecord:
        """Deserialize an AgentTransitionRecord from a plain dictionary.

        Args:
            data: A dictionary containing agent transition fields. Must include
                ``to_agent`` at minimum.

        Returns:
            A new ``AgentTransitionRecord`` instance populated from *data*.

        Raises:
            KeyError: If the required key ``to_agent`` is missing.

        Example:
            >>> data = {"from_agent": "a", "to_agent": "b"}
            >>> rec = AgentTransitionRecord.from_dict(data)
            >>> rec.to_agent
            'b'
        """
        return cls(
            from_agent=data.get("from_agent"),
            to_agent=data["to_agent"],
            reason=data.get("reason"),
            turn_id=data.get("turn_id", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class SessionRecord:
    """Complete record of a session.

    Attributes:
        session_id: Unique identifier for this session.
        workspace_id: Identifier for the workspace this session belongs to.
        created_at: ISO 8601 timestamp when the session was created.
        updated_at: ISO 8601 timestamp when the session was last updated.
        agent_id: Initial/primary agent for the session.
        turns: Ordered list of turn records.
        agent_transitions: List of agent transition events.
        metadata: Arbitrary metadata for extensibility.
    """

    session_id: str
    workspace_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    agent_id: str | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    agent_transitions: list[AgentTransitionRecord] = field(default_factory=list)
    metadata: dict[str, tp.Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialize the session record to a JSON-compatible dictionary.

        Nested ``TurnRecord`` and ``AgentTransitionRecord`` instances are
        recursively serialized via their own ``to_dict`` methods. Mutable
        containers are shallow-copied.

        Returns:
            A dictionary containing all session record fields with
            JSON-serializable values.

        Example:
            >>> session = SessionRecord(session_id="sess-1")
            >>> data = session.to_dict()
            >>> data["session_id"]
            'sess-1'
        """
        return {
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "agent_id": self.agent_id,
            "turns": [t.to_dict() for t in self.turns],
            "agent_transitions": [at.to_dict() for at in self.agent_transitions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> SessionRecord:
        """Deserialize a SessionRecord from a plain dictionary.

        Reconstructs a complete ``SessionRecord`` including all nested
        ``TurnRecord`` and ``AgentTransitionRecord`` objects from a dictionary
        previously produced by :meth:`to_dict` or any compatible mapping.

        Args:
            data: A dictionary containing session record fields. Must include
                ``session_id`` at minimum.

        Returns:
            A new ``SessionRecord`` instance populated from *data*.

        Raises:
            KeyError: If the required key ``session_id`` is missing.

        Example:
            >>> data = {"session_id": "sess-1", "turns": [], "agent_transitions": []}
            >>> session = SessionRecord.from_dict(data)
            >>> session.session_id
            'sess-1'
        """
        return cls(
            session_id=data["session_id"],
            workspace_id=data.get("workspace_id"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            agent_id=data.get("agent_id"),
            turns=[TurnRecord.from_dict(t) for t in data.get("turns", [])],
            agent_transitions=[AgentTransitionRecord.from_dict(at) for at in data.get("agent_transitions", [])],
            metadata=data.get("metadata", {}),
        )
