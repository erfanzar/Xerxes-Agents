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
"""Models module for Xerxes.

Exports:
    - SessionId
    - WorkspaceId
    - ToolCallRecord
    - TurnRecord
    - AgentTransitionRecord
    - SessionRecord"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

SessionId = tp.NewType("SessionId", str)

WorkspaceId = tp.NewType("WorkspaceId", str)


@dataclass
class ToolCallRecord:
    """Tool call record.

    Attributes:
        call_id (str): call id.
        tool_name (str): tool name.
        arguments (dict[str, tp.Any]): arguments.
        result (tp.Any): result.
        status (str): status.
        error (str | None): error.
        duration_ms (float | None): duration ms.
        sandbox_context (str | None): sandbox context.
        metadata (dict[str, tp.Any]): metadata."""

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
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """From dict.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            data (dict[str, tp.Any]): IN: data. OUT: Consumed during execution.
        Returns:
            ToolCallRecord: OUT: Result of the operation."""

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
    """Turn record.

    Attributes:
        turn_id (str): turn id.
        agent_id (str | None): agent id.
        prompt (str): prompt.
        response_content (str | None): response content.
        tool_calls (list[ToolCallRecord]): tool calls.
        started_at (str): started at.
        ended_at (str | None): ended at.
        status (str): status.
        error (str | None): error.
        audit_event_ids (list[str]): audit event ids.
        metadata (dict[str, tp.Any]): metadata."""

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
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """From dict.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            data (dict[str, tp.Any]): IN: data. OUT: Consumed during execution.
        Returns:
            TurnRecord: OUT: Result of the operation."""

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
    """Agent transition record.

    Attributes:
        from_agent (str | None): from agent.
        to_agent (str): to agent.
        reason (str | None): reason.
        turn_id (str): turn id.
        timestamp (str): timestamp."""

    from_agent: str | None
    to_agent: str
    reason: str | None = None
    turn_id: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, tp.Any]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "reason": self.reason,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> AgentTransitionRecord:
        """From dict.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            data (dict[str, tp.Any]): IN: data. OUT: Consumed during execution.
        Returns:
            AgentTransitionRecord: OUT: Result of the operation."""

        return cls(
            from_agent=data.get("from_agent"),
            to_agent=data["to_agent"],
            reason=data.get("reason"),
            turn_id=data.get("turn_id", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class SessionRecord:
    """Session record.

    Attributes:
        session_id (str): session id.
        workspace_id (str | None): workspace id.
        created_at (str): created at.
        updated_at (str): updated at.
        agent_id (str | None): agent id.
        turns (list[TurnRecord]): turns.
        agent_transitions (list[AgentTransitionRecord]): agent transitions.
        metadata (dict[str, tp.Any]): metadata."""

    session_id: str
    workspace_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    agent_id: str | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    agent_transitions: list[AgentTransitionRecord] = field(default_factory=list)
    metadata: dict[str, tp.Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, tp.Any]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """From dict.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            data (dict[str, tp.Any]): IN: data. OUT: Consumed during execution.
        Returns:
            SessionRecord: OUT: Result of the operation."""

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
