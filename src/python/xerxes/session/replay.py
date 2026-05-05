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
"""Replay module for Xerxes.

Exports:
    - TimelineEvent
    - ReplayView
    - SessionReplay"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from .models import AgentTransitionRecord, SessionRecord, ToolCallRecord, TurnRecord


@dataclass
class TimelineEvent:
    """Timeline event.

    Attributes:
        timestamp (str): timestamp.
        event_type (str): event type.
        summary (str): summary.
        data (dict[str, tp.Any]): data."""

    timestamp: str
    event_type: str
    summary: str
    data: dict[str, tp.Any] = field(default_factory=dict)


class ReplayView:
    """Replay view."""

    def __init__(
        self,
        session: SessionRecord,
        turns: list[TurnRecord] | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution.
            turns (list[TurnRecord] | None, optional): IN: turns. Defaults to None. OUT: Consumed during execution."""

        self.session = session
        self.turns: list[TurnRecord] = turns if turns is not None else list(session.turns)

    def get_turn(self, index_or_id: int | str) -> TurnRecord | None:
        """Retrieve the turn.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            index_or_id (int | str): IN: index or id. OUT: Consumed during execution.
        Returns:
            TurnRecord | None: OUT: Result of the operation."""

        if isinstance(index_or_id, int):
            if 0 <= index_or_id < len(self.turns):
                return self.turns[index_or_id]
            return None
        for turn in self.turns:
            if turn.turn_id == index_or_id:
                return turn
        return None

    def get_tool_calls(self) -> list[ToolCallRecord]:
        """Retrieve the tool calls.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[ToolCallRecord]: OUT: Result of the operation."""

        result: list[ToolCallRecord] = []
        for turn in self.turns:
            result.extend(turn.tool_calls)
        return result

    def get_agent_transitions(self) -> list[AgentTransitionRecord]:
        """Retrieve the agent transitions.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[AgentTransitionRecord]: OUT: Result of the operation."""

        return list(self.session.agent_transitions)

    def get_timeline(self) -> list[TimelineEvent]:
        """Retrieve the timeline.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[TimelineEvent]: OUT: Result of the operation."""

        events: list[TimelineEvent] = []

        for turn in self.turns:
            if turn.started_at:
                events.append(
                    TimelineEvent(
                        timestamp=turn.started_at,
                        event_type="turn_start",
                        summary=f"Turn {turn.turn_id} started (agent={turn.agent_id})",
                        data={"turn_id": turn.turn_id, "agent_id": turn.agent_id},
                    )
                )
            for tc in turn.tool_calls:
                events.append(
                    TimelineEvent(
                        timestamp=turn.started_at or "",
                        event_type="tool_call",
                        summary=f"Tool call: {tc.tool_name} ({tc.status})",
                        data={
                            "call_id": tc.call_id,
                            "tool_name": tc.tool_name,
                            "status": tc.status,
                        },
                    )
                )
            if turn.ended_at:
                events.append(
                    TimelineEvent(
                        timestamp=turn.ended_at,
                        event_type="turn_end",
                        summary=f"Turn {turn.turn_id} ended ({turn.status})",
                        data={"turn_id": turn.turn_id, "status": turn.status},
                    )
                )

        for at in self.session.agent_transitions:
            events.append(
                TimelineEvent(
                    timestamp=at.timestamp,
                    event_type="agent_transition",
                    summary=f"Agent switch: {at.from_agent} -> {at.to_agent}",
                    data={
                        "from_agent": at.from_agent,
                        "to_agent": at.to_agent,
                        "reason": at.reason,
                    },
                )
            )

        events.sort(key=lambda e: e.timestamp)
        return events

    def filter_by_agent(self, agent_id: str) -> ReplayView:
        """Filter by agent.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            agent_id (str): IN: agent id. OUT: Consumed during execution.
        Returns:
            ReplayView: OUT: Result of the operation."""

        filtered = [t for t in self.turns if t.agent_id == agent_id]
        return ReplayView(session=self.session, turns=filtered)

    def to_markdown(self) -> str:
        """To markdown.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        lines: list[str] = []
        lines.append(f"# Session {self.session.session_id}")
        lines.append("")
        lines.append(f"- **Workspace:** {self.session.workspace_id or 'N/A'}")
        lines.append(f"- **Created:** {self.session.created_at}")
        lines.append(f"- **Updated:** {self.session.updated_at}")
        lines.append(f"- **Initial Agent:** {self.session.agent_id or 'N/A'}")
        lines.append(f"- **Turns:** {len(self.turns)}")
        all_tc = self.get_tool_calls()
        lines.append(f"- **Tool Calls:** {len(all_tc)}")
        lines.append("")

        if self.session.agent_transitions:
            lines.append("## Agent Transitions")
            lines.append("")
            for at in self.session.agent_transitions:
                lines.append(
                    f"- [{at.timestamp}] {at.from_agent} -> {at.to_agent}" + (f" ({at.reason})" if at.reason else "")
                )
            lines.append("")

        lines.append("## Turns")
        lines.append("")
        for i, turn in enumerate(self.turns):
            lines.append(f"### Turn {i + 1}: {turn.turn_id}")
            lines.append("")
            lines.append(f"- **Agent:** {turn.agent_id or 'N/A'}")
            lines.append(f"- **Status:** {turn.status}")
            lines.append(f"- **Started:** {turn.started_at}")
            lines.append(f"- **Ended:** {turn.ended_at or 'N/A'}")
            if turn.prompt:
                lines.append(f"- **Prompt:** {turn.prompt}")
            if turn.response_content:
                content_preview = turn.response_content[:200]
                if len(turn.response_content) > 200:
                    content_preview += "..."
                lines.append(f"- **Response:** {content_preview}")
            if turn.error:
                lines.append(f"- **Error:** {turn.error}")
            if turn.tool_calls:
                lines.append(f"- **Tool Calls ({len(turn.tool_calls)}):**")
                for tc in turn.tool_calls:
                    status_str = f" [{tc.status}]"
                    dur_str = f" ({tc.duration_ms:.0f}ms)" if tc.duration_ms else ""
                    lines.append(f"  - `{tc.tool_name}`{status_str}{dur_str}")
            lines.append("")

        return "\n".join(lines)


class SessionReplay:
    """Session replay."""

    @staticmethod
    def load(session: SessionRecord) -> ReplayView:
        """Load.

        Args:
            session (SessionRecord): IN: session. OUT: Consumed during execution.
        Returns:
            ReplayView: OUT: Result of the operation."""

        return ReplayView(session=session)
