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

"""Session replay and timeline inspection.

Provides ReplayView for exploring recorded session data --
filtering by agent, aggregating tool calls, generating timelines,
and producing human-readable markdown summaries.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from .models import AgentTransitionRecord, SessionRecord, ToolCallRecord, TurnRecord


@dataclass
class TimelineEvent:
    """A single event on the session timeline.

    Attributes:
        timestamp: ISO 8601 timestamp of the event.
        event_type: Category string (turn_start, turn_end, tool_call, agent_transition).
        summary: Short human-readable description.
        data: Arbitrary data payload for the event.
    """

    timestamp: str
    event_type: str
    summary: str
    data: dict[str, tp.Any] = field(default_factory=dict)


class ReplayView:
    """Read-only view over a recorded session for inspection and replay.

    Provides methods for querying turns, aggregating tool calls, building
    chronological timelines, filtering by agent, and rendering the session
    as a human-readable markdown summary.

    Attributes:
        session: The underlying ``SessionRecord`` being viewed.
        turns: The list of ``TurnRecord`` instances visible in this view.
            May be a subset of the full session when created via
            :meth:`filter_by_agent`.

    Example:
        >>> from calute.session.models import SessionRecord
        >>> record = SessionRecord(session_id="s1")
        >>> view = ReplayView(session=record)
        >>> len(view.turns)
        0
    """

    def __init__(
        self,
        session: SessionRecord,
        turns: list[TurnRecord] | None = None,
    ) -> None:
        """Initialise the replay view over a session record.

        Args:
            session: The session record to inspect.
            turns: Optional subset of turns to include. When ``None``,
                all turns from *session* are used.
        """
        self.session = session
        self.turns: list[TurnRecord] = turns if turns is not None else list(session.turns)

    def get_turn(self, index_or_id: int | str) -> TurnRecord | None:
        """Retrieve a turn by integer index or turn_id string.

        Args:
            index_or_id: Zero-based index or turn_id.

        Returns:
            The matching TurnRecord, or None if not found.
        """
        if isinstance(index_or_id, int):
            if 0 <= index_or_id < len(self.turns):
                return self.turns[index_or_id]
            return None
        for turn in self.turns:
            if turn.turn_id == index_or_id:
                return turn
        return None

    def get_tool_calls(self) -> list[ToolCallRecord]:
        """Aggregate all tool calls across all turns.

        Returns:
            Flat list of ToolCallRecord from every turn.
        """
        result: list[ToolCallRecord] = []
        for turn in self.turns:
            result.extend(turn.tool_calls)
        return result

    def get_agent_transitions(self) -> list[AgentTransitionRecord]:
        """Return all agent transitions for the session.

        Returns:
            List of AgentTransitionRecord.
        """
        return list(self.session.agent_transitions)

    def get_timeline(self) -> list[TimelineEvent]:
        """Build a chronologically sorted timeline of session events.

        Events include turn starts/ends, tool calls, and agent transitions.

        Returns:
            Chronologically sorted list of TimelineEvent.
        """
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
        """Create a new ReplayView containing only turns from a specific agent.

        Args:
            agent_id: The agent ID to filter by.

        Returns:
            A new ReplayView with filtered turns.
        """
        filtered = [t for t in self.turns if t.agent_id == agent_id]
        return ReplayView(session=self.session, turns=filtered)

    def to_markdown(self) -> str:
        """Render the session as a human-readable markdown summary.

        Returns:
            Markdown string summarizing the session.
        """
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
    """Factory for creating :class:`ReplayView` instances from session records.

    This class acts as the primary entry point for replaying recorded
    sessions. It is stateless and exposes only a static factory method.

    Example:
        >>> from calute.session.models import SessionRecord
        >>> record = SessionRecord(session_id="s1")
        >>> view = SessionReplay.load(record)
        >>> isinstance(view, ReplayView)
        True
    """

    @staticmethod
    def load(session: SessionRecord) -> ReplayView:
        """Create a ReplayView from a SessionRecord.

        This is the recommended way to start inspecting a recorded session.

        Args:
            session: The session record to replay.

        Returns:
            A new :class:`ReplayView` over the entire session, including
            all turns and agent transitions.

        Example:
            >>> view = SessionReplay.load(SessionRecord(session_id="s1"))
            >>> view.session.session_id
            's1'
        """
        return ReplayView(session=session)
