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
"""Read-only views over recorded sessions for replay and inspection.

:class:`ReplayView` slices and reformats a :class:`SessionRecord` without
mutating it: filter by agent, list tool calls, flatten everything to a sorted
:class:`TimelineEvent` stream, or render Markdown. :class:`SessionReplay` is a
thin factory kept for symmetry with the rest of the session API.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from .models import AgentTransitionRecord, SessionRecord, ToolCallRecord, TurnRecord


@dataclass
class TimelineEvent:
    """One row in the flattened replay timeline.

    Attributes:
        timestamp: ISO-8601 string used for chronological sorting.
        event_type: ``"turn_start"``, ``"turn_end"``, ``"tool_call"`` or
            ``"agent_transition"``.
        summary: Short human-readable description for display.
        data: Structured payload (turn/tool ids, agent names).
    """

    timestamp: str
    event_type: str
    summary: str
    data: dict[str, tp.Any] = field(default_factory=dict)


class ReplayView:
    """Filtered, read-only projection of a :class:`SessionRecord`.

    Constructing a view never mutates the source session. ``turns`` defaults
    to a shallow copy of the session's turns; pass an explicit list (e.g.
    from :meth:`filter_by_agent`) to slice the view further.
    """

    def __init__(
        self,
        session: SessionRecord,
        turns: list[TurnRecord] | None = None,
    ) -> None:
        """Wrap ``session`` with an optional turn-list override."""

        self.session = session
        self.turns: list[TurnRecord] = turns if turns is not None else list(session.turns)

    def get_turn(self, index_or_id: int | str) -> TurnRecord | None:
        """Look a turn up by positional index or ``turn_id``."""

        if isinstance(index_or_id, int):
            if 0 <= index_or_id < len(self.turns):
                return self.turns[index_or_id]
            return None
        for turn in self.turns:
            if turn.turn_id == index_or_id:
                return turn
        return None

    def get_tool_calls(self) -> list[ToolCallRecord]:
        """Return every tool call across the view's turns in order."""

        result: list[ToolCallRecord] = []
        for turn in self.turns:
            result.extend(turn.tool_calls)
        return result

    def get_agent_transitions(self) -> list[AgentTransitionRecord]:
        """Return a copy of the source session's agent-transition list."""

        return list(self.session.agent_transitions)

    def get_timeline(self) -> list[TimelineEvent]:
        """Flatten turns, tool calls, and transitions into a sorted timeline."""

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
        """Return a new view containing only turns produced by ``agent_id``."""

        filtered = [t for t in self.turns if t.agent_id == agent_id]
        return ReplayView(session=self.session, turns=filtered)

    def to_markdown(self) -> str:
        """Render the view as a human-readable Markdown report.

        Includes a header with session metadata, agent transitions, and a
        per-turn breakdown with prompt previews, response previews (truncated
        to 200 chars), and tool-call summaries.
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
    """Static factory for :class:`ReplayView` objects."""

    @staticmethod
    def load(session: SessionRecord) -> ReplayView:
        """Wrap ``session`` in a default :class:`ReplayView`."""

        return ReplayView(session=session)
