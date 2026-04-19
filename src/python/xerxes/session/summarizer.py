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

"""Session summariser.

Compresses a finished :class:`SessionRecord` into a short structured
summary that the runtime can show in lists ("recent sessions"), feed
back into the prompt ("here's what we discussed last time"), or store
for compliance reporting. A pure heuristic implementation is the
default; an LLM hook is offered for richer summaries.

The output is intentionally compact:

- ``title``: 5-10 word headline derived from the first user prompt.
- ``synopsis``: 1-3 sentence recap.
- ``key_actions``: bulleted list of distinct tools used.
- ``outcome``: ``"success"`` / ``"mixed"`` / ``"failure"`` based on
  per-turn statuses.
- ``tokens``: rough total of all turn-prompt + response chars.
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import asdict, dataclass, field

from .models import SessionRecord

logger = logging.getLogger(__name__)


@dataclass
class SessionSummary:
    """Compressed representation of a session.

    Attributes:
        session_id: Originating session.
        title: Short headline (5-10 words).
        synopsis: 1-3 sentence overview.
        key_actions: Distinct tool names used, in invocation order.
        outcome: ``"success"`` / ``"mixed"`` / ``"failure"``.
        turn_count: Number of turns in the session.
        agent_ids: Distinct agents that participated.
        char_count: Total characters across prompts + responses.
    """

    session_id: str
    title: str = ""
    synopsis: str = ""
    key_actions: list[str] = field(default_factory=list)
    outcome: str = "unknown"
    turn_count: int = 0
    agent_ids: list[str] = field(default_factory=list)
    char_count: int = 0

    def to_dict(self) -> dict[str, tp.Any]:
        """Return the summary as a plain ``dict`` suitable for JSON serialisation."""
        return asdict(self)


class SessionSummarizer:
    """Heuristic session summariser with optional LLM enhancement.

    Example:
        >>> s = SessionSummarizer()
        >>> summary = s.summarize(session_record)
        >>> print(summary.title, summary.outcome)
    """

    def __init__(
        self,
        llm_client: tp.Callable[[str], str] | None = None,
    ) -> None:
        """Initialise the summariser.

        Args:
            llm_client: Optional callable ``(prompt) -> str`` that
                rewrites the heuristic synopsis into a richer paragraph.
        """
        self.llm_client = llm_client

    def summarize(self, session: SessionRecord) -> SessionSummary:
        """Produce a :class:`SessionSummary` for *session*."""
        title = self._derive_title(session)
        synopsis = self._derive_synopsis(session)
        key_actions = self._collect_tools(session)
        outcome = self._derive_outcome(session)
        agents = self._distinct_agents(session)
        char_count = sum(len(t.prompt or "") + len(t.response_content or "") for t in session.turns)
        if self.llm_client is not None and session.turns:
            try:
                synopsis = self._refine_with_llm(session, synopsis) or synopsis
            except Exception:
                logger.debug("LLM refinement failed", exc_info=True)
        return SessionSummary(
            session_id=session.session_id,
            title=title,
            synopsis=synopsis,
            key_actions=key_actions,
            outcome=outcome,
            turn_count=len(session.turns),
            agent_ids=agents,
            char_count=char_count,
        )

    def _derive_title(self, session: SessionRecord) -> str:
        """Build a short headline from the session's first user prompt.

        Uses the first 10 whitespace-separated words (with an ellipsis
        when more than 12 exist), trims to 80 characters otherwise, and
        falls back to ``"Session <id-prefix>"`` when the session or its
        opening prompt is empty.

        Args:
            session: The :class:`SessionRecord` being summarised.

        Returns:
            A 5-10 word title string.
        """
        if not session.turns:
            return f"Session {session.session_id[:8]}"
        first = session.turns[0]
        prompt = (first.prompt or "").strip()
        if not prompt:
            return f"Session {session.session_id[:8]}"
        words = prompt.split()
        if len(words) > 12:
            return " ".join(words[:10]) + "…"
        return prompt[:80]

    def _derive_synopsis(self, session: SessionRecord) -> str:
        """Compose a 1-3 sentence recap from first prompt, tool usage, and last reply.

        Quotes the opening user prompt (truncated to 120 chars), reports
        the aggregate tool-call count across all turns, and quotes the
        last non-empty agent response (truncated to 200 chars).

        Args:
            session: The :class:`SessionRecord` being summarised.

        Returns:
            A whitespace-joined synopsis string; ``"Empty session."``
            when no turns exist.
        """
        if not session.turns:
            return "Empty session."
        first_prompt = (session.turns[0].prompt or "").strip()
        last_response = ""
        for t in reversed(session.turns):
            if t.response_content:
                last_response = t.response_content.strip()
                break
        n_tools = sum(len(t.tool_calls) for t in session.turns)
        sentences: list[str] = []
        if first_prompt:
            sentences.append(f'User asked: "{_truncate(first_prompt, 120)}".')
        if n_tools:
            sentences.append(f"Agent used {n_tools} tool call(s) across {len(session.turns)} turn(s).")
        else:
            sentences.append(f"Agent answered in {len(session.turns)} turn(s) without tools.")
        if last_response:
            sentences.append(f'Final answer: "{_truncate(last_response, 200)}".')
        return " ".join(sentences)

    def _collect_tools(self, session: SessionRecord) -> list[str]:
        """Return distinct tool names used, preserving first-invocation order.

        Scans each turn's ``tool_calls`` list, reading ``tool_name`` (or
        the legacy ``name`` attribute) and de-duplicating via a seen set.

        Args:
            session: The :class:`SessionRecord` being summarised.

        Returns:
            A list of tool-name strings in order of first appearance.
        """
        seen: set[str] = set()
        out: list[str] = []
        for turn in session.turns:
            for call in turn.tool_calls:
                name = getattr(call, "tool_name", None) or getattr(call, "name", "")
                if name and name not in seen:
                    seen.add(name)
                    out.append(name)
        return out

    def _derive_outcome(self, session: SessionRecord) -> str:
        """Collapse per-turn ``status`` values into a single outcome label.

        Returns ``"success"`` only when every turn succeeded,
        ``"failure"`` when no turn succeeded, ``"mixed"`` when there is
        at least one of each, and ``"unknown"`` for empty sessions.

        Args:
            session: The :class:`SessionRecord` being summarised.

        Returns:
            One of ``"success"``, ``"failure"``, ``"mixed"``, or ``"unknown"``.
        """
        if not session.turns:
            return "unknown"
        statuses = [t.status for t in session.turns]
        if all(s == "success" for s in statuses):
            return "success"
        if all(s != "success" for s in statuses):
            return "failure"
        return "mixed"

    def _distinct_agents(self, session: SessionRecord) -> list[str]:
        """Return unique ``agent_id`` values in first-seen order.

        Args:
            session: The :class:`SessionRecord` being summarised.

        Returns:
            A de-duplicated list of agent identifiers, skipping falsy values.
        """
        seen: set[str] = set()
        out: list[str] = []
        for turn in session.turns:
            if turn.agent_id and turn.agent_id not in seen:
                seen.add(turn.agent_id)
                out.append(turn.agent_id)
        return out

    def _refine_with_llm(self, session: SessionRecord, draft: str) -> str:
        """Ask the optional LLM client to rewrite *draft* into a cleaner synopsis.

        Builds a prompt that includes the heuristic draft plus the last
        three turns (user prompt and agent response, each truncated to
        120 chars) and returns the stripped LLM output.

        Args:
            session: The :class:`SessionRecord` being summarised.
            draft: Heuristic synopsis to rewrite.

        Returns:
            The LLM-rewritten synopsis, or an empty string if the client
            returned something non-string.
        """
        prompt = (
            "Rewrite this session synopsis as 1-3 short, neutral sentences. "
            "Preserve all factual claims; do not invent details.\n\n"
            f"Draft:\n{draft}\n\n"
            "Recent turns (newest last):\n"
            + "\n".join(
                f"- USER: {(t.prompt or '')[:120]} | AGENT: {(t.response_content or '')[:120]}"
                for t in session.turns[-3:]
            )
        )
        out = self.llm_client(prompt)
        return out.strip() if isinstance(out, str) else ""


def _truncate(text: str, n: int) -> str:
    """Collapse whitespace and cap *text* to at most *n* characters.

    Normalises any run of whitespace to a single space. If the result is
    longer than *n*, trims to ``n - 1`` characters and appends an ellipsis.

    Args:
        text: Input string to shorten.
        n: Maximum output length (including the ellipsis character).

    Returns:
        A whitespace-normalised, length-bounded string.
    """
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


__all__ = ["SessionSummarizer", "SessionSummary"]
