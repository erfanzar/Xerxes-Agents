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
"""Summarizer module for Xerxes.

Exports:
    - logger
    - SessionSummary
    - SessionSummarizer"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import asdict, dataclass, field

from .models import SessionRecord

logger = logging.getLogger(__name__)


@dataclass
class SessionSummary:
    """Session summary.

    Attributes:
        session_id (str): session id.
        title (str): title.
        synopsis (str): synopsis.
        key_actions (list[str]): key actions.
        outcome (str): outcome.
        turn_count (int): turn count.
        agent_ids (list[str]): agent ids.
        char_count (int): char count."""

    session_id: str
    title: str = ""
    synopsis: str = ""
    key_actions: list[str] = field(default_factory=list)
    outcome: str = "unknown"
    turn_count: int = 0
    agent_ids: list[str] = field(default_factory=list)
    char_count: int = 0

    def to_dict(self) -> dict[str, tp.Any]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return asdict(self)


class SessionSummarizer:
    """Session summarizer."""

    def __init__(
        self,
        llm_client: tp.Callable[[str], str] | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            llm_client (tp.Callable[[str], str] | None, optional): IN: llm client. Defaults to None. OUT: Consumed during execution."""

        self.llm_client = llm_client

    def summarize(self, session: SessionRecord) -> SessionSummary:
        """Summarize.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution.
        Returns:
            SessionSummary: OUT: Result of the operation."""

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
        """Internal helper to derive title.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Internal helper to derive synopsis.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        """Internal helper to collect tools.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

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
        """Internal helper to derive outcome.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        if not session.turns:
            return "unknown"
        statuses = [t.status for t in session.turns]
        if all(s == "success" for s in statuses):
            return "success"
        if all(s != "success" for s in statuses):
            return "failure"
        return "mixed"

    def _distinct_agents(self, session: SessionRecord) -> list[str]:
        """Internal helper to distinct agents.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        seen: set[str] = set()
        out: list[str] = []
        for turn in session.turns:
            if turn.agent_id and turn.agent_id not in seen:
                seen.add(turn.agent_id)
                out.append(turn.agent_id)
        return out

    def _refine_with_llm(self, session: SessionRecord, draft: str) -> str:
        """Internal helper to refine with llm.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            session (SessionRecord): IN: session. OUT: Consumed during execution.
            draft (str): IN: draft. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

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
        if self.llm_client is None:
            return ""
        out = self.llm_client(prompt)
        return out.strip() if isinstance(out, str) else ""


def _truncate(text: str, n: int) -> str:
    """Internal helper to truncate.

    Args:
        text (str): IN: text. OUT: Consumed during execution.
        n (int): IN: n. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


__all__ = ["SessionSummarizer", "SessionSummary"]
