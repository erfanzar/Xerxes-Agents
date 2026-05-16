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
"""``clarify`` tool — interactive clarification protocol for agent turns.

Defines the abstract :class:`InteractiveAsker` contract that the TUI and
channel gateways implement to surface a clarifying prompt to the human user.
The agent calls :func:`clarify`; the UX layer presents the question and feeds
either a selected option, a free-text answer, or a skip back via
:class:`ClarifyResult`. Tests stub interaction via :class:`StaticAsker`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ClarifyResult:
    """Outcome of a single :meth:`InteractiveAsker.ask` interaction.

    Attributes:
        answered: True when the user provided some response (text or
            option). Mutually exclusive with :attr:`skipped`.
        answer: Free-text answer or the text of the selected option.
        selected_index: Index into the ``options`` list when the user
            picked an option; ``None`` for free-text or skip responses.
        skipped: True when the user declined to answer.
    """

    answered: bool
    answer: str = ""
    selected_index: int | None = None
    skipped: bool = False


class InteractiveAsker(ABC):
    """Abstract UX-layer interface for surfacing clarification prompts."""

    @abstractmethod
    def ask(self, question: str, options: list[str], allow_freeform: bool) -> ClarifyResult:
        """Display ``question`` to the user and return their response.

        Args:
            question: Prompt text shown to the user.
            options: Optional multiple-choice list; may be empty.
            allow_freeform: When True the user may type a custom answer in
                addition to picking an option.
        """


class StaticAsker(InteractiveAsker):
    """Deterministic asker that returns a pre-canned answer; used in tests."""

    def __init__(self, *, answer: str = "", index: int | None = None, skip: bool = False) -> None:
        """Pre-configure the response that :meth:`ask` will return.

        Args:
            answer: Free-text answer to surface when ``index`` is unset.
            index: Index into the ``options`` list to "select" when valid.
            skip: When True the asker reports the user skipped the prompt.
        """
        self._answer = answer
        self._index = index
        self._skip = skip

    def ask(self, question: str, options: list[str], allow_freeform: bool) -> ClarifyResult:
        """Return the pre-configured :class:`ClarifyResult` ignoring the prompt."""
        if self._skip:
            return ClarifyResult(answered=False, skipped=True)
        if self._index is not None and 0 <= self._index < len(options):
            return ClarifyResult(answered=True, answer=options[self._index], selected_index=self._index)
        return ClarifyResult(answered=True, answer=self._answer)


def clarify(
    *,
    question: str,
    options: list[str] | None = None,
    allow_freeform: bool = True,
    asker: InteractiveAsker | None = None,
) -> dict[str, Any]:
    """Surface a clarification prompt and return the user's structured reply.

    Args:
        question: Non-empty prompt text; whitespace-only inputs are rejected.
        options: Multiple-choice list to display; may be empty when free-form
            input is allowed.
        allow_freeform: When True, the user may type a free-text reply.
        asker: UX-layer asker responsible for presenting the prompt. When
            ``None``, the call returns ``needs_ui=True`` so the agent knows
            no asker is wired up yet.

    Returns:
        Dict with ``ok`` plus, when an asker is configured, ``answered``,
        ``answer``, ``selected_index``, and ``skipped`` mirroring the
        :class:`ClarifyResult`. Validation failures return
        ``{"ok": False, "error": ...}``.
    """
    if not question.strip():
        return {"ok": False, "error": "question required"}
    opts = list(options or [])
    if not opts and not allow_freeform:
        return {"ok": False, "error": "either options must be supplied or allow_freeform=True"}
    if asker is None:
        return {"ok": True, "answered": False, "needs_ui": True}
    result = asker.ask(question, opts, allow_freeform)
    return {
        "ok": True,
        "answered": result.answered,
        "answer": result.answer,
        "selected_index": result.selected_index,
        "skipped": result.skipped,
    }


__all__ = ["ClarifyResult", "InteractiveAsker", "StaticAsker", "clarify"]
