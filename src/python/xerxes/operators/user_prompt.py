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
"""Coordinator for blocking ``ask_user`` calls during a turn.

Holds at most one in-flight :class:`PendingUserPrompt` and its awaiting
future. The TUI polls :meth:`get_pending`, displays the prompt, then calls
:meth:`answer` with the user's reply to resolve the awaiting coroutine.
"""

from __future__ import annotations

import asyncio
import typing as tp
import uuid

from .types import PendingUserPrompt, UserPromptOption


class UserPromptManager:
    """Tracks the single outstanding user-prompt request for a session.

    The manager enforces "one question at a time" semantics: a second
    :meth:`request` call while a prompt is still pending raises rather than
    silently dropping the first prompt. The waiting future is resolved by
    :meth:`answer`, which performs option matching and freeform validation
    against the original :class:`PendingUserPrompt`.
    """

    def __init__(self) -> None:
        """Start with no pending prompt and no awaiting future."""

        self._pending: PendingUserPrompt | None = None
        self._pending_future: asyncio.Future[dict[str, tp.Any]] | None = None

    def get_pending(self) -> dict[str, tp.Any] | None:
        """Return the wire dict for the pending prompt, or ``None`` if idle."""

        return self._pending.to_dict() if self._pending is not None else None

    def has_pending(self) -> bool:
        """Return ``True`` while a user prompt is awaiting an answer."""

        return self._pending is not None

    async def request(
        self,
        question: str,
        *,
        options: list[str] | None = None,
        allow_freeform: bool = True,
        placeholder: str | None = None,
    ) -> dict[str, tp.Any]:
        """Queue a question and await the user's reply.

        Blocks the calling tool coroutine until :meth:`answer` resolves the
        underlying future. Both the pending state and the future are cleared
        before returning, regardless of success or cancellation.

        Args:
            question: Text shown to the user (whitespace-stripped).
            options: Optional list of choice strings; empty entries are
                dropped and each non-empty entry becomes a
                :class:`UserPromptOption`.
            allow_freeform: Permit the user to type a custom answer in
                addition to (or instead of) selecting an option.
            placeholder: Optional placeholder for the input field.

        Raises:
            RuntimeError: If another prompt is already pending.
        """

        if self._pending is not None:
            raise RuntimeError("Another user question is already pending")

        loop = asyncio.get_running_loop()
        self._pending = PendingUserPrompt(
            request_id=f"user_prompt_{uuid.uuid4().hex[:10]}",
            question=question.strip(),
            options=[
                UserPromptOption(label=option.strip(), value=option.strip())
                for option in (options or [])
                if option.strip()
            ],
            allow_freeform=allow_freeform,
            placeholder=placeholder,
        )
        self._pending_future = loop.create_future()

        try:
            return await self._pending_future
        finally:
            self._pending = None
            self._pending_future = None

    def answer(self, raw_input: str) -> dict[str, tp.Any]:
        """Resolve the pending prompt with the user's reply.

        Numeric input is interpreted as a 1-based option index; text input
        is matched against option labels and values (case-insensitive). If
        the prompt disallows freeform answers and no option matches, this
        method raises ``ValueError`` and leaves the prompt pending so the
        TUI can re-ask.

        Raises:
            ValueError: When the answer is empty or violates the prompt's
                constraints.
        """

        pending = self._require_pending()
        cleaned = raw_input.strip()
        if not cleaned:
            raise ValueError("Answer cannot be empty.")

        selected_option: dict[str, str] | None = None
        answer_value = cleaned

        if pending.options:
            if cleaned.isdigit():
                index = int(cleaned) - 1
                if 0 <= index < len(pending.options):
                    option = pending.options[index]
                    selected_option = option.to_dict()
                    answer_value = selected_option["value"]
                elif not pending.allow_freeform:
                    raise ValueError(self._invalid_choice_message(pending))
            else:
                for option in pending.options:
                    normalized = option.to_dict()
                    if cleaned.casefold() in {normalized["label"].casefold(), normalized["value"].casefold()}:
                        selected_option = normalized
                        answer_value = normalized["value"]
                        break
                if selected_option is None and not pending.allow_freeform:
                    raise ValueError(self._invalid_choice_message(pending))
        elif not pending.allow_freeform:
            raise ValueError("This question requires choosing one of the provided options.")

        result = {
            "request_id": pending.request_id,
            "question": pending.question,
            "answer": answer_value,
            "raw_input": cleaned,
            "selected_option": selected_option,
            "used_freeform": selected_option is None,
        }

        if self._pending_future is not None and not self._pending_future.done():
            self._pending_future.set_result(result)
        return result

    def _require_pending(self) -> PendingUserPrompt:
        """Return the pending prompt or raise if none is awaiting."""

        if self._pending is None:
            raise ValueError("No pending user question.")
        return self._pending

    @staticmethod
    def _invalid_choice_message(pending: PendingUserPrompt) -> str:
        """Format the human-readable error listing the valid choices."""

        labels = ", ".join(f"{index + 1}:{option.label}" for index, option in enumerate(pending.options))
        return f"Choose one of the listed options: {labels}"
