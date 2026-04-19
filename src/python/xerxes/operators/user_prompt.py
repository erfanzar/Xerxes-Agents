# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive user-question manager for operator tooling.

Provides :class:`UserPromptManager`, which manages one live user
clarification question at a time.  The ``ask_user`` operator tool
creates a :class:`~xerxes.operators.types.PendingUserPrompt` via
:meth:`UserPromptManager.request` and awaits a future that the TUI
resolves by calling :meth:`UserPromptManager.answer`.
"""

from __future__ import annotations

import asyncio
import typing as tp
import uuid

from .types import PendingUserPrompt, UserPromptOption


class UserPromptManager:
    """Manage one live user clarification question at a time.

    At most one question can be pending.  The ``ask_user`` tool calls
    :meth:`request`, which creates a :class:`PendingUserPrompt` and
    returns a future.  The TUI polls :meth:`get_pending` to discover
    the question and calls :meth:`answer` to resolve the future.

    Attributes:
        _pending: The currently active :class:`PendingUserPrompt`, or
            ``None`` when no question is outstanding.
        _pending_future: The :class:`asyncio.Future` that will be
            resolved with the answer dictionary when the user responds.
    """

    def __init__(self) -> None:
        """Initialise the manager with no pending question."""
        self._pending: PendingUserPrompt | None = None
        self._pending_future: asyncio.Future[dict[str, tp.Any]] | None = None

    def get_pending(self) -> dict[str, tp.Any] | None:
        """Return the current pending question, if any.

        Returns:
            A serialised dictionary of the pending prompt (via
            :meth:`PendingUserPrompt.to_dict`), or ``None`` when no
            question is outstanding.
        """
        return self._pending.to_dict() if self._pending is not None else None

    def has_pending(self) -> bool:
        """Report whether the runtime is currently waiting on the user.

        Returns:
            ``True`` if a question is pending, ``False`` otherwise.
        """
        return self._pending is not None

    async def request(
        self,
        question: str,
        *,
        options: list[str] | None = None,
        allow_freeform: bool = True,
        placeholder: str | None = None,
    ) -> dict[str, tp.Any]:
        """Create a pending question and wait until the UI submits an answer.

        Builds a :class:`PendingUserPrompt`, stores it as the current
        pending prompt, creates an :class:`asyncio.Future`, and awaits
        it.  The future is resolved when :meth:`answer` is called by
        the TUI layer.

        Args:
            question: The question text to display to the user.
            options: Optional list of string choices presented as
                numbered options.  Each string is converted into a
                :class:`~xerxes.operators.types.UserPromptOption`.
            allow_freeform: When ``True``, the user may type a custom
                answer in addition to (or instead of) the listed
                options.
            placeholder: Optional hint text shown in the input field
                while waiting for a response.

        Returns:
            The resolved answer dictionary containing fields such as
            ``request_id``, ``question``, ``answer``, ``raw_input``,
            ``selected_option``, and ``used_freeform``.

        Raises:
            RuntimeError: If another question is already pending.
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
        """Resolve the pending question from typed UI input.

        Matches the user's raw input against the pending question's
        options (by index or label/value text) and resolves the
        internal future so that the awaiting ``ask_user`` tool call
        receives the result.

        Args:
            raw_input: The raw string entered by the user in the TUI.
                May be a numeric index (1-based) referring to a listed
                option, the full option label or value text, or
                arbitrary freeform text when allowed.

        Returns:
            A dictionary containing:

            - ``request_id``: The prompt request identifier.
            - ``question``: The original question text.
            - ``answer``: The normalised answer value.
            - ``raw_input``: The cleaned user input.
            - ``selected_option``: The matched option dictionary, or
              ``None`` if no listed option was selected.
            - ``used_freeform``: ``True`` when the answer did not match
              any listed option.

        Raises:
            ValueError: If the input is empty, or if freeform is
                disallowed and the input does not match any option.
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
        """Return the current pending prompt or raise.

        Returns:
            The active :class:`PendingUserPrompt` instance.

        Raises:
            ValueError: If no question is currently pending.
        """
        if self._pending is None:
            raise ValueError("No pending user question.")
        return self._pending

    @staticmethod
    def _invalid_choice_message(pending: PendingUserPrompt) -> str:
        """Build an error message listing valid option choices.

        Args:
            pending: The :class:`PendingUserPrompt` whose options
                should be listed.

        Returns:
            A human-readable error string enumerating the valid
            numbered choices.
        """
        labels = ", ".join(f"{index + 1}:{option.label}" for index, option in enumerate(pending.options))
        return f"Choose one of the listed options: {labels}"
