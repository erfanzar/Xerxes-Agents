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
"""User prompt module for Xerxes.

Exports:
    - UserPromptManager"""

from __future__ import annotations

import asyncio
import typing as tp
import uuid

from .types import PendingUserPrompt, UserPromptOption


class UserPromptManager:
    """User prompt manager."""

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self._pending: PendingUserPrompt | None = None
        self._pending_future: asyncio.Future[dict[str, tp.Any]] | None = None

    def get_pending(self) -> dict[str, tp.Any] | None:
        """Retrieve the pending.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any] | None: OUT: Result of the operation."""

        return self._pending.to_dict() if self._pending is not None else None

    def has_pending(self) -> bool:
        """Check whether pending.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""

        return self._pending is not None

    async def request(
        self,
        question: str,
        *,
        options: list[str] | None = None,
        allow_freeform: bool = True,
        placeholder: str | None = None,
    ) -> dict[str, tp.Any]:
        """Asynchronously Request.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            question (str): IN: question. OUT: Consumed during execution.
            options (list[str] | None, optional): IN: options. Defaults to None. OUT: Consumed during execution.
            allow_freeform (bool, optional): IN: allow freeform. Defaults to True. OUT: Consumed during execution.
            placeholder (str | None, optional): IN: placeholder. Defaults to None. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Answer.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            raw_input (str): IN: raw input. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """Internal helper to require pending.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            PendingUserPrompt: OUT: Result of the operation."""

        if self._pending is None:
            raise ValueError("No pending user question.")
        return self._pending

    @staticmethod
    def _invalid_choice_message(pending: PendingUserPrompt) -> str:
        """Internal helper to invalid choice message.

        Args:
            pending (PendingUserPrompt): IN: pending. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        labels = ", ".join(f"{index + 1}:{option.label}" for index, option in enumerate(pending.options))
        return f"Choose one of the listed options: {labels}"
