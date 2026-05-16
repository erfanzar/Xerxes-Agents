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
"""Cheap LLM client for auxiliary work (summaries, titles, memory ops).

A deliberately narrow client for tasks that shouldn't burn the main reasoning
model. Defaults to a small, fast model (``claude-haiku-4-5``) and delegates
the actual LLM invocation to an injected callable so production code can plug
in real provider clients while tests can pass in a deterministic fake.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Signature for the injected backend. Accepts a list of OpenAI-shaped
# messages and a max_tokens budget; returns the model's text output.
LLMCallable = Callable[[list[dict[str, Any]], int, str], str]
"""Backend signature: ``(messages, max_tokens, model) -> response_text``."""


@dataclass
class AuxiliaryRequest:
    """One auxiliary-LLM call request.

    Attributes:
        purpose: What the call is for (``"summarize"``, ``"title"`` etc.).
        messages: OpenAI-shaped chat messages to send to the backend.
        max_tokens: Response budget enforced by the backend.
        temperature: Sampling temperature; aux work is usually deterministic.
        metadata: Free-form bag for caller use; not interpreted here.
    """

    purpose: str
    messages: list[dict[str, Any]]
    max_tokens: int = 1_000
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuxiliaryResponse:
    """Outcome of an auxiliary-LLM call.

    Attributes:
        text: Raw text returned by the backend.
        purpose: Echoes :attr:`AuxiliaryRequest.purpose`.
        model: Model identifier that actually ran the call.
        duration_ms: Wall-clock duration in milliseconds.
        request_tokens: Input token count reported by the backend (when known).
        response_tokens: Output token count reported by the backend (when known).
    """

    text: str
    purpose: str
    model: str
    duration_ms: float
    request_tokens: int = 0
    response_tokens: int = 0


class AuxiliaryClient:
    """Send focused, cheap LLM requests through a configured small model.

    Convenience methods (:meth:`summarize`, :meth:`title`, :meth:`extract`)
    cover the common cases; arbitrary requests go through :meth:`call`. The
    actual LLM dispatch is delegated to the ``backend`` callable injected at
    construction time.
    """

    def __init__(
        self,
        backend: LLMCallable,
        *,
        model: str = "claude-haiku-4-5",
        default_max_tokens: int = 1_000,
    ) -> None:
        """Create a client bound to ``backend`` and a default cheap model."""
        self._backend = backend
        self._model = model
        self._default_max_tokens = default_max_tokens

    @property
    def model(self) -> str:
        """Identifier of the small model this client targets."""
        return self._model

    def call(self, request: AuxiliaryRequest) -> AuxiliaryResponse:
        """Dispatch ``request`` through the backend and wrap the result."""
        start = time.monotonic()
        try:
            text = self._backend(request.messages, request.max_tokens, self._model)
        except Exception as exc:
            logger.warning("AuxiliaryClient backend raised %s: %s", type(exc).__name__, exc)
            raise
        duration_ms = (time.monotonic() - start) * 1000.0
        return AuxiliaryResponse(
            text=text,
            purpose=request.purpose,
            model=self._model,
            duration_ms=duration_ms,
        )

    # ---------------------------- canned shapes

    def summarize(self, messages: list[dict[str, Any]], *, budget_tokens: int | None = None) -> str:
        """Produce a concise summary of ``messages`` suitable for context compaction.

        Wraps the messages in a system instruction so the model treats the
        request as a summarisation, not a continuation; the budget defaults
        to :attr:`_default_max_tokens` when not provided.
        """

        budget = budget_tokens or self._default_max_tokens
        system = (
            "You are a context-compaction assistant. Summarize the following conversation "
            "concisely, preserving facts the model will need later. Capture decisions, "
            "user preferences, error states, and partial work. Do NOT continue the "
            "conversation. Output plain prose."
        )
        # Stringify messages into a single prompt.
        rendered_lines: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            rendered_lines.append(f"[{role}] {content}")
        user_text = "\n".join(rendered_lines)
        req = AuxiliaryRequest(
            purpose="summarize",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            max_tokens=budget,
        )
        return self.call(req).text

    def title(self, first_turns: list[dict[str, Any]]) -> str:
        """Generate a short (8-word) descriptive title for a session."""
        system = "Generate a short, descriptive title (max 8 words) for this conversation. Output only the title."
        rendered_lines: list[str] = []
        for m in first_turns:
            role = m.get("role", "user")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            rendered_lines.append(f"[{role}] {content}")
        user_text = "\n".join(rendered_lines)
        req = AuxiliaryRequest(
            purpose="title",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            max_tokens=64,
        )
        text = self.call(req).text.strip().strip("\"'`")
        return text

    def extract(self, text: str, *, instruction: str) -> str:
        """Run a single-shot extraction over ``text`` using ``instruction`` as the system prompt."""
        req = AuxiliaryRequest(
            purpose="extract",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            max_tokens=self._default_max_tokens,
        )
        return self.call(req).text


__all__ = ["AuxiliaryClient", "AuxiliaryRequest", "AuxiliaryResponse", "LLMCallable"]
