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
"""Shared data structures for the operator subsystem.

Defines the lightweight, JSON-serialisable value objects exchanged between
operator managers (PTY, browser, plan, user-prompt, subagent) and the
streaming runtime. Each dataclass owns a single piece of operator state and
exposes ``to_dict`` so it can be embedded in wire events.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from datetime import UTC, datetime

from PIL import Image


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(UTC).isoformat()


@dataclass
class ImageInspectionResult:
    """In-memory record produced when the model loads an image for inspection.

    Attributes:
        path: Filesystem path the image was loaded from.
        format: PIL-detected format name (``"PNG"``, ``"JPEG"``, ...), or
            ``None`` when PIL cannot determine it.
        mode: PIL colour mode (e.g. ``"RGB"``, ``"RGBA"``, ``"L"``).
        width: Pixel width of the image.
        height: Pixel height of the image.
        image: The decoded ``PIL.Image.Image`` instance held in memory.
        detail: Caller-supplied detail level forwarded to the multimodal
            model (``"auto"``, ``"low"`` or ``"high"``).
    """

    path: str
    format: str | None
    mode: str
    width: int
    height: int
    image: Image.Image
    detail: str = "auto"

    def summary(self) -> str:
        """Return a single-line human-readable summary of the inspected image."""

        return (
            f"Image loaded from {self.path} "
            f"({self.width}x{self.height}, mode={self.mode}, format={self.format or 'unknown'})"
        )

    def tool_metadata(self) -> dict[str, tp.Any]:
        """Return the JSON-safe subset of the result for tool envelopes."""

        return {
            "path": self.path,
            "format": self.format,
            "mode": self.mode,
            "width": self.width,
            "height": self.height,
            "detail": self.detail,
        }


@dataclass
class UserPromptOption:
    """One selectable choice rendered in a user-prompt dialog.

    Attributes:
        label: Text shown to the user.
        value: Value returned when this option is selected. Defaults to
            ``label`` if left ``None``.
    """

    label: str
    value: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Serialise the option to its wire representation."""

        return {
            "label": self.label,
            "value": self.value or self.label,
        }


@dataclass
class PendingUserPrompt:
    """An outstanding ``ask_user`` request awaiting a reply.

    Attributes:
        request_id: Identifier the TUI uses to correlate the reply.
        question: Free-form question shown to the user.
        options: Optional pre-defined choices; empty for free-form prompts.
        allow_freeform: Whether the user may type a response in addition to
            (or instead of) selecting an option.
        placeholder: Optional placeholder text shown in the input box.
        created_at: ISO-8601 timestamp captured when the prompt was queued.
    """

    request_id: str
    question: str
    options: list[UserPromptOption] = field(default_factory=list)
    allow_freeform: bool = True
    placeholder: str | None = None
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialise the prompt to its wire representation."""

        return {
            "request_id": self.request_id,
            "question": self.question,
            "options": [option.to_dict() for option in self.options],
            "allow_freeform": self.allow_freeform,
            "placeholder": self.placeholder,
            "created_at": self.created_at,
        }


@dataclass
class OperatorPlanStep:
    """Single line item inside an :class:`OperatorPlanState`.

    Attributes:
        step: Imperative description of the work to perform.
        status: Lifecycle marker — typically ``"pending"``, ``"in_progress"``
            or ``"done"``.
    """

    step: str
    status: str = "pending"

    def to_dict(self) -> dict[str, str]:
        """Serialise the step to its wire representation."""

        return {"step": self.step, "status": self.status}


@dataclass
class OperatorPlanState:
    """Mutable plan tracked by the operator across a session.

    Attributes:
        explanation: Optional preamble describing the overall plan intent.
        steps: Ordered list of plan items.
        revision: Monotonic counter incremented on every successful update.
        updated_at: ISO-8601 timestamp of the most recent update.
    """

    explanation: str | None = None
    steps: list[OperatorPlanStep] = field(default_factory=list)
    revision: int = 0
    updated_at: str = field(default_factory=now_iso)

    def update(self, explanation: str | None, plan: list[dict[str, str]]) -> dict[str, tp.Any]:
        """Replace the plan contents and bump revision metadata.

        Args:
            explanation: New plan preamble; pass ``None`` to clear.
            plan: Ordered list of ``{"step": ..., "status": ...}`` dicts.
                Missing ``status`` keys default to ``"pending"``.

        Returns:
            The post-update wire dict, identical to :meth:`to_dict`.
        """

        self.explanation = explanation
        self.steps = [OperatorPlanStep(step=item["step"], status=item.get("status", "pending")) for item in plan]
        self.revision += 1
        self.updated_at = now_iso()
        return self.to_dict()

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialise the full plan state to its wire representation."""

        return {
            "explanation": self.explanation,
            "revision": self.revision,
            "updated_at": self.updated_at,
            "steps": [step.to_dict() for step in self.steps],
        }
