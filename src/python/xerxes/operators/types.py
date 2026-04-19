# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared operator tool datatypes.

Defines the core value types used across the operator subsystem:

- :class:`ImageInspectionResult` -- structured result for local image
  inspection via the ``view_image`` tool.
- :class:`UserPromptOption` and :class:`PendingUserPrompt` -- types
  that model user clarification questions.
- :class:`OperatorPlanStep` and :class:`OperatorPlanState` -- types
  that represent the structured execution plan.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from datetime import datetime, timezone, UTC

from PIL import Image


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        A timezone-aware ISO 8601 timestamp string representing the
        current moment in UTC.

    Example:
        >>> ts = now_iso()
        >>> "T" in ts
        True
    """
    return datetime.now(UTC).isoformat()


@dataclass
class ImageInspectionResult:
    """Structured result for local image inspection.

    Returned by the ``view_image`` operator tool and carries both
    serialisable metadata (path, dimensions, format) and the in-memory
    PIL image for multimodal reinvocation.

    Attributes:
        path: Absolute filesystem path to the inspected image.
        format: Image format string reported by PIL (e.g. ``"PNG"``,
            ``"JPEG"``).  May be ``None`` if PIL could not determine
            the format.
        mode: PIL image mode string (e.g. ``"RGB"``, ``"RGBA"``).
        width: Image width in pixels.
        height: Image height in pixels.
        image: In-memory PIL :class:`~PIL.Image.Image` instance.
        detail: Requested inspection detail level.  Defaults to
            ``"auto"``.
    """

    path: str
    format: str | None
    mode: str
    width: int
    height: int
    image: Image.Image
    detail: str = "auto"

    def summary(self) -> str:
        """Return a compact text summary safe for tool-message persistence.

        Returns:
            A single-line string describing the image path, dimensions,
            mode, and format.
        """
        return (
            f"Image loaded from {self.path} "
            f"({self.width}x{self.height}, mode={self.mode}, format={self.format or 'unknown'})"
        )

    def tool_metadata(self) -> dict[str, tp.Any]:
        """Return serialisable metadata for session persistence.

        Produces a dictionary that can be safely JSON-encoded and
        stored alongside tool call records, without including the
        heavy PIL image object.

        Returns:
            A dictionary with keys ``path``, ``format``, ``mode``,
            ``width``, ``height``, and ``detail``.
        """
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
    """Single selectable option for a pending user question.

    Attributes:
        label: Human-readable display text for the option.
        value: Machine-readable value submitted when this option is
            selected.  Defaults to the label when ``None``.
    """

    label: str
    value: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Serialise the option for UI and persistence use.

        Returns:
            A dictionary with ``label`` and ``value`` keys.  When
            ``value`` was ``None``, the label is used as the value.
        """
        return {
            "label": self.label,
            "value": self.value or self.label,
        }


@dataclass
class PendingUserPrompt:
    """Live question awaiting user input from the UI.

    Created by :class:`~xerxes.operators.user_prompt.UserPromptManager`
    when the ``ask_user`` tool is invoked, and resolved once the user
    submits an answer through the TUI.

    Attributes:
        request_id: Unique identifier for this prompt request.
        question: The question text displayed to the user.
        options: Optional list of :class:`UserPromptOption` instances
            the user can choose from.
        allow_freeform: When ``True``, the user may type a custom
            answer instead of selecting a listed option.
        placeholder: Optional placeholder hint shown in the input
            field.
        created_at: ISO 8601 timestamp of when the prompt was created.
    """

    request_id: str
    question: str
    options: list[UserPromptOption] = field(default_factory=list)
    allow_freeform: bool = True
    placeholder: str | None = None
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialise the pending prompt for UI polling.

        Returns:
            A dictionary containing all prompt fields, with options
            serialised via :meth:`UserPromptOption.to_dict`.
        """
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
    """One step in the operator plan state.

    Attributes:
        step: Short description of what this step involves.
        status: Current status label (e.g. ``"pending"``,
            ``"in_progress"``, ``"completed"``).  Defaults to
            ``"pending"``.
    """

    step: str
    status: str = "pending"

    def to_dict(self) -> dict[str, str]:
        """Serialise the plan step.

        Returns:
            A dictionary with ``step`` and ``status`` keys.
        """
        return {"step": self.step, "status": self.status}


@dataclass
class OperatorPlanState:
    """Structured plan state updated by the operator plan tool.

    Tracks a sequence of :class:`OperatorPlanStep` items together with
    an explanation and a monotonically increasing revision counter.

    Attributes:
        explanation: Optional text explaining the current plan state
            or the most recent change.
        steps: Ordered list of :class:`OperatorPlanStep` instances.
        revision: Monotonically increasing counter bumped on every
            call to :meth:`update`.
        updated_at: ISO 8601 timestamp of the last update.
    """

    explanation: str | None = None
    steps: list[OperatorPlanStep] = field(default_factory=list)
    revision: int = 0
    updated_at: str = field(default_factory=now_iso)

    def update(self, explanation: str | None, plan: list[dict[str, str]]) -> dict[str, tp.Any]:
        """Replace the current plan state.

        Atomically swaps the explanation and step list, increments the
        revision counter, and refreshes the timestamp.

        Args:
            explanation: Optional short note describing the plan change.
            plan: List of step dictionaries.  Each must contain a
                ``"step"`` key; ``"status"`` defaults to ``"pending"``
                when absent.

        Returns:
            The serialised plan state produced by :meth:`to_dict`.
        """
        self.explanation = explanation
        self.steps = [OperatorPlanStep(step=item["step"], status=item.get("status", "pending")) for item in plan]
        self.revision += 1
        self.updated_at = now_iso()
        return self.to_dict()

    def to_dict(self) -> dict[str, tp.Any]:
        """Serialise the plan state.

        Returns:
            A dictionary with ``explanation``, ``revision``,
            ``updated_at``, and ``steps`` (each serialised via
            :meth:`OperatorPlanStep.to_dict`).
        """
        return {
            "explanation": self.explanation,
            "revision": self.revision,
            "updated_at": self.updated_at,
            "steps": [step.to_dict() for step in self.steps],
        }
