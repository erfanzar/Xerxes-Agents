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
"""Types module for Xerxes.

Exports:
    - now_iso
    - ImageInspectionResult
    - UserPromptOption
    - PendingUserPrompt
    - OperatorPlanStep
    - OperatorPlanState"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field
from datetime import UTC, datetime

from PIL import Image


def now_iso() -> str:
    """Now iso.

    Returns:
        str: OUT: Result of the operation."""

    return datetime.now(UTC).isoformat()
    """Now iso.

    Returns:
        str: OUT: Result of the operation."""
    """Now iso.

    Returns:
        str: OUT: Result of the operation."""


@dataclass
class ImageInspectionResult:
    """Image inspection result.

    Attributes:
        path (str): path.
        format (str | None): format.
        mode (str): mode.
        width (int): width.
        height (int): height.
        image (Image.Image): image.
        detail (str): detail."""

    path: str
    format: str | None
    mode: str
    width: int
    height: int
    image: Image.Image
    detail: str = "auto"

    def summary(self) -> str:
        """Summary.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        return (
            """Summary.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""
            """Summary.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""
            f"Image loaded from {self.path} "
            f"({self.width}x{self.height}, mode={self.mode}, format={self.format or 'unknown'})"
        )

    def tool_metadata(self) -> dict[str, tp.Any]:
        """Tool metadata.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return {
            """Tool metadata.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""
            """Tool metadata.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""
            "path": self.path,
            "format": self.format,
            "mode": self.mode,
            "width": self.width,
            "height": self.height,
            "detail": self.detail,
        }


@dataclass
class UserPromptOption:
    """User prompt option.

    Attributes:
        label (str): label.
        value (str | None): value."""

    label: str
    value: str | None = None

    def to_dict(self) -> dict[str, str]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str]: OUT: Result of the operation."""

        return {
            """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str]: OUT: Result of the operation."""
            """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str]: OUT: Result of the operation."""
            "label": self.label,
            "value": self.value or self.label,
        }


@dataclass
class PendingUserPrompt:
    """Pending user prompt.

    Attributes:
        request_id (str): request id.
        question (str): question.
        options (list[UserPromptOption]): options.
        allow_freeform (bool): allow freeform.
        placeholder (str | None): placeholder.
        created_at (str): created at."""

    request_id: str
    question: str
    options: list[UserPromptOption] = field(default_factory=list)
    allow_freeform: bool = True
    placeholder: str | None = None
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, tp.Any]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return {
            """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""
            """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""
            "request_id": self.request_id,
            "question": self.question,
            "options": [option.to_dict() for option in self.options],
            "allow_freeform": self.allow_freeform,
            "placeholder": self.placeholder,
            "created_at": self.created_at,
        }


@dataclass
class OperatorPlanStep:
    """Operator plan step.

    Attributes:
        step (str): step.
        status (str): status."""

    step: str
    status: str = "pending"

    def to_dict(self) -> dict[str, str]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str]: OUT: Result of the operation."""

        return {"step": self.step, "status": self.status}
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str]: OUT: Result of the operation."""
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, str]: OUT: Result of the operation."""


@dataclass
class OperatorPlanState:
    """Operator plan state.

    Attributes:
        explanation (str | None): explanation.
        steps (list[OperatorPlanStep]): steps.
        revision (int): revision.
        updated_at (str): updated at."""

    explanation: str | None = None
    steps: list[OperatorPlanStep] = field(default_factory=list)
    revision: int = 0
    updated_at: str = field(default_factory=now_iso)

    def update(self, explanation: str | None, plan: list[dict[str, str]]) -> dict[str, tp.Any]:
        """Update.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            explanation (str | None): IN: explanation. OUT: Consumed during execution.
            plan (list[dict[str, str]]): IN: plan. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        self.explanation = explanation
        """Update.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            explanation (str | None): IN: explanation. OUT: Consumed during execution.
            plan (list[dict[str, str]]): IN: plan. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""
        """Update.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            explanation (str | None): IN: explanation. OUT: Consumed during execution.
            plan (list[dict[str, str]]): IN: plan. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""
        self.steps = [OperatorPlanStep(step=item["step"], status=item.get("status", "pending")) for item in plan]
        self.revision += 1
        self.updated_at = now_iso()
        return self.to_dict()

    def to_dict(self) -> dict[str, tp.Any]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return {
            """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""
            """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""
            "explanation": self.explanation,
            "revision": self.revision,
            "updated_at": self.updated_at,
            "steps": [step.to_dict() for step in self.steps],
        }
