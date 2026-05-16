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
"""Pure data layer behind ``xerxes setup``.

Defines :class:`SetupStep` (one question), :data:`DEFAULT_STEPS` (the canonical
question list), and :func:`run_wizard` (a scripted runner that takes a
pre-collected answers dict). Keeping prompt UI out of this module lets the
real CLI plug in :mod:`prompt_toolkit` while tests drive it with plain
dicts. :func:`write_config` serialises the result as a flat YAML-ish file
without taking a PyYAML dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SetupStep:
    """Describes one question the wizard asks.

    Attributes:
        key: Key under which the answer is stored on :class:`WizardResult`.
        prompt: Human-facing prompt string rendered by the CLI.
        default: Value used when the user accepts the default or skips.
        validator: Callable returning ``True`` when the collected value is
            acceptable; the default accepts everything.
        optional: When ``True``, blank answers are skipped instead of failing
            validation.
    """

    key: str
    prompt: str
    default: Any = None
    validator: Callable[[Any], bool] = lambda v: True
    optional: bool = False


DEFAULT_STEPS: tuple[SetupStep, ...] = (
    SetupStep(
        key="provider",
        prompt="Which LLM provider? [anthropic/openai/openrouter/ollama]",
        default="anthropic",
    ),
    SetupStep(
        key="model",
        prompt="Default model id",
        default="claude-opus-4-7",
    ),
    SetupStep(
        key="api_key",
        prompt="API key (paste or skip if env var is set)",
        optional=True,
    ),
    SetupStep(
        key="permission_mode",
        prompt="Permission mode [auto/manual/accept-all]",
        default="auto",
    ),
    SetupStep(
        key="enable_voice",
        prompt="Enable voice mode? [y/N]",
        default="n",
    ),
    SetupStep(
        key="messaging_platform",
        prompt="Bridge a messaging platform? [none/telegram/discord/slack]",
        default="none",
    ),
)


def _coerce_default(value: Any, step: SetupStep) -> Any:
    """Return ``step.default`` when ``value`` is ``None`` or the empty string."""
    if value is None or value == "":
        return step.default
    return value


@dataclass
class WizardResult:
    """Outcome of :func:`run_wizard`.

    Attributes:
        answers: ``{step.key: value}`` for every successfully collected step.
        skipped: Keys of optional steps the user left blank.
    """

    answers: dict[str, Any] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


def run_wizard(
    answers: dict[str, Any] | None = None,
    *,
    steps: tuple[SetupStep, ...] | None = None,
) -> WizardResult:
    """Drive the wizard against a pre-collected answer dict.

    Each step reads ``answers[step.key]`` (or its default), validates the
    result, and records it on the returned :class:`WizardResult`. Optional
    steps with blank answers are reported via :attr:`WizardResult.skipped`.

    Raises:
        ValueError: A non-optional step's value failed its validator.
    """
    out = WizardResult()
    answers = answers or {}
    for step in steps or DEFAULT_STEPS:
        raw = answers.get(step.key)
        value = _coerce_default(raw, step)
        if step.optional and (value is None or value == ""):
            out.skipped.append(step.key)
            continue
        if not step.validator(value):
            raise ValueError(f"invalid value for {step.key}: {value!r}")
        out.answers[step.key] = value
    return out


def write_config(answers: dict[str, Any], *, target: Path) -> Path:
    """Persist ``answers`` as a flat YAML-ish config file at ``target``.

    Only emits ``key: value`` lines (quoting strings); does not depend on
    PyYAML. The real CLI later re-parses these files with PyYAML, while tests
    only need to verify the expected keys were written.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, v in answers.items():
        if isinstance(v, str):
            lines.append(f'{k}: "{v}"')
        else:
            lines.append(f"{k}: {v}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


__all__ = ["DEFAULT_STEPS", "SetupStep", "WizardResult", "run_wizard", "write_config"]
