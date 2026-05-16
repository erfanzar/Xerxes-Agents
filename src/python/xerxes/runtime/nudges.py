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
"""Periodic nudges that shape agent self-improvement.

Two built-in rules ship by default:

    * :class:`MemoryNudge` — every ``interval`` turns, if no memory writes
      have happened since the last fire and the conversation contains
      durable info, inject a one-shot system note suggesting
      ``save_memory(...)``.
    * :class:`SkillNudge` — after a turn with ``>= threshold`` successful
      tool calls, suggest ``skill_manage`` to capture the pattern.

Nudges are *suggestions*, not interruptions: each fires by appending a
short system-role message that the LLM reads on the *next* turn. Rules
can be silenced individually via :meth:`NudgeManager.disable`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NudgeContext:
    """Snapshot of session state the nudge rules consult.

    Attributes:
        turn_index: 0-based count of completed turns in the session.
        tool_calls_this_turn: Tool calls issued during the just-finished turn.
        successful_tool_calls_this_turn: Subset of ``tool_calls_this_turn``
            that returned without error.
        memory_writes_this_turn: Memory-writing calls (``save_memory`` etc.)
            during this turn.
        memory_writes_since_last_fire: Manager-tracked rolling count used
            for rate-limiting :class:`MemoryNudge`.
        last_user_message: Most recent user turn text, or ``""``.
        last_assistant_message: Most recent assistant turn text, or ``""``.
        metadata: Free-form bag for rule-specific extras.
    """

    turn_index: int
    tool_calls_this_turn: int = 0
    successful_tool_calls_this_turn: int = 0
    memory_writes_this_turn: int = 0
    memory_writes_since_last_fire: int = 0
    last_user_message: str = ""
    last_assistant_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class NudgeRule(ABC):
    """Abstract base class for nudge rules.

    Subclasses set a ``name`` class attribute and implement
    :meth:`should_fire` and :meth:`message`.
    """

    name: str = ""

    @abstractmethod
    def should_fire(self, ctx: NudgeContext) -> bool:
        """Return ``True`` when this rule should emit a nudge for ``ctx``."""
        ...

    @abstractmethod
    def message(self, ctx: NudgeContext) -> str:
        """Return the system-role message to inject before the next turn."""
        ...


class MemoryNudge(NudgeRule):
    """Reminds the agent to persist durable info via ``save_memory``."""

    name = "memory"

    DURABLE_HINTS = ("decided", "prefers", "will always", "should never", "username", "deadline", "remember")
    """Substring markers that suggest the conversation produced durable info."""

    def __init__(self, *, interval: int = 8) -> None:
        """Fire at most once every ``interval`` turns (default ``8``)."""
        self.interval = int(interval)

    def should_fire(self, ctx: NudgeContext) -> bool:
        """Return ``True`` when the interval is up and durable hints are present."""
        if ctx.memory_writes_since_last_fire > 0:
            return False
        if (ctx.turn_index + 1) % self.interval != 0:
            return False
        blob = f"{ctx.last_user_message} {ctx.last_assistant_message}".lower()
        return any(hint in blob for hint in self.DURABLE_HINTS)

    def message(self, ctx: NudgeContext) -> str:
        """Return the canned save-to-memory suggestion."""
        return (
            "[NUDGE] You haven't written to memory in a while and the recent turn looks like it had "
            "durable user info (preferences, decisions, deadlines). Consider calling "
            "`save_memory(content=...)` so this doesn't get lost across sessions."
        )


class SkillNudge(NudgeRule):
    """Suggests formalising a heavy tool chain into a reusable skill."""

    name = "skill"

    def __init__(self, *, threshold: int = 6) -> None:
        """Fire when a single turn made at least ``threshold`` successful tool calls."""
        self.threshold = int(threshold)

    def should_fire(self, ctx: NudgeContext) -> bool:
        """Return ``True`` when the successful tool count crosses ``threshold``."""
        return ctx.successful_tool_calls_this_turn >= self.threshold

    def message(self, ctx: NudgeContext) -> str:
        """Return the canned skill-capture suggestion."""
        return (
            "[NUDGE] The just-finished turn used many tools successfully. If this pattern is likely "
            "to recur, consider `skill_manage(intent='create', name=..., body=...)` to capture it "
            "as a reusable skill."
        )


class NudgeManager:
    """Coordinate a list of :class:`NudgeRule` instances and their fired state.

    Each call to :meth:`check` returns the messages every active rule wants to
    inject before the next turn. Per-rule fire counts are tracked so external
    callers can rate-limit display or reset state.
    """

    def __init__(self, rules: list[NudgeRule] | None = None) -> None:
        """Initialise with ``rules`` or the default ``[MemoryNudge, SkillNudge]``."""
        self._rules = rules or [MemoryNudge(), SkillNudge()]
        self._disabled: set[str] = set()
        self._fired_count: dict[str, int] = {}

    @property
    def rules(self) -> list[NudgeRule]:
        """Return a shallow copy of the registered rule list."""
        return list(self._rules)

    def disable(self, rule_name: str) -> None:
        """Silence the rule with ``rule_name`` until :meth:`enable` is called."""
        self._disabled.add(rule_name)

    def enable(self, rule_name: str) -> None:
        """Re-enable a previously disabled rule."""
        self._disabled.discard(rule_name)

    def disabled(self) -> set[str]:
        """Return a copy of the currently disabled rule names."""
        return set(self._disabled)

    def fired_count(self, rule_name: str) -> int:
        """Return how many times the named rule has fired so far."""
        return self._fired_count.get(rule_name, 0)

    def check(self, ctx: NudgeContext) -> list[tuple[str, str]]:
        """Return ``[(rule_name, message), ...]`` for every rule that fires."""
        out: list[tuple[str, str]] = []
        for rule in self._rules:
            if rule.name in self._disabled:
                continue
            if rule.should_fire(ctx):
                out.append((rule.name, rule.message(ctx)))
                self._fired_count[rule.name] = self._fired_count.get(rule.name, 0) + 1
        return out


__all__ = ["MemoryNudge", "NudgeContext", "NudgeManager", "NudgeRule", "SkillNudge"]
