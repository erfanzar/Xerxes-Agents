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
"""User profile module for Xerxes.

Exports:
    - logger
    - PROFILE_KEY_PREFIX
    - ConfidentValue
    - UserProfile
    - UserProfileStore"""

from __future__ import annotations

import logging
import threading
import typing as tp
from dataclasses import asdict, dataclass, field
from datetime import datetime

if tp.TYPE_CHECKING:
    from .storage import MemoryStorage
logger = logging.getLogger(__name__)
PROFILE_KEY_PREFIX = "_profile_"


@dataclass
class ConfidentValue:
    """Confident value.

    Attributes:
        value (tp.Any): value.
        confidence (float): confidence.
        last_updated (datetime): last updated.
        evidence_count (int): evidence count."""

    value: tp.Any
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    evidence_count: int = 0

    def reinforce(self, weight: float = 1.0) -> None:
        """Reinforce.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            weight (float, optional): IN: weight. Defaults to 1.0. OUT: Consumed during execution."""

        self.confidence = min(1.0, self.confidence + weight)
        self.evidence_count += 1
        self.last_updated = datetime.now()

    def demote(self, weight: float = 0.5) -> None:
        """Demote.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            weight (float, optional): IN: weight. Defaults to 0.5. OUT: Consumed during execution."""

        self.confidence = max(0.0, self.confidence - weight)
        self.last_updated = datetime.now()


@dataclass
class UserProfile:
    """User profile.

    Attributes:
        user_id (str): user id.
        expertise (dict[str, ConfidentValue]): expertise.
        domains (list[str]): domains.
        tone (ConfidentValue | None): tone.
        recurring_goals (list[str]): recurring goals.
        explicit_preferences (dict[str, ConfidentValue]): explicit preferences.
        implicit_preferences (dict[str, ConfidentValue]): implicit preferences.
        notes (list[str]): notes.
        last_seen (datetime): last seen.
        feedback_history (list[dict[str, tp.Any]]): feedback history."""

    user_id: str
    expertise: dict[str, ConfidentValue] = field(default_factory=dict)
    domains: list[str] = field(default_factory=list)
    tone: ConfidentValue | None = None
    recurring_goals: list[str] = field(default_factory=list)
    explicit_preferences: dict[str, ConfidentValue] = field(default_factory=dict)
    implicit_preferences: dict[str, ConfidentValue] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)
    feedback_history: list[dict[str, tp.Any]] = field(default_factory=list)

    def render(self, *, max_lines: int = 12, min_confidence: float = 0.3) -> str:
        """Render.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            max_lines (int, optional): IN: max lines. Defaults to 12. OUT: Consumed during execution.
            min_confidence (float, optional): IN: min confidence. Defaults to 0.3. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""

        lines: list[str] = []
        if self.domains:
            lines.append(f"- Active domains: {', '.join(self.domains[:5])}")
        if self.tone and self.tone.confidence >= min_confidence:
            lines.append(f"- Preferred tone: {self.tone.value} (confidence {self.tone.confidence:.2f})")
        for k, v in self.expertise.items():
            if v.confidence >= min_confidence:
                lines.append(f"- Expertise in {k}: {v.value} (confidence {v.confidence:.2f})")
                if len(lines) >= max_lines:
                    break
        for k, v in self.explicit_preferences.items():
            if v.confidence >= min_confidence:
                lines.append(f"- Prefers {k}: {v.value}")
                if len(lines) >= max_lines:
                    break
        for k, v in self.implicit_preferences.items():
            if v.confidence >= min_confidence and len(lines) < max_lines:
                lines.append(f"- Likely prefers {k}: {v.value} (inferred)")
        if self.recurring_goals and len(lines) < max_lines:
            lines.append(f"- Recurring goals: {'; '.join(self.recurring_goals[:3])}")
        for n in self.notes:
            if len(lines) >= max_lines:
                break
            lines.append(f"- Note: {n}")
        return "\n".join(lines)

    def record_feedback(
        self,
        signal: str,
        *,
        target: str = "",
        delta: float = 1.0,
    ) -> None:
        """Record feedback.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            signal (str): IN: signal. OUT: Consumed during execution.
            target (str, optional): IN: target. Defaults to ''. OUT: Consumed during execution.
            delta (float, optional): IN: delta. Defaults to 1.0. OUT: Consumed during execution."""

        self.feedback_history.append(
            {
                "signal": signal,
                "target": target,
                "delta": delta,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(self.feedback_history) > 256:
            self.feedback_history = self.feedback_history[-128:]

    def to_dict(self) -> dict[str, tp.Any]:
        """To dict.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return {
            "user_id": self.user_id,
            "domains": list(self.domains),
            "recurring_goals": list(self.recurring_goals),
            "notes": list(self.notes),
            "last_seen": self.last_seen.isoformat(),
            "feedback_history": list(self.feedback_history),
            "tone": _cv_to_dict(self.tone) if self.tone else None,
            "expertise": {k: _cv_to_dict(v) for k, v in self.expertise.items()},
            "explicit_preferences": {k: _cv_to_dict(v) for k, v in self.explicit_preferences.items()},
            "implicit_preferences": {k: _cv_to_dict(v) for k, v in self.implicit_preferences.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, tp.Any]) -> UserProfile:
        """From dict.

        Args:
            cls: IN: The class. OUT: Used for class-level operations.
            data (dict[str, tp.Any]): IN: data. OUT: Consumed during execution.
        Returns:
            UserProfile: OUT: Result of the operation."""

        tone_data = data.get("tone")
        return cls(
            user_id=data["user_id"],
            domains=list(data.get("domains", [])),
            recurring_goals=list(data.get("recurring_goals", [])),
            notes=list(data.get("notes", [])),
            last_seen=_parse_dt(data.get("last_seen")),
            feedback_history=list(data.get("feedback_history", [])),
            tone=_cv_from_dict(tone_data) if tone_data else None,
            expertise={k: _cv_from_dict(v) for k, v in data.get("expertise", {}).items()},
            explicit_preferences={k: _cv_from_dict(v) for k, v in data.get("explicit_preferences", {}).items()},
            implicit_preferences={k: _cv_from_dict(v) for k, v in data.get("implicit_preferences", {}).items()},
        )


def _cv_to_dict(cv: ConfidentValue) -> dict[str, tp.Any]:
    """Internal helper to cv to dict.

    Args:
        cv (ConfidentValue): IN: cv. OUT: Consumed during execution.
    Returns:
        dict[str, tp.Any]: OUT: Result of the operation."""

    return {
        "value": cv.value,
        "confidence": cv.confidence,
        "last_updated": cv.last_updated.isoformat(),
        "evidence_count": cv.evidence_count,
    }


def _cv_from_dict(d: dict[str, tp.Any]) -> ConfidentValue:
    """Internal helper to cv from dict.

    Args:
        d (dict[str, tp.Any]): IN: d. OUT: Consumed during execution.
    Returns:
        ConfidentValue: OUT: Result of the operation."""

    return ConfidentValue(
        value=d.get("value"),
        confidence=float(d.get("confidence", 0.0)),
        last_updated=_parse_dt(d.get("last_updated")),
        evidence_count=int(d.get("evidence_count", 0)),
    )


def _decay_value(cv: ConfidentValue, *, now: datetime, half_life_days: float) -> None:
    """Internal helper to decay value.

    Args:
        cv (ConfidentValue): IN: cv. OUT: Consumed during execution.
        now (datetime): IN: now. OUT: Consumed during execution.
        half_life_days (float): IN: half life days. OUT: Consumed during execution."""

    age = max(0.0, (now - cv.last_updated).total_seconds() / 86400.0)
    factor = 0.5 ** (age / max(half_life_days, 0.001))
    cv.confidence = max(0.0, cv.confidence * factor)


def _parse_dt(s: tp.Any) -> datetime:
    """Internal helper to parse dt.

    Args:
        s (tp.Any): IN: s. OUT: Consumed during execution.
    Returns:
        datetime: OUT: Result of the operation."""

    if isinstance(s, datetime):
        return s
    if isinstance(s, str):
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return datetime.now()
    return datetime.now()


class UserProfileStore:
    """User profile store."""

    def __init__(self, storage: MemoryStorage | None = None) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            storage (MemoryStorage | None, optional): IN: storage. Defaults to None. OUT: Consumed during execution."""

        self.storage = storage
        self._profiles: dict[str, UserProfile] = {}
        self._lock = threading.RLock()
        self._hydrate()

    def _hydrate(self) -> None:
        """Internal helper to hydrate.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        if self.storage is None:
            return
        try:
            keys = self.storage.list_keys(PROFILE_KEY_PREFIX)
        except Exception:
            return
        for k in keys:
            if not k.startswith(PROFILE_KEY_PREFIX):
                continue
            try:
                data = self.storage.load(k)
                if data:
                    p = UserProfile.from_dict(data)
                    self._profiles[p.user_id] = p
            except Exception:
                logger.debug("Failed to hydrate profile %s", k, exc_info=True)

    def get(self, user_id: str) -> UserProfile | None:
        """Get.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            user_id (str): IN: user id. OUT: Consumed during execution.
        Returns:
            UserProfile | None: OUT: Result of the operation."""

        with self._lock:
            return self._profiles.get(user_id)

    def get_or_create(self, user_id: str) -> UserProfile:
        """Retrieve the or create.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            user_id (str): IN: user id. OUT: Consumed during execution.
        Returns:
            UserProfile: OUT: Result of the operation."""

        with self._lock:
            p = self._profiles.get(user_id)
            if p is None:
                p = UserProfile(user_id=user_id)
                self._profiles[user_id] = p
                self.save(p)
            return p

    def save(self, profile: UserProfile) -> None:
        """Save.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            profile (UserProfile): IN: profile. OUT: Consumed during execution."""

        profile.last_seen = datetime.now()
        with self._lock:
            self._profiles[profile.user_id] = profile
            if self.storage is not None:
                try:
                    self.storage.save(PROFILE_KEY_PREFIX + profile.user_id, profile.to_dict())
                except Exception:
                    logger.warning("Failed to persist profile for %s", profile.user_id, exc_info=True)

    def delete(self, user_id: str) -> bool:
        """Delete.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            user_id (str): IN: user id. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        with self._lock:
            removed = self._profiles.pop(user_id, None)
            if self.storage is not None:
                try:
                    self.storage.delete(PROFILE_KEY_PREFIX + user_id)
                except Exception:
                    pass
            return removed is not None

    def all_user_ids(self) -> list[str]:
        """All user ids.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[str]: OUT: Result of the operation."""

        with self._lock:
            return list(self._profiles.keys())

    def render_for(self, user_id: str, **kwargs: tp.Any) -> str:
        """Render for.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            user_id (str): IN: user id. OUT: Consumed during execution.
            **kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            str: OUT: Result of the operation."""

        p = self.get(user_id)
        if p is None:
            return ""
        return p.render(**kwargs)

    def decay_all(
        self,
        *,
        half_life_days: float = 30.0,
        prune_threshold: float = 0.05,
    ) -> dict[str, int]:
        """Decay all.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            half_life_days (float, optional): IN: half life days. Defaults to 30.0. OUT: Consumed during execution.
            prune_threshold (float, optional): IN: prune threshold. Defaults to 0.05. OUT: Consumed during execution.
        Returns:
            dict[str, int]: OUT: Result of the operation."""

        from datetime import datetime as _dt

        now = _dt.now()
        prunes: dict[str, int] = {}
        with self._lock:
            for uid, profile in list(self._profiles.items()):
                pruned = self._decay_profile(
                    profile, now=now, half_life_days=half_life_days, prune_threshold=prune_threshold
                )
                prunes[uid] = pruned
                if pruned:
                    self.save(profile)
        return prunes

    @staticmethod
    def _decay_profile(
        profile: UserProfile,
        *,
        now: datetime,
        half_life_days: float,
        prune_threshold: float,
    ) -> int:
        """Internal helper to decay profile.

        Args:
            profile (UserProfile): IN: profile. OUT: Consumed during execution.
            now (datetime): IN: now. OUT: Consumed during execution.
            half_life_days (float): IN: half life days. OUT: Consumed during execution.
            prune_threshold (float): IN: prune threshold. OUT: Consumed during execution.
        Returns:
            int: OUT: Result of the operation."""

        pruned = 0
        if profile.tone is not None:
            _decay_value(profile.tone, now=now, half_life_days=half_life_days)
            if profile.tone.confidence < prune_threshold:
                profile.tone = None
                pruned += 1
        for bag in (profile.expertise, profile.explicit_preferences, profile.implicit_preferences):
            for k in list(bag.keys()):
                _decay_value(bag[k], now=now, half_life_days=half_life_days)
                if bag[k].confidence < prune_threshold:
                    del bag[k]
                    pruned += 1
        return pruned


__all__ = [
    "PROFILE_KEY_PREFIX",
    "ConfidentValue",
    "UserProfile",
    "UserProfileStore",
]


def _drop_dataclass_warning() -> dict[str, tp.Any]:
    """Internal helper to drop dataclass warning.

    Returns:
        dict[str, tp.Any]: OUT: Result of the operation."""

    return asdict(ConfidentValue(value="x"))
