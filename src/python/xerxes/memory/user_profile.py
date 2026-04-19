# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-deepening user profile.

Hermes' user model deepens across sessions: it tracks the user's
expertise, recurring goals, tone preferences, and explicit + implicit
feedback signals. This module provides the storage primitive
(:class:`UserProfile`) and a feedback recorder; the profile updater
agent that turns conversation data into profile updates lives in
:mod:`xerxes.agents.profile_agent`.
"""

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
    """A profile attribute with a confidence score.

    Attributes:
        value: The current best estimate.
        confidence: ``[0.0, 1.0]`` — fraction of evidence supporting *value*.
        last_updated: Timestamp of the most recent reinforcement.
        evidence_count: How many signals have contributed.
    """

    value: tp.Any
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    evidence_count: int = 0

    def reinforce(self, weight: float = 1.0) -> None:
        """Boost the confidence by *weight* (clamped to ``[0, 1]``)."""
        self.confidence = min(1.0, self.confidence + weight)
        self.evidence_count += 1
        self.last_updated = datetime.now()

    def demote(self, weight: float = 0.5) -> None:
        """Reduce confidence (e.g. after an explicit user correction)."""
        self.confidence = max(0.0, self.confidence - weight)
        self.last_updated = datetime.now()


@dataclass
class UserProfile:
    """The deepening per-user model.

    Attributes:
        user_id: Stable identifier.
        expertise: Per-domain expertise level (e.g. ``{"python": ConfidentValue("expert", 0.9)}``).
        domains: Active domains the user works in.
        tone: Preferred tone (formal / casual / terse / verbose).
        recurring_goals: Goals observed across sessions.
        explicit_preferences: Preferences the user stated literally.
        implicit_preferences: Preferences derived from behaviour.
        notes: Free-text observations the profile agent has saved.
        last_seen: Most recent activity timestamp.
        feedback_history: Append-only log of feedback signals.
    """

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
        """Format the profile as a compact string for system prompt injection.

        Filters values below *min_confidence* and caps total lines.

        Args:
            max_lines: Hard cap on output lines.
            min_confidence: Skip values below this confidence.

        Returns:
            A multi-line string suitable for ``[User Profile]`` injection.
            Empty if no high-confidence info is known yet.
        """
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
        """Append a feedback signal to the history log.

        Common signals: ``"correction"``, ``"compliment"``, ``"revert"``,
        ``"retry"``, ``"explicit_preference"``.

        Args:
            signal: Free-form signal name.
            target: Optional target the signal applies to (tool, domain, etc.).
            delta: Signed magnitude (negative = demote).
        """
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
        """JSON-serialisable snapshot."""
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
        """Round-trip a profile saved via :meth:`to_dict`."""
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
    """Serialise a :class:`ConfidentValue` to a JSON-friendly dict."""
    return {
        "value": cv.value,
        "confidence": cv.confidence,
        "last_updated": cv.last_updated.isoformat(),
        "evidence_count": cv.evidence_count,
    }


def _cv_from_dict(d: dict[str, tp.Any]) -> ConfidentValue:
    """Rebuild a :class:`ConfidentValue` from its serialised dict form."""
    return ConfidentValue(
        value=d.get("value"),
        confidence=float(d.get("confidence", 0.0)),
        last_updated=_parse_dt(d.get("last_updated")),
        evidence_count=int(d.get("evidence_count", 0)),
    )


def _decay_value(cv: ConfidentValue, *, now: datetime, half_life_days: float) -> None:
    """Apply half-life confidence decay in-place based on last_updated age."""
    age = max(0.0, (now - cv.last_updated).total_seconds() / 86400.0)
    factor = 0.5 ** (age / max(half_life_days, 0.001))
    cv.confidence = max(0.0, cv.confidence * factor)


def _parse_dt(s: tp.Any) -> datetime:
    """Coerce ISO strings / datetimes / ``None`` into a :class:`datetime` (falls back to now)."""
    if isinstance(s, datetime):
        return s
    if isinstance(s, str):
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return datetime.now()
    return datetime.now()


class UserProfileStore:
    """Thread-safe per-user profile store with optional persistence.

    Profiles are keyed by user_id and persisted under
    ``_profile_<user_id>`` in the supplied :class:`MemoryStorage` (when
    one is provided). Without storage, the store is purely in-memory.

    Example:
        >>> store = UserProfileStore()
        >>> p = store.get_or_create("alice")
        >>> p.domains.append("python")
        >>> store.save(p)
    """

    def __init__(self, storage: MemoryStorage | None = None) -> None:
        """Initialise the store; eagerly hydrates known profiles from disk."""
        self.storage = storage
        self._profiles: dict[str, UserProfile] = {}
        self._lock = threading.RLock()
        self._hydrate()

    def _hydrate(self) -> None:
        """Load any previously persisted profiles from ``storage`` into the cache."""
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
        """Return the cached profile for *user_id*, or ``None`` if absent."""
        with self._lock:
            return self._profiles.get(user_id)

    def get_or_create(self, user_id: str) -> UserProfile:
        """Return the profile for *user_id*, creating and persisting a blank one if needed."""
        with self._lock:
            p = self._profiles.get(user_id)
            if p is None:
                p = UserProfile(user_id=user_id)
                self._profiles[user_id] = p
                self.save(p)
            return p

    def save(self, profile: UserProfile) -> None:
        """Persist *profile* and stamp ``last_seen``; logs on storage failure."""
        profile.last_seen = datetime.now()
        with self._lock:
            self._profiles[profile.user_id] = profile
            if self.storage is not None:
                try:
                    self.storage.save(PROFILE_KEY_PREFIX + profile.user_id, profile.to_dict())
                except Exception:
                    logger.warning("Failed to persist profile for %s", profile.user_id, exc_info=True)

    def delete(self, user_id: str) -> bool:
        """Remove the profile from cache and storage; returns ``True`` when one existed."""
        with self._lock:
            removed = self._profiles.pop(user_id, None)
            if self.storage is not None:
                try:
                    self.storage.delete(PROFILE_KEY_PREFIX + user_id)
                except Exception:
                    pass
            return removed is not None

    def all_user_ids(self) -> list[str]:
        """Return the ids of every profile currently cached in memory."""
        with self._lock:
            return list(self._profiles.keys())

    def render_for(self, user_id: str, **kwargs: tp.Any) -> str:
        """Format the profile for prompt injection. Returns ``""`` if unknown."""
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
        """Apply time-based confidence decay to every stored profile.

        Each :class:`ConfidentValue` loses confidence according to a
        half-life curve based on its ``last_updated`` timestamp:
        ``new_conf = old_conf * 0.5 ** (age_days / half_life_days)``.
        Values that fall below ``prune_threshold`` are removed from
        their containing dict (or, for the singular ``tone`` field,
        cleared to ``None``).

        Args:
            half_life_days: Number of days for confidence to halve.
            prune_threshold: Confidence floor; below this the entry is
                deleted.

        Returns:
            Mapping ``user_id`` → number of pruned attributes.
        """
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
        """Apply decay in-place to one profile; return # pruned attributes."""
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
    """Internal: silence ``asdict`` import warning when not used."""
    return asdict(ConfidentValue(value="x"))
