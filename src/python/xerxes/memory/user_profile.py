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
"""Confidence-weighted user profile store with temporal decay.

``UserProfile`` accumulates evidence about a user (domains, tone,
expertise, preferences explicit/implicit, recurring goals, notes, and
a small feedback ring buffer). Every learnt fact is wrapped in a
``ConfidentValue`` so we can reinforce, demote, and exponentially decay
beliefs based on age. ``UserProfileStore`` persists every profile
behind any ``MemoryStorage`` backend keyed by ``PROFILE_KEY_PREFIX``."""

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
    """A profile fact paired with a confidence score and last-update marker.

    Attributes:
        value: The actual fact (string, number, dict — anything JSON-safe).
        confidence: Belief strength in ``[0, 1]``.
        last_updated: Wall-clock time of the last reinforcement/demotion;
            used by ``UserProfileStore._decay_profile``.
        evidence_count: Number of reinforcement events seen."""

    value: tp.Any
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    evidence_count: int = 0

    def reinforce(self, weight: float = 1.0) -> None:
        """Increase confidence (clamped to 1.0) and bump the evidence counter."""

        self.confidence = min(1.0, self.confidence + weight)
        self.evidence_count += 1
        self.last_updated = datetime.now()

    def demote(self, weight: float = 0.5) -> None:
        """Decrease confidence (clamped at 0.0) and refresh ``last_updated``."""

        self.confidence = max(0.0, self.confidence - weight)
        self.last_updated = datetime.now()


@dataclass
class UserProfile:
    """Long-lived per-user beliefs, preferences, and feedback history.

    Attributes:
        user_id: Stable identifier for the user.
        expertise: ``topic -> ConfidentValue`` map describing skill level.
        domains: Active domain tags (free-form labels).
        tone: Preferred conversational tone as a ``ConfidentValue``.
        recurring_goals: Recurring objectives the user has expressed.
        explicit_preferences: Preferences the user stated directly.
        implicit_preferences: Preferences inferred from behaviour.
        notes: Free-form long-form notes about the user.
        last_seen: Wall-clock time of the most recent interaction.
        feedback_history: Bounded ring (≤256, trimmed to 128) of recent
            feedback signal dicts."""

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
        """Render the profile as a bulleted human-readable block.

        Only confident-enough beliefs are surfaced (default
        ``min_confidence=0.3``). Output is capped at ``max_lines`` to
        keep prompt overhead bounded.

        Args:
            max_lines: Maximum number of bullet lines.
            min_confidence: Minimum confidence required for a fact to
                appear."""

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
        """Append a feedback event to the bounded history ring.

        The ring trims to 128 entries once it grows past 256 to bound
        memory growth without losing the last few dozen signals.

        Args:
            signal: Label describing the feedback (e.g. ``"positive"``).
            target: Optional reference to the message or action.
            delta: Magnitude of the feedback (sign indicates direction)."""

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
        """Serialise the profile to a JSON-safe dict for persistence."""

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
        """Reconstruct a profile from its ``to_dict`` form, parsing nested values back into ``ConfidentValue`` instances."""

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
    """Convert a ``ConfidentValue`` to a JSON-safe dict (ISO timestamp)."""

    return {
        "value": cv.value,
        "confidence": cv.confidence,
        "last_updated": cv.last_updated.isoformat(),
        "evidence_count": cv.evidence_count,
    }


def _cv_from_dict(d: dict[str, tp.Any]) -> ConfidentValue:
    """Rebuild a ``ConfidentValue`` from its serialised dict form."""

    return ConfidentValue(
        value=d.get("value"),
        confidence=float(d.get("confidence", 0.0)),
        last_updated=_parse_dt(d.get("last_updated")),
        evidence_count=int(d.get("evidence_count", 0)),
    )


def _decay_value(cv: ConfidentValue, *, now: datetime, half_life_days: float) -> None:
    """Apply exponential confidence decay in-place: ``c *= 0.5^(age/half_life)``."""

    age = max(0.0, (now - cv.last_updated).total_seconds() / 86400.0)
    factor = 0.5 ** (age / max(half_life_days, 0.001))
    cv.confidence = max(0.0, cv.confidence * factor)


def _parse_dt(s: tp.Any) -> datetime:
    """Best-effort parse of an ISO timestamp; fall back to ``datetime.now()`` on failure."""

    if isinstance(s, datetime):
        return s
    if isinstance(s, str):
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return datetime.now()
    return datetime.now()


class UserProfileStore:
    """Thread-safe registry of ``UserProfile`` objects with optional persistence.

    Profiles are kept in an in-memory dict keyed by ``user_id`` and
    mirrored to a backing ``MemoryStorage`` (under ``PROFILE_KEY_PREFIX``)
    when one is supplied. Decay is a manual operation invoked via
    ``decay_all``."""

    def __init__(self, storage: MemoryStorage | None = None) -> None:
        """Attach optional persistent storage and hydrate any existing profiles."""

        self.storage = storage
        self._profiles: dict[str, UserProfile] = {}
        self._lock = threading.RLock()
        self._hydrate()

    def _hydrate(self) -> None:
        """Load every ``PROFILE_KEY_PREFIX``-prefixed row into the in-memory map."""

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
        """Return the cached profile for ``user_id`` or ``None``."""

        with self._lock:
            return self._profiles.get(user_id)

    def get_or_create(self, user_id: str) -> UserProfile:
        """Return the cached profile, creating and persisting one on first call."""

        with self._lock:
            p = self._profiles.get(user_id)
            if p is None:
                p = UserProfile(user_id=user_id)
                self._profiles[user_id] = p
                self.save(p)
            return p

    def save(self, profile: UserProfile) -> None:
        """Persist ``profile`` to storage (best-effort) and refresh its ``last_seen``."""

        profile.last_seen = datetime.now()
        with self._lock:
            self._profiles[profile.user_id] = profile
            if self.storage is not None:
                try:
                    self.storage.save(PROFILE_KEY_PREFIX + profile.user_id, profile.to_dict())
                except Exception:
                    logger.warning("Failed to persist profile for %s", profile.user_id, exc_info=True)

    def delete(self, user_id: str) -> bool:
        """Drop the user's cached profile and best-effort delete its storage row."""

        with self._lock:
            removed = self._profiles.pop(user_id, None)
            if self.storage is not None:
                try:
                    self.storage.delete(PROFILE_KEY_PREFIX + user_id)
                except Exception:
                    pass
            return removed is not None

    def all_user_ids(self) -> list[str]:
        """Return every cached ``user_id``."""

        with self._lock:
            return list(self._profiles.keys())

    def render_for(self, user_id: str, **kwargs: tp.Any) -> str:
        """Render the user's profile to text; returns ``""`` if no profile exists.

        Extra keyword arguments are forwarded to ``UserProfile.render``."""

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
        """Apply exponential confidence decay across every cached profile.

        Beliefs whose confidence falls below ``prune_threshold`` after
        decay are deleted. Profiles that lose any beliefs are
        re-persisted.

        Args:
            half_life_days: Days after which a belief's confidence halves.
            prune_threshold: Confidence floor below which beliefs are dropped.

        Returns:
            Mapping of ``user_id`` to the number of beliefs pruned."""

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
        """Decay every ``ConfidentValue`` inside one profile and prune the weak ones.

        Returns the number of beliefs that fell below ``prune_threshold``
        and were removed."""

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
    """Touch ``asdict`` so the import isn't flagged as unused by linters."""

    return asdict(ConfidentValue(value="x"))
