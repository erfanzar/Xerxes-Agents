# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Background user-profile updater.

Runs at the end of each turn (or session) and folds the new evidence
into the persistent :class:`UserProfile`. Heuristics-only by default —
no LLM is required — so the updater works in tests, CI, and offline.
An optional LLM hook (``llm_summariser``) can produce richer notes when
available.

Signal sources:
    - **NER** over the user's prompts → domains.
    - **Tone heuristics** (length / punctuation / exclamation density).
    - **Behavioural** (retry rate, edit distance) → confidence updates.
    - **Explicit phrasing** (``"I prefer X"``, ``"don't ever Y"``) →
      explicit_preferences with high initial confidence.
"""

from __future__ import annotations

import logging
import re
import typing as tp
from dataclasses import dataclass
from datetime import datetime

from ..memory.user_profile import ConfidentValue, UserProfile, UserProfileStore

logger = logging.getLogger(__name__)
_TECH_DOMAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("python", ("python", "pytest", "django", "flask", "fastapi", "uv", "poetry", "pip")),
    ("rust", ("rust", "cargo", "tokio", "serde", "axum")),
    ("javascript", ("javascript", "typescript", "node", "react", "next", "vite", "npm", "yarn", "pnpm")),
    ("go", ("golang", " go ")),
    ("devops", ("docker", "kubernetes", "k8s", "terraform", "ansible", "ci/cd", "github actions", "gitlab ci")),
    ("ml", ("pytorch", "tensorflow", "huggingface", "embedding", "fine-tune", "lora")),
    ("databases", ("postgres", "postgresql", "mysql", "sqlite", "redis", "mongodb")),
    ("security", ("owasp", "csrf", "xss", "sql injection", "auth", "oauth", "jwt", "tls")),
)
_PREFERENCE_PHRASES = (
    re.compile(r"\bi (?:prefer|want|like|wish|need)(?: to)?\s+(.{3,80})", re.I),
    re.compile(r"\b(?:please|always|make sure to)\s+(.{3,80})", re.I),
    re.compile(r"\b(?:don'?t|do not|never)\s+(.{3,80})", re.I),
)


@dataclass
class ProfileUpdate:
    """Outcome of a single :meth:`ProfileAgent.update` call.

    Attributes:
        user_id: Profile that was updated.
        domains_added: Newly inferred domains.
        prefs_added: Newly extracted explicit preference phrases.
        confidence_changes: Map of attribute → delta.
    """

    user_id: str
    domains_added: list[str]
    prefs_added: list[str]
    confidence_changes: dict[str, float]


class ProfileAgent:
    """Heuristic profile updater.

    Pure-Python; can run synchronously after every turn or offline as a
    batch over a session. Set ``llm_summariser=...`` to enrich the
    ``notes`` list with a free-text summary.
    """

    def __init__(
        self,
        store: UserProfileStore,
        *,
        llm_summariser: tp.Callable[[str, UserProfile], str] | None = None,
        ner_extractor: tp.Callable[[str], dict[str, list[str]]] | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            store: The :class:`UserProfileStore` that owns the profiles.
            llm_summariser: Optional callable
                ``(turn_text, profile) -> note`` for richer notes. Errors
                are swallowed silently.
            ner_extractor: Optional callable ``(text) -> {entity_type: [values]}``.
                Defaults to :func:`_default_ner` (tools/ai_tools).
        """
        self.store = store
        self.llm_summariser = llm_summariser
        self.ner_extractor = ner_extractor or _default_ner

    def update(
        self,
        user_id: str,
        *,
        user_prompt: str = "",
        agent_response: str = "",
        signals: tp.Iterable[str] = (),
    ) -> ProfileUpdate:
        """Fold a single turn's evidence into the profile.

        Args:
            user_id: Stable user identifier.
            user_prompt: What the user said in this turn.
            agent_response: What the agent answered.
            signals: Extra behavioural signal names
                (e.g. ``["correction", "revert"]``).

        Returns:
            A :class:`ProfileUpdate` describing what changed.
        """
        profile = self.store.get_or_create(user_id)
        profile.last_seen = datetime.now()
        domains_added: list[str] = []
        prefs_added: list[str] = []
        confidence_changes: dict[str, float] = {}
        if user_prompt:
            for d in self._infer_domains(user_prompt):
                if d not in profile.domains:
                    profile.domains.append(d)
                    domains_added.append(d)
            tone_value = self._infer_tone(user_prompt)
            if tone_value:
                if profile.tone is None:
                    profile.tone = ConfidentValue(value=tone_value, confidence=0.2)
                elif profile.tone.value == tone_value:
                    profile.tone.reinforce(0.1)
                else:
                    profile.tone.demote(0.1)
                    if profile.tone.confidence < 0.05:
                        profile.tone = ConfidentValue(value=tone_value, confidence=0.2)
                confidence_changes["tone"] = profile.tone.confidence
            for phrase in self._extract_preference_phrases(user_prompt):
                key = phrase.lower()[:60]
                cv = profile.explicit_preferences.get(key)
                if cv is None:
                    cv = ConfidentValue(value=phrase, confidence=0.6)
                    profile.explicit_preferences[key] = cv
                    prefs_added.append(phrase)
                else:
                    cv.reinforce(0.2)
        for sig in signals:
            profile.record_feedback(sig)
            if sig in ("correction", "revert", "retry"):
                if profile.tone is not None:
                    profile.tone.demote(0.1)
        if self.llm_summariser is not None and (user_prompt or agent_response):
            try:
                note = self.llm_summariser(
                    f"USER: {user_prompt}\nAGENT: {agent_response}",
                    profile,
                )
                if isinstance(note, str) and note.strip() and note.strip() not in profile.notes:
                    profile.notes.append(note.strip()[:500])
                    if len(profile.notes) > 50:
                        profile.notes = profile.notes[-50:]
            except Exception:
                logger.debug("llm_summariser failed", exc_info=True)
        self.store.save(profile)
        return ProfileUpdate(
            user_id=user_id,
            domains_added=domains_added,
            prefs_added=prefs_added,
            confidence_changes=confidence_changes,
        )

    def _infer_domains(self, text: str) -> list[str]:
        """Return the domains whose keyword set matches anywhere in ``text``.

        Args:
            text: Free-form user text to scan (matched case-insensitively).

        Returns:
            List of domain labels drawn from ``_TECH_DOMAINS``.
        """
        text_lower = text.lower()
        out: list[str] = []
        for domain, keywords in _TECH_DOMAINS:
            if any(k in text_lower for k in keywords):
                out.append(domain)
        return out

    def _infer_tone(self, text: str) -> str:
        """Classify tone as ``terse``/``casual``/``verbose``/``balanced``.

        Args:
            text: User utterance to inspect.

        Returns:
            One of ``""`` (empty input), ``"terse"`` (<=6 words),
            ``"casual"`` (>5% exclamation marks), ``"verbose"`` (>80
            words), or ``"balanced"``.
        """
        n_words = len(text.split())
        if n_words == 0:
            return ""
        n_excl = text.count("!")
        if n_words <= 6:
            return "terse"
        if n_excl / max(n_words, 1) > 0.05:
            return "casual"
        if n_words > 80:
            return "verbose"
        return "balanced"

    def _extract_preference_phrases(self, text: str) -> list[str]:
        """Pull preference-signalling phrases (``I prefer …``, etc.) from ``text``.

        Args:
            text: User utterance to scan with ``_PREFERENCE_PHRASES`` regexes.

        Returns:
            List of matched phrases, trimmed of trailing punctuation and
            filtered to the 3-200 character range.
        """
        out: list[str] = []
        for pat in _PREFERENCE_PHRASES:
            for m in pat.finditer(text):
                phrase = m.group(0).strip().rstrip(".!?,")
                if 3 <= len(phrase) <= 200:
                    out.append(phrase)
        return out


def _default_ner(text: str) -> dict[str, list[str]]:
    """Optional NER pass via :class:`EntityExtractor`. Failures yield ``{}``."""
    try:
        from ..tools.ai_tools import EntityExtractor

        result = EntityExtractor.static_call(text)
        return result.get("entities", {}) if isinstance(result, dict) else {}
    except Exception:
        return {}


__all__ = ["ProfileAgent", "ProfileUpdate"]
