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
"""User profile inference and update agent.

This module provides :class:`ProfileAgent`, which analyzes conversation text
to infer user domains, tone, and explicit preferences, then persists them via
a :class:`~xerxes.memory.user_profile.UserProfileStore`.
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
    """Result of a profile update operation.

    Attributes:
        user_id (str): Identifier of the updated user.
        domains_added (list[str]): Newly inferred domains.
        prefs_added (list[str]): Newly extracted preference phrases.
        confidence_changes (dict[str, float]): Updated confidence scores by key.
    """

    user_id: str
    domains_added: list[str]
    prefs_added: list[str]
    confidence_changes: dict[str, float]


class ProfileAgent:
    """Analyzes interactions to build and update a user profile.

    Uses keyword matching for domains, heuristics for tone, regex for
    preferences, and an optional LLM summarizer for notes.
    """

    def __init__(
        self,
        store: UserProfileStore,
        *,
        llm_summariser: tp.Callable[[str, UserProfile], str] | None = None,
        ner_extractor: tp.Callable[[str], dict[str, list[str]]] | None = None,
    ) -> None:
        """Initialize the profile agent.

        Args:
            store (UserProfileStore): IN: Persistent store for user profiles. OUT:
                Used for reading and saving profiles.
            llm_summariser (Callable[[str, UserProfile], str] | None): IN: Optional
                callback to summarize a turn into a profile note. OUT: Called during
                updates when both prompt and response are present.
            ner_extractor (Callable[[str], dict[str, list[str]]] | None): IN:
                Optional NER extractor. OUT: Defaults to :func:`_default_ner` if
                not provided.
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
        """Update a user profile based on a single interaction turn.

        Args:
            user_id (str): IN: User identifier. OUT: Used to fetch/create the profile.
            user_prompt (str): IN: Text from the user. OUT: Analyzed for domains,
                tone, and preferences.
            agent_response (str): IN: Text from the agent. OUT: Passed to the LLM
                summarizer along with the user prompt.
            signals (Iterable[str]): IN: Feedback signals (e.g., ``"correction"``).
                OUT: Recorded as feedback and may affect tone confidence.

        Returns:
            ProfileUpdate: OUT: Summary of changes made to the profile.
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
        """Infer technology domains from text via keyword matching.

        Args:
            text (str): IN: Text to analyze. OUT: Lowercased and scanned for keywords.

        Returns:
            list[str]: OUT: Matched domain names.
        """
        text_lower = text.lower()
        out: list[str] = []
        for domain, keywords in _TECH_DOMAINS:
            if any(k in text_lower for k in keywords):
                out.append(domain)
        return out

    def _infer_tone(self, text: str) -> str:
        """Infer the user's tone from message characteristics.

        Args:
            text (str): IN: User message text. OUT: Analyzed for length and
                punctuation density.

        Returns:
            str: OUT: One of ``"terse"``, ``"casual"``, ``"verbose"``, or
                ``"balanced"``.
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
        """Extract explicit preference phrases from text using regex.

        Args:
            text (str): IN: Text to search. OUT: Scanned against preference patterns.

        Returns:
            list[str]: OUT: Matched preference phrases.
        """
        out: list[str] = []
        for pat in _PREFERENCE_PHRASES:
            for m in pat.finditer(text):
                phrase = m.group(0).strip().rstrip(".!?,")
                if 3 <= len(phrase) <= 200:
                    out.append(phrase)
        return out


def _default_ner(text: str) -> dict[str, list[str]]:
    """Default NER fallback using the EntityExtractor tool.

    Args:
        text (str): IN: Text to extract entities from. OUT: Passed to the static
            tool call.

    Returns:
        dict[str, list[str]]: OUT: Extracted entities by category, or an empty
            dict on failure.
    """
    try:
        from ..tools.ai_tools import EntityExtractor

        result = EntityExtractor.static_call(text)
        return result.get("entities", {}) if isinstance(result, dict) else {}
    except Exception:
        return {}


__all__ = ["ProfileAgent", "ProfileUpdate"]
