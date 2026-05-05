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
"""Tests for the heuristic ProfileAgent."""

from __future__ import annotations

from xerxes.agents.profile_agent import ProfileAgent
from xerxes.memory import UserProfileStore


def _agent():
    return ProfileAgent(UserProfileStore())


class TestProfileAgent:
    def test_infers_python_domain(self):
        a = _agent()
        result = a.update("u1", user_prompt="please refactor this pytest fixture")
        assert "python" in result.domains_added
        assert "python" in a.store.get("u1").domains

    def test_infers_devops_domain(self):
        a = _agent()
        a.update("u1", user_prompt="set up GitHub Actions for the docker build")
        assert "devops" in a.store.get("u1").domains

    def test_terse_tone_short_prompt(self):
        a = _agent()
        a.update("u1", user_prompt="fix it")
        assert a.store.get("u1").tone.value == "terse"

    def test_verbose_tone_long_prompt(self):
        a = _agent()
        words = " ".join("hello" for _ in range(100))
        a.update("u1", user_prompt=words)
        assert a.store.get("u1").tone.value == "verbose"

    def test_explicit_preference_phrase_captured(self):
        a = _agent()
        result = a.update("u1", user_prompt="I prefer terse responses without preamble")
        assert any("prefer" in p.lower() for p in result.prefs_added)

    def test_dont_phrase_captured(self):
        a = _agent()
        result = a.update("u1", user_prompt="don't ever apologize in your responses")
        assert any("don" in p.lower() for p in result.prefs_added)

    def test_repeated_prompt_reinforces_pref(self):
        a = _agent()
        a.update("u1", user_prompt="I prefer terse responses")
        a.update("u1", user_prompt="I prefer terse responses")
        prefs = a.store.get("u1").explicit_preferences
        cv = next(iter(prefs.values()))
        assert cv.evidence_count >= 1
        assert cv.confidence >= 0.6

    def test_signals_recorded_in_history(self):
        a = _agent()
        a.update("u1", user_prompt="hi", signals=["correction", "revert"])
        history = a.store.get("u1").feedback_history
        assert any(h["signal"] == "correction" for h in history)
        assert any(h["signal"] == "revert" for h in history)

    def test_correction_demotes_tone_confidence(self):
        a = _agent()
        a.update("u1", user_prompt=" ".join("hello" for _ in range(100)))
        before = a.store.get("u1").tone.confidence
        a.update("u1", user_prompt="ok", signals=["correction", "correction"])
        after = a.store.get("u1").tone.confidence
        assert after <= before

    def test_llm_summariser_failure_does_not_break(self):
        def broken(_text, _prof):
            raise RuntimeError("boom")

        a = ProfileAgent(UserProfileStore(), llm_summariser=broken)
        a.update("u1", user_prompt="hi", agent_response="hello")
        assert a.store.get("u1") is not None

    def test_llm_summariser_appends_note(self):
        def s(_text, _prof):
            return "the user is friendly"

        a = ProfileAgent(UserProfileStore(), llm_summariser=s)
        a.update("u1", user_prompt="hi", agent_response="hello")
        assert "the user is friendly" in a.store.get("u1").notes

    def test_persists_via_store(self):
        store = UserProfileStore()
        a = ProfileAgent(store)
        a.update("alice", user_prompt="docker compose")
        assert "alice" in store.all_user_ids()
