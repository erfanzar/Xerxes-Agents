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
"""Tests for UserProfile, ConfidentValue, and UserProfileStore."""

from __future__ import annotations

from xerxes.memory import (
    ConfidentValue,
    SimpleStorage,
    UserProfile,
    UserProfileStore,
)


class TestConfidentValue:
    def test_reinforce_increments_confidence(self):
        cv = ConfidentValue(value="x", confidence=0.3)
        cv.reinforce(0.4)
        assert cv.confidence == 0.7
        assert cv.evidence_count == 1

    def test_confidence_clamped_to_one(self):
        cv = ConfidentValue(value="x", confidence=0.8)
        cv.reinforce(0.5)
        assert cv.confidence == 1.0

    def test_demote_reduces_confidence(self):
        cv = ConfidentValue(value="x", confidence=0.9)
        cv.demote(0.4)
        assert abs(cv.confidence - 0.5) < 1e-9

    def test_demote_clamped_to_zero(self):
        cv = ConfidentValue(value="x", confidence=0.1)
        cv.demote(0.5)
        assert cv.confidence == 0.0


class TestUserProfile:
    def test_render_filters_low_confidence(self):
        p = UserProfile(user_id="u1")
        p.expertise["python"] = ConfidentValue(value="expert", confidence=0.9)
        p.expertise["rust"] = ConfidentValue(value="novice", confidence=0.1)
        rendered = p.render(min_confidence=0.5)
        assert "Expertise in python" in rendered
        assert "rust" not in rendered

    def test_render_caps_lines(self):
        p = UserProfile(user_id="u1")
        p.notes = [f"note {i}" for i in range(50)]
        rendered = p.render(max_lines=5)
        assert rendered.count("\n") < 5

    def test_render_empty_when_low_confidence(self):
        p = UserProfile(user_id="u1")
        p.expertise["python"] = ConfidentValue(value="x", confidence=0.05)
        assert p.render(min_confidence=0.5) == ""

    def test_record_feedback_appends(self):
        p = UserProfile(user_id="u1")
        p.record_feedback("correction", target="response_style")
        assert len(p.feedback_history) == 1
        assert p.feedback_history[0]["signal"] == "correction"

    def test_feedback_history_capped(self):
        p = UserProfile(user_id="u1")
        for _i in range(300):
            p.record_feedback("ping")
        assert len(p.feedback_history) <= 256

    def test_round_trip_dict(self):
        p = UserProfile(
            user_id="u1",
            domains=["python"],
            recurring_goals=["refactor"],
            notes=["likes terse"],
        )
        p.expertise["python"] = ConfidentValue(value="expert", confidence=0.9)
        p.tone = ConfidentValue(value="terse", confidence=0.8)
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        assert p2.user_id == "u1"
        assert p2.domains == ["python"]
        assert p2.expertise["python"].value == "expert"
        assert p2.tone.value == "terse"


class TestUserProfileStore:
    def test_get_or_create_persists_default(self):
        store = SimpleStorage()
        s = UserProfileStore(store)
        p = s.get_or_create("u1")
        assert isinstance(p, UserProfile)
        assert "_profile_u1" in store.list_keys()

    def test_save_round_trip(self):
        store = SimpleStorage()
        s = UserProfileStore(store)
        p = s.get_or_create("u1")
        p.domains.append("python")
        s.save(p)
        loaded = UserProfile.from_dict(store.load("_profile_u1"))
        assert "python" in loaded.domains

    def test_hydrate_on_init(self):
        store = SimpleStorage()
        s1 = UserProfileStore(store)
        p = s1.get_or_create("alice")
        p.notes.append("vegetarian")
        s1.save(p)
        s2 = UserProfileStore(store)
        assert s2.get("alice").notes == ["vegetarian"]

    def test_render_for_unknown_returns_empty(self):
        s = UserProfileStore()
        assert s.render_for("unknown") == ""

    def test_delete_removes_persisted_entry(self):
        store = SimpleStorage()
        s = UserProfileStore(store)
        s.get_or_create("u1")
        assert s.delete("u1") is True
        assert "_profile_u1" not in store.list_keys()

    def test_thread_safe(self):
        import threading

        store = UserProfileStore()

        def worker(i):
            p = store.get_or_create(f"u{i % 3}")
            p.notes.append(f"by-{i}")
            store.save(p)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(store.all_user_ids()) == ["u0", "u1", "u2"]
