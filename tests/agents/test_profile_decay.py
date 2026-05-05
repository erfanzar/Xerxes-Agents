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
"""Tests for UserProfileStore.decay_all profile aging."""

from __future__ import annotations

from datetime import datetime, timedelta

from xerxes.memory import ConfidentValue, UserProfileStore


def _aged_value(value, days_old, confidence=1.0):
    cv = ConfidentValue(value=value, confidence=confidence)
    cv.last_updated = datetime.now() - timedelta(days=days_old)
    return cv


class TestDecay:
    def test_recent_value_barely_decays(self):
        s = UserProfileStore()
        p = s.get_or_create("u1")
        p.tone = _aged_value("terse", days_old=1, confidence=0.8)
        s.save(p)
        s.decay_all(half_life_days=30, prune_threshold=0.05)
        assert s.get("u1").tone is not None
        assert 0.7 < s.get("u1").tone.confidence < 0.8

    def test_old_value_pruned(self):
        s = UserProfileStore()
        p = s.get_or_create("u1")
        p.tone = _aged_value("terse", days_old=180, confidence=0.5)
        p.expertise["python"] = _aged_value("expert", days_old=180, confidence=0.5)
        s.save(p)
        prunes = s.decay_all(half_life_days=30, prune_threshold=0.05)
        assert prunes["u1"] >= 1
        assert s.get("u1").tone is None or s.get("u1").tone.confidence < 0.05

    def test_high_confidence_survives_one_half_life(self):
        s = UserProfileStore()
        p = s.get_or_create("u1")
        p.expertise["python"] = _aged_value("expert", days_old=30, confidence=1.0)
        s.save(p)
        s.decay_all(half_life_days=30, prune_threshold=0.05)
        assert "python" in s.get("u1").expertise
        assert abs(s.get("u1").expertise["python"].confidence - 0.5) < 0.05

    def test_decay_persists(self):
        from xerxes.memory import SimpleStorage

        backing = SimpleStorage()
        s = UserProfileStore(backing)
        p = s.get_or_create("u1")
        p.expertise["x"] = _aged_value("y", days_old=180, confidence=1.0)
        s.save(p)
        s.decay_all(half_life_days=30, prune_threshold=0.05)
        s2 = UserProfileStore(backing)
        assert "x" not in s2.get("u1").expertise

    def test_decay_returns_per_user_prune_counts(self):
        s = UserProfileStore()
        a = s.get_or_create("a")
        b = s.get_or_create("b")
        a.expertise["k1"] = _aged_value("x", days_old=300, confidence=1.0)
        a.expertise["k2"] = _aged_value("x", days_old=300, confidence=1.0)
        b.expertise["k"] = _aged_value("x", days_old=1, confidence=0.9)
        s.save(a)
        s.save(b)
        prunes = s.decay_all(half_life_days=30, prune_threshold=0.05)
        assert prunes["a"] >= 1
        assert prunes["b"] == 0
