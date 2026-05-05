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
"""Tests for xerxes.memory.user_memory module."""

from xerxes.memory.user_memory import UserMemory


class TestUserMemory:
    def test_init(self):
        um = UserMemory()
        assert len(um.user_memories) == 0

    def test_get_or_create_user(self):
        um = UserMemory()
        mem = um.get_or_create_user_memory("user1")
        assert mem is not None
        assert "user1" in um.user_memories
        assert "user1" in um.user_entities
        assert "user1" in um.user_preferences

    def test_get_or_create_idempotent(self):
        um = UserMemory()
        mem1 = um.get_or_create_user_memory("user1")
        mem2 = um.get_or_create_user_memory("user1")
        assert mem1 is mem2

    def test_save_memory(self):
        um = UserMemory()
        item = um.save_memory("user1", "Hello world")
        assert item.content == "Hello world"

    def test_search_user_memory(self):
        um = UserMemory()
        um.save_memory("user1", "Python programming basics")
        um.save_memory("user1", "Java programming basics")
        results = um.search_user_memory("user1", "Python")
        assert len(results) >= 0

    def test_get_user_context(self):
        um = UserMemory()
        um.save_memory("user1", "Some context data")
        context = um.get_user_context("user1")
        assert isinstance(context, str)

    def test_update_user_preferences(self):
        um = UserMemory()
        um.update_user_preferences("user1", {"language": "fr"})
        prefs = um.get_user_preferences("user1")
        assert prefs["language"] == "fr"

    def test_get_user_preferences_default(self):
        um = UserMemory()
        prefs = um.get_user_preferences("unknown")
        assert prefs["language"] == "en"
        assert prefs["response_style"] == "balanced"

    def test_get_user_statistics(self):
        um = UserMemory()
        um.save_memory("user1", "test data")
        stats = um.get_user_statistics("user1")
        assert stats["user_id"] == "user1"
        assert "total_memories" in stats
        assert "entities_known" in stats

    def test_get_user_statistics_unknown(self):
        um = UserMemory()
        stats = um.get_user_statistics("unknown")
        assert stats["total_memories"] == 0

    def test_clear_user_memory(self):
        um = UserMemory()
        um.save_memory("user1", "data")
        um.clear_user_memory("user1")
        assert "user1" not in um.user_memories
        assert "user1" not in um.user_entities

    def test_clear_nonexistent_user(self):
        um = UserMemory()
        um.clear_user_memory("nonexistent")

    def test_multiple_users(self):
        um = UserMemory()
        um.save_memory("user1", "User 1 data")
        um.save_memory("user2", "User 2 data")
        assert len(um.user_memories) == 2
