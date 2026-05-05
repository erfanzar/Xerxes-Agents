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
"""Tests for memory + user-profile injection in PromptContextBuilder."""

from __future__ import annotations

from xerxes.runtime.context import PromptContextBuilder
from xerxes.runtime.profiles import PromptProfile, PromptProfileConfig


class TestMemoryInjection:
    def test_no_provider_yields_empty_section(self):
        b = PromptContextBuilder()
        ctx = b.build()
        assert ctx.memory_section == ""

    def test_provider_emits_relevant_memories_block(self):
        def provider(agent_id, k):
            return ["Project deadline is March 15", "User prefers terse output"]

        b = PromptContextBuilder(memory_provider=provider)
        ctx = b.build()
        assert "[Relevant Memories]" in ctx.memory_section
        assert "Project deadline is March 15" in ctx.memory_section
        assert "User prefers terse output" in ctx.memory_section

    def test_provider_respects_max_memories_injected(self):
        def provider(agent_id, k):
            return [f"memory {i}" for i in range(20)]

        cfg = PromptProfileConfig(profile=PromptProfile.FULL, max_memories_injected=3)
        b = PromptContextBuilder(memory_provider=provider, profile=cfg)
        ctx = b.build()
        assert ctx.memory_section.count("memory ") == 3

    def test_provider_failure_swallowed(self):
        def provider(agent_id, k):
            raise RuntimeError("vector store down")

        b = PromptContextBuilder(memory_provider=provider)
        ctx = b.build()
        assert ctx.memory_section == ""

    def test_provider_returns_empty_yields_no_section(self):
        b = PromptContextBuilder(memory_provider=lambda aid, k: [])
        ctx = b.build()
        assert ctx.memory_section == ""

    def test_minimal_profile_disables_memory_injection(self):
        b = PromptContextBuilder(memory_provider=lambda aid, k: ["x"])
        ctx = b.build(profile=PromptProfile.MINIMAL)
        assert ctx.memory_section == ""

    def test_memory_section_appears_in_assembled_prefix(self):
        b = PromptContextBuilder(memory_provider=lambda aid, k: ["The project is named Xerxes."])
        prefix = b.assemble_system_prompt_prefix()
        assert "[Relevant Memories]" in prefix
        assert "Xerxes" in prefix


class TestUserProfileInjection:
    def test_no_provider_yields_empty(self):
        b = PromptContextBuilder()
        ctx = b.build()
        assert ctx.user_profile_section == ""

    def test_provider_emits_block(self):
        b = PromptContextBuilder(user_profile_provider=lambda aid: "expertise: deep Python")
        ctx = b.build()
        assert "[User Profile]" in ctx.user_profile_section
        assert "deep Python" in ctx.user_profile_section

    def test_provider_failure_swallowed(self):
        def provider(aid):
            raise RuntimeError("db down")

        b = PromptContextBuilder(user_profile_provider=provider)
        ctx = b.build()
        assert ctx.user_profile_section == ""

    def test_minimal_profile_disables_profile_injection(self):
        b = PromptContextBuilder(user_profile_provider=lambda aid: "expert user")
        ctx = b.build(profile=PromptProfile.MINIMAL)
        assert ctx.user_profile_section == ""
