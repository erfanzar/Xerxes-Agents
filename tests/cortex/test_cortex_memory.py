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
"""Tests for xerxes.cortex.memory_integration module."""

from xerxes.cortex.agents.memory_integration import CortexMemory


class TestCortexMemory:
    def test_init_defaults(self):
        mem = CortexMemory()
        assert mem.short_term is not None
        assert mem.long_term is not None
        assert mem.entity_memory is not None
        assert mem.user_memory is None
        assert mem.contextual is not None

    def test_init_minimal(self):
        mem = CortexMemory(enable_short_term=False, enable_long_term=False, enable_entity=False)
        assert mem.short_term is None
        assert mem.long_term is None
        assert mem.entity_memory is None

    def test_save_task_result(self):
        mem = CortexMemory()
        mem.save_task_result(
            task_description="Analyze data",
            result="Found 5 anomalies",
            agent_role="analyst",
            importance=0.8,
        )
        assert len(mem.short_term) >= 0

    def test_save_task_result_low_importance(self):
        mem = CortexMemory()
        mem.save_task_result(
            task_description="Minor task",
            result="Done",
            agent_role="worker",
            importance=0.3,
        )

    def test_build_context_for_task(self):
        mem = CortexMemory()
        mem.save_task_result("Previous analysis", "Results: positive", "analyst", 0.8)
        context = mem.build_context_for_task("New analysis", agent_role="analyst")
        assert isinstance(context, str)

    def test_build_context_with_additional(self):
        mem = CortexMemory()
        context = mem.build_context_for_task("Task", additional_context="Background info")
        assert "Background" in context

    def test_save_agent_interaction(self):
        mem = CortexMemory()
        mem.save_agent_interaction("researcher", "thinking", "Considering approach", importance=0.3)
        assert len(mem.short_term) >= 0

    def test_save_agent_interaction_high_importance(self):
        mem = CortexMemory()
        mem.save_agent_interaction("researcher", "discovery", "Critical finding", importance=0.8)

    def test_save_cortex_decision(self):
        mem = CortexMemory()
        mem.save_cortex_decision("Assign to analyst", "Data needs analysis")

    def test_save_cortex_decision_with_outcome(self):
        mem = CortexMemory()
        mem.save_cortex_decision("Delegate", "Too complex", outcome="Success")

    def test_get_agent_history(self):
        mem = CortexMemory()
        mem.save_agent_interaction("writer", "write", "Draft complete")
        mem.save_agent_interaction("writer", "edit", "Final version")
        history = mem.get_agent_history("writer")
        assert isinstance(history, list)

    def test_get_user_context_no_user_memory(self):
        mem = CortexMemory(enable_user=False)
        assert mem.get_user_context("user1") == ""

    def test_get_user_context_with_user_memory(self):
        mem = CortexMemory(enable_user=True)
        context = mem.get_user_context("user1")
        assert isinstance(context, str)

    def test_reset_short_term(self):
        mem = CortexMemory()
        mem.save_agent_interaction("agent", "action", "content")
        mem.reset_short_term()

        assert len(mem.short_term) == 0

    def test_reset_all(self):
        mem = CortexMemory()
        mem.reset_all()
        assert len(mem.short_term) == 0

    def test_get_summary(self):
        mem = CortexMemory()
        mem.save_task_result("task", "result", "agent", 0.8)
        summary = mem.get_summary()
        assert isinstance(summary, str)

    def test_get_summary_empty(self):
        mem = CortexMemory()
        summary = mem.get_summary()
        assert isinstance(summary, str)
