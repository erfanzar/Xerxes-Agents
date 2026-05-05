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
"""Tests for OTelCollector — works whether or not OTel is installed."""

from __future__ import annotations

from xerxes.audit import (
    AuditEmitter,
    CompositeCollector,
    InMemoryCollector,
    OTelCollector,
)


class TestOTelCollector:
    def test_emit_turn_lifecycle(self):
        otel = OTelCollector(service_name="test")
        em = AuditEmitter(collector=otel)
        tid = em.emit_turn_start(agent_id="a", prompt="hello")
        em.emit_turn_end(agent_id="a", turn_id=tid, content="bye", fc_count=2)
        otel.flush()

    def test_emit_tool_calls(self):
        otel = OTelCollector(service_name="test")
        em = AuditEmitter(collector=otel)
        em.emit_tool_call_attempt("Read", args="{}")
        em.emit_tool_call_complete("Read", status="success", duration_ms=12.0)
        em.emit_tool_call_failure("Bash", error_type="ValueError", error_msg="x")
        em.emit_skill_used(skill_name="ci", outcome="success", duration_ms=42.0)
        em.emit_error(error_type="X", error_msg="boom")
        # No exceptions raised → API surface is sound regardless of OTel install state.

    def test_fanout_with_inmemory(self):
        inmem = InMemoryCollector()
        otel = OTelCollector()
        em = AuditEmitter(collector=CompositeCollector([inmem, otel]))
        em.emit_turn_start(agent_id="a", prompt="hi")
        em.emit_turn_end(agent_id="a")
        assert len(inmem.get_events()) == 2

    def test_fallback_log_when_no_otel(self):
        otel = OTelCollector()
        if otel.has_otel:
            return
        em = AuditEmitter(collector=otel)
        em.emit_tool_call_attempt("Read", args="{}")
        log = otel.fallback_log
        assert any(e["name"].startswith("tool.attempt") for e in log)

    def test_unknown_event_type_does_not_crash(self):
        from xerxes.audit.events import HookMutationEvent

        otel = OTelCollector()
        otel.emit(HookMutationEvent(hook_name="x", mutated_field="y"))

    def test_flush_clears_open_spans(self):
        otel = OTelCollector()
        em = AuditEmitter(collector=otel)
        em.emit_turn_start(agent_id="a", prompt="hi")
        otel.flush()
        # No turn_end called — flush should clean up regardless.
        assert otel._open_turn_spans == {}
