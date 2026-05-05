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
"""Tests for the xerxes.audit structured event system."""

from __future__ import annotations

import json
import threading
from io import StringIO

from xerxes.audit import (
    AuditEmitter,
    AuditEvent,
    CompositeCollector,
    ErrorEvent,
    HookMutationEvent,
    InMemoryCollector,
    JSONLSinkCollector,
    SandboxDecisionEvent,
    ToolCallAttemptEvent,
    ToolCallCompleteEvent,
    ToolCallFailureEvent,
    ToolLoopBlockEvent,
    ToolLoopWarningEvent,
    ToolPolicyDecisionEvent,
    TurnEndEvent,
    TurnStartEvent,
)


class TestEventCreation:
    def test_base_event_defaults(self):
        ev = AuditEvent()
        assert ev.event_type == "base"
        assert ev.severity == "info"
        assert ev.agent_id is None
        assert ev.turn_id is None
        assert ev.session_id is None
        assert isinstance(ev.metadata, dict)
        assert ev.timestamp

    def test_turn_start_event(self):
        ev = TurnStartEvent(agent_id="a1", turn_id="t1", prompt_preview="hello")
        assert ev.event_type == "turn_start"
        assert ev.agent_id == "a1"
        assert ev.prompt_preview == "hello"

    def test_turn_end_event(self):
        ev = TurnEndEvent(content_preview="done", function_calls_count=3)
        assert ev.event_type == "turn_end"
        assert ev.function_calls_count == 3

    def test_tool_call_attempt_event(self):
        ev = ToolCallAttemptEvent(tool_name="search", arguments_preview='{"q":"hi"}')
        assert ev.event_type == "tool_call_attempt"
        assert ev.tool_name == "search"

    def test_tool_policy_decision_event(self):
        ev = ToolPolicyDecisionEvent(tool_name="rm", action="deny", policy_source="global")
        assert ev.event_type == "tool_policy_decision"
        assert ev.action == "deny"

    def test_tool_loop_warning_event(self):
        ev = ToolLoopWarningEvent(tool_name="fetch", pattern="repeat", severity_level="warning", call_count=5)
        assert ev.event_type == "tool_loop_warning"
        assert ev.severity == "warning"
        assert ev.call_count == 5

    def test_tool_loop_block_event(self):
        ev = ToolLoopBlockEvent(tool_name="fetch", pattern="repeat", call_count=10)
        assert ev.event_type == "tool_loop_block"
        assert ev.severity == "error"

    def test_sandbox_decision_event(self):
        ev = SandboxDecisionEvent(tool_name="exec", context="unsafe", reason="pattern", backend_type="docker")
        assert ev.event_type == "sandbox_decision"
        assert ev.backend_type == "docker"

    def test_tool_call_complete_event(self):
        ev = ToolCallCompleteEvent(tool_name="search", status="success", duration_ms=123.4, result_preview="ok")
        assert ev.event_type == "tool_call_complete"
        assert ev.duration_ms == 123.4

    def test_tool_call_failure_event(self):
        ev = ToolCallFailureEvent(tool_name="search", error_type="Timeout", error_message="timed out")
        assert ev.event_type == "tool_call_failure"
        assert ev.severity == "error"

    def test_hook_mutation_event(self):
        ev = HookMutationEvent(hook_name="before_tool_call", tool_name="search", agent_id="a1", mutated_field="args")
        assert ev.event_type == "hook_mutation"

    def test_error_event(self):
        ev = ErrorEvent(error_type="RuntimeError", error_message="boom", error_context="executor")
        assert ev.event_type == "error"
        assert ev.severity == "error"


class TestEventSerialization:
    def test_to_dict_is_plain_dict(self):
        ev = TurnStartEvent(agent_id="a1", prompt_preview="hello")
        d = ev.to_dict()
        assert isinstance(d, dict)
        assert d["event_type"] == "turn_start"
        assert d["agent_id"] == "a1"

    def test_to_dict_json_serializable(self):
        ev = ToolCallCompleteEvent(tool_name="t", duration_ms=1.5, result_preview="r")
        text = json.dumps(ev.to_dict())
        parsed = json.loads(text)
        assert parsed["tool_name"] == "t"

    def test_to_json_produces_valid_json(self):
        ev = ErrorEvent(error_type="E", error_message="msg")
        parsed = json.loads(ev.to_json())
        assert parsed["event_type"] == "error"

    def test_all_event_types_serializable(self):
        events = [
            AuditEvent(),
            TurnStartEvent(),
            TurnEndEvent(),
            ToolCallAttemptEvent(),
            ToolCallCompleteEvent(),
            ToolCallFailureEvent(),
            ToolPolicyDecisionEvent(),
            ToolLoopWarningEvent(),
            ToolLoopBlockEvent(),
            SandboxDecisionEvent(),
            HookMutationEvent(),
            ErrorEvent(),
        ]
        for ev in events:
            d = ev.to_dict()
            text = json.dumps(d, default=str)
            assert isinstance(json.loads(text), dict)

    def test_all_event_types_have_correct_event_type(self):
        mapping = {
            AuditEvent: "base",
            TurnStartEvent: "turn_start",
            TurnEndEvent: "turn_end",
            ToolCallAttemptEvent: "tool_call_attempt",
            ToolCallCompleteEvent: "tool_call_complete",
            ToolCallFailureEvent: "tool_call_failure",
            ToolPolicyDecisionEvent: "tool_policy_decision",
            ToolLoopWarningEvent: "tool_loop_warning",
            ToolLoopBlockEvent: "tool_loop_block",
            SandboxDecisionEvent: "sandbox_decision",
            HookMutationEvent: "hook_mutation",
            ErrorEvent: "error",
        }
        for cls, expected_type in mapping.items():
            ev = cls()
            assert ev.event_type == expected_type, f"{cls.__name__}.event_type={ev.event_type!r}"

    def test_metadata_round_trips(self):
        ev = AuditEvent(metadata={"key": [1, 2, 3]})
        d = ev.to_dict()
        assert d["metadata"] == {"key": [1, 2, 3]}
        text = json.dumps(d)
        assert json.loads(text)["metadata"]["key"] == [1, 2, 3]


class TestInMemoryCollector:
    def test_emit_and_get(self):
        c = InMemoryCollector()
        e1 = TurnStartEvent(agent_id="a1")
        e2 = TurnEndEvent(agent_id="a1")
        c.emit(e1)
        c.emit(e2)
        assert len(c) == 2
        assert c.get_events() == [e1, e2]

    def test_get_events_returns_copy(self):
        c = InMemoryCollector()
        c.emit(AuditEvent())
        events = c.get_events()
        events.clear()
        assert len(c) == 1

    def test_get_events_by_type(self):
        c = InMemoryCollector()
        c.emit(TurnStartEvent())
        c.emit(TurnEndEvent())
        c.emit(TurnStartEvent())
        assert len(c.get_events_by_type("turn_start")) == 2
        assert len(c.get_events_by_type("turn_end")) == 1
        assert len(c.get_events_by_type("nonexistent")) == 0

    def test_clear(self):
        c = InMemoryCollector()
        c.emit(AuditEvent())
        c.clear()
        assert len(c) == 0

    def test_flush_is_noop(self):
        c = InMemoryCollector()
        c.flush()

    def test_event_ordering_preserved(self):
        c = InMemoryCollector()
        for i in range(100):
            c.emit(AuditEvent(metadata={"i": i}))
        events = c.get_events()
        for i, ev in enumerate(events):
            assert ev.metadata["i"] == i


class TestJSONLSinkCollector:
    def test_write_to_stringio(self):
        buf = StringIO()
        c = JSONLSinkCollector(buf)
        c.emit(TurnStartEvent(agent_id="a1", prompt_preview="hi"))
        c.emit(TurnEndEvent(content_preview="bye"))
        c.flush()

        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["event_type"] == "turn_start"
        assert first["agent_id"] == "a1"

        second = json.loads(lines[1])
        assert second["event_type"] == "turn_end"

    def test_write_to_file(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        c = JSONLSinkCollector(path)
        c.emit(AuditEvent(metadata={"x": 1}))
        c.flush()
        c.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["metadata"]["x"] == 1

    def test_flush_forces_write(self):
        buf = StringIO()
        c = JSONLSinkCollector(buf)
        c.emit(AuditEvent())
        c.flush()
        assert len(buf.getvalue()) > 0


class TestCompositeCollector:
    def test_fanout(self):
        m1 = InMemoryCollector()
        m2 = InMemoryCollector()
        comp = CompositeCollector([m1, m2])

        ev = TurnStartEvent()
        comp.emit(ev)

        assert len(m1) == 1
        assert len(m2) == 1

    def test_add(self):
        comp = CompositeCollector()
        m = InMemoryCollector()
        comp.add(m)
        comp.emit(AuditEvent())
        assert len(m) == 1

    def test_flush_propagates(self):
        buf = StringIO()
        jsonl = JSONLSinkCollector(buf)
        mem = InMemoryCollector()
        comp = CompositeCollector([jsonl, mem])
        comp.emit(AuditEvent())
        comp.flush()
        assert len(buf.getvalue()) > 0
        assert len(mem) == 1

    def test_mixed_collectors(self):
        buf = StringIO()
        mem = InMemoryCollector()
        jsonl = JSONLSinkCollector(buf)
        comp = CompositeCollector([mem, jsonl])

        for i in range(5):
            comp.emit(AuditEvent(metadata={"i": i}))
        comp.flush()

        assert len(mem) == 5
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) == 5


class TestAuditEmitter:
    def test_default_collector(self):
        em = AuditEmitter()
        assert isinstance(em.collector, InMemoryCollector)

    def test_session_id_stamped(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem, session_id="sess-1")
        em.emit_turn_start(agent_id="a1", prompt="hello")
        ev = mem.get_events()[0]
        assert ev.session_id == "sess-1"

    def test_emit_turn_start_returns_turn_id(self):
        em = AuditEmitter()
        tid = em.emit_turn_start(agent_id="a1")
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_emit_turn_start_custom_turn_id(self):
        em = AuditEmitter()
        tid = em.emit_turn_start(turn_id="custom-id")
        assert tid == "custom-id"

    def test_emit_turn_end(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_turn_end(agent_id="a1", turn_id="t1", content="done", fc_count=2)
        ev = mem.get_events()[0]
        assert isinstance(ev, TurnEndEvent)
        assert ev.function_calls_count == 2

    def test_emit_tool_call_attempt(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_tool_call_attempt("search", '{"q":"hi"}', "a1")
        ev = mem.get_events()[0]
        assert isinstance(ev, ToolCallAttemptEvent)
        assert ev.tool_name == "search"

    def test_emit_tool_policy_decision(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_tool_policy_decision("rm", agent_id="a1", action="deny", source="global")
        ev = mem.get_events()[0]
        assert isinstance(ev, ToolPolicyDecisionEvent)
        assert ev.action == "deny"

    def test_emit_tool_loop_warning(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_tool_loop_warning("fetch", pattern="repeat", severity="warning", count=5)
        ev = mem.get_events()[0]
        assert isinstance(ev, ToolLoopWarningEvent)
        assert ev.call_count == 5

    def test_emit_tool_loop_block(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_tool_loop_block("fetch", pattern="repeat", count=10)
        ev = mem.get_events()[0]
        assert isinstance(ev, ToolLoopBlockEvent)

    def test_emit_sandbox_decision(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_sandbox_decision("exec", context="unsafe", reason="pattern", backend_type="docker")
        ev = mem.get_events()[0]
        assert isinstance(ev, SandboxDecisionEvent)
        assert ev.backend_type == "docker"

    def test_emit_tool_call_complete(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_tool_call_complete("search", status="success", duration_ms=42.0, result="ok")
        ev = mem.get_events()[0]
        assert isinstance(ev, ToolCallCompleteEvent)
        assert ev.duration_ms == 42.0

    def test_emit_tool_call_failure(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_tool_call_failure("search", error_type="Timeout", error_msg="timed out")
        ev = mem.get_events()[0]
        assert isinstance(ev, ToolCallFailureEvent)
        assert ev.error_type == "Timeout"

    def test_emit_hook_mutation(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_hook_mutation("before_tool_call", tool_name="search", agent_id="a1", field="args")
        ev = mem.get_events()[0]
        assert isinstance(ev, HookMutationEvent)
        assert ev.mutated_field == "args"

    def test_emit_error(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_error(error_type="RuntimeError", error_msg="boom", context="exec")
        ev = mem.get_events()[0]
        assert isinstance(ev, ErrorEvent)
        assert ev.error_message == "boom"

    def test_emit_agent_switch(self):
        """Test that AgentSwitchEvent is emitted correctly."""
        from xerxes.audit.events import AgentSwitchEvent

        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        em.emit_agent_switch(
            from_agent="agent1",
            to_agent="agent2",
            reason="Function foo requested handoff to agent agent2",
            agent_id="agent1",
            turn_id="turn-1",
        )
        ev = mem.get_events()[0]
        assert isinstance(ev, AgentSwitchEvent)
        assert ev.from_agent == "agent1"
        assert ev.to_agent == "agent2"
        assert ev.reason == "Function foo requested handoff to agent agent2"
        assert ev.agent_id == "agent1"
        assert ev.turn_id == "turn-1"

    def test_prompt_preview_truncated(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        long_prompt = "x" * 500
        em.emit_turn_start(prompt=long_prompt)
        ev = mem.get_events()[0]
        assert len(ev.prompt_preview) == 200

    def test_flush(self):
        buf = StringIO()
        jsonl = JSONLSinkCollector(buf)
        em = AuditEmitter(collector=jsonl)
        em.emit_error(error_type="E", error_msg="m")
        em.flush()
        assert len(buf.getvalue()) > 0


class TestThreadSafety:
    def test_concurrent_emit_to_in_memory(self):
        mem = InMemoryCollector()
        em = AuditEmitter(collector=mem)
        n_threads = 10
        n_per_thread = 100

        def emit_batch(thread_idx: int):
            for i in range(n_per_thread):
                em.emit_turn_start(agent_id=f"t{thread_idx}", prompt=f"msg-{i}")

        threads = [threading.Thread(target=emit_batch, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(mem) == n_threads * n_per_thread

    def test_concurrent_emit_to_jsonl(self):
        buf = StringIO()
        jsonl = JSONLSinkCollector(buf)
        em = AuditEmitter(collector=jsonl)
        n_threads = 8
        n_per_thread = 50

        def emit_batch(thread_idx: int):
            for i in range(n_per_thread):
                em.emit_tool_call_attempt(f"tool-{thread_idx}", f"arg-{i}", f"a{thread_idx}")

        threads = [threading.Thread(target=emit_batch, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        em.flush()
        lines = [el for el in buf.getvalue().strip().split("\n") if el]
        assert len(lines) == n_threads * n_per_thread

        for line in lines:
            json.loads(line)

    def test_concurrent_emit_to_composite(self):
        mem = InMemoryCollector()
        buf = StringIO()
        jsonl = JSONLSinkCollector(buf)
        comp = CompositeCollector([mem, jsonl])
        em = AuditEmitter(collector=comp)
        n_threads = 6
        n_per_thread = 50

        def emit_batch(thread_idx: int):
            for i in range(n_per_thread):
                em.emit_error(error_type="E", error_msg=f"{thread_idx}-{i}")

        threads = [threading.Thread(target=emit_batch, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        em.flush()
        total = n_threads * n_per_thread
        assert len(mem) == total
        lines = [el for el in buf.getvalue().strip().split("\n") if el]
        assert len(lines) == total
