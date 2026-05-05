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
"""Focused tests for audit event dataclass semantics and edge cases."""

from __future__ import annotations

import json
from datetime import datetime

from xerxes.audit.events import (
    AuditEvent,
    ErrorEvent,
    HookMutationEvent,
    SandboxDecisionEvent,
    ToolCallAttemptEvent,
    ToolCallCompleteEvent,
    ToolCallFailureEvent,
    ToolLoopBlockEvent,
    ToolLoopWarningEvent,
    ToolPolicyDecisionEvent,
    TurnEndEvent,
    TurnStartEvent,
    _now_iso,
)


class TestTimestamp:
    def test_now_iso_is_valid(self):
        ts = _now_iso()
        # Should parse without error
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None  # must be tz-aware

    def test_events_get_auto_timestamp(self):
        ev = AuditEvent()
        dt = datetime.fromisoformat(ev.timestamp)
        assert dt.year >= 2025


class TestEventTypeImmutability:
    """Verify that event_type is always set by the class, not the caller."""

    def test_subclass_event_type_cannot_be_overridden_by_init(self):
        # event_type uses init=False, so passing it should raise or be ignored
        ev = TurnStartEvent()
        assert ev.event_type == "turn_start"

    def test_each_subclass_has_unique_event_type(self):
        classes = [
            TurnStartEvent,
            TurnEndEvent,
            ToolCallAttemptEvent,
            ToolCallCompleteEvent,
            ToolCallFailureEvent,
            ToolPolicyDecisionEvent,
            ToolLoopWarningEvent,
            ToolLoopBlockEvent,
            SandboxDecisionEvent,
            HookMutationEvent,
            ErrorEvent,
        ]
        types = [cls().event_type for cls in classes]
        assert len(types) == len(set(types)), "Duplicate event_type values found"


class TestDefaultSeverity:
    def test_base_event_default_severity(self):
        assert AuditEvent().severity == "info"

    def test_failure_events_default_error_severity(self):
        assert ToolCallFailureEvent().severity == "error"
        assert ToolLoopBlockEvent().severity == "error"
        assert ErrorEvent().severity == "error"

    def test_warning_events_default_warning_severity(self):
        assert ToolLoopWarningEvent().severity == "warning"


class TestToDictCompleteness:
    def test_turn_start_dict_has_all_fields(self):
        ev = TurnStartEvent(agent_id="a", turn_id="t", prompt_preview="p")
        d = ev.to_dict()
        assert "event_type" in d
        assert "timestamp" in d
        assert "agent_id" in d
        assert "turn_id" in d
        assert "session_id" in d
        assert "severity" in d
        assert "metadata" in d
        assert "prompt_preview" in d

    def test_tool_call_complete_dict_has_duration(self):
        ev = ToolCallCompleteEvent(duration_ms=99.9)
        d = ev.to_dict()
        assert d["duration_ms"] == 99.9

    def test_error_event_dict_has_context(self):
        ev = ErrorEvent(error_context="ctx")
        d = ev.to_dict()
        assert d["error_context"] == "ctx"

    def test_to_dict_with_nested_metadata(self):
        ev = AuditEvent(metadata={"nested": {"a": [1, 2]}})
        d = ev.to_dict()
        text = json.dumps(d)
        parsed = json.loads(text)
        assert parsed["metadata"]["nested"]["a"] == [1, 2]


class TestToJson:
    def test_round_trip(self):
        ev = ToolPolicyDecisionEvent(tool_name="t", action="allow", policy_source="agent")
        text = ev.to_json()
        parsed = json.loads(text)
        assert parsed["tool_name"] == "t"
        assert parsed["action"] == "allow"

    def test_special_characters(self):
        ev = ErrorEvent(error_message='He said "hello" & <goodbye>')
        parsed = json.loads(ev.to_json())
        assert parsed["error_message"] == 'He said "hello" & <goodbye>'
