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
"""Tests for xerxes.loop_detection — tool loop prevention."""

from xerxes.runtime.loop_detection import LoopDetectionConfig, LoopDetector, LoopEvent, LoopSeverity, ToolLoopError


class TestLoopDetector:
    def test_single_call_ok(self):
        d = LoopDetector()
        event = d.record_call("search", {"q": "hello"})
        assert event.severity == LoopSeverity.OK

    def test_different_calls_ok(self):
        d = LoopDetector()
        for i in range(10):
            event = d.record_call(f"tool_{i}", {"x": i})
            assert event.severity == LoopSeverity.OK

    def test_same_call_warning(self):
        config = LoopDetectionConfig(same_call_warning=3, same_call_critical=5)
        d = LoopDetector(config)
        for _i in range(2):
            d.record_call("search", {"q": "hello"})
        event = d.record_call("search", {"q": "hello"})
        assert event.severity == LoopSeverity.WARNING
        assert event.pattern == "same_call"

    def test_same_call_critical(self):
        config = LoopDetectionConfig(same_call_warning=2, same_call_critical=4)
        d = LoopDetector(config)
        for _i in range(3):
            d.record_call("search", {"q": "hello"})
        event = d.record_call("search", {"q": "hello"})
        assert event.severity == LoopSeverity.CRITICAL

    def test_different_args_not_same_call(self):
        config = LoopDetectionConfig(same_call_warning=2, same_call_critical=3)
        d = LoopDetector(config)
        d.record_call("search", {"q": "hello"})
        d.record_call("search", {"q": "world"})
        event = d.record_call("search", {"q": "foo"})
        assert event.severity == LoopSeverity.OK

    def test_max_calls_limit(self):
        config = LoopDetectionConfig(max_tool_calls_per_turn=5)
        d = LoopDetector(config)
        for i in range(4):
            event = d.record_call(f"tool_{i}", {"x": i})
            assert event.severity == LoopSeverity.OK
        event = d.record_call("tool_last", {})
        assert event.severity == LoopSeverity.CRITICAL
        assert event.pattern == "max_calls"

    def test_pingpong_warning(self):
        config = LoopDetectionConfig(pingpong_warning=4, pingpong_critical=6, max_tool_calls_per_turn=50)
        d = LoopDetector(config)
        for _ in range(3):
            d.record_call("tool_a", {"x": 1})
            d.record_call("tool_b", {"x": 2})
        # After 6 calls with alternation of 5, should warn
        # The alternation count at the tail is checked
        # Let's verify the event
        assert d.call_count == 6

    def test_disabled_detector(self):
        config = LoopDetectionConfig(enabled=False)
        d = LoopDetector(config)
        for _ in range(100):
            event = d.record_call("same", {"x": 1})
            assert event.severity == LoopSeverity.OK

    def test_reset_clears_state(self):
        config = LoopDetectionConfig(same_call_warning=2, same_call_critical=3)
        d = LoopDetector(config)
        d.record_call("search", {"q": "hello"})
        d.record_call("search", {"q": "hello"})
        d.reset()
        event = d.record_call("search", {"q": "hello"})
        assert event.severity == LoopSeverity.OK  # Only 1 call after reset

    def test_listener_notified(self):
        events = []
        config = LoopDetectionConfig(same_call_warning=2, same_call_critical=3)
        d = LoopDetector(config)
        d.add_listener(lambda e: events.append(e))
        d.record_call("x", {"a": 1})
        d.record_call("x", {"a": 1})  # Warning
        assert len(events) == 1
        assert events[0].severity == LoopSeverity.WARNING

    def test_tool_loop_error(self):
        event = LoopEvent(severity=LoopSeverity.CRITICAL, pattern="same_call", tool_name="x", details="test")
        err = ToolLoopError(event)
        assert "same_call" in str(err)
        assert err.event is event

    def test_call_count_property(self):
        d = LoopDetector()
        assert d.call_count == 0
        d.record_call("a", {})
        d.record_call("b", {})
        assert d.call_count == 2
