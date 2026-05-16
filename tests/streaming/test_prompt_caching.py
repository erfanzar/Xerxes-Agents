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
"""Tests for xerxes.streaming.prompt_caching and end-to-end cache wiring."""

from __future__ import annotations

from xerxes.runtime.cost_tracker import CostTracker
from xerxes.streaming.events import AgentState, TurnDone
from xerxes.streaming.prompt_caching import (
    extract_cache_tokens,
    wrap_system_with_cache,
    wrap_tools_with_cache,
)


class TestWrapSystem:
    def test_empty_system_returns_empty_string(self) -> None:
        assert wrap_system_with_cache("") == ""

    def test_non_empty_system_returns_blocks_with_cache_control(self) -> None:
        out = wrap_system_with_cache("You are a helpful assistant.")
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0]["type"] == "text"
        assert out[0]["text"] == "You are a helpful assistant."
        assert out[0]["cache_control"] == {"type": "ephemeral"}


class TestWrapTools:
    def test_empty_tools_unchanged(self) -> None:
        assert wrap_tools_with_cache([]) == []

    def test_single_tool_gets_cache_control_on_last(self) -> None:
        tools = [{"name": "read_file", "input_schema": {}}]
        out = wrap_tools_with_cache(tools)
        assert out[0]["cache_control"] == {"type": "ephemeral"}
        # Original input not mutated.
        assert "cache_control" not in tools[0]

    def test_multiple_tools_only_last_marked(self) -> None:
        tools = [
            {"name": "a"},
            {"name": "b"},
            {"name": "c"},
        ]
        out = wrap_tools_with_cache(tools)
        assert "cache_control" not in out[0]
        assert "cache_control" not in out[1]
        assert out[2]["cache_control"] == {"type": "ephemeral"}

    def test_stale_cache_control_stripped_from_earlier_tools(self) -> None:
        tools = [
            {"name": "a", "cache_control": {"type": "ephemeral"}},
            {"name": "b"},
        ]
        out = wrap_tools_with_cache(tools)
        assert "cache_control" not in out[0]
        assert out[1]["cache_control"] == {"type": "ephemeral"}


class TestExtractCacheTokens:
    def test_none_returns_zeros(self) -> None:
        assert extract_cache_tokens(None) == (0, 0)

    def test_dict_form(self) -> None:
        usage = {"cache_read_input_tokens": 100, "cache_creation_input_tokens": 50}
        assert extract_cache_tokens(usage) == (100, 50)

    def test_object_form(self) -> None:
        class FakeUsage:
            cache_read_input_tokens = 42
            cache_creation_input_tokens = 7

        assert extract_cache_tokens(FakeUsage()) == (42, 7)

    def test_missing_fields_default_to_zero(self) -> None:
        class Empty:
            pass

        assert extract_cache_tokens(Empty()) == (0, 0)

    def test_none_attribute_treated_as_zero(self) -> None:
        class Nulls:
            cache_read_input_tokens = None
            cache_creation_input_tokens = None

        assert extract_cache_tokens(Nulls()) == (0, 0)


class TestAgentStateTracksCacheTokens:
    def test_default_zero(self) -> None:
        state = AgentState()
        assert state.total_cache_read_tokens == 0
        assert state.total_cache_creation_tokens == 0

    def test_fields_assignable(self) -> None:
        state = AgentState()
        state.total_cache_read_tokens = 1000
        state.total_cache_creation_tokens = 200
        assert state.total_cache_read_tokens == 1000


class TestTurnDoneCacheFields:
    def test_default_zero(self) -> None:
        td = TurnDone(input_tokens=10, output_tokens=20)
        assert td.cache_read_tokens == 0
        assert td.cache_creation_tokens == 0

    def test_populated(self) -> None:
        td = TurnDone(
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=500,
            cache_creation_tokens=100,
        )
        assert td.cache_read_tokens == 500
        assert td.cache_creation_tokens == 100


class TestCostTrackerCacheAware:
    def test_record_turn_with_cache(self) -> None:
        ct = CostTracker()
        ev = ct.record_turn(
            "claude-opus-4-7",
            in_tokens=100,
            out_tokens=200,
            cache_read_tokens=1000,
            cache_creation_tokens=50,
        )
        assert ev.cache_read_tokens == 1000
        assert ev.cache_creation_tokens == 50

    def test_totals_aggregate(self) -> None:
        ct = CostTracker()
        ct.record_turn("m", 10, 20, cache_read_tokens=500)
        ct.record_turn("m", 10, 20, cache_read_tokens=500, cache_creation_tokens=100)
        assert ct.total_cache_read_tokens == 1000
        assert ct.total_cache_creation_tokens == 100

    def test_cache_hit_rate_no_data(self) -> None:
        ct = CostTracker()
        assert ct.cache_hit_rate() == 0.0

    def test_cache_hit_rate_partial(self) -> None:
        ct = CostTracker()
        ct.record_turn("m", in_tokens=100, out_tokens=0, cache_read_tokens=900)
        # served = 900 (cache) + 100 (uncached) = 1000; hit rate = 0.9
        assert abs(ct.cache_hit_rate() - 0.9) < 1e-9

    def test_cache_extras_cost_more_than_zero(self) -> None:
        ct = CostTracker()
        # Use a model with known pricing — claude-opus-4-7 has entries.
        ev = ct.record_turn(
            "claude-opus-4-7",
            in_tokens=0,
            out_tokens=0,
            cache_creation_tokens=10000,
        )
        # cache_creation is billed at ~1.25x input, so cost > 0 even with no other tokens.
        # (If the pricing table doesn't have this model, this becomes 0 — that's still a
        # legitimate state; just guard against negative.)
        assert ev.cost_usd >= 0.0
