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
"""Tests for runtime.pricing + runtime.insights."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from xerxes.runtime.cost_tracker import CostEvent
from xerxes.runtime.insights import build_report, format_report
from xerxes.runtime.pricing import compute_cost, get_pricing, known_models


class TestPricing:
    def test_at_least_30_models(self):
        # Plan 26 targets 50+; defend a conservative lower bound.
        assert len(known_models()) >= 30

    def test_known_keys_include_each_provider(self):
        ks = set(known_models())
        assert "claude-opus-4-7" in ks
        assert "gpt-4o" in ks
        assert "gemini-2.5-pro" in ks
        assert any(k.startswith("mistralai/") for k in ks)

    def test_compute_cost_no_model_returns_zero(self):
        assert compute_cost(model="not-a-real-model", input_tokens=1_000_000) == 0.0

    def test_compute_cost_basic(self):
        cost = compute_cost(model="claude-opus-4-7", input_tokens=1_000_000, output_tokens=0)
        assert cost == 15.0  # $15/M input

    def test_compute_cost_with_cache(self):
        # 1M input, 1M cache_read, 1M cache_write
        cost = compute_cost(
            model="claude-opus-4-7",
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=1_000_000,
            cache_write_tokens=1_000_000,
        )
        # $15 input + $1.5 cache_read + $18.75 cache_write = $35.25
        assert abs(cost - 35.25) < 1e-6

    def test_get_pricing(self):
        p = get_pricing("gpt-4o")
        assert p is not None
        assert p.input_per_million == 5.0


class TestInsights:
    def _ev(
        self, ts_offset_days=0, model="claude-opus-4-7", in_tokens=100, out_tokens=200, cost=0.01, label="", cache_read=0
    ):
        now = datetime.now(UTC) - timedelta(days=ts_offset_days)
        return CostEvent(
            model=model,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            cost_usd=cost,
            label=label,
            timestamp=now.isoformat(),
            cache_read_tokens=cache_read,
        )

    def test_empty_events(self):
        rpt = build_report([])
        assert rpt.total_events == 0
        assert rpt.projected_monthly_cost == 0.0

    def test_aggregates_totals(self):
        events = [
            self._ev(in_tokens=100, out_tokens=200, cost=0.01),
            self._ev(in_tokens=50, out_tokens=100, cost=0.005),
        ]
        rpt = build_report(events)
        assert rpt.total_events == 2
        assert rpt.total_input_tokens == 150
        assert rpt.total_output_tokens == 300
        assert abs(rpt.total_cost_usd - 0.015) < 1e-9

    def test_filters_by_days(self):
        events = [
            self._ev(ts_offset_days=10, cost=0.01),
            self._ev(ts_offset_days=1, cost=0.02),
        ]
        rpt = build_report(events, days=7)
        # Only the recent one is counted.
        assert rpt.total_events == 1
        assert abs(rpt.total_cost_usd - 0.02) < 1e-9

    def test_by_model_breakdown(self):
        events = [
            self._ev(model="A", cost=0.01),
            self._ev(model="A", cost=0.02),
            self._ev(model="B", cost=0.05),
        ]
        rpt = build_report(events)
        assert rpt.by_model["A"]["events"] == 2
        assert abs(rpt.by_model["A"]["cost_usd"] - 0.03) < 1e-9

    def test_by_label_counts(self):
        events = [self._ev(label="user_turn"), self._ev(label="user_turn"), self._ev(label="aux")]
        rpt = build_report(events)
        assert rpt.by_label["user_turn"] == 2

    def test_cache_hit_rate(self):
        events = [self._ev(in_tokens=200, cache_read=800)]
        rpt = build_report(events)
        # served = 200 + 800 = 1000; hit rate = 800 / 1000 = 0.8.
        assert abs(rpt.cache_hit_rate - 0.8) < 1e-9

    def test_projected_monthly(self):
        events = [self._ev(ts_offset_days=0, cost=10.0)]
        rpt = build_report(events)
        assert rpt.projected_monthly_cost == 300.0

    def test_format_report_returns_text(self):
        events = [self._ev(model="A", cost=0.05)]
        out = format_report(build_report(events))
        assert "Events: 1" in out
        assert "Top models" in out
