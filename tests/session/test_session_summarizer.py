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
"""Tests for SessionSummarizer and SearchHistoryTool."""

from __future__ import annotations

from datetime import datetime

from xerxes.session.models import SessionRecord, ToolCallRecord, TurnRecord
from xerxes.session.store import InMemorySessionStore
from xerxes.session.summarizer import SessionSummarizer
from xerxes.tools.history_tool import SearchHistoryTool


def _now() -> str:
    return datetime.now().isoformat()


def _turn(turn_id, prompt, response="", agent="a", status="success", tools=()):
    tcs = [
        ToolCallRecord(
            call_id=f"{turn_id}-{i}",
            tool_name=name,
            arguments={},
            status="success",
            duration_ms=1.0,
        )
        for i, name in enumerate(tools)
    ]
    return TurnRecord(
        turn_id=turn_id,
        agent_id=agent,
        prompt=prompt,
        response_content=response,
        started_at=_now(),
        status=status,
        tool_calls=tcs,
    )


def _session(sid, turns):
    return SessionRecord(session_id=sid, created_at=_now(), turns=turns)


class TestSessionSummarizer:
    def test_empty_session(self):
        s = SessionSummarizer().summarize(_session("s1", []))
        assert s.title.startswith("Session ")
        assert "Empty" in s.synopsis
        assert s.outcome == "unknown"

    def test_title_from_first_prompt(self):
        s = SessionSummarizer().summarize(
            _session("s1", [_turn("t1", "configure github actions for my python project")])
        )
        assert "github actions" in s.title.lower()

    def test_long_prompt_truncated(self):
        words = " ".join("word" for _ in range(50))
        s = SessionSummarizer().summarize(_session("s1", [_turn("t1", words)]))
        assert "…" in s.title

    def test_outcome_success_when_all_ok(self):
        s = SessionSummarizer().summarize(_session("s1", [_turn("t1", "ok"), _turn("t2", "ok")]))
        assert s.outcome == "success"

    def test_outcome_mixed(self):
        s = SessionSummarizer().summarize(_session("s1", [_turn("t1", "ok"), _turn("t2", "x", status="error")]))
        assert s.outcome == "mixed"

    def test_outcome_failure(self):
        s = SessionSummarizer().summarize(_session("s1", [_turn("t1", "x", status="error")]))
        assert s.outcome == "failure"

    def test_key_actions_dedup_in_order(self):
        sess = _session(
            "s1",
            [
                _turn("t1", "x", tools=("Read", "Bash")),
                _turn("t2", "y", tools=("Bash", "Edit")),
            ],
        )
        s = SessionSummarizer().summarize(sess)
        assert s.key_actions == ["Read", "Bash", "Edit"]

    def test_distinct_agents(self):
        sess = _session(
            "s1",
            [
                _turn("t1", "x", agent="coder"),
                _turn("t2", "y", agent="reviewer"),
                _turn("t3", "z", agent="coder"),
            ],
        )
        s = SessionSummarizer().summarize(sess)
        assert s.agent_ids == ["coder", "reviewer"]

    def test_llm_refinement_applied(self):
        called = {"n": 0}

        def llm(_p):
            called["n"] += 1
            return "Refined synopsis from LLM."

        s = SessionSummarizer(llm_client=llm).summarize(_session("s1", [_turn("t1", "p", "r")]))
        assert s.synopsis == "Refined synopsis from LLM."
        assert called["n"] == 1

    def test_llm_failure_falls_back(self):
        def boom(_p):
            raise RuntimeError("api down")

        s = SessionSummarizer(llm_client=boom).summarize(_session("s1", [_turn("t1", "hi")]))
        assert "User asked" in s.synopsis


class TestSearchHistoryTool:
    def test_via_store(self):
        store = InMemorySessionStore()
        store.save_session(_session("s1", [_turn("t1", "set up CI", "use actions")]))
        tool = SearchHistoryTool(store=store)
        result = tool("CI", limit=5)
        assert result["count"] == 1
        assert result["hits"][0]["session_id"] == "s1"

    def test_via_index(self):
        from xerxes.session.index import SessionIndex

        idx = SessionIndex(":memory:")
        idx.index_session(_session("s1", [_turn("t1", "deploy with terraform")]))
        tool = SearchHistoryTool(index=idx)
        result = tool("terraform")
        assert result["hits"][0]["turn_id"] == "t1"

    def test_filter_by_agent(self):
        store = InMemorySessionStore()
        store.save_session(
            _session(
                "s1",
                [
                    _turn("t1", "X", agent="a"),
                    _turn("t2", "X", agent="b"),
                ],
            )
        )
        tool = SearchHistoryTool(store=store)
        result = tool("X", agent_id="a")
        assert {h["agent_id"] for h in result["hits"]} == {"a"}

    def test_requires_store_or_index(self):
        import pytest

        with pytest.raises(ValueError):
            SearchHistoryTool()

    def test_empty_query_returns_zero_hits(self):
        store = InMemorySessionStore()
        store.save_session(_session("s1", [_turn("t1", "x")]))
        tool = SearchHistoryTool(store=store)
        result = tool("")
        assert result["count"] == 0
