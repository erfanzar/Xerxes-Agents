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
"""Tests for agent meta-tools (mixture_of_agents + skill_* + aliases)."""

from __future__ import annotations

import pytest
from xerxes.extensions.skills import SkillRegistry
from xerxes.tools.agent_meta_tools import (
    configure_mixture_of_agents,
    mixture_of_agents,
    session_search,
    set_session_searcher,
    set_skill_registry,
    skill_manage,
    skill_view,
    skills_list,
)


class TestMoA:
    def test_no_members_returns_error(self):
        configure_mixture_of_agents({})
        out = mixture_of_agents.static_call(prompt="x")
        assert "error" in out

    def test_runs_each_member(self):
        configure_mixture_of_agents(
            {
                "alpha": lambda p: f"alpha says: {p}",
                "beta": lambda p: f"beta says: {p}",
            }
        )
        out = mixture_of_agents.static_call(prompt="hi", synthesise=False)
        assert set(out["answers"].keys()) == {"alpha", "beta"}
        assert "alpha says: hi" in out["answers"]["alpha"]

    def test_synthesizer_combines(self):
        configure_mixture_of_agents(
            {"a": lambda p: "yes", "b": lambda p: "no"},
            synthesizer=lambda combined: f"final({combined.count('yes')})",
        )
        out = mixture_of_agents.static_call(prompt="vote")
        assert out["final"] == "final(1)"

    def test_voting_picks_majority(self):
        configure_mixture_of_agents(
            {"a": lambda p: "Paris", "b": lambda p: "Paris", "c": lambda p: "London"},
            voting=True,
        )
        out = mixture_of_agents.static_call(prompt="capital of france?")
        assert out["voted"] == "paris"

    def test_member_failure_captured(self):
        def boom(_):
            raise RuntimeError("rate limited")

        configure_mixture_of_agents({"a": boom, "b": lambda p: "ok"})
        out = mixture_of_agents.static_call(prompt="x", synthesise=False)
        assert "[error" in out["answers"]["a"]
        assert out["answers"]["b"] == "ok"


class TestSessionSearchAlias:
    def test_no_searcher_returns_empty(self):
        set_session_searcher(None)
        out = session_search.static_call(query="x")
        assert out["error"] == "no session searcher configured"

    def test_dispatches_to_callable(self):
        captured = {}

        def fake(query, limit, agent_id, session_id):
            captured.update(query=query, limit=limit, agent_id=agent_id, session_id=session_id)
            return {"count": 0, "hits": []}

        set_session_searcher(fake)
        try:
            session_search.static_call(query="hello", limit=3, agent_id="a", session_id="s")
            assert captured == {"query": "hello", "limit": 3, "agent_id": "a", "session_id": "s"}
        finally:
            set_session_searcher(None)


@pytest.fixture
def registry(tmp_path):
    reg = SkillRegistry()
    set_skill_registry(reg)
    yield reg
    set_skill_registry(None)


class TestSkillTools:
    def test_skills_list_empty(self, registry):
        out = skills_list.static_call()
        assert out == {"count": 0, "skills": []}

    def test_skill_manage_create_and_view(self, registry, tmp_path):
        out = skill_manage.static_call(
            action="create",
            name="ci-bootstrap",
            description="set up CI",
            instructions="Step 1: …",
            tags=["ci", "github"],
            skills_dir=str(tmp_path),
        )
        assert out["ok"] is True
        assert out["path"]

        registry.discover(tmp_path)
        listed = skills_list.static_call()
        assert any(s["name"] == "ci-bootstrap" for s in listed["skills"])

        viewed = skill_view.static_call(name="ci-bootstrap")
        assert viewed["description"] == "set up CI"
        assert "Step 1" in viewed["instructions"]
        assert "ci" in viewed["tags"]

    def test_skill_manage_delete(self, registry, tmp_path):
        skill_manage.static_call(
            action="create",
            name="x",
            instructions="body",
            skills_dir=str(tmp_path),
        )
        deleted = skill_manage.static_call(action="delete", name="x", skills_dir=str(tmp_path))
        assert deleted["ok"] is True

    def test_skill_view_missing(self, registry):
        out = skill_view.static_call(name="never")
        assert out["error"] == "not_found"

    def test_skill_manage_unknown_action(self, registry, tmp_path):
        out = skill_manage.static_call(action="patch", name="x", instructions="b", skills_dir=str(tmp_path))
        assert out["ok"] is False
