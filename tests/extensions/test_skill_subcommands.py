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
"""Skill sub-commands — parsing + daemon dispatch."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from xerxes.extensions.skills import SkillRegistry


def _write_skill(dir_: Path, *, frontmatter: dict, refs: list[str] | None = None) -> Path:
    skill_dir = dir_ / frontmatter["name"]
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for k, v in frontmatter.items():
        if isinstance(v, list):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("# Body of the skill")
    (skill_dir / "SKILL.md").write_text("\n".join(lines))
    if refs:
        (skill_dir / "references").mkdir(parents=True, exist_ok=True)
        for ref in refs:
            (skill_dir / "references" / ref).write_text(f"# {ref}\n")
    return skill_dir


class TestSubcommandFrontmatter:
    def test_explicit_subcommands_list(self, tmp_path):
        _write_skill(tmp_path, frontmatter={"name": "alpha", "description": "x", "subcommands": ["one", "two"]})
        reg = SkillRegistry()
        reg.discover(tmp_path)
        skill = reg.get("alpha")
        assert skill.metadata.subcommands == ["one", "two"]

    def test_no_subcommands_field_means_empty(self, tmp_path):
        _write_skill(tmp_path, frontmatter={"name": "beta", "description": "x"})
        reg = SkillRegistry()
        reg.discover(tmp_path)
        assert reg.get("beta").metadata.subcommands == []

    def test_string_subcommand_normalized_to_list(self, tmp_path):
        _write_skill(tmp_path, frontmatter={"name": "gamma", "description": "x", "subcommands": "only-one"})
        reg = SkillRegistry()
        reg.discover(tmp_path)
        assert reg.get("gamma").metadata.subcommands == ["only-one"]


class TestSubcommandAutoDetect:
    def test_workflow_references_become_subcommands(self, tmp_path):
        _write_skill(
            tmp_path,
            frontmatter={"name": "auto", "description": "x"},
            refs=[
                "fix-workflow.md",
                "debug-workflow.md",
                "plan-workflow.md",
                "core-principles.md",
                "results-logging.md",
            ],
        )
        reg = SkillRegistry()
        reg.discover(tmp_path)
        subs = reg.get("auto").metadata.subcommands
        # core-principles.md and results-logging.md aren't *-workflow.md → excluded.
        assert sorted(subs) == ["debug", "fix", "plan"]

    def test_explicit_field_wins_over_autodetect(self, tmp_path):
        _write_skill(
            tmp_path,
            frontmatter={"name": "explicit", "description": "x", "subcommands": ["only-this"]},
            refs=["fix-workflow.md", "debug-workflow.md"],
        )
        reg = SkillRegistry()
        reg.discover(tmp_path)
        assert reg.get("explicit").metadata.subcommands == ["only-this"]

    def test_no_references_dir_no_subcommands(self, tmp_path):
        _write_skill(tmp_path, frontmatter={"name": "noref", "description": "x"})
        reg = SkillRegistry()
        reg.discover(tmp_path)
        assert reg.get("noref").metadata.subcommands == []


class TestRealAutoresearchSkill:
    """End-to-end against the bundled autoresearch skill."""

    def test_autoresearch_subcommands_detected(self):
        skill_path = Path(__file__).resolve().parents[2] / "src" / "python" / "xerxes" / "skills" / "autoresearch"
        if not skill_path.is_dir():
            pytest.skip("autoresearch skill not present in this checkout")
        reg = SkillRegistry()
        reg.discover(skill_path.parent)
        skill = reg.get("autoresearch")
        assert skill is not None
        subs = set(skill.metadata.subcommands)
        # The 9 workflows declared in SKILL.md (plus reason added by Plan TUI work).
        # Allow for SKILL.md drift — assert at least the most common subset.
        must_have = {"debug", "fix", "plan", "predict", "security", "ship"}
        assert must_have.issubset(subs), f"missing core subs in {subs}"


class TestDaemonDispatchSubcommands:
    """Drive the daemon's _handle_slash directly to verify routing."""

    def _make_server(self, tmp_path):
        from xerxes.daemon.config import DaemonConfig
        from xerxes.daemon.runtime import RuntimeManager
        from xerxes.daemon.server import DaemonServer

        # Seed a real skill that uses subcommands.
        skills_dir = tmp_path / "skills"
        _write_skill(
            skills_dir,
            frontmatter={"name": "demo", "description": "the demo skill", "subcommands": ["fix", "debug"]},
        )

        server = DaemonServer.__new__(DaemonServer)
        server.config = DaemonConfig(project_dir=str(tmp_path))
        server.runtime = RuntimeManager(server.config)
        server.runtime.runtime_config = {"permission_mode": "auto"}
        server.runtime.skills_dir = skills_dir
        server.runtime.skill_registry = type(server.runtime.skill_registry)()
        server.runtime.skill_registry.discover(skills_dir)
        return server

    def _drive(self, server, command: str) -> list[str]:
        events: list[tuple[str, dict]] = []

        async def emit(event_type, payload):
            events.append((event_type, payload))

        asyncio.new_event_loop().run_until_complete(server._handle_slash(command, emit))
        return [
            payload.get("body", "")
            for (etype, payload) in events
            if etype == "notification" and payload.get("category") == "slash"
        ]

    def test_unknown_subcommand_reports_available(self, tmp_path):
        server = self._make_server(tmp_path)
        # Capture the skill invocation path — patch _slash_skill to record.
        called: dict = {}

        async def fake_slash_skill(args, emit, *, run_now=True):
            called["args"] = args

        server._slash_skill = fake_slash_skill  # type: ignore[assignment]
        out = self._drive(server, "/demo:bogus")
        assert out and "no sub-command 'bogus'" in out[0]
        assert "/demo:fix" in out[0]
        assert "/demo:debug" in out[0]
        # No invocation happened.
        assert called == {}

    def test_known_subcommand_passes_through(self, tmp_path):
        server = self._make_server(tmp_path)
        called: dict = {}

        async def fake_slash_skill(args, emit, *, run_now=True):
            called["args"] = args

        server._slash_skill = fake_slash_skill  # type: ignore[assignment]
        self._drive(server, "/demo:fix")
        assert called["args"] == "demo:fix"

    def test_subcommand_with_extra_args(self, tmp_path):
        server = self._make_server(tmp_path)
        called: dict = {}

        async def fake_slash_skill(args, emit, *, run_now=True):
            called["args"] = args

        server._slash_skill = fake_slash_skill  # type: ignore[assignment]
        self._drive(server, "/demo:fix make it faster")
        assert called["args"] == "demo:fix make it faster"

    def test_root_skill_still_works(self, tmp_path):
        server = self._make_server(tmp_path)
        called: dict = {}

        async def fake_slash_skill(args, emit, *, run_now=True):
            called["args"] = args

        server._slash_skill = fake_slash_skill  # type: ignore[assignment]
        self._drive(server, "/demo")
        assert called["args"] == "demo"


class TestRuntimeManagerSkillsList:
    def test_skills_list_includes_subcommands(self, tmp_path):
        from xerxes.daemon.config import DaemonConfig
        from xerxes.daemon.runtime import RuntimeManager

        rm = RuntimeManager(DaemonConfig())
        rm.skills_dir = tmp_path / "skills"
        _write_skill(
            rm.skills_dir,
            frontmatter={"name": "auto", "description": "x", "subcommands": ["a", "b"]},
        )
        rm.skill_registry = type(rm.skill_registry)()
        rm.skill_registry.discover(rm.skills_dir)
        names = rm.skill_names_with_subs()
        assert "auto" in names
        assert "auto:a" in names
        assert "auto:b" in names

    def test_skills_list_text_renders_subs(self, tmp_path):
        from xerxes.daemon.config import DaemonConfig
        from xerxes.daemon.runtime import RuntimeManager

        rm = RuntimeManager(DaemonConfig())
        rm.skills_dir = tmp_path / "skills"
        _write_skill(
            rm.skills_dir,
            frontmatter={"name": "auto", "description": "Autoresearch skill", "subcommands": ["fix", "debug"]},
        )
        rm.skill_registry = type(rm.skill_registry)()
        text = rm.skills_list_text()
        assert "/auto" in text
        assert "/auto:fix" in text
        assert "/auto:debug" in text
