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
"""Skill sub-command trigger format — regression for the autoresearch bug.

Typing ``/autoresearch:learn`` used to send the model ``User request: learn``,
making it look like an ambiguous one-word prompt. The skill instructions
key off the canonical ``/name:sub`` form. These tests pin that the daemon
reconstructs the canonical form before handing off to the model."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import RuntimeManager, SessionManager, WorkspaceManager
from xerxes.daemon.server import DaemonServer


def _write_skill(skills_dir: Path, name: str, *, subcommands: list[str] | None = None) -> Path:
    sk = skills_dir / name
    sk.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", "description: test skill"]
    if subcommands is not None:
        lines.append(f"subcommands: {subcommands}")
    lines.append("---")
    lines.append("# Body")
    (sk / "SKILL.md").write_text("\n".join(lines))
    return sk


def _make_server(tmp_path) -> DaemonServer:
    server = DaemonServer.__new__(DaemonServer)
    cfg = DaemonConfig(project_dir=str(tmp_path))
    cfg.workspace = {"root": str(tmp_path / "agents"), "default_agent_id": "default"}
    server.config = cfg
    server.workspaces = WorkspaceManager(cfg)
    server.runtime = RuntimeManager(cfg)
    server.runtime.runtime_config = {"permission_mode": "auto"}
    server.runtime.skills_dir = tmp_path / "skills"
    server.runtime.skill_registry = type(server.runtime.skill_registry)()
    server.sessions = SessionManager(server.workspaces, store_dir=tmp_path / "sessions")
    server._current_session_key = "tui:default"
    server._current_mode = "code"
    server._current_plan_mode = False
    return server


def _drive(server, command: str, *, captured: dict | None = None) -> dict:
    """Drive _handle_slash; intercept the submitted turn payload.

    Returns the captured submit_turn args dict (with ``text``)."""
    captured = captured if captured is not None else {}

    async def fake_submit(params, emit):
        captured.update(params)

    server._submit_turn = fake_submit  # type: ignore[assignment]

    async def emit(et, payload):
        pass

    asyncio.new_event_loop().run_until_complete(server._handle_slash(command, emit))
    return captured


class TestSubcommandTrigger:
    def test_autoresearch_learn_includes_canonical_form(self, tmp_path):
        server = _make_server(tmp_path)
        _write_skill(server.runtime.skills_dir, "autoresearch", subcommands=["learn", "fix", "plan"])
        server.runtime.skill_registry.discover(server.runtime.skills_dir)

        captured = _drive(server, "/autoresearch:learn")
        text = captured["text"]
        # The canonical form goes into the user-request trailer.
        assert "User request: /autoresearch:learn" in text
        # NOT the bare sub-command name on its own.
        assert "User request: learn" not in text

    def test_subcommand_plus_free_form_args(self, tmp_path):
        server = _make_server(tmp_path)
        _write_skill(server.runtime.skills_dir, "autoresearch", subcommands=["fix"])
        server.runtime.skill_registry.discover(server.runtime.skills_dir)

        captured = _drive(server, "/autoresearch:fix the failing CI run")
        text = captured["text"]
        assert "User request: /autoresearch:fix the failing CI run" in text

    def test_no_subcommand_passes_args_directly(self, tmp_path):
        server = _make_server(tmp_path)
        _write_skill(server.runtime.skills_dir, "researcher", subcommands=[])
        server.runtime.skill_registry.discover(server.runtime.skills_dir)

        captured = _drive(server, "/researcher find a good paper on diffusion")
        text = captured["text"]
        assert "User request: find a good paper on diffusion" in text

    def test_bare_skill_invocation_uses_default_trigger(self, tmp_path):
        server = _make_server(tmp_path)
        _write_skill(server.runtime.skills_dir, "lonely", subcommands=[])
        server.runtime.skill_registry.discover(server.runtime.skills_dir)

        captured = _drive(server, "/lonely")
        text = captured["text"]
        assert "Execute the 'lonely' skill now" in text

    def test_first_token_matching_subcommand_recognised(self, tmp_path):
        """``/skill <sub>`` (no colon) where <sub> is a declared sub-command
        is treated the same as ``/skill:<sub>``."""
        server = _make_server(tmp_path)
        _write_skill(server.runtime.skills_dir, "autoresearch", subcommands=["fix"])
        server.runtime.skill_registry.discover(server.runtime.skills_dir)

        # The daemon's _handle_slash composes ``autoresearch:fix make it faster``
        # for ``/autoresearch fix make it faster``. _slash_skill sees the
        # ``:fix`` form and reconstructs the canonical trigger.
        captured = _drive(server, "/autoresearch fix make it faster")
        text = captured["text"]
        assert "User request: /autoresearch:fix make it faster" in text

    def test_non_subcommand_first_token_kept_as_free_form(self, tmp_path):
        """When the first token isn't a declared sub-command, the args
        pass through unchanged — we don't false-positive into ``:foo``
        forms for skills that legitimately take free-form text."""
        server = _make_server(tmp_path)
        _write_skill(server.runtime.skills_dir, "autoresearch", subcommands=["fix", "learn"])
        server.runtime.skill_registry.discover(server.runtime.skills_dir)

        captured = _drive(server, "/autoresearch make it faster please")
        text = captured["text"]
        # `make` isn't a declared sub-command, so the args pass through verbatim.
        assert "User request: make it faster please" in text
        assert "/autoresearch:make" not in text


class TestRealAutoresearchSkillSubcommand:
    """Drive against the actual bundled skill to mirror the user's bug."""

    def test_real_autoresearch_learn(self, tmp_path):
        bundled = Path(__file__).resolve().parents[2] / "src" / "python" / "xerxes" / "skills"
        if not (bundled / "autoresearch" / "SKILL.md").exists():
            pytest.skip("autoresearch skill not present")
        server = _make_server(tmp_path)
        server.runtime.skill_registry.discover(bundled)

        captured = _drive(server, "/autoresearch:learn")
        text = captured["text"]
        assert "User request: /autoresearch:learn" in text
        # Sanity: the skill body got included.
        assert "## Skill: autoresearch" in text or "[Skill 'autoresearch' activated]" in text
