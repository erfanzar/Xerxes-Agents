# Copyright 2026 Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
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
"""Coverage for every command in :data:`COMMAND_REGISTRY` resolving in the daemon.

Catches the regression of "command listed in ``/help`` but daemon prints
``Unknown command``" by:

1. Asserting every registered command (and alias) has either an explicit
   ``cmd == "..."`` branch in ``_handle_slash`` or an entry in
   ``_BULK_SLASH_HANDLERS``.
2. Smoke-testing each bulk-dispatched command with a minimal daemon
   fixture so a future refactor that breaks one of them shows up here.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from xerxes.bridge.commands import COMMAND_REGISTRY
from xerxes.daemon import server as daemon_server
from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import RuntimeManager


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))

    def slash_bodies(self) -> list[str]:
        return [
            p.get("body", "") for (etype, p) in self.events if etype == "notification" and p.get("category") == "slash"
        ]


def _direct_handler_names() -> set[str]:
    """Parse cmd == "..." / cmd in {...} branches out of the daemon source."""
    src = open(daemon_server.__file__).read()
    out: set[str] = set()
    for match in re.finditer(r'cmd == "([\w-]+)"|cmd in \{([^}]+)\}', src):
        if match.group(1):
            out.add(match.group(1))
        else:
            for tok in match.group(2).split(","):
                tok = tok.strip().strip('"').strip("'")
                if tok:
                    out.add(tok)
    return out


def test_every_registered_command_has_a_handler():
    """Each name and alias in the registry resolves to a daemon dispatcher."""
    direct = _direct_handler_names()
    bulk = set(daemon_server._BULK_SLASH_HANDLERS)
    canonical = {c.name for c in COMMAND_REGISTRY}
    alias_map = {alias: c.name for c in COMMAND_REGISTRY for alias in c.aliases}

    handled = direct | bulk
    missing: list[str] = []
    for name in canonical:
        if name not in handled:
            missing.append(name)
    for alias, target in alias_map.items():
        # Aliases get canonicalised by resolve_command — they're fine as long
        # as their canonical name is handled.
        if target not in handled:
            missing.append(f"alias `{alias}` → `{target}`")
    assert missing == [], f"Registered but unhandled: {missing}"


# --- Smoke-test fixture for the bulk handlers ------------------------------


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    from xerxes.daemon.server import DaemonServer

    server = DaemonServer.__new__(DaemonServer)
    server.config = DaemonConfig(project_dir=str(tmp_path))
    server.runtime = RuntimeManager(server.config)
    server.runtime.runtime_config = {"permission_mode": "auto", "model": "fake-model"}
    # ``model`` is a property — set via runtime_config which it reads from.
    server.runtime.tool_schemas = [
        {"function": {"name": "Bash"}},
        {"function": {"name": "Read"}},
    ]
    # Avoid touching the real filesystem skills tree.
    monkeypatch.setattr(server.runtime, "discover_skills", lambda: ["test-skill"])
    server.runtime.skills_dir = tmp_path / "skills"
    server.runtime.skills_dir.mkdir()
    server._current_session_key = "tui:default"
    server._current_mode = "code"
    server._current_plan_mode = False
    server._pending_slash_arg = None
    server._pending_skill_create = None
    server._background_tasks = set()
    server.workspaces = type("W", (), {"default_agent_id": "default"})()
    # Stub out subsystems the handlers reach into.
    server.channels = type("C", (), {"channels": {}})()
    server.sessions = type(
        "S",
        (),
        {
            "_sessions": {},
            "evict": lambda self_, key: None,
            "open": lambda self_, key, agent: type(
                "Sess",
                (),
                {"id": "sess123", "agent_id": agent, "state": type("St", (), {})()},
            )(),
            "get": lambda self_, key: None,
            "save": lambda self_, sess: None,
            "cancel": lambda self_, key: True,
            "cancel_all": lambda self_: 1,
            "list": lambda self_: [],
            "_session_path": lambda self_, sid: tmp_path / f"{sid}.json",
        },
    )()

    captured_submit: list[dict[str, Any]] = []

    async def _fake_submit_turn(params: dict[str, Any], emit) -> dict[str, Any]:
        captured_submit.append(params)
        return {"ok": True, "session": {}, "turn_task": None}

    server._submit_turn = _fake_submit_turn  # type: ignore[assignment]
    server.captured_submit = captured_submit  # type: ignore[attr-defined]

    def _fake_track_task(coro):
        try:
            coro.close()
        except Exception:
            pass

        class _T:
            def done(self):
                return True

            def get_name(self):
                return "fake"

        return _T()

    server._track_task = _fake_track_task  # type: ignore[assignment]
    return server


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _drive(server, command: str) -> _Recorder:
    rec = _Recorder()
    _run(server._handle_slash(command, rec))
    return rec


# Each command, plus a check the response body contains the listed snippet.
# Snippet is empty when we just care that *something* came back.
_SMOKE_CASES: list[tuple[str, str]] = [
    ("/new", "New session"),
    ("/stop", ""),
    ("/cancel-all", "Cancelled"),
    ("/compact", "Compaction"),
    ("/btw hello", "Steer"),
    ("/steer hello", "Steer"),
    ("/model", "fake-model"),
    ("/sampling", "Sampling"),
    ("/config", "config"),
    ("/title my chat", ""),
    ("/workspace", "Agent workspace"),
    ("/save", ""),
    ("/personality", ""),
    ("/soul", ""),
    ("/tools", "Tools"),
    ("/toolsets", ""),
    ("/agents", "Agents"),
    ("/reload", "Reloaded"),
    ("/reload-mcp", ""),
    ("/memory", "Memory"),
    ("/history", ""),
    ("/usage", ""),
    ("/cost", ""),
    ("/insights", ""),
    ("/budget", ""),
    ("/doctor", "Diagnostics"),
    ("/update", "Xerxes"),
    ("/nudge", "Nudge"),
    ("/feedback", "Feedback"),
    ("/plugins", "Plugins"),
    ("/platforms", "Channel platforms"),
    ("/browser", ""),
    ("/image red car", "queued"),
    ("/cron", ""),
    ("/fast", "Fast mode"),
    ("/skin", "TUI"),
    ("/statusbar", "TUI"),
    ("/paste", "TUI"),
    ("/voice", "TUI"),
    ("/queue", "TUI"),
    ("/background", "background"),
    ("/resume", "saved sessions"),
    ("/undo", "Nothing to undo"),
    ("/retry", "Nothing to retry"),
    ("/branches", ""),
    ("/snapshots", ""),
]


@pytest.mark.parametrize("command,snippet", _SMOKE_CASES, ids=[c[0] for c in _SMOKE_CASES])
def test_command_smoke(daemon, command, snippet):
    rec = _drive(daemon, command)
    bodies = rec.slash_bodies()
    assert bodies, f"`{command}` emitted nothing"
    if snippet:
        joined = "\n".join(bodies)
        assert snippet.lower() in joined.lower(), f"`{command}` response missing `{snippet}`: {bodies[0][:200]}"
