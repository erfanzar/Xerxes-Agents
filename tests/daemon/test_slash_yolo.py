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
"""Daemon-side slash dispatch — regression coverage for /yolo and friends.

The TUI talks to ``daemon/server.py``, not ``bridge/server.py``, so the
daemon must handle the common slash commands or users see
``Unknown command: /yolo``. These tests pin that contract."""

from __future__ import annotations

import asyncio

import pytest
from xerxes.daemon.config import DaemonConfig
from xerxes.daemon.runtime import RuntimeManager


class TestRuntimeManagerPermissions:
    def _rm(self, mode: str = "auto") -> RuntimeManager:
        rm = RuntimeManager(DaemonConfig())
        rm.runtime_config = {"permission_mode": mode}
        return rm

    def test_yolo_toggle_auto_to_accept_all(self):
        rm = self._rm("auto")
        assert rm.toggle_yolo() == "accept-all"
        assert rm.permission_mode == "accept-all"

    def test_yolo_toggle_accept_all_to_auto(self):
        rm = self._rm("accept-all")
        assert rm.toggle_yolo() == "auto"

    def test_set_known_modes(self):
        rm = self._rm()
        for mode in ("auto", "manual", "accept-all"):
            assert rm.set_permission_mode(mode) == mode

    def test_set_unknown_falls_back_to_auto(self):
        rm = self._rm("manual")
        assert rm.set_permission_mode("nuke-everything") == "auto"

    def test_toggle_flag_round_trip(self):
        rm = self._rm()
        assert rm.toggle_flag("debug") is True
        assert rm.toggle_flag("debug") is False
        assert rm.toggle_flag("verbose") is True
        assert rm.toggle_flag("thinking") is True

    def test_status_includes_permission_mode(self):
        rm = self._rm("manual")
        assert rm.status()["permission_mode"] == "manual"


class _Recorder:
    """Async EmitFn stand-in for the daemon's slash output channel."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def __call__(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))

    def slash_outputs(self) -> list[str]:
        return [
            payload.get("body", "")
            for (etype, payload) in self.events
            if etype == "notification" and payload.get("category") == "slash"
        ]


@pytest.fixture
def daemon_with_runtime(tmp_path):
    """Build a ``DaemonServer`` whose ``runtime`` is hand-seeded.

    We deliberately skip ``runtime.reload()`` (which requires a real
    provider profile) and seed ``runtime_config`` directly."""
    from xerxes.daemon.server import DaemonServer

    # Use defaults; the test path never touches sockets or the real bootstrap.
    server = DaemonServer.__new__(DaemonServer)
    server.config = DaemonConfig(project_dir=str(tmp_path))
    server.runtime = RuntimeManager(server.config)
    server.runtime.runtime_config = {"permission_mode": "auto"}
    return server


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _drive(server, command: str) -> list[str]:
    rec = _Recorder()
    asyncio.new_event_loop().run_until_complete(server._handle_slash(command, rec))
    return rec.slash_outputs()


class TestSlashYolo:
    def test_yolo_toggles_on(self, daemon_with_runtime):
        out = _drive(daemon_with_runtime, "/yolo")
        assert out
        assert "YOLO mode ON" in out[0]
        assert daemon_with_runtime.runtime.permission_mode == "accept-all"

    def test_yolo_toggles_off(self, daemon_with_runtime):
        daemon_with_runtime.runtime.set_permission_mode("accept-all")
        out = _drive(daemon_with_runtime, "/yolo")
        assert "YOLO mode OFF" in out[0]
        assert daemon_with_runtime.runtime.permission_mode == "auto"


class TestSlashPermissions:
    def test_permissions_explicit_manual(self, daemon_with_runtime):
        out = _drive(daemon_with_runtime, "/permissions manual")
        assert "manual" in out[0]
        assert daemon_with_runtime.runtime.permission_mode == "manual"

    def test_permissions_cycle(self, daemon_with_runtime):
        # auto → accept-all → manual → auto
        modes = []
        for _ in range(3):
            _drive(daemon_with_runtime, "/permissions")
            modes.append(daemon_with_runtime.runtime.permission_mode)
        assert modes == ["accept-all", "manual", "auto"]

    def test_unknown_mode_falls_back_to_auto(self, daemon_with_runtime):
        _drive(daemon_with_runtime, "/permissions garbage")
        assert daemon_with_runtime.runtime.permission_mode == "auto"


class TestSlashFlags:
    def test_thinking(self, daemon_with_runtime):
        out = _drive(daemon_with_runtime, "/thinking")
        assert "Thinking: True" in out[0]
        out2 = _drive(daemon_with_runtime, "/thinking")
        assert "Thinking: False" in out2[0]

    def test_verbose(self, daemon_with_runtime):
        out = _drive(daemon_with_runtime, "/verbose")
        assert "Verbose: True" in out[0]

    def test_debug(self, daemon_with_runtime):
        out = _drive(daemon_with_runtime, "/debug")
        assert "Debug: True" in out[0]


class TestSlashInfo:
    def test_help_lists_commands_by_category(self, daemon_with_runtime):
        out = _drive(daemon_with_runtime, "/help")
        text = out[0]
        # Help renders by category.
        assert "[session]" in text
        assert "[config]" in text
        assert "/yolo" in text
        assert "/permissions" in text

    def test_commands_lists_flat(self, daemon_with_runtime):
        out = _drive(daemon_with_runtime, "/commands")
        assert "/yolo" in out[0]
        assert "/help" in out[0]

    def test_context_includes_perm_mode(self, daemon_with_runtime):
        daemon_with_runtime.runtime.set_permission_mode("manual")
        out = _drive(daemon_with_runtime, "/context")
        assert "Permission mode: manual" in out[0]

    def test_status_json(self, daemon_with_runtime):
        import json

        out = _drive(daemon_with_runtime, "/status")
        # The /status response is a JSON dump; make sure it parses + has mode.
        parsed = json.loads(out[0])
        assert "permission_mode" in parsed


class TestSlashAliases:
    def test_q_resolves_to_exit(self, daemon_with_runtime):
        out = _drive(daemon_with_runtime, "/q")
        assert "Ctrl+D" in out[0] or "close" in out[0].lower()

    def test_quit_alias(self, daemon_with_runtime):
        out = _drive(daemon_with_runtime, "/quit")
        assert out  # any response is fine; what matters is it's not "Unknown".
        assert "Unknown command" not in out[0]


class TestSlashPlugins:
    def test_registered_plugin_runs(self, daemon_with_runtime):
        from xerxes.extensions.slash_plugins import register_slash, registry

        # Clean slate.
        registry()._plugins.clear()
        register_slash("test_plugin_cmd", lambda *a: "plugin ran")
        try:
            out = _drive(daemon_with_runtime, "/test_plugin_cmd")
            assert "plugin ran" in out[0]
        finally:
            registry()._plugins.clear()


class TestSlashUnknown:
    def test_truly_unknown_still_falls_through(self, daemon_with_runtime):
        out = _drive(daemon_with_runtime, "/this-command-does-not-exist")
        assert "Unknown command" in out[0]
