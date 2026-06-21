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
"""Bridge runtime project-directory binding."""

from __future__ import annotations

import io
import json

from xerxes.bridge import server as bridge_server
from xerxes.bridge.server import BridgeServer


def test_bridge_init_honors_project_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge_server.profiles, "get_active_profile", lambda: None)
    project = tmp_path / "jax-metallib"
    server = BridgeServer()
    server._stdout = io.StringIO()

    server.handle_init({"model": "test-model", "project_dir": str(project), "thinking": False})

    assert server._session_cwd == str(project.resolve())
    assert server.config["project_dir"] == str(project.resolve())


def test_bridge_load_session_restores_persisted_cwd(tmp_path) -> None:
    project = tmp_path / "jax-metallib"
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    record = {
        "session_id": "abcd1234",
        "model": "test-model",
        "cwd": str(project),
        "messages": [{"role": "user", "content": "hello"}],
    }
    (sessions_dir / "abcd1234.json").write_text(json.dumps(record), encoding="utf-8")
    server = BridgeServer()
    server.SESSIONS_DIR = sessions_dir

    assert server._load_session("abcd1234")
    assert server._session_cwd == str(project.resolve())
