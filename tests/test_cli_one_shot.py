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

from argparse import Namespace
from types import SimpleNamespace

import pytest
from xerxes import __main__
from xerxes.__main__ import _discord_child_argv, _discord_service_name, _resolve_one_shot_prompt


def test_resolve_one_shot_prompt_from_args() -> None:
    prompt, one_shot = _resolve_one_shot_prompt(["hello", "world"], stdin_is_tty=True)

    assert one_shot is True
    assert prompt == "hello world"


def test_resolve_one_shot_prompt_drops_separator() -> None:
    prompt, one_shot = _resolve_one_shot_prompt(["--", "review", "--flag"], stdin_is_tty=True)

    assert one_shot is True
    assert prompt == "review --flag"


def test_resolve_one_shot_prompt_from_stdin() -> None:
    prompt, one_shot = _resolve_one_shot_prompt([], stdin_is_tty=False, stdin_text="\nwrite a plan\n")

    assert one_shot is True
    assert prompt == "write a plan"


def test_resolve_one_shot_prompt_opens_tui_for_empty_tty() -> None:
    prompt, one_shot = _resolve_one_shot_prompt([], stdin_is_tty=True)

    assert one_shot is False
    assert prompt == ""


def test_main_interactive_uses_new_tui_by_default(monkeypatch) -> None:
    seen: dict[str, str] = {}

    monkeypatch.setattr(__main__.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        __main__, "_run_new_tui", lambda *, resume_session_id: seen.setdefault("resume", resume_session_id)
    )

    __main__.main(["-r", "abc123"])

    assert seen == {"resume": "abc123"}


def test_run_new_tui_invokes_node_and_preserves_python_env(monkeypatch, tmp_path) -> None:
    entry = tmp_path / "entry.js"
    entry.write_text("console.log('tui')\n", encoding="utf-8")
    monkeypatch.setenv("XERXES_TUI_ENTRY", str(entry))
    monkeypatch.setenv("XERXES_PYTHON", "/custom/python")
    monkeypatch.setattr(__main__.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    seen: dict[str, object] = {}

    def run(argv, **kwargs):
        env = kwargs.get("env")
        if env is None:
            return SimpleNamespace(returncode=1, stdout="")
        seen["argv"] = argv
        seen["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(__main__.subprocess, "run", run)

    __main__._run_new_tui(resume_session_id="abc123")

    assert seen["argv"] == ["/usr/bin/node", str(entry)]
    env = seen["env"]
    assert isinstance(env, dict)
    assert env["XERXES_PYTHON"] == "/custom/python"
    assert env["XERXES_TUI_RESUME"] == "abc123"


def test_run_new_tui_prefers_managed_node(monkeypatch, tmp_path) -> None:
    entry = tmp_path / "entry.js"
    entry.write_text("console.log('tui')\n", encoding="utf-8")
    managed = tmp_path / "home" / "node" / "current" / "bin" / "node"
    managed.parent.mkdir(parents=True)
    managed.write_text("#!/bin/sh\n", encoding="utf-8")
    managed.chmod(0o755)
    monkeypatch.setenv("XERXES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XERXES_TUI_ENTRY", str(entry))
    monkeypatch.delenv("XERXES_NODE", raising=False)
    monkeypatch.setattr(__main__.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    seen: dict[str, object] = {}

    def run(argv, **kwargs):
        if argv and argv[0] == "git":
            return SimpleNamespace(returncode=1, stdout="")
        seen["argv"] = argv
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(__main__.subprocess, "run", run)

    __main__._run_new_tui()

    assert seen["argv"] == [str(managed), str(entry)]


def test_run_new_tui_uses_current_python_when_unset(monkeypatch, tmp_path) -> None:
    entry = tmp_path / "entry.js"
    entry.write_text("console.log('tui')\n", encoding="utf-8")
    monkeypatch.setenv("XERXES_TUI_ENTRY", str(entry))
    monkeypatch.setenv("DEV", "true")
    monkeypatch.delenv("XERXES_PYTHON", raising=False)
    monkeypatch.setattr(__main__.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    seen: dict[str, object] = {}

    def run(argv, **kwargs):
        env = kwargs.get("env")
        if env is None:
            return SimpleNamespace(returncode=1, stdout="")
        seen["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(__main__.subprocess, "run", run)

    __main__._run_new_tui()

    env = seen["env"]
    assert isinstance(env, dict)
    assert env["XERXES_PYTHON"] == __main__.sys.executable
    assert env["NODE_ENV"] == "production"
    assert env["DEV"] == "false"
    assert "XERXES_TUI_RESUME" not in env


def test_run_new_tui_errors_when_node_is_missing(monkeypatch, tmp_path) -> None:
    entry = tmp_path / "entry.js"
    entry.write_text("console.log('tui')\n", encoding="utf-8")
    monkeypatch.setenv("XERXES_TUI_ENTRY", str(entry))
    monkeypatch.delenv("XERXES_NODE", raising=False)
    monkeypatch.setattr(__main__.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match=r"xerxes install --node"):
        __main__._run_new_tui()


def _discord_args(**overrides) -> Namespace:
    base = {
        "service_name": "",
        "project_dir": "",
        "host": "",
        "port": 0,
        "always_reply": False,
        "no_message_content_intent": False,
        "no_discord_commands": False,
        "allowed_channel": [],
        "allowed_channel_names": [],
        "allowed_guild": [],
        "instance_name": "",
        "address_names": [],
        "token": "",
    }
    base.update(overrides)
    return Namespace(**base)


def test_discord_service_name_prefers_channel_name() -> None:
    args = _discord_args(allowed_channel_names=["macbook"])

    assert _discord_service_name(args) == "discord-macbook"


def test_discord_service_name_prefers_explicit_name() -> None:
    args = _discord_args(service_name="GPU Mac")

    assert _discord_service_name(args) == "gpu-mac"


def test_discord_child_argv_does_not_embed_token() -> None:
    args = _discord_args(token="secret-token", allowed_channel_names=["macbook"], instance_name="macbook")

    argv = _discord_child_argv(args, "discord-macbook")

    assert "secret-token" not in argv
    assert "--foreground" in argv
    assert argv[argv.index("--channel-name") + 1] == "macbook"
    assert argv[argv.index("--device-name") + 1] == "macbook"
