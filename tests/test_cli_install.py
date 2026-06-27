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
"""CLI installer command tests."""

from __future__ import annotations

from xerxes import __main__ as cli


def test_install_node_dry_run_prints_official_archive(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("XERXES_NODE_VERSION", "22.17.1")
    monkeypatch.setenv("XERXES_HOME", str(tmp_path / "home"))

    cli._run_install_command(["--node", "--dry-run"])

    out = capsys.readouterr().out
    assert "https://nodejs.org/dist/v22.17.1/node-v22.17.1-darwin-arm64.tar.gz" in out
    assert "Would install to:" in out


def test_install_node_skips_when_managed_runtime_exists(monkeypatch, tmp_path, capsys) -> None:
    managed = tmp_path / "home" / "node" / "node-v22.17.1-linux-x64" / "bin" / "node"
    managed.parent.mkdir(parents=True)
    managed.write_text("#!/bin/sh\n", encoding="utf-8")
    managed.chmod(0o755)
    monkeypatch.setenv("XERXES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")

    cli._run_install_command(["--node"])

    out = capsys.readouterr().out
    assert "already installed" in out
    assert (tmp_path / "home" / "node" / "current").is_symlink()


def test_install_cloud_code_dry_run_uses_npm_package(monkeypatch, capsys) -> None:
    calls: list[list[str]] = []

    def _which(command: str) -> str | None:
        if command == "claude":
            return None
        if command == "npm":
            return "/usr/local/bin/npm"
        return None

    monkeypatch.setattr(cli.shutil, "which", _which)
    monkeypatch.setattr(cli.subprocess, "run", lambda argv, **_kwargs: calls.append(argv))

    cli._run_install_command(["--cloud-code", "--dry-run"])

    assert calls == []
    assert "npm install -g @anthropic-ai/claude-code" in capsys.readouterr().out


def test_install_cloud_code_uses_managed_npm(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []
    npm = tmp_path / "home" / "node" / "current" / "bin" / "npm"
    npm.parent.mkdir(parents=True)
    npm.write_text("#!/bin/sh\n", encoding="utf-8")
    npm.chmod(0o755)
    monkeypatch.setenv("XERXES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli.shutil, "which", lambda _command: None)

    def run(argv, **_kwargs):
        calls.append(argv)
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(cli.subprocess, "run", run)

    cli._run_install_command(["--cloud-code"])

    assert calls == [[str(npm), "install", "-g", "@anthropic-ai/claude-code"]]


def test_install_claude_code_skips_when_already_available(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda command: "/usr/local/bin/claude" if command == "claude" else None)

    cli._run_install_command(["--claude-code"])

    out = capsys.readouterr().out
    assert "already installed" in out
    assert "claude auth login" in out
