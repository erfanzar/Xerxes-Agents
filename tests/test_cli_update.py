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
"""Tests for the ``xerxes update`` CLI command."""

from __future__ import annotations

from xerxes import __main__
from xerxes.runtime import update


def test_update_cli_check_reports_package_and_git_status(monkeypatch, capsys):
    monkeypatch.setattr(update, "installed_version", lambda: "0.2.3")
    monkeypatch.setattr(
        update,
        "check_for_update",
        lambda: update.UpdateAvailable("0.2.3", "0.2.4", update.InstallMode.PIP_SYSTEM),
    )
    monkeypatch.setattr(
        update,
        "git_update_status",
        lambda *, fetch, timeout: update.GitUpdateStatus(
            is_git=True,
            branch="main",
            upstream="origin/main",
            head_hash="abc1234",
            upstream_hash="def5678",
            behind_count=2,
        ),
    )

    __main__.main(["update", "--check", "--no-fetch"])

    out = capsys.readouterr().out
    assert "Xerxes 0.2.3" in out
    assert "0.2.4 available" in out
    assert "2 ahead available (origin/main def5678; HEAD abc1234)" in out


def test_update_cli_dry_run_reports_command(monkeypatch, capsys):
    monkeypatch.setattr(update, "installed_version", lambda: "0.2.3")
    monkeypatch.setattr(update, "check_for_update", lambda: None)
    monkeypatch.setattr(update, "git_update_status", lambda *, fetch, timeout: update.GitUpdateStatus(is_git=False))
    monkeypatch.setattr(
        update,
        "apply_update",
        lambda *, dry_run: {
            "ok": True,
            "dry_run": dry_run,
            "mode": "uv_tool",
            "argv": ["uv", "tool", "upgrade", "xerxes-agent"],
        },
    )

    __main__.main(["update", "--dry-run", "--no-fetch"])

    out = capsys.readouterr().out
    assert "Package: current or PyPI unavailable" in out
    assert "Git: not a git checkout" in out
    assert "Would run: uv tool upgrade xerxes-agent" in out


def test_update_cli_does_not_apply_when_current(monkeypatch, capsys):
    monkeypatch.setattr(update, "installed_version", lambda: "0.2.3")
    monkeypatch.setattr(update, "check_for_update", lambda: None)
    monkeypatch.setattr(
        update,
        "git_update_status",
        lambda *, fetch, timeout: update.GitUpdateStatus(is_git=True, head_hash="abc1234"),
    )

    def fail_apply(*, dry_run):
        raise AssertionError("apply_update should not run when update status is current")

    monkeypatch.setattr(update, "apply_update", fail_apply)

    __main__.main(["update", "--no-fetch"])

    out = capsys.readouterr().out
    assert "Git: current (HEAD abc1234)" in out
    assert "Already current" in out
