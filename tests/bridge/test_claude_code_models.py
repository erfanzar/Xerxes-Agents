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
"""Claude Code model discovery tests."""

from __future__ import annotations

from types import SimpleNamespace

from xerxes.bridge import profiles


def test_builtin_claude_code_profile_is_available_and_active_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(profiles, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(profiles, "PROFILES_FILE", tmp_path / "profiles.json")

    listed = profiles.list_profiles()
    cc = next(profile for profile in listed if profile["name"] == "cc")

    assert cc["active"] is True
    assert cc["provider"] == "claude-code"
    assert cc["base_url"] == "claude-code://local"
    assert cc["model"] == "claude-code/default"
    assert profiles.get_active_profile() == {
        "name": "cc",
        "base_url": "claude-code://local",
        "api_key": "",
        "model": "claude-code/default",
        "provider": "claude-code",
        "sampling": {},
    }


def test_builtin_claude_code_profile_can_be_selected_and_edited(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(profiles, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(profiles, "PROFILES_FILE", tmp_path / "profiles.json")

    assert profiles.set_active("cc") is True

    updated = profiles.update_active_model("claude-code/opus")

    assert updated is not None
    assert updated["model"] == "claude-code/opus"
    assert profiles.get_active_profile()["model"] == "claude-code/opus"


def test_fetch_claude_code_models_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_MODELS", "sonnet, opus claude-fable-5")

    assert profiles.fetch_models("claude-code://local", "") == [
        "claude-code/default",
        "claude-code/sonnet",
        "claude-code/opus",
        "claude-code/claude-fable-5",
    ]


def test_fetch_claude_code_models_parses_installed_cli_help(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CODE_MODELS", raising=False)
    monkeypatch.setattr(profiles.shutil, "which", lambda command: "/bin/claude" if command == "claude" else None)

    def _run(argv, **kwargs):
        assert argv == ["/bin/claude", "--help"]
        assert kwargs["timeout"] == 5
        return SimpleNamespace(
            stdout=(
                "  --model <model>  Model for the current session. Provide an alias for the latest model "
                "(e.g. 'fable', 'opus', or 'sonnet') or a model's full name (e.g. 'claude-fable-5').\n"
                "  --tools <tools...>  Specify tools\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(profiles.subprocess, "run", _run)

    assert profiles.fetch_models("claude-code://local", "") == [
        "claude-code/default",
        "claude-code/fable",
        "claude-code/opus",
        "claude-code/sonnet",
        "claude-code/claude-fable-5",
    ]
