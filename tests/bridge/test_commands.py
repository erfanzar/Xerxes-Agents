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
"""Tests for xerxes.bridge.commands."""

from __future__ import annotations

from xerxes.bridge.commands import (
    CATEGORIES,
    COMMAND_REGISTRY,
    list_commands,
    resolve_command,
    telegram_bot_commands,
)


class TestRegistry:
    def test_at_least_50_commands(self):
        # Plan 19 says expand from 28 → 56+; defend the lower bound.
        assert len(COMMAND_REGISTRY) >= 50

    def test_categories_all_known(self):
        for cmd in COMMAND_REGISTRY:
            assert cmd.category in CATEGORIES

    def test_unique_canonical_names(self):
        names = [c.name for c in COMMAND_REGISTRY]
        assert len(names) == len(set(names))

    def test_aliases_dont_clash(self):
        names = {c.name for c in COMMAND_REGISTRY}
        for cmd in COMMAND_REGISTRY:
            for alias in cmd.aliases:
                assert alias not in names, f"alias {alias} also a canonical name"


class TestResolveCommand:
    def test_canonical_lookup(self):
        cmd = resolve_command("/model")
        assert cmd is not None
        assert cmd.name == "model"

    def test_without_leading_slash(self):
        assert resolve_command("model") is not None

    def test_with_args_strips_argument(self):
        cmd = resolve_command("/model gpt-4o")
        assert cmd is not None and cmd.name == "model"

    def test_alias_resolves(self):
        cmd = resolve_command("/q")
        assert cmd is not None and cmd.name == "exit"

    def test_unknown_returns_none(self):
        assert resolve_command("/bogus") is None

    def test_thinking_alias(self):
        cmd = resolve_command("/thinking")
        assert cmd is not None and cmd.name == "reasoning"


class TestListCommands:
    def test_no_filter_returns_all(self):
        assert list_commands() == list(COMMAND_REGISTRY)

    def test_filter_by_category(self):
        snapshots = list_commands(category="snapshots")
        names = {c.name for c in snapshots}
        assert {"snapshot", "snapshots", "rollback"}.issubset(names)


class TestTelegramBotCommands:
    def test_excludes_cli_only(self):
        names = {c["command"] for c in telegram_bot_commands()}
        assert "skin" not in names  # cli_only
        assert "verbose" not in names  # cli_only

    def test_includes_messaging_friendly(self):
        names = {c["command"] for c in telegram_bot_commands()}
        assert {"new", "model", "compact", "skills", "exit"}.issubset(names)

    def test_names_are_telegram_safe(self):
        import re

        pat = re.compile(r"^[a-z0-9_]{1,32}$")
        for c in telegram_bot_commands():
            assert pat.match(c["command"]), c["command"]

    def test_descriptions_clamped(self):
        for c in telegram_bot_commands():
            assert len(c["description"]) <= 256
