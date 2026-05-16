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
"""Tests for xerxes.tools.workspace_tools."""

from __future__ import annotations

import pytest
from xerxes.channels.workspace import MarkdownAgentWorkspace
from xerxes.tools.workspace_tools import (
    workspace_append,
    workspace_diff,
    workspace_list,
    workspace_read,
    workspace_write,
)


@pytest.fixture
def ws(tmp_path):
    return MarkdownAgentWorkspace(tmp_path / "agent")


class TestWorkspaceList:
    def test_lists_default_files(self, ws) -> None:
        result = workspace_list(ws)
        names = {r["path"] for r in result}
        # Defaults created by ws.ensure()
        for expected in ("AGENTS.md", "SOUL.md", "USER.md", "MEMORY.md", "TOOLS.md"):
            assert expected in names


class TestWorkspaceRead:
    def test_reads_existing(self, ws) -> None:
        out = workspace_read("SOUL.md", ws)
        assert "Xerxes" in out

    def test_missing_raises(self, ws) -> None:
        with pytest.raises(FileNotFoundError):
            workspace_read("nope.md", ws)

    def test_path_escape_blocked(self, ws) -> None:
        with pytest.raises(ValueError):
            workspace_read("../../etc/passwd", ws)

    def test_empty_path_blocked(self, ws) -> None:
        with pytest.raises(ValueError):
            workspace_read("", ws)

    def test_truncate_at_max_bytes(self, ws) -> None:
        workspace_write("big.md", "x" * 1000, ws)
        out = workspace_read("big.md", ws, max_bytes=100)
        assert "truncated" in out


class TestWorkspaceWrite:
    def test_creates_new_file(self, ws) -> None:
        result = workspace_write("notes.md", "hello", ws)
        assert result["created"] is True
        assert result["bytes"] == 5
        assert workspace_read("notes.md", ws) == "hello"

    def test_overwrites_existing(self, ws) -> None:
        workspace_write("notes.md", "first", ws)
        result = workspace_write("notes.md", "second", ws)
        assert result["created"] is False
        assert workspace_read("notes.md", ws) == "second"

    def test_create_subdirs(self, ws) -> None:
        workspace_write("memory/2026-05-15.md", "today", ws)
        assert workspace_read("memory/2026-05-15.md", ws) == "today"

    def test_path_escape_blocked(self, ws) -> None:
        with pytest.raises(ValueError):
            workspace_write("../escape.md", "x", ws)


class TestWorkspaceAppend:
    def test_appends_with_newline(self, ws) -> None:
        workspace_write("notes.md", "first line", ws)
        workspace_append("notes.md", "second line", ws)
        content = workspace_read("notes.md", ws)
        assert "first line\nsecond line" in content

    def test_appends_to_missing_file(self, ws) -> None:
        result = workspace_append("new.md", "hi", ws)
        assert result["created"] is True
        assert workspace_read("new.md", ws) == "hi"

    def test_does_not_double_newline(self, ws) -> None:
        workspace_write("notes.md", "first\n", ws)
        workspace_append("notes.md", "second", ws)
        # No double newline expected.
        assert workspace_read("notes.md", ws) == "first\nsecond"


class TestWorkspaceDiff:
    def test_diff_against_existing(self, ws) -> None:
        workspace_write("a.md", "hello\nworld\n", ws)
        diff = workspace_diff("a.md", "hello\nuniverse\n", ws)
        assert "-world" in diff
        assert "+universe" in diff

    def test_diff_against_missing_returns_full_new(self, ws) -> None:
        diff = workspace_diff("brand-new.md", "fresh\n", ws)
        assert "+fresh" in diff

    def test_diff_no_changes_empty(self, ws) -> None:
        workspace_write("a.md", "same", ws)
        diff = workspace_diff("a.md", "same", ws)
        assert diff == ""
