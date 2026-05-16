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
"""Tests for xerxes.channels.workspace_import."""

from __future__ import annotations

import pytest
from xerxes.channels.workspace import MarkdownAgentWorkspace
from xerxes.channels.workspace_import import import_workspace


@pytest.fixture
def source_dir(tmp_path):
    src = tmp_path / "external_ws"
    src.mkdir()
    (src / "SOUL.md").write_text("# SOUL\nI am the imported soul.\n" * 30)
    (src / "USER.md").write_text("# USER\nLikes long walks." * 50)
    (src / "MEMORY.md").write_text("# Past memories" * 100)
    return src


@pytest.fixture
def target(tmp_path):
    return MarkdownAgentWorkspace(tmp_path / "xerxes")


class TestImportWorkspace:
    def test_missing_source_raises(self, tmp_path, target):
        with pytest.raises(FileNotFoundError):
            import_workspace(tmp_path / "doesnt-exist", target_workspace=target)

    def test_copies_files(self, source_dir, target):
        res = import_workspace(source_dir, target_workspace=target)
        assert "SOUL.md" in res.copied
        assert "USER.md" in res.copied
        assert "MEMORY.md" in res.copied
        # AGENTS.md / TOOLS.md were not in source.
        assert "AGENTS.md" in res.skipped
        assert "TOOLS.md" in res.skipped
        # File actually written.
        soul = (target.path / "SOUL.md").read_text()
        assert "imported soul" in soul

    def test_conflicts_when_target_has_content(self, source_dir, target):
        target.ensure()
        # Replace default SOUL.md with substantial content (>600 bytes).
        (target.path / "SOUL.md").write_text("CUSTOM " * 200)
        res = import_workspace(source_dir, target_workspace=target)
        # SOUL.md is now a conflict.
        assert "SOUL.md" in res.conflicts
        # Target unchanged.
        assert "CUSTOM" in (target.path / "SOUL.md").read_text()

    def test_overwrite_force(self, source_dir, target):
        target.ensure()
        (target.path / "SOUL.md").write_text("CUSTOM " * 200)
        res = import_workspace(source_dir, target_workspace=target, overwrite=True)
        assert "SOUL.md" in res.copied
        assert "imported soul" in (target.path / "SOUL.md").read_text()

    def test_dry_run_does_not_write(self, source_dir, target):
        res = import_workspace(source_dir, target_workspace=target, dry_run=True)
        assert "SOUL.md" in res.copied
        # No actual file written.
        assert not (target.path / "SOUL.md").exists()

    def test_memory_dir_imported(self, source_dir, target):
        (source_dir / "memory").mkdir()
        (source_dir / "memory" / "2026-05-01.md").write_text("notes for 5/1")
        res = import_workspace(source_dir, target_workspace=target)
        assert any(p.startswith("memory/") for p in res.copied)
        assert (target.path / "memory" / "2026-05-01.md").exists()

    def test_default_target_files_overwritten_silently(self, source_dir, tmp_path):
        # Tiny default files should be replaced without listing as conflicts.
        target = MarkdownAgentWorkspace(tmp_path / "xerxes-default")
        target.ensure()  # default tiny files in place
        res = import_workspace(source_dir, target_workspace=target)
        assert "SOUL.md" in res.copied
        assert "SOUL.md" not in res.conflicts
