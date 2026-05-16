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
"""Import workspace files from a compatible Markdown workspace layout.

``import_workspace`` copies any of the canonical Markdown files
Xerxes consumes (``AGENTS.md``, ``SOUL.md``, ``USER.md``, ``MEMORY.md``,
``TOOLS.md``, plus a ``memory/`` subfolder) from an external source
directory into a Xerxes ``MarkdownAgentWorkspace``. Users migrating
from any project that uses the same on-disk layout can point
``source_dir`` at it to avoid retyping their persona or preferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .workspace import MarkdownAgentWorkspace

IMPORTABLE_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "SOUL.md",
    "USER.md",
    "MEMORY.md",
    "TOOLS.md",
    "IDENTITY.md",
)


@dataclass
class ImportResult:
    """Summary returned by ``import_workspace``.

    Attributes:
        source: Directory the import read from.
        target: Xerxes workspace path that was (or would be) written to.
        copied: Names of files that were copied (or would be, when
            ``dry_run`` is set). Includes ``memory/<file>`` entries.
        skipped: Names that did not exist in the source.
        conflicts: Names skipped because the target already held
            substantial content and ``overwrite`` was ``False``.
    """

    source: Path
    target: Path
    copied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


def _file_looks_default(path: Path) -> bool:
    """Heuristic for "still the default template, safe to overwrite".

    The seeded templates in ``workspace.py`` all sit under 600 bytes; real
    user content tends to grow past that quickly. Returns ``True`` for
    unreadable paths so a half-broken workspace does not block migration.
    """
    try:
        return path.stat().st_size < 600
    except OSError:
        return True


def import_workspace(
    source_dir: str | Path,
    *,
    target_workspace: MarkdownAgentWorkspace | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> ImportResult:
    """Copy compatible Markdown files into a Xerxes workspace.

    Per file, the function checks the target. If the target is missing or
    looks like the default template, it is overwritten; otherwise it is
    only overwritten when ``overwrite=True``. The ``memory/`` subdirectory
    is copied entry by entry under the same rules.

    Args:
        source_dir: External workspace path holding the canonical Markdown
            files (e.g. ``~/.openclaw`` or any other directory using the
            same layout).
        target_workspace: Destination workspace. Defaults to a fresh
            ``MarkdownAgentWorkspace()`` (i.e. the Xerxes default agent).
        overwrite: When ``True`` non-default target files are clobbered;
            when ``False`` they are reported in ``conflicts``.
        dry_run: When ``True`` no writes happen but the returned
            ``ImportResult`` still reflects what would have been copied.

    Returns:
        An ``ImportResult`` describing the outcome.

    Raises:
        FileNotFoundError: ``source_dir`` does not exist or is not a directory.
    """

    src = Path(source_dir).expanduser()
    if not src.is_dir():
        raise FileNotFoundError(f"workspace source directory not found: {src}")

    ws = target_workspace or MarkdownAgentWorkspace()
    if not dry_run:
        ws.ensure()

    res = ImportResult(source=src, target=ws.path)
    for name in IMPORTABLE_FILES:
        src_file = src / name
        if not src_file.is_file():
            res.skipped.append(name)
            continue
        target_file = ws.path / name
        if target_file.exists() and not overwrite and not _file_looks_default(target_file):
            res.conflicts.append(name)
            continue
        if not dry_run:
            target_file.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")
        res.copied.append(name)

    # Memory subdir: copy daily notes too.
    memory_src = src / "memory"
    if memory_src.is_dir():
        memory_target = ws.path / "memory"
        if not dry_run:
            memory_target.mkdir(parents=True, exist_ok=True)
        for entry in memory_src.glob("*.md"):
            target = memory_target / entry.name
            if target.exists() and not overwrite:
                res.conflicts.append(f"memory/{entry.name}")
                continue
            if not dry_run:
                target.write_text(entry.read_text(encoding="utf-8"), encoding="utf-8")
            res.copied.append(f"memory/{entry.name}")

    return res


__all__ = ["IMPORTABLE_FILES", "ImportResult", "import_workspace"]
