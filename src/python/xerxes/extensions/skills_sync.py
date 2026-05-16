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
"""Manifest-driven skill sync.

The manifest at ``~/.xerxes/skills/manifest.yaml`` lists ``(source,
identifier)`` pairs. ``sync_manifest`` reconciles the on-disk skill
directory with the manifest, installing missing skills and optionally
removing strays.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .skill_sources.base import SkillBundle, SkillSource


@dataclass
class ManifestEntry:
    """One ``(source, identifier)`` row from the skills manifest.

    Attributes:
        source: Name of the ``SkillSource`` backend to consult.
        identifier: Backend-specific lookup string.
    """

    source: str
    identifier: str


@dataclass
class SyncResult:
    """Outcome of a single ``sync_manifest`` pass.

    Attributes:
        installed: Identifiers newly installed during the sync.
        skipped: Identifiers already present and left untouched.
        removed: Identifiers pruned because they were absent from the manifest.
        failed: ``(identifier, reason)`` pairs for failures.
    """

    installed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def install_bundle(bundle: SkillBundle, target_dir: Path) -> Path:
    """Write ``bundle.body_markdown`` to ``target_dir/<bundle.name>/SKILL.md``."""
    skill_dir = target_dir / bundle.name
    skill_dir.mkdir(parents=True, exist_ok=True)
    out = skill_dir / "SKILL.md"
    out.write_text(bundle.body_markdown, encoding="utf-8")
    return out


def sync_manifest(
    manifest: Iterable[ManifestEntry],
    sources: dict[str, SkillSource],
    *,
    target_dir: Path,
    prune: bool = False,
) -> SyncResult:
    """Reconcile ``target_dir`` against ``manifest`` by installing missing skills.

    Args:
        manifest: Iterable of ``ManifestEntry`` rows describing desired skills.
        sources: Mapping from source name to ``SkillSource`` backend.
        target_dir: Directory holding installed skills.
        prune: When True, delete on-disk skills that are not in the manifest.

    Returns:
        ``SyncResult`` summarising installed, skipped, removed, and failed entries.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    result = SyncResult()
    requested: set[str] = set()
    for entry in manifest:
        requested.add(entry.identifier)
        if (target_dir / entry.identifier / "SKILL.md").exists():
            result.skipped.append(entry.identifier)
            continue
        source = sources.get(entry.source)
        if source is None:
            result.failed.append((entry.identifier, f"unknown source: {entry.source}"))
            continue
        try:
            bundle = source.fetch(entry.identifier)
            install_bundle(bundle, target_dir)
            result.installed.append(entry.identifier)
        except Exception as exc:
            result.failed.append((entry.identifier, str(exc)))
    if prune:
        for existing in target_dir.iterdir():
            if not existing.is_dir():
                continue
            if existing.name not in requested:
                try:
                    import shutil

                    shutil.rmtree(existing)
                    result.removed.append(existing.name)
                except OSError as exc:
                    result.failed.append((existing.name, str(exc)))
    return result


__all__ = ["ManifestEntry", "SyncResult", "install_bundle", "sync_manifest"]
