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
"""Security scanning and quarantine utilities for skills.

``scan_skill`` checks a skill directory for prompt injection, hash mismatch,
and untrusted sources. ``quarantine_skill`` and ``approve_skill`` move skills
between the active directory and quarantine.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from xerxes.core.paths import xerxes_subdir
from xerxes.security.prompt_scanner import scan_context_content

logger = logging.getLogger(__name__)

TRUSTED_REPOS: set[str] = {
    "NousResearch/hermes-agent",
    "erfanzar/xerxes",
}

_TRUSTED_HASHES_PATH = xerxes_subdir("skills", ".hub", "trusted_hashes.json")


def _load_trusted_hashes() -> dict[str, str]:
    """Load the trusted hash database.

    Returns:
        dict[str, str]: OUT: Mapping from file path to SHA-256 hex digest.
        Empty if the file does not exist or is unreadable.
    """

    if not _TRUSTED_HASHES_PATH.exists():
        return {}
    try:
        return json.loads(_TRUSTED_HASHES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_trusted_hashes(data: dict[str, str]) -> None:
    """Persist the trusted hash database to disk.

    Args:
        data (dict[str, str]): IN: Mapping from file path to SHA-256 digest.
            OUT: Serialized as JSON to ``_TRUSTED_HASHES_PATH``.

    Returns:
        None: OUT: File is written (parent directories created as needed).
    """
    _TRUSTED_HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRUSTED_HASHES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


@dataclass
class ScanResult:
    """Outcome of a security scan on a skill.

    Attributes:
        is_safe (bool): IN: Computed flag. OUT: ``True`` only if ``reasons``
            is empty.
        reasons (list[str]): IN: Empty initially. OUT: Populated with
            human-readable failure descriptions.
        hash_mismatch (bool): IN: ``False`` initially. OUT: Set when the file
            hash differs from the trusted database.
        injection_detected (bool): IN: ``False`` initially. OUT: Set when the
            scanner blocks the content.
        untrusted_source (bool): IN: ``False`` initially. OUT: Set when the
            source repo is not in ``TRUSTED_REPOS``.
    """

    is_safe: bool
    reasons: list[str] = field(default_factory=list)
    hash_mismatch: bool = False
    injection_detected: bool = False
    untrusted_source: bool = False

    @property
    def summary(self) -> str:
        """Return a one-line summary of the scan.

        Returns:
            str: OUT: ``"Safe"`` or a semicolon-separated reason string.
        """
        if self.is_safe:
            return "Safe"
        return "; ".join(self.reasons) if self.reasons else "Unsafe"


def _hash_file(path: Path) -> str:
    """Compute the SHA-256 digest of a single file.

    Args:
        path (Path): IN: File to hash. OUT: Read in full.

    Returns:
        str: OUT: Lowercase hex digest.
    """

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _hash_directory(dir_path: Path) -> str:
    """Compute a deterministic SHA-256 digest of an entire directory tree.

    Args:
        dir_path (Path): IN: Root directory to hash. OUT: Recursively
            enumerated for files.

    Returns:
        str: OUT: Lowercase hex digest covering relative paths and contents.
    """

    h = hashlib.sha256()
    for path in sorted(dir_path.rglob("*")):
        if path.is_file():
            rel = path.relative_to(dir_path).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()


def scan_skill(
    skill_path: Path,
    *,
    source_repo: str | None = None,
    trusted_hashes: dict[str, str] | None = None,
) -> ScanResult:
    """Run security checks on a skill file or directory.

    Args:
        skill_path (Path): IN: Path to ``SKILL.md`` or the skill directory.
            OUT: Used to locate the markdown file.
        source_repo (str | None): IN: Origin repository identifier. OUT:
            Compared against ``TRUSTED_REPOS``.
        trusted_hashes (dict[str, str] | None): IN: Known-good SHA-256
            digests. OUT: Used to detect tampering.

    Returns:
        ScanResult: OUT: Detailed scan outcome.
    """

    reasons: list[str] = []
    injection = False
    hash_mismatch = False
    untrusted = False

    if skill_path.is_file() and skill_path.name == "SKILL.md":
        skill_dir = skill_path.parent
        skill_md = skill_path
    else:
        skill_dir = skill_path
        skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        return ScanResult(
            is_safe=False,
            reasons=["Missing SKILL.md"],
        )

    try:
        content = skill_md.read_text(encoding="utf-8")
    except Exception as exc:
        return ScanResult(is_safe=False, reasons=[f"Unreadable SKILL.md: {exc}"])

    safe = scan_context_content(content, filename=str(skill_md))
    if safe.startswith("[BLOCKED:"):
        injection = True
        reasons.append("Prompt injection detected in SKILL.md")

    if trusted_hashes is not None:
        current_hash = _hash_file(skill_md)
        key = str(skill_md)
        expected = trusted_hashes.get(key)
        if expected is not None and current_hash != expected:
            hash_mismatch = True
            reasons.append("Content hash mismatch")

    if source_repo is not None:
        if source_repo not in TRUSTED_REPOS:
            untrusted = True
            reasons.append(f"Source repo '{source_repo}' not in trusted list")

    is_safe = not reasons
    return ScanResult(
        is_safe=is_safe,
        reasons=reasons,
        hash_mismatch=hash_mismatch,
        injection_detected=injection,
        untrusted_source=untrusted,
    )


def quarantine_skill(skill_path: Path) -> Path:
    """Move a skill into the quarantine directory.

    Args:
        skill_path (Path): IN: Path to the skill directory. OUT: Moved to
            ``QUARANTINE_DIR``.

    Returns:
        Path: OUT: Destination path in quarantine.
    """

    from xerxes.extensions.skills_hub import QUARANTINE_DIR

    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    dest = QUARANTINE_DIR / skill_path.name
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    skill_path.rename(dest)
    logger.info("Quarantined skill %s → %s", skill_path, dest)
    return dest


def approve_skill(skill_name: str) -> str:
    """Move a quarantined skill back into the active skills directory.

    Args:
        skill_name (str): IN: Name of the quarantined skill directory. OUT:
            Looked up in ``QUARANTINE_DIR``.

    Returns:
        str: OUT: Human-readable result message.
    """

    from xerxes.extensions.skills_hub import QUARANTINE_DIR, SKILLS_DIR

    quarantined = QUARANTINE_DIR / skill_name
    if not quarantined.exists():
        return f"[Error] Skill '{skill_name}' not found in quarantine."

    target = SKILLS_DIR / skill_name
    if target.exists():
        import shutil

        shutil.rmtree(target)
    quarantined.rename(target)
    return f"Approved and activated skill '{skill_name}'"
