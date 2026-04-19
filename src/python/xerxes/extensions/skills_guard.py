# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Skill security scanner — quarantine, hash verification, trusted repos.

Before a skill from an external source is activated, it passes through
``skills_guard`` for:

1. **Content hash** verification against a trusted-repo allowlist.
2. **Prompt-injection** scanning (reuses ``xerxes.security.prompt_scanner``).
3. **Quarantine** — untrusted skills are staged in ``~/.xerxes/skills/.hub/quarantine/``
   until explicitly approved.

Usage::

    from xerxes.extensions.skills_guard import ScanResult, scan_skill

    result = scan_skill(skill_path)
    if result.is_safe:
        activate_skill(skill_path)
    else:
        print(result.reasons)
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
    """Load known-good SHA-256 hashes for skill files.

    Returns:
        Mapping of ``filepath → sha256_hex``.
    """
    if not _TRUSTED_HASHES_PATH.exists():
        return {}
    try:
        return json.loads(_TRUSTED_HASHES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_trusted_hashes(data: dict[str, str]) -> None:
    _TRUSTED_HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRUSTED_HASHES_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


@dataclass
class ScanResult:
    """Outcome of a skill security scan."""

    is_safe: bool
    reasons: list[str] = field(default_factory=list)
    hash_mismatch: bool = False
    injection_detected: bool = False
    untrusted_source: bool = False

    @property
    def summary(self) -> str:
        if self.is_safe:
            return "Safe"
        return "; ".join(self.reasons) if self.reasons else "Unsafe"


def _hash_file(path: Path) -> str:
    """Return SHA-256 hex digest of *path* contents."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _hash_directory(dir_path: Path) -> str:
    """Return a deterministic SHA-256 of all files under *dir_path*."""
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
    """Run a full security scan on a skill directory or SKILL.md file.

    Checks:
    1. SKILL.md exists and is readable.
    2. No prompt-injection patterns in SKILL.md.
    3. Content hash matches known-good hash (if provided).
    4. Source repo is in the trusted allowlist (if provided).

    Args:
        skill_path: Path to the skill directory or SKILL.md file.
        source_repo: Optional ``owner/repo`` string for trust checking.
        trusted_hashes: Optional mapping of known-good hashes.

    Returns:
        A :class:`ScanResult` with ``is_safe`` and detailed flags.
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
        skill_path: Path to the skill directory.

    Returns:
        The new quarantine path.
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
    """Approve a quarantined skill and move it to the active skills dir.

    Args:
        skill_name: Name of the quarantined skill.

    Returns:
        Status message.
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
