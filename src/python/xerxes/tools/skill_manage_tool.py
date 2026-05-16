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
"""Agent-facing tool for managing markdown skills mid-conversation.

Exposes :func:`skill_manage` with the actions ``list``, ``view``, ``create``,
``edit`` and ``delete``. Created or edited skills land under
``~/.xerxes/skills/agent-authored/`` as ``<name>.md`` files with a small YAML
front-matter. Bodies pass through :func:`scan_context_content` before being
written — if the security scan empties the body, the write aborts with an
explicit error so prompt-injection payloads cannot persist themselves as
skills.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.paths import xerxes_subdir
from ..security.prompt_scanner import scan_context_content

AUTHORED_DIR = xerxes_subdir("skills", "agent-authored")


@dataclass
class SkillManageResult:
    """Structured outcome of a :func:`skill_manage` invocation.

    Attributes:
        intent: One of ``create``, ``edit``, ``view``, ``delete``, ``list``.
        name: Skill filename stem (no extension).
        path: Absolute path of the affected file on disk.
        ok: True when the action succeeded.
        error: Human-readable error message when ``ok`` is False.
    """

    intent: str
    name: str
    path: str
    ok: bool
    error: str = ""


def _frontmatter(name: str, description: str, version: str) -> str:
    """Build the YAML front-matter block prepended to authored skills."""
    return f"---\nname: {name}\ndescription: {description}\nversion: {version}\nauthor: xerxes-agent\n---\n\n"


def _skill_path(name: str) -> Path:
    """Return the on-disk path for a skill named ``name``, ensuring the directory exists."""
    AUTHORED_DIR.mkdir(parents=True, exist_ok=True)
    return AUTHORED_DIR / f"{name}.md"


def _validate_name(name: str) -> None:
    """Reject empty or path-traversing skill names.

    Raises:
        ValueError: ``name`` is empty or contains ``/`` or ``..``.
    """
    if not name or "/" in name or ".." in name:
        raise ValueError(f"invalid skill name: {name!r}")


def skill_manage(
    intent: str,
    *,
    name: str = "",
    body: str = "",
    description: str = "",
    version: str = "0.1.0",
) -> dict[str, Any]:
    """Create, edit, view, delete, or list agent-authored skills.

    Args:
        intent: Action to perform — ``create``, ``edit``, ``view``, ``delete``,
            or ``list``. ``list`` is the only action that ignores ``name``.
        name: Skill filename stem (no extension). Cannot contain ``/`` or
            ``..``.
        body: Markdown body for ``create``/``edit``. Routed through
            :func:`scan_context_content` and rejected if the scanner empties
            it.
        description: Short description written into the YAML front-matter
            for new or edited skills.
        version: Semantic version stamped into the front-matter.

    Returns:
        Mapping with ``ok`` plus ``intent`` and ``name``. ``list`` returns
        a ``skills`` array of names; ``view`` adds ``body`` and ``path``;
        ``create``/``edit``/``delete`` return ``path``. Failures carry an
        ``error`` field.
    """
    if intent == "list":
        AUTHORED_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "ok": True,
            "intent": "list",
            "skills": sorted(p.stem for p in AUTHORED_DIR.glob("*.md")),
        }

    _validate_name(name)
    path = _skill_path(name)

    if intent == "view":
        if not path.exists():
            return {"ok": False, "intent": "view", "name": name, "error": "not found"}
        return {"ok": True, "intent": "view", "name": name, "path": str(path), "body": path.read_text(encoding="utf-8")}

    if intent == "delete":
        if not path.exists():
            return {"ok": False, "intent": "delete", "name": name, "error": "not found"}
        path.unlink()
        return {"ok": True, "intent": "delete", "name": name, "path": str(path)}

    if intent in ("create", "edit"):
        if not body.strip():
            return {"ok": False, "intent": intent, "name": name, "error": "body must be non-empty"}
        # scan_context_content returns the sanitized text; if it stripped
        # significant content we surface a soft warning but still write
        # (the operator owns ``~/.xerxes/skills/agent-authored/``).
        scanned = scan_context_content(body, filename=str(path))
        if scanned.strip() == "":
            return {
                "ok": False,
                "intent": intent,
                "name": name,
                "error": "security scan stripped all content",
            }
        body = scanned
        if intent == "create" and path.exists():
            return {"ok": False, "intent": "create", "name": name, "error": "skill already exists; use intent=edit"}
        content = _frontmatter(name, description or "Agent-authored skill", version) + body.lstrip()
        path.write_text(content, encoding="utf-8")
        return {"ok": True, "intent": intent, "name": name, "path": str(path)}

    return {"ok": False, "intent": intent, "name": name, "error": f"unknown intent: {intent}"}


__all__ = ["AUTHORED_DIR", "SkillManageResult", "skill_manage"]
