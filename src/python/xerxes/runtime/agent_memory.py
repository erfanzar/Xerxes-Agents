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
"""Agent-managed cross-project plus project-scoped persistent memory.

The agent owns two memory tiers it can read and write across a session:

    * **Global / cross-project** — lives under ``~/.xerxes/memory/``.
      Things that should apply everywhere the agent runs (user
      preferences, persona notes, durable knowledge).
    * **Project-scoped** — lives under ``~/.xerxes/projects/<hash>/memory/``
      where ``<hash>`` is derived from the project root path. Things that
      only matter for one codebase (architecture notes, weird file layouts,
      cross-session work logs).

Each scope ships the same canonical files (``IDENTITY.md``, ``SOUL.md``,
``USER.md``, ``MEMORY.md``, ``KNOWLEDGE.md``, ``INSIGHTS.md``,
``EXPERIENCES.md``) plus a ``journal/<date>.md`` daily-notes directory.
Reads, writes, appends, journaling, and search are exposed through
:class:`AgentMemory`; the same instance also renders the system-prompt
preamble that turns the files into an always-on second brain.
"""

from __future__ import annotations

import enum
import hashlib
import os
import re
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .._compat_shims import xerxes_subdir_safe

# Process-wide lock keyed by absolute target path. The append() path is
# read-modify-write, so two concurrent callers can race and lose one of the
# entries. The lock collapses each path's appenders into a critical section.
# Threading is sufficient here because all AgentMemory access is in-process
# (the daemon owns memory writes; other processes have their own daemons).
_APPEND_LOCKS: dict[str, threading.Lock] = {}
_APPEND_LOCKS_GUARD = threading.Lock()


def _append_lock_for(path: Path) -> threading.Lock:
    """Return (lazily creating) the process-wide append lock for ``path``."""
    key = str(path)
    with _APPEND_LOCKS_GUARD:
        lock = _APPEND_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _APPEND_LOCKS[key] = lock
        return lock


CANONICAL_FILES: tuple[str, ...] = (
    "IDENTITY.md",
    "SOUL.md",
    "USER.md",
    "MEMORY.md",
    "KNOWLEDGE.md",
    "INSIGHTS.md",
    "EXPERIENCES.md",
)

_DEFAULTS: dict[str, str] = {
    "IDENTITY.md": (
        "# Identity\n\n"
        "You are Xerxes. Track here what you've learned about *yourself* —\n"
        "voice, defaults, rituals, the lens you bring to work.\n"
    ),
    "SOUL.md": (
        "# Soul\n\n"
        "The persona, values, and orientation that show up no matter the\n"
        "task. Direct, pragmatic, technically careful, action-oriented.\n"
        "Prefers evidence from the workspace over guesses. Keeps memory\n"
        "private unless asked. Preserves user trust over task completion.\n"
    ),
    "USER.md": (
        "# User profile\n\n"
        "Track stable user preferences you've observed: tone, vocabulary,\n"
        "tools they like, the way they want plans presented, the things\n"
        "they hate seeing again.\n"
    ),
    "MEMORY.md": (
        "# Memory\n\n"
        "Durable facts, decisions, and project context worth carrying\n"
        "forward. One bullet per fact. Date entries when timing matters.\n"
    ),
    "KNOWLEDGE.md": (
        "# Knowledge\n\n"
        "Cumulative understanding — explanations of how things work,\n"
        "mental models, vocab. The handbook you wish existed.\n"
    ),
    "INSIGHTS.md": (
        "# Insights\n\nShort aha-moments, surprises, anti-patterns spotted. Append; do\nnot rewrite history.\n"
    ),
    "EXPERIENCES.md": (
        "# Experiences\n\n"
        "Persistent log of what's been tried — successes that surprised you,\n"
        "failures that should be remembered, and the heuristic for next\n"
        "time. **Search this before attempting something risky or novel.**\n"
        "Append after every meaningful failure and every non-obvious win.\n"
        "\n"
        "## Format\n\n"
        "Use this shape per entry:\n"
        "\n"
        "```\n"
        "### YYYY-MM-DD — short title\n"
        "\n"
        "**Tried:** what you attempted\n"
        "**Result:** what happened (success / failure / partial)\n"
        "**Lesson:** what to do (or avoid) next time\n"
        "```\n"
        "\n"
        "Append; never rewrite history.\n"
    ),
}


class AgentMemoryScope(enum.StrEnum):
    """Memory tier identifier; serialised as a plain string in tool outputs.

    Attributes:
        GLOBAL: Cross-project memory shared by every Xerxes session.
        PROJECT: Memory bound to a specific project root.
    """

    GLOBAL = "global"
    PROJECT = "project"


def default_global_memory_dir() -> Path:
    """Return the global memory directory (``~/.xerxes/memory/`` by default)."""
    return xerxes_subdir_safe("memory")


def _project_hash(project_root: Path) -> str:
    """Return a deterministic, salted 12-char id for ``project_root``.

    The salt (``XERXES_PROJECT_SALT`` env var, with a built-in default) keeps
    project directories distinct across users on a shared host.
    """
    salt = os.environ.get("XERXES_PROJECT_SALT", "xerxes-project-salt")
    payload = f"{salt}|{Path(project_root).expanduser().resolve()}".encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def project_memory_dir_for(project_root: Path | str) -> Path:
    """Return ``~/.xerxes/projects/<hash>/memory/`` for ``project_root``."""
    root = Path(project_root).expanduser().resolve()
    digest = _project_hash(root)
    base = xerxes_subdir_safe("projects", digest)
    return base / "memory"


@dataclass(frozen=True)
class AgentMemoryFile:
    """One file inside one of the agent's memory scopes.

    Attributes:
        scope: Which scope the file belongs to.
        path: Absolute filesystem path.
        relative: Path relative to the scope root (e.g. ``"MEMORY.md"``).
        bytes: File size in bytes at observation time.
    """

    scope: AgentMemoryScope
    path: Path
    relative: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly serialisation of this record."""
        return {"scope": self.scope.value, "relative": self.relative, "bytes": self.bytes}


@dataclass
class AgentMemory:
    """Two-tier persistent memory (global + project) backing the agent.

    Attributes:
        project_root: Current project's root path. The project memory dir is
            derived from this; ``None`` disables project scope for the session.
        global_dir: Override for the cross-project memory directory.
        project_dir: Override for the project-scoped memory directory; falls
            back to :func:`project_memory_dir_for` when ``project_root`` is
            set and ``project_dir`` is left unset.
    """

    project_root: Path | None = None
    global_dir: Path = field(default_factory=default_global_memory_dir)
    project_dir: Path | None = None

    def __post_init__(self) -> None:
        """Derive ``project_dir`` from ``project_root`` when it wasn't explicit."""
        if self.project_dir is None and self.project_root is not None:
            object.__setattr__(self, "project_dir", project_memory_dir_for(self.project_root))

    # ---------------------------- discovery ----------------------------

    def has_project_scope(self) -> bool:
        """Return ``True`` when this instance has a usable project scope."""
        return self.project_dir is not None

    def scope_dir(self, scope: AgentMemoryScope) -> Path:
        """Return the on-disk directory backing ``scope``.

        Raises:
            FileNotFoundError: ``scope`` is ``PROJECT`` but no ``project_dir``
                is configured.
        """
        if scope is AgentMemoryScope.GLOBAL:
            return self.global_dir
        if self.project_dir is None:
            raise FileNotFoundError("project memory scope is unavailable (no project_root configured)")
        return self.project_dir

    def ensure(self) -> None:
        """Create the canonical files (and ``journal/``) for every configured scope.

        Existing files are left untouched; missing ones get the default
        seed text from :data:`_DEFAULTS`.
        """
        for scope_dir in self._configured_dirs():
            scope_dir.mkdir(parents=True, exist_ok=True)
            (scope_dir / "journal").mkdir(parents=True, exist_ok=True)
            for name in CANONICAL_FILES:
                target = scope_dir / name
                if not target.exists():
                    target.write_text(_DEFAULTS[name], encoding="utf-8")

    def _configured_dirs(self) -> Iterable[Path]:
        """Yield each configured scope directory (global, then project)."""
        yield self.global_dir
        if self.project_dir is not None:
            yield self.project_dir

    # ---------------------------- read / write -------------------------

    def _resolve_inside(self, scope: AgentMemoryScope, rel: str) -> Path:
        """Resolve ``rel`` under ``scope``'s root and reject escapes.

        Raises:
            ValueError: ``rel`` is empty or its resolved form escapes the
                scope root.
        """
        if not rel:
            raise ValueError("relative path must be non-empty")
        scope_root = self.scope_dir(scope).resolve()
        candidate = (scope_root / rel).resolve()
        try:
            candidate.relative_to(scope_root)
        except ValueError as exc:
            raise ValueError(f"path {rel!r} escapes the {scope.value} memory scope") from exc
        return candidate

    def read(self, scope: AgentMemoryScope | str, relative: str) -> str:
        """Return the UTF-8 contents of ``scope/relative``.

        Raises:
            FileNotFoundError: The file does not exist in ``scope``.
            ValueError: ``relative`` escapes the scope root or is empty.
        """
        scope = AgentMemoryScope(scope) if isinstance(scope, str) else scope
        target = self._resolve_inside(scope, relative)
        if not target.exists():
            raise FileNotFoundError(f"{scope.value} memory has no file {relative!r}")
        return target.read_text(encoding="utf-8")

    def write(self, scope: AgentMemoryScope | str, relative: str, content: str) -> dict[str, Any]:
        """Atomically replace ``scope/relative`` with ``content``.

        The write goes through a ``tempfile`` + :func:`os.replace` so partial
        writes never leave a corrupt file. Returns a summary dict with the
        scope name, relative path, and byte count actually written.
        """
        scope = AgentMemoryScope(scope) if isinstance(scope, str) else scope
        target = self._resolve_inside(scope, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via tempfile + os.replace.
        fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return {
            "scope": scope.value,
            "path": str(target.relative_to(self.scope_dir(scope).resolve())),
            "bytes": len(content),
        }

    def append(
        self,
        scope: AgentMemoryScope | str,
        relative: str,
        content: str,
        *,
        section: str = "",
        timestamp: bool = True,
    ) -> dict[str, Any]:
        """Append ``content`` to ``scope/relative``, with optional section header.

        Args:
            scope: Target scope.
            relative: Path under the scope root.
            content: Body to append (whitespace-stripped).
            section: When non-empty, a ``## <section>`` header is inserted
                before the body.
            timestamp: When ``True``, prepend an HTML comment with the UTC
                ISO timestamp so future readers can sort entries.

        Returns:
            A summary dict with ``scope``, ``path`` (relative) and
            ``appended_bytes``.

        Note:
            Concurrent appenders against the same path are serialised via
            :func:`_append_lock_for` to avoid lost writes from the
            read-modify-write sequence.
        """
        scope = AgentMemoryScope(scope) if isinstance(scope, str) else scope
        target = self._resolve_inside(scope, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = content.strip()
        if section:
            body = f"## {section}\n\n{body}"
        if timestamp:
            body = f"<!-- {datetime.now(UTC).isoformat()} -->\n{body}"
        # Serialize concurrent appenders to the same file. Without this lock
        # two callers can both read the existing content, both compose a
        # write, and one writer's payload overwrites the other — silently
        # losing one of the journal entries.
        with _append_lock_for(target):
            existing = ""
            if target.exists():
                existing = target.read_text(encoding="utf-8")
                if existing and not existing.endswith("\n"):
                    existing += "\n"
                existing += "\n"
            target.write_text(existing + body + "\n", encoding="utf-8")
        return {
            "scope": scope.value,
            "path": str(target.relative_to(self.scope_dir(scope).resolve())),
            "appended_bytes": len(body),
        }

    def journal(self, scope: AgentMemoryScope | str, note: str, *, when: datetime | None = None) -> dict[str, Any]:
        """Append a timestamped one-liner to ``journal/<YYYY-MM-DD>.md``.

        Args:
            scope: Memory scope to write into.
            note: Free-form note text.
            when: Override clock for the timestamp (UTC); defaults to ``now``.
        """
        scope = AgentMemoryScope(scope) if isinstance(scope, str) else scope
        now = when or datetime.now(UTC)
        day_path = f"journal/{now.date().isoformat()}.md"
        return self.append(scope, day_path, f"- {now.strftime('%H:%M:%S')}  {note.strip()}", timestamp=False)

    def list_files(self, scope: AgentMemoryScope | str | None = None) -> list[AgentMemoryFile]:
        """Walk one (or both) scope directories and return every file.

        When ``scope`` is ``None``, both scopes are returned (global, then
        project if configured). Files that fail to stat are silently skipped.
        """
        out: list[AgentMemoryFile] = []
        scopes: list[AgentMemoryScope]
        if scope is None:
            scopes = [AgentMemoryScope.GLOBAL]
            if self.project_dir is not None:
                scopes.append(AgentMemoryScope.PROJECT)
        else:
            scopes = [AgentMemoryScope(scope) if isinstance(scope, str) else scope]
        for sc in scopes:
            try:
                root = self.scope_dir(sc).resolve()
            except FileNotFoundError:
                continue
            if not root.is_dir():
                continue
            for current, _dirs, files in os.walk(root):
                for name in sorted(files):
                    full = Path(current) / name
                    try:
                        rel = str(full.relative_to(root))
                        size = full.stat().st_size
                    except OSError:
                        continue
                    out.append(AgentMemoryFile(scope=sc, path=full, relative=rel, bytes=size))
        return out

    def search(
        self, query: str, *, scope: AgentMemoryScope | str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Run a case-insensitive substring search across memory files.

        Returns up to ``limit`` hits, each containing ``scope``, ``path``
        (relative), and a short ``snippet`` of surrounding text. At most
        three snippets per file are emitted so a single huge file can't
        dominate the result set.
        """
        q = query.lower().strip()
        if not q:
            return []
        hits: list[dict[str, Any]] = []
        for entry in self.list_files(scope):
            try:
                text = entry.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lowered = text.lower()
            idx = 0
            for _ in range(3):  # at most 3 snippets per file
                pos = lowered.find(q, idx)
                if pos == -1:
                    break
                start = max(0, pos - 60)
                end = min(len(text), pos + len(q) + 60)
                snippet = text[start:end].replace("\n", " / ")
                hits.append({"scope": entry.scope.value, "path": entry.relative, "snippet": snippet})
                idx = pos + len(q)
                if len(hits) >= limit:
                    return hits
        return hits

    # ---------------------------- prompt section -----------------------

    def to_prompt_section(self, *, max_bytes_per_file: int = 4_000) -> str:
        """Render the persistent-memory protocol plus current file contents.

        The returned string is pasted into the system prompt at every session
        start. It contains three sections:

            * **Wake-up read** announcing that identity, soul, knowledge,
              experiences, and journal are already loaded.
            * **Operational protocol** with concrete rules for when to update
              which file, with explicit tool examples.
            * **Current contents** of every canonical file (truncated to
              ``max_bytes_per_file``) plus journal entries from the last
              seven days.
        """

        self.ensure()
        sections: list[str] = []
        sections.append("# Your persistent memory — wake up, read, update, repeat\n")

        scope_lines = [
            f"- **global** (`{self.global_dir}`): facts that apply across every project.",
        ]
        if self.project_dir is not None:
            scope_lines.append(f"- **project** (`{self.project_dir}`): facts about this codebase only.")
        else:
            scope_lines.append("- (project memory unavailable — no project root configured)")
        sections.append("You own two memory scopes:\n" + "\n".join(scope_lines) + "\n")

        sections.append(
            "## How you must use these — operational protocol\n"
            "\n"
            "Treat persistent memory as an active loop, not a passive archive:\n"
            "\n"
            "1. **Right now (wake-up):** Everything below is the current state of\n"
            "   your memory. You've already read it. Use it.\n"
            "2. **Before attempting anything risky, novel, or that's failed before:**\n"
            '   call `agent_memory_search("<short keyword>")` against\n'
            "   `EXPERIENCES.md` to see if there's a prior outcome you should\n"
            "   respect. Hit it BEFORE you act, not after.\n"
            "3. **After every meaningful failure:** append to `EXPERIENCES.md`\n"
            '   using `agent_memory_append(scope, "EXPERIENCES.md", body,\n'
            '   section="YYYY-MM-DD — short title")`. Format each entry as:\n'
            "   ```\n"
            "   **Tried:** <what you attempted>\n"
            "   **Result:** failure — <what actually happened>\n"
            "   **Lesson:** <what to do or avoid next time>\n"
            "   ```\n"
            "4. **After non-obvious successes** (the trick wasn't obvious from the\n"
            "   docs): also append to `EXPERIENCES.md` so future-you remembers.\n"
            "5. **When you learn a durable fact** about the project / codebase /\n"
            "   user setup: append to `MEMORY.md` (project scope is usually right).\n"
            "6. **When you build a mental model** (how a subsystem fits together,\n"
            "   why a design choice was made): write it to `KNOWLEDGE.md`.\n"
            '7. **When the user signals a preference** ("I like terse answers",\n'
            '   "never use emoji", "prefer x over y"): append to `USER.md`\n'
            "   in the **global** scope.\n"
            "8. **At the end of each substantive turn** (you made progress, you\n"
            "   shipped something, you got blocked): call\n"
            '   `agent_memory_journal("project", "<one-line summary>")` so\n'
            "   tomorrow-you sees today's progress.\n"
            "9. **Soul / identity drift** — if the user reshapes how you should\n"
            "   show up, update `SOUL.md` or `IDENTITY.md` accordingly.\n"
            "\n"
            "**Choosing scope.** Global is for *you across every project* and\n"
            "*the user wherever they show up*. Project is for *this codebase*.\n"
            "If you're not sure, prefer **project** — it's always safe to\n"
            "promote a project entry to global later, but cluttering global\n"
            "with project-only details is hard to undo.\n"
            "\n"
            "**Tools.** `agent_memory_read | write | append | journal | search | list | status`.\n"
            'All return `{ok, ...}` dicts; check `.get("ok")` and surface\n'
            "errors. Path escapes outside the scope root are blocked.\n"
        )

        # Now the current contents. Order matters: SOUL/IDENTITY first
        # (who you are), then USER (who you serve), then EXPERIENCES
        # (what you've learned the hard way), then MEMORY/KNOWLEDGE/INSIGHTS.
        order = ["SOUL.md", "IDENTITY.md", "USER.md", "EXPERIENCES.md", "MEMORY.md", "KNOWLEDGE.md", "INSIGHTS.md"]
        priority = {name: i for i, name in enumerate(order)}

        sections.append("## Current memory contents\n")

        def _entry_sort_key(entry: AgentMemoryFile) -> tuple[int, int, str]:
            """Sort key putting project scope first, then a canonical-file order."""
            scope_order = 0 if entry.scope is AgentMemoryScope.PROJECT else 1
            order_idx = priority.get(entry.relative, 99)
            return (scope_order, order_idx, entry.relative)

        entries = sorted(self.list_files(), key=_entry_sort_key)
        for entry in entries:
            if entry.bytes == 0:
                continue
            if not entry.relative.endswith(".md"):
                continue
            # Journal entries: only surface the last 7 days.
            if entry.relative.startswith("journal/"):
                day = entry.relative.removeprefix("journal/").removesuffix(".md")
                try:
                    parsed = date.fromisoformat(day)
                except ValueError:
                    continue
                if (date.today() - parsed).days > 7:
                    continue
            try:
                text = entry.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            text = text.strip()
            if len(text) > max_bytes_per_file:
                text = (
                    text[:max_bytes_per_file].rstrip()
                    + f'\n\n[... truncated; agent_memory_read("{entry.scope.value}", "{entry.relative}") for full text ...]'
                )
            sections.append(f"### [{entry.scope.value}] {entry.relative}\n\n{text}\n")
        return "\n".join(sections).rstrip() + "\n"

    # ---------------------------- export -------------------------------

    def status(self) -> dict[str, Any]:
        """Return a small dict summarising configured directories and file counts."""
        files = self.list_files()
        by_scope: dict[str, int] = {}
        for f in files:
            by_scope[f.scope.value] = by_scope.get(f.scope.value, 0) + 1
        return {
            "global_dir": str(self.global_dir),
            "project_dir": str(self.project_dir) if self.project_dir else None,
            "files_by_scope": by_scope,
            "total_files": len(files),
        }


_HEX_SAFE_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")


def safe_relative(text: str) -> bool:
    """Return ``True`` when ``text`` is a safe relative path (no escapes or NULs)."""
    return bool(text) and ".." not in Path(text).parts and bool(_HEX_SAFE_RE.match(text))


__all__ = [
    "CANONICAL_FILES",
    "AgentMemory",
    "AgentMemoryFile",
    "AgentMemoryScope",
    "default_global_memory_dir",
    "project_memory_dir_for",
    "safe_relative",
]
