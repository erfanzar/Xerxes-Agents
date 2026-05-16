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
"""@-mention completer for files / folders / git refs / URLs.

The completer surfaces matches for:

    * ``@file:/some/path.py``  → file path completion
    * ``@folder:/some/dir``    → directory completion
    * ``@diff``                → ``git diff`` of unstaged changes
    * ``@staged``              → ``git diff --staged``
    * ``@git:<ref>``           → arbitrary git ref / branch
    * ``@url:<u>``             → raw URL placeholder

It also resolves a typed token into a concrete payload via
``expand_mention`` so the TUI can substitute the mention with the
referenced content when the user submits.

Exports:
    - AT_TRIGGERS
    - AtMentionCompleter
    - expand_mention"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

AT_TRIGGERS: tuple[str, ...] = (
    "@file:",
    "@folder:",
    "@diff",
    "@staged",
    "@git:",
    "@url:",
)

_GIT_REFS = ("HEAD", "main", "master", "develop")


@dataclass
class _ParsedAt:
    """The ``@…`` token under the cursor, parsed into kind + remainder.

    Attributes:
        trigger: One of :data:`AT_TRIGGERS` (e.g. ``"@file:"`` / ``"@diff"``)
            — or the bare ``"@"`` when no concrete trigger matched yet.
        remainder: Text after the trigger; the completion candidates filter
            against this prefix.
        start: Absolute position of the ``@`` in ``document.text_before_cursor``.
        text_before_token: Everything left of the trigger; useful for
            substituting the token in-place on commit.
    """

    trigger: str  # one of AT_TRIGGERS, e.g. "@file:" or "@diff"
    remainder: str  # what comes after the trigger
    start: int  # absolute position in the document text
    text_before_token: str  # text before the trigger (for replacement)


def _parse_at_under_cursor(document: Document) -> _ParsedAt | None:
    """Detect an active ``@…`` token at the cursor; ``None`` when not inside one."""
    text = document.text_before_cursor
    # Look back for the last "@". Bail out if there's a space between
    # cursor and the "@" — we're not inside a token.
    at_idx = text.rfind("@")
    if at_idx == -1:
        return None
    span = text[at_idx:]
    if " " in span or "\t" in span or "\n" in span:
        return None
    for trigger in AT_TRIGGERS:
        if span.startswith(trigger):
            return _ParsedAt(
                trigger=trigger,
                remainder=span[len(trigger) :],
                start=at_idx,
                text_before_token=text[:at_idx],
            )
    # Bare "@..." without a trigger — surface every trigger.
    return _ParsedAt(trigger="@", remainder=span[1:], start=at_idx, text_before_token=text[:at_idx])


class AtMentionCompleter(Completer):
    """Yield completions for @-mentions.

    Constructed with a workspace root so file/folder completion stays
    inside the project. ``git_root`` defaults to ``workspace_root``."""

    def __init__(
        self,
        workspace_root: Path | str = ".",
        *,
        git_root: Path | str | None = None,
        max_results: int = 30,
    ) -> None:
        """Anchor the completer to ``workspace_root`` and an optional ``git_root``.

        Args:
            workspace_root: Directory used as the base for ``@file:`` /
                ``@folder:`` completions; relative prefixes are resolved
                under it.
            git_root: Repository used for ``@git:`` / ``@diff`` /
                ``@staged``; defaults to ``workspace_root``.
            max_results: Hard cap on completions per request so deep
                directories don't flood the menu.
        """
        # Resolve once so symlink-resolved entries can be made relative
        # cleanly (macOS: /var → /private/var).
        self._root = Path(workspace_root).expanduser().resolve()
        self._git_root = Path(git_root).expanduser().resolve() if git_root else self._root
        self._max = int(max_results)

    def get_completions(self, document: Document, _complete_event: CompleteEvent) -> Iterable[Completion]:
        """Yield completions for the ``@…`` token under the cursor, if any."""
        parsed = _parse_at_under_cursor(document)
        if parsed is None:
            return
        if parsed.trigger == "@":
            # Suggest every trigger.
            for t in AT_TRIGGERS:
                yield Completion(t, start_position=-len(parsed.trigger) - len(parsed.remainder), display=t)
            return
        if parsed.trigger == "@file:":
            yield from self._complete_path(parsed, files=True, dirs=False)
            return
        if parsed.trigger == "@folder:":
            yield from self._complete_path(parsed, files=False, dirs=True)
            return
        if parsed.trigger == "@git:":
            yield from self._complete_git_refs(parsed)
            return
        if parsed.trigger == "@diff" or parsed.trigger == "@staged" or parsed.trigger == "@url:":
            # Singletons — nothing else to complete.
            return

    def _complete_path(self, parsed: _ParsedAt, *, files: bool, dirs: bool) -> Iterable[Completion]:
        """Yield path entries under the prefix in ``parsed.remainder``.

        ``files``/``dirs`` toggle which kinds are surfaced; absolute prefixes
        are walked from disk root, relative prefixes from the workspace root."""
        prefix = parsed.remainder
        # Determine the base directory: relative prefixes are anchored
        # to the workspace root, absolute prefixes are honored as-is.
        if prefix.startswith("/"):
            base = Path(prefix).parent
            stem = Path(prefix).name
        else:
            rel_parent = Path(prefix).parent if prefix else Path()
            base = (self._root / rel_parent).resolve()
            stem = Path(prefix).name
        try:
            entries = sorted(base.iterdir(), key=lambda p: p.name)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return
        count = 0
        for entry in entries:
            if stem and not entry.name.lower().startswith(stem.lower()):
                continue
            is_dir = entry.is_dir()
            if is_dir and not dirs and not files:
                continue
            if is_dir and not dirs:
                continue
            if not is_dir and not files:
                continue
            display = entry.name + ("/" if is_dir else "")
            try:
                rel = str(entry.resolve().relative_to(self._root))
            except ValueError:
                rel = str(entry.resolve())
            replacement = rel + ("/" if is_dir else "")
            yield Completion(
                replacement,
                start_position=-len(prefix),
                display=display,
            )
            count += 1
            if count >= self._max:
                return

    def _complete_git_refs(self, parsed: _ParsedAt) -> Iterable[Completion]:
        """Yield git refs (HEAD plus local branches) matching ``parsed.remainder``."""
        prefix = parsed.remainder.lower()
        refs = list(_GIT_REFS)
        if shutil.which("git") is not None:
            try:
                out = subprocess.run(
                    ["git", "-C", str(self._git_root), "branch", "--format=%(refname:short)"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if out.returncode == 0:
                    for line in out.stdout.splitlines():
                        line = line.strip()
                        if line and line not in refs:
                            refs.append(line)
            except (subprocess.SubprocessError, OSError):
                pass
        count = 0
        for ref in refs:
            if prefix and not ref.lower().startswith(prefix):
                continue
            yield Completion(ref, start_position=-len(parsed.remainder), display=ref)
            count += 1
            if count >= self._max:
                return


# ---------------------------- expansion ------------------------------------


_TOKEN_RE = re.compile(r"@(?:file|folder|git|url):[^\s]+|@diff\b|@staged\b")


@dataclass
class ExpandedMention:
    """One ``@`` token expanded into a concrete payload.

    Attributes:
        token: Original token text, e.g. ``"@file:src/main.py"``.
        kind: Token category (``"file"``, ``"folder"``, ``"diff"``,
            ``"staged"``, ``"git"``, ``"url"``, or ``"unknown"``).
        payload: Resolved content (file body, directory listing, git
            output, raw URL, ...). Empty on error.
        error: Human-readable error message; empty on success.
    """

    token: str
    kind: str
    payload: str = ""
    error: str = ""


def expand_mention(token: str, *, workspace_root: Path | str = ".") -> ExpandedMention:
    """Resolve a single ``@…`` token into a payload string.

    File and folder targets are constrained to ``workspace_root`` to keep
    accidental ``../etc/passwd`` escapes from leaking onto the wire.
    Git operations shell out to ``git -C workspace_root``; URLs are
    returned verbatim for the agent to fetch later."""
    root = Path(workspace_root).expanduser()
    if token == "@diff":
        return _git_run(root, ["diff"], "diff")
    if token == "@staged":
        return _git_run(root, ["diff", "--staged"], "staged")
    if token.startswith("@git:"):
        return _git_run(root, ["log", "-1", "--format=%H %s", token[5:]], "git")
    if token.startswith("@url:"):
        return ExpandedMention(token=token, kind="url", payload=token[5:])
    if token.startswith("@file:") or token.startswith("@folder:"):
        kind, _, rel = token.partition(":")
        target = (root / rel).resolve()
        try:
            if not str(target).startswith(str(root.resolve())):
                return ExpandedMention(token=token, kind=kind[1:], error="escapes workspace root")
        except OSError as exc:
            return ExpandedMention(token=token, kind=kind[1:], error=str(exc))
        if not target.exists():
            return ExpandedMention(token=token, kind=kind[1:], error="not found")
        if target.is_file():
            try:
                return ExpandedMention(
                    token=token, kind="file", payload=target.read_text(encoding="utf-8", errors="replace")
                )
            except OSError as exc:
                return ExpandedMention(token=token, kind="file", error=str(exc))
        if target.is_dir():
            entries = sorted(p.name for p in target.iterdir())
            return ExpandedMention(token=token, kind="folder", payload="\n".join(entries))
    return ExpandedMention(token=token, kind="unknown", error="unrecognized trigger")


def _git_run(root: Path, args: list[str], kind: str) -> ExpandedMention:
    """Run ``git -C root args`` with a 5 s timeout and box the result."""
    if shutil.which("git") is None:
        return ExpandedMention(token="@" + kind, kind=kind, error="git binary not on PATH")
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return ExpandedMention(token="@" + kind, kind=kind, error=str(exc))
    if out.returncode != 0:
        return ExpandedMention(token="@" + kind, kind=kind, error=out.stderr.strip())
    return ExpandedMention(token="@" + kind, kind=kind, payload=out.stdout)


def expand_mentions_in_text(text: str, *, workspace_root: Path | str = ".") -> list[ExpandedMention]:
    """Resolve every ``@`` token found in ``text`` in document order."""
    return [expand_mention(m.group(0), workspace_root=workspace_root) for m in _TOKEN_RE.finditer(text)]


__all__ = ["AT_TRIGGERS", "AtMentionCompleter", "ExpandedMention", "expand_mention", "expand_mentions_in_text"]
