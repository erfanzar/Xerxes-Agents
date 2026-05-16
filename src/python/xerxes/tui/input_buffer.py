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
"""Multiline input buffer + key-binding helpers.

Provides the multiline editor, auto-suggestion, and file-history wiring
that back the TUI prompt, plus the file-mention completer hook."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from .._compat_shims import xerxes_subdir_safe


@dataclass
class InputBufferConfig:
    """Tunables for the multiline input buffer.

    Attributes:
        multiline: True to allow Enter-without-submit. Submission must
            be wired separately via the accept handler.
        history_path: absolute path for persisted Up/Down history.
        max_history: rolling cap; older entries are dropped on save.
        auto_suggest: enable ghost-text suggestions from history.
        complete_while_typing: show completion menu as user types.
        on_accept: callback that receives the submitted text."""

    multiline: bool = True
    history_path: Path | None = None
    max_history: int = 2_000
    auto_suggest: bool = True
    complete_while_typing: bool = True
    on_accept: Callable[[str], None] | None = None
    completer: Completer | None = None
    completer_callable: Callable[[Buffer], Completer | None] | None = field(default=None, repr=False)


def history_file_path() -> Path:
    """Return the default ``$XERXES_HOME/history`` path, ensuring its dir."""
    path = xerxes_subdir_safe("history")
    path.mkdir(parents=True, exist_ok=True)
    return path / "tui_history.txt"


def _make_history(config: InputBufferConfig):
    """Return a ``FileHistory`` (when ``history_path`` is set) or in-memory store."""
    if config.history_path is None:
        return InMemoryHistory()
    config.history_path.parent.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(config.history_path))


def build_input_buffer(config: InputBufferConfig | None = None) -> Buffer:
    """Build a prompt_toolkit ``Buffer`` configured per ``config``.

    Sets multiline, attaches FileHistory (or InMemoryHistory), enables
    ``AutoSuggestFromHistory``, and registers the accept handler if
    supplied. Doesn't bind keys — see ``build_multiline_key_bindings``."""

    cfg = config or InputBufferConfig()
    history = _make_history(cfg)

    accept = None
    if cfg.on_accept is not None:
        cb = cfg.on_accept

        def accept(buf: Buffer) -> bool:
            text = buf.text
            cb(text)
            return False  # don't keep the buffer text

    buf = Buffer(
        history=history,
        auto_suggest=AutoSuggestFromHistory() if cfg.auto_suggest else None,
        complete_while_typing=cfg.complete_while_typing,
        completer=cfg.completer,
        multiline=cfg.multiline,
        accept_handler=accept,
    )
    return buf


def build_multiline_key_bindings(
    *,
    submit_on_plain_enter: bool = True,
) -> KeyBindings:
    """Return a `KeyBindings` registry for multiline submission.

    Plain Enter submits (calls ``buffer.validate_and_handle``); Alt+Enter
    and Ctrl+J insert a literal newline. Caller adds these to its own
    KeyBindings via ``merge_key_bindings`` so the existing TUI bindings
    keep working."""

    kb = KeyBindings()

    if submit_on_plain_enter:

        @kb.add(Keys.Enter)
        def _accept(event):
            event.current_buffer.validate_and_handle()

    @kb.add(Keys.Escape, Keys.Enter)
    def _alt_enter_newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add(Keys.ControlJ)
    def _ctrl_j_newline(event):
        event.current_buffer.insert_text("\n")

    return kb


__all__ = [
    "InputBufferConfig",
    "build_input_buffer",
    "build_multiline_key_bindings",
    "history_file_path",
]
