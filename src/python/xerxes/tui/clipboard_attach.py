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
"""Pending-attachment state for the TUI.

Wraps ``clipboard.save_clipboard_image`` with a small bag of pending
attachments that the next message inherits.
The TUI key handler (Ctrl+V / Alt+V) writes here; the submit path
reads-and-clears."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .._compat_shims import xerxes_subdir_safe


@dataclass(frozen=True)
class Attachment:
    """One staged attachment to be sent alongside the next submission.

    Attributes:
        path: Absolute path on disk to the captured file.
        bytes: File size in bytes (``0`` when stat fails).
        kind: Either ``"image"`` (clipboard capture) or ``"file"``
            (user-attached arbitrary file).
    """

    path: Path
    bytes: int
    kind: str  # "image" | "file"


class AttachmentBuffer:
    """Stage clipboard images / file paths for the next submission.

    The TUI surfaces ``Ctrl+V`` → ``capture_clipboard_image`` (or
    ``attach_path`` from a slash command). ``drain`` returns and clears
    the staged attachments — call from your submit handler."""

    def __init__(
        self,
        *,
        capture_image: Callable[[Path], Path | None] | None = None,
        store_dir: Path | None = None,
    ) -> None:
        """Initialize the buffer with a capture callable and a staging directory.

        Args:
            capture_image: Function that writes the clipboard image into the
                given directory and returns the resulting path (or ``None``
                when the clipboard is empty/unsupported). Defaults to
                :func:`xerxes.tui.clipboard.save_clipboard_image`.
            store_dir: Directory holding captured images. Defaults to
                ``$XERXES_HOME/clipboard``.
        """
        from .clipboard import save_clipboard_image as _default_capture

        self._capture = capture_image or _default_capture
        self._dir = store_dir or xerxes_subdir_safe("clipboard")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._items: list[Attachment] = []

    def capture_clipboard_image(self) -> Attachment | None:
        """Try to grab the OS clipboard image and stage it.

        Returns the new ``Attachment`` (also added to the buffer) or
        ``None`` if the clipboard had no image."""
        path = self._capture(self._dir)
        if path is None:
            return None
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        att = Attachment(path=Path(path), bytes=size, kind="image")
        with self._lock:
            self._items.append(att)
        return att

    def attach_path(self, file_path: Path | str, *, kind: str = "file") -> Attachment | None:
        """Stage an existing file path; returns ``None`` when the path doesn't exist."""
        p = Path(file_path).expanduser()
        if not p.exists():
            return None
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        att = Attachment(path=p, bytes=size, kind=kind)
        with self._lock:
            self._items.append(att)
        return att

    def pending(self) -> list[Attachment]:
        """Return a snapshot copy of staged attachments without consuming them."""
        with self._lock:
            return list(self._items)

    def drain(self) -> list[Attachment]:
        """Return + clear the staged attachments atomically (called on submit)."""
        with self._lock:
            out = list(self._items)
            self._items.clear()
            return out

    def clear(self) -> int:
        """Discard staged attachments; returns the number that were dropped."""
        with self._lock:
            n = len(self._items)
            self._items.clear()
            return n


__all__ = ["Attachment", "AttachmentBuffer"]
