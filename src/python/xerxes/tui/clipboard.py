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
"""Clipboard helpers for the TUI.

Two surfaces:
    * Image: ``grab_clipboard_image`` returns a Pillow ``Image`` (or
      ``None`` on Linux/headless where ``ImageGrab`` is unsupported).
    * Text: ``grab_clipboard_text`` returns the text contents.

Plus a small ``PromptQueue`` that buffers prompts the user typed
while the agent was generating — the TUI fires them in order as the
agent comes free."""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Any


def grab_clipboard_image() -> Any | None:
    """Return a Pillow Image of the clipboard contents, or ``None``."""
    try:
        from PIL import ImageGrab  # type: ignore
    except ImportError:
        return None
    try:
        return ImageGrab.grabclipboard()
    except (NotImplementedError, OSError):
        return None


def save_clipboard_image(target_dir: Path, *, prefix: str = "xerxes-clip-") -> Path | None:
    """Persist the clipboard image to ``target_dir`` and return the path."""
    img = grab_clipboard_image()
    if img is None or not hasattr(img, "save"):
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    import tempfile

    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".png", dir=str(target_dir))
    import os

    os.close(fd)
    out = Path(name)
    img.save(out, "PNG")
    return out


def grab_clipboard_text() -> str | None:
    """Return clipboard text or None if no clipboard backend exists."""
    try:
        import subprocess

        # macOS first.
        try:
            out = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2)
            if out.returncode == 0:
                return out.stdout
        except FileNotFoundError:
            pass
        # Linux: xclip / wl-paste.
        for binary in ("wl-paste", "xclip"):
            try:
                args = [binary] if binary == "wl-paste" else [binary, "-selection", "clipboard", "-o"]
                out = subprocess.run(args, capture_output=True, text=True, timeout=2)
                if out.returncode == 0:
                    return out.stdout
            except FileNotFoundError:
                continue
    except Exception:
        pass
    return None


class PromptQueue:
    """Thread-safe FIFO of pending user prompts.

    The TUI's input loop pushes here while an agent turn is running;
    after the turn finishes, the loop drains and feeds them in order."""

    def __init__(self) -> None:
        """Construct an empty thread-safe FIFO."""
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()

    def push(self, prompt: str) -> int:
        """Enqueue ``prompt`` (ignored when empty); returns the new queue length."""
        if not prompt:
            return self.size()
        with self._lock:
            self._queue.append(prompt)
            return len(self._queue)

    def pop(self) -> str | None:
        """Remove and return the next queued prompt, or ``None`` when empty."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def drain(self) -> list[str]:
        """Atomically return + clear every pending prompt in FIFO order."""
        with self._lock:
            out = list(self._queue)
            self._queue.clear()
            return out

    def size(self) -> int:
        """Return the number of queued prompts."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> int:
        """Drop every queued prompt; returns the number that were discarded."""
        with self._lock:
            n = len(self._queue)
            self._queue.clear()
            return n


__all__ = [
    "PromptQueue",
    "grab_clipboard_image",
    "grab_clipboard_text",
    "save_clipboard_image",
]
