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
"""Structured JSONL logger used by the daemon.

Each call writes a single JSON object per line to a daily-rotated file
(``daemon-YYYY-MM-DD.jsonl``) and mirrors a short ``[level] event`` line to
``stderr`` so an attached operator still sees activity.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


class DaemonLogger:
    """JSONL logger with daily file rotation.

    ``log_dir`` is created on instantiation. Files are named
    ``daemon-YYYY-MM-DD.jsonl`` and rotated lazily on the first write of a new
    UTC day. Every entry is also echoed as a single ``[level] event`` line on
    stderr.
    """

    def __init__(self, log_dir: str) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = None
        self._current_date = ""

    def _ensure_file(self) -> None:
        """Open or rotate the daily log file for the current UTC date."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._file:
                self._file.close()
            self._current_date = today
            path = self._dir / f"daemon-{today}.jsonl"
            self._file = open(path, "a", encoding="utf-8")

    def log(self, level: str, event: str, **kwargs: Any) -> None:
        """Write one ``{ts, level, event, ...kwargs}`` JSON object.

        Args:
            level: Severity label (``"info"``, ``"error"``, ...).
            event: Short event name.
            **kwargs: Extra structured fields merged into the line.
        """
        self._ensure_file()
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "level": level,
            "event": event,
            **kwargs,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        assert self._file is not None
        self._file.write(line + "\n")
        self._file.flush()

        print(f"[{level}] {event}", file=sys.stderr)

    def info(self, event: str, **kwargs: Any) -> None:
        """Write an ``info``-level entry."""
        self.log("info", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Write an ``error``-level entry."""
        self.log("error", event, **kwargs)

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file:
            self._file.close()
            self._file = None
