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
"""Structured JSONL logging for the Xerxes daemon.

``DaemonLogger`` writes timestamped log entries to daily rotating files and
mirrors each line to ``stderr``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


class DaemonLogger:
    """JSONL logger with daily file rotation.

    Args:
        log_dir (str): IN: Directory path for log files. OUT: Created if
            missing; files named ``daemon-{YYYY-MM-DD}.jsonl`` are written
            inside it.
    """

    def __init__(self, log_dir: str) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            log_dir (str): IN: log dir. OUT: Consumed during execution."""
        self._dir = Path(log_dir)
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            log_dir (str): IN: log dir. OUT: Consumed during execution."""
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            log_dir (str): IN: log dir. OUT: Consumed during execution."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = None
        self._current_date = ""

    def _ensure_file(self) -> None:
        """Open or rotate the log file based on the current UTC date.

        Returns:
            None: OUT: ``self._file`` points to the correct daily file.
        """
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._file:
                self._file.close()
            self._current_date = today
            path = self._dir / f"daemon-{today}.jsonl"
            self._file = open(path, "a", encoding="utf-8")

    def log(self, level: str, event: str, **kwargs: Any) -> None:
        """Write a structured log entry.

        Args:
            level (str): IN: Severity label (e.g. ``"info"``, ``"error"``).
                OUT: Written as ``"level"`` in the JSON object.
            event (str): IN: Short event name. OUT: Written as ``"event"``.
            **kwargs: IN: Arbitrary extra fields. OUT: Merged into the JSON
                object and printed to ``stderr``.

        Returns:
            None: OUT: Line appended to the current daily log file and echoed
            to ``stderr``.
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
        """Convenience wrapper for ``log("info", ...)``.

        Args:
            event (str): IN: Event name. OUT: Passed to ``log``.
            **kwargs: IN: Extra fields. OUT: Passed to ``log``.

        Returns:
            None: OUT: Log line written at ``"info"`` level.
        """
        self.log("info", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Convenience wrapper for ``log("error", ...)``.

        Args:
            event (str): IN: Event name. OUT: Passed to ``log``.
            **kwargs: IN: Extra fields. OUT: Passed to ``log``.

        Returns:
            None: OUT: Log line written at ``"error"`` level.
        """
        self.log("error", event, **kwargs)

    def close(self) -> None:
        """Close the current log file handle.

        Returns:
            None: OUT: ``self._file`` is closed and nulled.
        """
        if self._file:
            self._file.close()
            self._file = None
