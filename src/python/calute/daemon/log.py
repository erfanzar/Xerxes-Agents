# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Structured JSONL logging for the daemon.

Writes one JSON object per line to daily log files under
``~/.calute/daemon/logs/``. Also mirrors log entries to stderr
when the daemon runs in foreground mode.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class DaemonLogger:
    """Append-only JSONL logger writing to ~/.calute/daemon/logs/."""

    def __init__(self, log_dir: str) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._current_date = ""

    def _ensure_file(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._file:
                self._file.close()
            self._current_date = today
            path = self._dir / f"daemon-{today}.jsonl"
            self._file = open(path, "a", encoding="utf-8")

    def log(self, level: str, event: str, **kwargs: Any) -> None:
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
        # Also print to stderr for foreground mode.
        print(f"[{level}] {event}", file=sys.stderr)

    def info(self, event: str, **kwargs: Any) -> None:
        self.log("info", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self.log("error", event, **kwargs)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
