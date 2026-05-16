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
"""Cron job dataclass, JSON job store, and schedule arithmetic.

Schedules are stored as 5-field Vixie-style cron expressions. The
parser handles ``*``, lists, ``a-b`` ranges, and ``*/N`` steps — the
subset Xerxes uses in practice. Day-of-week is Sunday-zero like
standard cron, not ISO. All datetimes are UTC; ``next_fire_at``
returns UTC instants for ``CronScheduler`` to store on each job.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class CronJob:
    """A scheduled or one-shot job.

    Attributes:
        id: stable identifier.
        prompt: text passed to the agent on each fire.
        schedule: 5-field cron expression (``""`` for one-shot jobs).
        deliver: platform to route output to (``"none"`` to archive only).
        recipient: platform-specific recipient (chat id, email, ...).
        paused: when ``True``, the scheduler skips this job.
        oneshot: when ``True`` the job deletes itself after firing.
        last_run_at: ISO timestamp of the last execution, UTC.
        next_run_at: ISO timestamp of the next planned fire, UTC.
        workspace_id: optional workspace scope.
        metadata: free-form annotations.
    """

    id: str
    prompt: str
    schedule: str = ""
    deliver: str = "none"
    recipient: str = ""
    paused: bool = False
    oneshot: bool = False
    last_run_at: str | None = None
    next_run_at: str | None = None
    workspace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict for persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CronJob:
        """Inflate a ``CronJob`` from its persisted dict form."""
        return cls(
            **{
                k: data.get(k, v)
                for k, v in cls.__dataclass_fields__.items()  # type: ignore[attr-defined]
                if not isinstance(v, type)
            }
            | {k: data[k] for k in ("id", "prompt") if k in data}
        )


def _parse_cron_field(spec: str, lo: int, hi: int) -> set[int]:
    """Parse one cron field into the set of permitted integer values.

    Supports ``*``, lists, ``a-b`` ranges, ``a-b/step``, and ``*/N``.
    Used for minute/hour/dom/month/dow fields.
    """
    if spec == "*":
        return set(range(lo, hi + 1))
    out: set[int] = set()
    for part in spec.split(","):
        if part.startswith("*/"):
            step = int(part[2:])
            out.update(range(lo, hi + 1, step))
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if "/" in b:
                b, step_s = b.split("/", 1)
                step = int(step_s)
            else:
                step = 1
            out.update(range(int(a), int(b) + 1, step))
            continue
        out.add(int(part))
    return out


def _matches(dt: datetime, fields: tuple[set[int], set[int], set[int], set[int], set[int]]) -> bool:
    """Return ``True`` when ``dt`` matches the five parsed cron fields."""
    minute, hour, dom, month, dow = fields
    if dt.minute not in minute:
        return False
    if dt.hour not in hour:
        return False
    if dt.month not in month:
        return False
    # cron day-of-week: 0 = Sunday; isoweekday: 1=Mon..7=Sun → convert.
    dow_value = dt.isoweekday() % 7  # 0=Sun..6=Sat
    if dow_value not in dow:
        return False
    if dt.day not in dom:
        return False
    return True


def next_fire_at(schedule: str, now: datetime | None = None) -> datetime:
    """Return the first UTC instant after ``now`` matching ``schedule``.

    Searches a one-year window in one-minute steps; this is bounded
    and cheap (well under a second for the worst case). Raises
    ``ValueError`` if ``schedule`` isn't a 5-field expression and
    ``RuntimeError`` if no match is found within a year.
    """
    fields_raw = schedule.split()
    if len(fields_raw) != 5:
        raise ValueError(f"expected 5-field cron expression, got {schedule!r}")
    fields = (
        _parse_cron_field(fields_raw[0], 0, 59),
        _parse_cron_field(fields_raw[1], 0, 23),
        _parse_cron_field(fields_raw[2], 1, 31),
        _parse_cron_field(fields_raw[3], 1, 12),
        _parse_cron_field(fields_raw[4], 0, 6),
    )
    base = (now or datetime.now(UTC)).replace(second=0, microsecond=0)
    candidate = base + timedelta(minutes=1)
    # Step one minute at a time. A year of minutes is ~525k iterations,
    # still well under a second; cheap and correct.
    for _ in range(60 * 24 * 366):
        if _matches(candidate, fields):
            return candidate
        candidate += timedelta(minutes=1)
    raise RuntimeError(f"no fire time found within a year for schedule {schedule!r}")


class JobStore:
    """Thread-safe JSON-backed CRUD store for :class:`CronJob` records.

    Records live in a single JSON file; every read/write is serialized
    through an internal :class:`threading.Lock`. A missing file is
    initialised to an empty array on construction.
    """

    def __init__(self, path: str | Path) -> None:
        """Open or create the store at ``path``."""
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self._path.exists():
            self._path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict[str, Any]]:
        """Read and decode the raw record list."""
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self, records: list[dict[str, Any]]) -> None:
        """Write ``records`` back to disk as pretty JSON."""
        self._path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def list_jobs(self) -> list[CronJob]:
        """Return every job in the store as a :class:`CronJob`."""
        with self._lock:
            return [self._inflate(r) for r in self._load()]

    @staticmethod
    def _inflate(record: dict[str, Any]) -> CronJob:
        """Build a :class:`CronJob` from one raw record dict."""
        return CronJob(
            id=record["id"],
            prompt=record["prompt"],
            schedule=record.get("schedule", ""),
            deliver=record.get("deliver", "none"),
            recipient=record.get("recipient", ""),
            paused=record.get("paused", False),
            oneshot=record.get("oneshot", False),
            last_run_at=record.get("last_run_at"),
            next_run_at=record.get("next_run_at"),
            workspace_id=record.get("workspace_id"),
            metadata=record.get("metadata", {}),
        )

    def add(self, job: CronJob) -> CronJob:
        """Upsert ``job`` (existing record with the same id is replaced)."""
        with self._lock:
            records = self._load()
            records = [r for r in records if r.get("id") != job.id]
            records.append(job.to_dict())
            self._save(records)
        return job

    def get(self, job_id: str) -> CronJob | None:
        """Return the job with ``job_id`` or ``None`` if absent."""
        for j in self.list_jobs():
            if j.id == job_id:
                return j
        return None

    def update(self, job_id: str, **changes: Any) -> CronJob | None:
        """Apply ``changes`` to ``job_id`` and return the updated record."""
        with self._lock:
            records = self._load()
            for r in records:
                if r.get("id") == job_id:
                    r.update(changes)
                    self._save(records)
                    return self._inflate(r)
            return None

    def remove(self, job_id: str) -> bool:
        """Delete ``job_id``; return ``True`` if a record was removed."""
        with self._lock:
            records = self._load()
            new = [r for r in records if r.get("id") != job_id]
            if len(new) == len(records):
                return False
            self._save(new)
            return True

    def new_id(self) -> str:
        """Mint a short uuid suitable as a :class:`CronJob` id."""
        return uuid.uuid4().hex[:12]


__all__ = ["CronJob", "JobStore", "next_fire_at"]
