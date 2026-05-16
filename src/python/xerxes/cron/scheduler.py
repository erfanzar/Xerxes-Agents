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
"""Daemon-side cron scheduler.

Sleeps for short intervals, looks for jobs whose ``next_run_at`` is
in the past, fires them via the injected ``run_job`` callable, then
updates ``next_run_at``. The scheduler is single-threaded internally
but runs in a background thread so the daemon's main loop is free.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from .jobs import CronJob, JobStore, next_fire_at

logger = logging.getLogger(__name__)

JobRunner = Callable[[CronJob], str]


class CronScheduler:
    """Background scheduler that fires due jobs and writes results.

    The scheduler resolves each job's next fire time using
    ``next_fire_at`` (or fires one-shot jobs immediately). In tests
    you can skip :meth:`start` and drive the scheduler with explicit
    :meth:`tick` calls.
    """

    def __init__(
        self,
        store: JobStore,
        run_job: JobRunner,
        *,
        on_complete: Callable[[CronJob, str], None] | None = None,
        sleep_seconds: float = 30.0,
    ) -> None:
        """Wire the scheduler to its store and execution callback.

        Args:
            store: backing :class:`JobStore`.
            run_job: invoked to execute a job; returns its text output.
            on_complete: optional post-run hook (e.g. delivery routing).
            sleep_seconds: poll interval between ticks.
        """
        self._store = store
        self._run = run_job
        self._on_complete = on_complete
        self._sleep_seconds = sleep_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the background thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="xerxes-cron", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float | None = None) -> None:
        """Signal the loop to exit and join the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        """Tick forever, swallowing per-tick exceptions to the log."""
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("CronScheduler tick failed")
            self._stop.wait(self._sleep_seconds)

    # ---------------------------- public surface

    def tick(self, *, now: datetime | None = None) -> list[str]:
        """Fire every due job; return ids of jobs that ran this tick.

        Newly-seen jobs are scheduled but not fired on the same tick,
        which avoids racing through their first run too eagerly.
        """
        current = (now or datetime.now(UTC)).replace(microsecond=0)
        ran: list[str] = []
        for job in self._store.list_jobs():
            if job.paused:
                continue
            if not self._is_due(job, current):
                continue
            output = self._run(job)
            ran.append(job.id)
            if self._on_complete is not None:
                try:
                    self._on_complete(job, output)
                except Exception:
                    logger.exception("on_complete failed for job %s", job.id)
            self._schedule_next(job, current)
        return ran

    def _is_due(self, job: CronJob, now: datetime) -> bool:
        """Return ``True`` when ``job.next_run_at`` is at or before ``now``."""
        if job.next_run_at is None:
            # First time we've seen it; schedule and let the next tick fire it.
            self._schedule_next(job, now, just_seen=True)
            return False
        try:
            next_dt = datetime.fromisoformat(job.next_run_at)
        except ValueError:
            return False
        if next_dt.tzinfo is None:
            next_dt = next_dt.replace(tzinfo=UTC)
        return next_dt <= now

    def _schedule_next(self, job: CronJob, now: datetime, *, just_seen: bool = False) -> None:
        """Compute and persist ``job``'s next fire time after ``now``."""
        if job.oneshot:
            if not just_seen:
                self._store.remove(job.id)
            else:
                # Fire ASAP — set next_run_at to now so the next tick picks it.
                self._store.update(job.id, next_run_at=now.isoformat())
            return
        if not job.schedule:
            return
        try:
            next_dt = next_fire_at(job.schedule, now)
        except (ValueError, RuntimeError):
            logger.warning("cron job %s has invalid schedule %r", job.id, job.schedule)
            return
        self._store.update(job.id, next_run_at=next_dt.isoformat(), last_run_at=now.isoformat())


__all__ = ["CronScheduler", "JobRunner"]
