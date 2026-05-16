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
"""Tests for the cron scheduler package."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from xerxes.cron import CronJob, CronScheduler, DeliveryTarget, JobStore, next_fire_at, route_output

# ---------------------------- next_fire_at ---------------------------------


class TestNextFireAt:
    def test_invalid_schedule(self):
        with pytest.raises(ValueError):
            next_fire_at("not enough fields")

    def test_every_minute(self):
        base = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
        out = next_fire_at("* * * * *", base)
        assert out == datetime(2026, 5, 15, 12, 1, tzinfo=UTC)

    def test_specific_hour_and_minute(self):
        base = datetime(2026, 5, 15, 8, 30, tzinfo=UTC)
        # Daily 09:00 UTC
        out = next_fire_at("0 9 * * *", base)
        assert out == datetime(2026, 5, 15, 9, 0, tzinfo=UTC)

    def test_step_minutes(self):
        base = datetime(2026, 5, 15, 12, 4, tzinfo=UTC)
        out = next_fire_at("*/15 * * * *", base)
        # Next 15-minute mark after 12:04 is 12:15.
        assert out == datetime(2026, 5, 15, 12, 15, tzinfo=UTC)

    def test_day_of_week_sunday(self):
        # Sunday is dow=0 in cron. Pick a known Saturday and find next Sunday.
        base = datetime(2026, 5, 16, 23, 59, tzinfo=UTC)  # Sat
        out = next_fire_at("0 9 * * 0", base)
        assert out.isoweekday() == 7  # Sunday
        assert out.hour == 9


# ---------------------------- JobStore -------------------------------------


class TestJobStore:
    def test_initialized_empty(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        assert store.list_jobs() == []

    def test_add_and_get(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        job = CronJob(id=store.new_id(), prompt="say hi", schedule="* * * * *")
        store.add(job)
        loaded = store.get(job.id)
        assert loaded is not None
        assert loaded.prompt == "say hi"

    def test_update(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        job = CronJob(id="abc", prompt="run", schedule="* * * * *")
        store.add(job)
        out = store.update("abc", paused=True)
        assert out is not None
        assert out.paused is True
        assert store.get("abc").paused is True

    def test_remove(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        store.add(CronJob(id="x", prompt="p"))
        assert store.remove("x") is True
        assert store.list_jobs() == []
        assert store.remove("x") is False

    def test_add_replaces_existing(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        store.add(CronJob(id="abc", prompt="first"))
        store.add(CronJob(id="abc", prompt="second"))
        assert len(store.list_jobs()) == 1
        assert store.get("abc").prompt == "second"


# ---------------------------- delivery -------------------------------------


class TestDelivery:
    def test_archive_only_when_platform_none(self, tmp_path):
        sent = []
        route_output(
            DeliveryTarget(platform="none"),
            "content",
            archive_dir=tmp_path,
            job_id="job1",
            sender=lambda *a: sent.append(a),
        )
        assert sent == []
        # Archive file exists.
        files = list((tmp_path / "job1").iterdir())
        assert len(files) == 1
        assert files[0].read_text() == "content"

    def test_routes_to_platform(self, tmp_path):
        sent = []
        route_output(
            DeliveryTarget(platform="telegram", recipient="123"),
            "Daily digest",
            archive_dir=tmp_path,
            job_id="job1",
            sender=lambda p, r, c: sent.append((p, r, c)),
        )
        assert sent == [("telegram", "123", "Daily digest")]


# ---------------------------- scheduler ------------------------------------


class TestCronScheduler:
    def test_tick_fires_due_job(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        job = CronJob(id="j1", prompt="hi", schedule="* * * * *")
        store.add(job)
        runs = []
        sched = CronScheduler(store, run_job=lambda j: runs.append(j.id) or "output")
        # First tick: just registers a next_run_at (no fire).
        now = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
        sched.tick(now=now)
        assert runs == []
        # Advance past next_run_at and fire.
        sched.tick(now=now + _minutes(2))
        assert runs == ["j1"]

    def test_paused_jobs_skipped(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        store.add(
            CronJob(
                id="j1",
                prompt="hi",
                schedule="* * * * *",
                paused=True,
                next_run_at=datetime(2026, 5, 15, 12, 0, tzinfo=UTC).isoformat(),
            )
        )
        runs = []
        sched = CronScheduler(store, run_job=lambda j: runs.append(j.id) or "")
        sched.tick(now=datetime(2026, 5, 15, 13, 0, tzinfo=UTC))
        assert runs == []

    def test_oneshot_removes_itself(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        store.add(CronJob(id="o1", prompt="once", oneshot=True))
        sched = CronScheduler(store, run_job=lambda j: "out")
        now = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
        # First tick: schedules ASAP.
        sched.tick(now=now)
        # Next tick: fires + removes.
        sched.tick(now=now + _minutes(1))
        assert store.list_jobs() == []

    def test_on_complete_callback(self, tmp_path):
        store = JobStore(tmp_path / "jobs.json")
        store.add(CronJob(id="j1", prompt="p", schedule="* * * * *"))
        completions = []
        sched = CronScheduler(
            store,
            run_job=lambda j: "result text",
            on_complete=lambda j, out: completions.append((j.id, out)),
        )
        now = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
        sched.tick(now=now)
        sched.tick(now=now + _minutes(2))
        assert completions == [("j1", "result text")]


def _minutes(n: int):
    from datetime import timedelta

    return timedelta(minutes=n)
