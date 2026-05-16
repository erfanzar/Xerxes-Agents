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
"""Tests for xerxes.runtime.process_registry."""

from __future__ import annotations

import subprocess
import sys

import pytest
from xerxes.runtime.process_registry import ProcessRegistry, get_default_registry


@pytest.fixture
def sleep_proc():
    """Spawn a tiny Python subprocess that exits on demand."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys, time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield proc
    try:
        proc.kill()
    except ProcessLookupError:
        pass


class TestProcessRegistry:
    def test_register_returns_id(self, sleep_proc) -> None:
        r = ProcessRegistry()
        pid = r.register(sleep_proc, name="sleeper", command="python -c sleep")
        assert isinstance(pid, str)
        assert len(pid) == 12

    def test_list_returns_records(self, sleep_proc) -> None:
        r = ProcessRegistry()
        r.register(sleep_proc, name="x")
        records = r.list()
        assert len(records) == 1
        assert records[0].name == "x"
        assert records[0].pid == sleep_proc.pid

    def test_poll_returns_none_for_running(self, sleep_proc) -> None:
        r = ProcessRegistry()
        pid = r.register(sleep_proc)
        assert r.poll(pid) is None

    def test_kill_signals_process(self, sleep_proc) -> None:
        r = ProcessRegistry()
        pid = r.register(sleep_proc)
        assert r.kill(pid, force=True) is True
        # Give the OS a moment.
        sleep_proc.wait(timeout=5)
        assert r.poll(pid) is not None

    def test_kill_unknown_returns_false(self) -> None:
        r = ProcessRegistry()
        assert r.kill("nope") is False

    def test_remove(self, sleep_proc) -> None:
        r = ProcessRegistry()
        pid = r.register(sleep_proc)
        assert r.remove(pid) is True
        assert r.remove(pid) is False
        assert r.list() == []

    def test_clear(self, sleep_proc) -> None:
        r = ProcessRegistry()
        r.register(sleep_proc)
        assert r.clear() == 1
        assert r.list() == []

    def test_wait_returns_exit_code(self) -> None:
        r = ProcessRegistry()
        proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(7)"])
        pid = r.register(proc)
        assert r.wait(pid, timeout=5) == 7

    def test_default_registry_is_singleton(self) -> None:
        a = get_default_registry()
        b = get_default_registry()
        assert a is b
