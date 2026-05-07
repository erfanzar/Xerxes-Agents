from __future__ import annotations

import json
import time

from xerxes.agents.subagent_manager import SubAgentManager, SubAgentTask
from xerxes.tools import claude_tools
from xerxes.tools.claude_tools import SpawnAgents


class _BlockingFuture:
    def result(self, timeout=None):
        raise AssertionError("wait() should not block on cleanup after terminal status")


def test_subagent_wait_returns_for_terminal_status_without_future_join():
    mgr = SubAgentManager()
    task = SubAgentTask(id="done", status="completed", result="ok")
    task._done_event.set()
    task._future = _BlockingFuture()
    mgr.tasks[task.id] = task

    assert mgr.wait(task.id, timeout=0) is task

    mgr.shutdown()


def test_spawn_agents_bounded_wait_returns_pending_snapshots(monkeypatch):
    def runner(prompt, config, system_prompt, depth, cancel_check):
        time.sleep(0.2)
        return f"done: {prompt}"

    mgr = SubAgentManager(max_concurrent=1)
    mgr.set_runner(runner)
    monkeypatch.setattr(claude_tools, "_agent_manager", mgr)

    started = time.monotonic()
    raw = SpawnAgents.static_call(
        [{"prompt": "one", "name": "a"}, {"prompt": "two", "name": "b"}],
        wait=True,
        timeout=0.01,
    )
    elapsed = time.monotonic() - started

    rows = json.loads(raw)
    assert elapsed < 0.15
    assert {row["name"] for row in rows} == {"a", "b"}
    assert any(row["status"] in {"pending", "running"} for row in rows)
    assert all("id" in row for row in rows)

    mgr.shutdown()


def test_cancel_all_wakes_waiting_spawn_agents(monkeypatch):
    def runner(prompt, config, system_prompt, depth, cancel_check):
        while not cancel_check():
            time.sleep(0.01)
        return "cancelled"

    mgr = SubAgentManager(max_concurrent=1)
    mgr.set_runner(runner)
    monkeypatch.setattr(claude_tools, "_agent_manager", mgr)

    task = mgr.spawn(prompt="wait", config={}, system_prompt="sys")
    time.sleep(0.02)

    assert mgr.cancel_all() == 1

    started = time.monotonic()
    assert mgr.wait(task.id, timeout=5) is task
    assert time.monotonic() - started < 0.1
    assert task.status == "cancelled"

    mgr.shutdown()
