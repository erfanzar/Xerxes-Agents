from __future__ import annotations

import json
import threading
import time

from xerxes.agents.definitions import get_agent_definition
from xerxes.agents.subagent_manager import SubAgentManager, SubAgentTask, _filter_subagent_tools
from xerxes.tools import claude_tools
from xerxes.tools.claude_tools import AgentTool, SpawnAgents


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


def test_agent_tool_bounded_wait_returns_running_snapshot(monkeypatch):
    def runner(prompt, config, system_prompt, depth, cancel_check):
        time.sleep(0.2)
        return f"done: {prompt}"

    mgr = SubAgentManager(max_concurrent=1)
    mgr.set_runner(runner)
    monkeypatch.setattr(claude_tools, "_agent_manager", mgr)

    started = time.monotonic()
    raw = AgentTool.static_call("slow task", name="slow", wait=True, timeout=0.01)
    elapsed = time.monotonic() - started

    row = json.loads(raw)
    assert elapsed < 0.15
    assert row["name"] == "slow"
    assert row["status"] in {"pending", "running"}
    assert "still running" in row["note"]

    mgr.shutdown()


def test_spawn_agents_grows_pool_to_batch_size(monkeypatch):
    agent_count = 12
    ready = threading.Barrier(agent_count + 1)
    release = threading.Event()
    started: list[str] = []

    def runner(prompt, config, system_prompt, depth, cancel_check):
        started.append(prompt)
        ready.wait(timeout=2)
        release.wait(timeout=2)
        return f"done: {prompt}"

    mgr = SubAgentManager(max_concurrent=5)
    mgr.set_runner(runner)
    monkeypatch.setattr(claude_tools, "_agent_manager", mgr)

    specs = [{"prompt": f"agent-{idx}", "name": f"a{idx}"} for idx in range(agent_count)]
    raw = SpawnAgents.static_call(specs, wait=False)

    ready.wait(timeout=2)
    release.set()
    rows = json.loads(raw)

    assert mgr.max_concurrent == agent_count
    assert len(rows) == agent_count
    assert len(started) == agent_count

    mgr.shutdown()


def test_subagent_manager_can_grow_capacity_while_tasks_are_active():
    release = threading.Event()
    started: list[str] = []

    def runner(prompt, config, system_prompt, depth, cancel_check):
        started.append(prompt)
        release.wait(timeout=2)
        return f"done: {prompt}"

    mgr = SubAgentManager(max_concurrent=1)
    mgr.set_runner(runner)
    first = mgr.spawn(prompt="first", config={}, system_prompt="sys")
    time.sleep(0.05)

    assert first.status == "running"
    assert mgr.ensure_capacity(3) is True

    more = [mgr.spawn(prompt=f"next-{idx}", config={}, system_prompt="sys") for idx in range(2)]
    deadline = time.monotonic() + 2
    while len(started) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    release.set()

    assert mgr.max_concurrent == 3
    assert set(started) == {"first", "next-0", "next-1"}
    assert mgr.wait(first.id, timeout=2) is first
    for task in more:
        assert mgr.wait(task.id, timeout=2) is task

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


def test_subagent_tool_filter_blocks_recursive_delegation_tools():
    schemas = [
        {"name": "ReadFile"},
        {"name": "SpawnAgents"},
        {"name": "AgentTool"},
        {"name": "SkillTool"},
    ]
    calls: list[str] = []

    def executor(tool_name, tool_input):
        calls.append(tool_name)
        return "ok"

    filtered, filtered_executor = _filter_subagent_tools(
        tool_schemas=schemas,
        tool_executor=executor,
        config={},
        is_subagent=True,
    )

    assert [schema["name"] for schema in filtered or []] == ["ReadFile"]
    assert filtered_executor("ReadFile", {}) == "ok"
    assert filtered_executor("SpawnAgents", {}) == "Error: tool 'SpawnAgents' is not allowed for this agent."
    assert calls == ["ReadFile"]


def test_subagent_system_prompt_does_not_include_active_skills_by_default(monkeypatch):
    class _Skill:
        def to_prompt_section(self):
            return "DEEPSCAN SpawnAgents instructions"

    monkeypatch.setattr(claude_tools, "get_active_skills", lambda: ["deepscan"], raising=False)
    monkeypatch.setattr(claude_tools, "_skill_registry", {"deepscan": _Skill()}, raising=False)

    assert claude_tools._build_subagent_system_prompt("base") == "base"


def test_agent_definition_spawn_adds_subagent_caller_prompt():
    captured: dict[str, object] = {}

    def runner(prompt, config, system_prompt, depth, cancel_check):
        captured["prompt"] = prompt
        captured["config"] = dict(config)
        captured["system_prompt"] = system_prompt
        captured["depth"] = depth
        return "done"

    agent_def = get_agent_definition("researcher")
    mgr = SubAgentManager(max_concurrent=1)
    mgr.set_runner(runner)
    try:
        task = mgr.spawn(
            prompt="inspect mode handling",
            config={},
            system_prompt="base system",
            agent_def=agent_def,
            name="researcher-check",
        )

        assert mgr.wait(task.id, timeout=2) is task
        assert task.status == "completed"
        assert "You are now running as a subagent" in str(captured["system_prompt"])
        assert "You are a research assistant focused on understanding codebases." in str(captured["system_prompt"])
        assert str(captured["system_prompt"]).rstrip().endswith("base system")
        assert captured["config"].get("_tools_allowed") == agent_def.allowed_tools
    finally:
        mgr.shutdown()
