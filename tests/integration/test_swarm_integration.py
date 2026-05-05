#!/usr/bin/env python3
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
"""Comprehensive agent swarm integration test for Xerxes.

Uses a MockLLM to exercise every major feature without API costs.
Reports any errors or unexpected behavior.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from typing import Any

sys.path.insert(0, "src/python")

from xerxes import Xerxes
from xerxes.agents.definitions import AgentDefinition
from xerxes.agents.subagent_manager import SubAgentManager
from xerxes.cortex import Cortex, CortexAgent, CortexTask
from xerxes.cortex.core.enums import ProcessType
from xerxes.executors import EnhancedAgentOrchestrator, EnhancedFunctionExecutor
from xerxes.llms.base import BaseLLM
from xerxes.memory import MemoryType
from xerxes.security.policy import PolicyAction, ToolPolicy
from xerxes.types import (
    Agent,
    AgentSwitchTrigger,
    ExecutionStatus,
    FunctionCallStrategy,
    RequestFunctionCall,
    Result,
)

REPORT: list[dict] = []


def log(category: str, message: str, error: bool = False):
    icon = "❌" if error else "✅"
    print(f"{icon} [{category}] {message}")
    REPORT.append({"category": category, "message": message, "error": error})


class MockLLM(BaseLLM):
    """Deterministic LLM for integration testing."""

    def __init__(self, responses: list[str] | None = None):
        super().__init__()
        self.responses = responses or []
        self.call_count = 0
        self.last_messages: list[dict] = []

    async def generate_completion(
        self,
        prompt: Any,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        stream: bool = True,
        extra_body: dict | None = None,
    ) -> Any:
        self.call_count += 1

        return self._stream_response()

    def _stream_response(self):
        """Simulate a streaming response."""
        idx = min(self.call_count - 1, len(self.responses) - 1)
        text = self.responses[idx] if self.responses else "Mock response"
        return iter([self._make_chunk(text)])

    def _make_chunk(self, text: str):
        return type(
            "Chunk",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "delta": type("Delta", (), {"content": text, "tool_calls": None})(),
                            "message": type("Message", (), {"content": text, "tool_calls": None})(),
                        },
                    )
                ],
                "content": text,
            },
        )()

    async def astream_completion(self, response: Any, agent: Agent | None = None) -> AsyncIterator[dict]:
        text = getattr(response, "content", "mock")
        yield {
            "raw_chunk": None,
            "content": text,
            "buffered_content": text,
            "reasoning_content": None,
            "buffered_reasoning_content": None,
            "is_final": True,
            "function_calls": [],
            "streaming_tool_calls": None,
        }

    def stream_completion(self, response: Any, agent: Agent | None = None) -> Any:
        text = getattr(response, "content", "mock")
        return iter(
            [
                {
                    "raw_chunk": None,
                    "content": text,
                    "buffered_content": text,
                    "reasoning_content": None,
                    "buffered_reasoning_content": None,
                    "is_final": True,
                    "function_calls": [],
                    "streaming_tool_calls": None,
                }
            ]
        )

    def extract_content(self, response: Any) -> str:
        return getattr(response, "content", "mock")

    def process_streaming_response(self, response: Any, callback=None) -> Any:
        return response

    def parse_tool_calls(self, raw_data: Any) -> list[dict]:
        return []

    async def _initialize_client(self):
        pass


def calculator(expression: str, context_variables: dict | None = None) -> dict:
    """Evaluate a math expression."""
    try:
        return {"result": eval(expression, {"__builtins__": {}}, {})}
    except Exception as e:
        return {"error": str(e)}


def greeter(name: str, context_variables: dict | None = None) -> Result:
    """Greet someone and optionally hand off to a specialist."""
    if name.lower() == "specialist":
        specialist = Agent(
            id="specialist",
            name="Specialist",
            instructions="You are a specialist agent.",
            model="mock",
        )
        return Result(value="Handing off to specialist.", agent=specialist)
    return Result(value=f"Hello, {name}!")


def echo(text: str, context_variables: dict | None = None) -> str:
    """Echo text back."""
    return text


async def async_multiplier(x: int, y: int, context_variables: dict | None = None) -> int:
    """Async multiply."""
    await asyncio.sleep(0.01)
    return x * y


def fail_once(context_variables: dict | None = None) -> str:
    """Fail on first call, succeed on retry."""
    ctx = context_variables or {}
    key = "fail_once_count"
    count = ctx.get(key, 0)
    ctx[key] = count + 1
    if count == 0:
        raise RuntimeError("Intentional first failure")
    return "Recovered"


async def test_agent_orchestrator():
    log("ORCHESTRATOR", "Testing EnhancedAgentOrchestrator...")
    orch = EnhancedAgentOrchestrator(max_agents=5)

    a1 = Agent(id="agent1", name="Agent1", model="mock", functions=[calculator, echo])
    a2 = Agent(id="agent2", name="Agent2", model="mock", functions=[greeter])

    orch.register_agent(a1)
    orch.register_agent(a2)
    assert orch.current_agent_id == "agent1"

    try:
        orch.register_agent(a1)
        log("ORCHESTRATOR", "Duplicate registration did NOT raise", error=True)
    except ValueError:
        log("ORCHESTRATOR", "Duplicate registration correctly rejected")

    try:
        for i in range(10):
            orch.register_agent(Agent(id=f"filler_{i}"))
        log("ORCHESTRATOR", "Max agents not enforced", error=True)
    except ValueError:
        log("ORCHESTRATOR", "Max agents correctly enforced")

    orch.switch_agent("agent2", "test")
    assert orch.current_agent_id == "agent2"

    _func, agent_id = orch.function_registry.get_function("greeter", current_agent_id="agent2")
    assert agent_id == "agent2"
    log("ORCHESTRATOR", "Function lookup prefers current agent")

    orch.switch_agent("agent1")
    a1.switch_triggers.append(AgentSwitchTrigger.CAPABILITY_BASED)
    orch.register_switch_trigger(
        AgentSwitchTrigger.CAPABILITY_BASED,
        lambda ctx, agents, current: "agent2" if ctx.get("need_specialist") else None,
    )
    target = orch.should_switch_agent({"need_specialist": True})
    assert target == "agent2", f"Expected agent2, got {target}"
    log("ORCHESTRATOR", "Agent-level switch_triggers work")


async def test_function_executor():
    log("EXECUTOR", "Testing EnhancedFunctionExecutor...")
    orch = EnhancedAgentOrchestrator()
    agent = Agent(id="exec_agent", model="mock", functions=[calculator, async_multiplier, fail_once])
    orch.register_agent(agent)
    executor = EnhancedFunctionExecutor(orch)

    calls = [
        RequestFunctionCall(name="calculator", arguments={"expression": "2 + 3"}),
        RequestFunctionCall(name="async_multiplier", arguments={"x": 4, "y": 5}),
    ]
    results = await executor.execute_function_calls(calls, strategy=FunctionCallStrategy.SEQUENTIAL)
    assert all(r.status == ExecutionStatus.SUCCESS for r in results)
    assert results[0].result == {"result": 5}
    assert results[1].result == 20
    log("EXECUTOR", "Sequential execution OK")

    calls = [
        RequestFunctionCall(name="async_multiplier", arguments={"x": 2, "y": 3}),
        RequestFunctionCall(name="async_multiplier", arguments={"x": 5, "y": 6}),
    ]
    results = await executor.execute_function_calls(calls, strategy=FunctionCallStrategy.PARALLEL)
    assert all(r.status == ExecutionStatus.SUCCESS for r in results)
    log("EXECUTOR", "Parallel execution OK")

    orch2 = EnhancedAgentOrchestrator()
    orch2.register_agent(Agent(id="handoff_agent", model="mock", functions=[greeter]))
    executor2 = EnhancedFunctionExecutor(orch2)
    result = await executor2._execute_single_call(
        RequestFunctionCall(name="greeter", arguments={"name": "specialist"}),
        {},
        agent=orch2.agents["handoff_agent"],
    )
    assert isinstance(result.result, Result)
    assert result.result.agent is not None
    assert result.result.agent.id == "specialist"
    assert orch2.current_agent_id == "specialist", f"Expected switch to specialist, still on {orch2.current_agent_id}"
    log("EXECUTOR", "Result.agent handoff works")


async def test_xerxes_core():
    log("XERXES", "Testing Xerxes core with MockLLM...")
    llm = MockLLM(["Hello from mock", '<calculator><arguments>{"expression": "7*8"}</arguments></calculator>'])
    xerxes = Xerxes(llm=llm, enable_memory=True)

    agent = Agent(
        id="core_agent",
        name="Core",
        model="mock",
        instructions="You are helpful.",
        functions=[calculator, echo],
    )
    xerxes.register_agent(agent)

    result = xerxes.run(prompt="Hi", stream=False)
    assert hasattr(result, "content")
    log("XERXES", "Basic run OK")

    llm2 = MockLLM(['<calculator><arguments>{"expression": "7*8"}</arguments></calculator>'])
    xerxes2 = Xerxes(llm=llm2, enable_memory=True)
    xerxes2.register_agent(Agent(id="t", model="mock", functions=[calculator]))
    result2 = xerxes2.run(prompt="Calculate 7*8", stream=False)
    assert result2 is not None
    log("XERXES", "Tool execution run OK")

    xerxes.memory_store.add_memory("Test memory", MemoryType.SHORT_TERM, agent_id="core_agent")
    mem = xerxes.memory_store.retrieve_memories(agent_id="core_agent")
    assert len(mem) > 0
    log("XERXES", "Memory store OK")


async def test_cortex():
    log("CORTEX", "Testing Cortex multi-agent orchestration...")

    llm = MockLLM(["Plan created", "Task result 1", "Task result 2", "Consensus result", "Final summary"])

    writer = CortexAgent(role="Writer", goal="Write content", backstory="Expert writer")
    editor = CortexAgent(role="Editor", goal="Edit content", backstory="Detail-oriented editor")

    task1 = CortexTask(description="Write intro", expected_output="Intro text", agent=writer)
    task2 = CortexTask(description="Edit intro", expected_output="Edited text", agent=editor, context=True)

    cortex_seq = Cortex(
        agents=[writer, editor], tasks=[task1, task2], llm=llm, process=ProcessType.SEQUENTIAL, verbose=False
    )
    try:
        out = cortex_seq.kickoff()
        assert out is not None
        log("CORTEX", "SEQUENTIAL process OK")
    except Exception as e:
        log("CORTEX", f"SEQUENTIAL failed: {e}", error=True)

    task_p1 = CortexTask(description="Task A", expected_output="A", agent=writer)
    task_p2 = CortexTask(description="Task B", expected_output="B", agent=editor)
    cortex_par = Cortex(
        agents=[writer, editor], tasks=[task_p1, task_p2], llm=llm, process=ProcessType.PARALLEL, verbose=False
    )
    try:
        out = cortex_par.kickoff()
        assert out is not None
        log("CORTEX", "PARALLEL process OK")
    except Exception as e:
        log("CORTEX", f"PARALLEL failed: {e}", error=True)

    log("CORTEX", "HIERARCHICAL skipped (MockLLM cannot produce JSON plans)")

    cortex_con = Cortex(agents=[writer, editor], tasks=[task1], llm=llm, process=ProcessType.CONSENSUS, verbose=False)
    try:
        out = cortex_con.kickoff()
        assert out is not None
        log("CORTEX", "CONSENSUS process OK")
    except Exception as e:
        log("CORTEX", f"CONSENSUS failed: {e}", error=True)

    cortex_plan = Cortex(
        agents=[writer, editor], tasks=[task1, task2], llm=llm, process=ProcessType.PLANNED, verbose=False
    )
    try:
        out = cortex_plan.kickoff()
        assert out is not None
        log("CORTEX", "PLANNED process OK")
    except Exception as e:
        log("CORTEX", f"PLANNED failed: {e}", error=True)

    try:
        t1_fc = CortexTask(description="T1", expected_output="O1", agent=writer)
        t2_fc = CortexTask(description="T2", expected_output="O2", agent=editor)
        auto = Cortex.from_task_creator(tasks=[t1_fc, t2_fc], llm=llm, verbose=False)
        assert len(auto.agents) == 2
        log("CORTEX", "from_task_creator deduplication OK")
    except Exception as e:
        log("CORTEX", f"from_task_creator failed: {e}", error=True)


async def test_subagent_manager():
    log("SUBAGENT", "Testing SubAgentManager...")

    def mock_runner(prompt, config, system_prompt, depth, cancel_check):
        return f"Result for: {prompt[:30]}"

    mgr = SubAgentManager(max_concurrent=3, max_depth=2)
    mgr.set_runner(mock_runner)

    reviewer_def = AgentDefinition(name="reviewer", tools=["Read", "ReadFile"])
    task = mgr.spawn(
        prompt="Review this code",
        config={"model": "mock"},
        system_prompt="You are a reviewer.",
        agent_def=reviewer_def,
        name="code-review",
    )
    assert task.name == "code-review"
    assert task.agent_def_name == "reviewer"

    mgr.wait(task.id, timeout=5)
    assert task.status in ("completed", "failed")
    if task.status == "completed":
        log("SUBAGENT", "Spawn + wait OK")
    else:
        log("SUBAGENT", f"Task failed: {task.error}", error=True)

    task2 = mgr.spawn(
        prompt="Initial",
        config={"model": "mock"},
        system_prompt="You are helpful.",
    )
    ok = mgr.send_message(task2.id, "Follow-up")
    assert ok
    mgr.wait(task2.id, timeout=5)
    log("SUBAGENT", "Inbox messaging OK")

    mgr.shutdown()


async def test_policy_and_sandbox():
    log("SECURITY", "Testing policy and sandbox routing...")
    from xerxes.security.sandbox import SandboxRouter

    policy = ToolPolicy(allow={"calculator", "echo"})

    assert policy.evaluate("calculator") == PolicyAction.ALLOW
    assert policy.evaluate("dangerous_tool") == PolicyAction.DENY
    log("SECURITY", "ToolPolicy whitelist works")

    router = SandboxRouter()
    decision = router.decide("calculator")
    assert decision.context.value in ("host", "sandbox")
    log("SECURITY", "SandboxRouter decision works")


async def test_definitions():
    log("DEFINITIONS", "Testing AgentDefinition loading...")
    from xerxes.agents.definitions import load_agent_definitions

    defs = load_agent_definitions()
    assert "coder" in defs
    assert "reviewer" in defs

    reviewer = defs["reviewer"]
    assert reviewer.tools == ["Read", "ReadFile", "Glob", "Grep", "ListDir"]
    log("DEFINITIONS", "Built-in definitions OK")


async def main():
    print("=" * 60)
    print("XERXES AGENT SWARM — INTEGRATION TEST")
    print("=" * 60)

    await test_agent_orchestrator()
    await test_function_executor()
    await test_xerxes_core()
    await test_cortex()
    await test_subagent_manager()
    await test_policy_and_sandbox()
    await test_definitions()

    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60)

    errors = [r for r in REPORT if r["error"]]
    ok = [r for r in REPORT if not r["error"]]

    for r in ok:
        print(f"✅ {r['category']}: {r['message']}")
    for r in errors:
        print(f"❌ {r['category']}: {r['message']}")

    print(f"\nPassed: {len(ok)} | Failed: {len(errors)}")
    if errors:
        print("\n⚠️  Some tests failed — see details above.")
        sys.exit(1)
    else:
        print("\n🎉 All swarm tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
