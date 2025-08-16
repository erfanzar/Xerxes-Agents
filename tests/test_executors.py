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

import asyncio

import pytest

from calute import Agent
from calute.executors import AgentOrchestrator, FunctionExecutor, FunctionRegistry
from calute.types import (
    AgentSwitchTrigger,
    ExecutionStatus,
    FunctionCallStrategy,
    RequestFunctionCall,
)


class TestFunctionRegistry:
    """Test suite for FunctionRegistry."""

    def test_register_function(self):
        """Test registering a function."""
        registry = FunctionRegistry()

        def test_func():
            return "test"

        registry.register(test_func, "agent1", {"description": "Test function"})

        assert "test_func" in registry._functions
        assert registry._function_agents["test_func"] == "agent1"
        assert registry._function_metadata["test_func"]["description"] == "Test function"

    def test_get_function(self):
        """Test getting a function and its agent."""
        registry = FunctionRegistry()

        def test_func():
            return "test"

        registry.register(test_func, "agent1")

        func, agent_id = registry.get_function("test_func")
        assert func == test_func
        assert agent_id == "agent1"

        # Test non-existent function
        func, agent_id = registry.get_function("non_existent")
        assert func is None
        assert agent_id is None

    def test_get_functions_by_agent(self):
        """Test getting all functions for an agent."""
        registry = FunctionRegistry()

        def func1():
            return "1"

        def func2():
            return "2"

        def func3():
            return "3"

        registry.register(func1, "agent1")
        registry.register(func2, "agent1")
        registry.register(func3, "agent2")

        agent1_funcs = registry.get_functions_by_agent("agent1")
        assert len(agent1_funcs) == 2
        assert func1 in agent1_funcs
        assert func2 in agent1_funcs

        agent2_funcs = registry.get_functions_by_agent("agent2")
        assert len(agent2_funcs) == 1
        assert func3 in agent2_funcs


class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator."""

    def test_register_agent(self, sample_agent):
        """Test registering an agent."""
        orchestrator = AgentOrchestrator()
        orchestrator.register_agent(sample_agent)

        assert sample_agent.id in orchestrator.agents
        assert orchestrator.current_agent_id == sample_agent.id
        assert orchestrator.agents[sample_agent.id] == sample_agent

    def test_register_multiple_agents(self):
        """Test registering multiple agents."""
        orchestrator = AgentOrchestrator()

        agent1 = Agent(id="agent1", model="gpt-4")
        agent2 = Agent(id="agent2", model="gpt-4")

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        assert len(orchestrator.agents) == 2
        assert "agent1" in orchestrator.agents
        assert "agent2" in orchestrator.agents
        assert orchestrator.current_agent_id == "agent1"  # First registered

    def test_switch_agent(self):
        """Test switching between agents."""
        orchestrator = AgentOrchestrator()

        agent1 = Agent(id="agent1", model="gpt-4")
        agent2 = Agent(id="agent2", model="gpt-4")

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        orchestrator.switch_agent("agent2", "Test switch")
        assert orchestrator.current_agent_id == "agent2"
        assert len(orchestrator.execution_history) == 1
        assert orchestrator.execution_history[0]["action"] == "agent_switch"

    def test_switch_to_non_existent_agent(self):
        """Test switching to a non-existent agent raises error."""
        orchestrator = AgentOrchestrator()
        agent = Agent(id="agent1", model="gpt-4")
        orchestrator.register_agent(agent)

        with pytest.raises(ValueError, match="Agent non_existent not found"):
            orchestrator.switch_agent("non_existent")

    def test_get_current_agent(self):
        """Test getting the current agent."""
        orchestrator = AgentOrchestrator()
        agent = Agent(id="test", model="gpt-4")
        orchestrator.register_agent(agent)

        current = orchestrator.get_current_agent()
        assert current == agent

    def test_register_switch_trigger(self):
        """Test registering custom switch triggers."""
        orchestrator = AgentOrchestrator()

        def custom_trigger(context, agents, current_agent_id):
            return "agent2" if context.get("switch") else None

        orchestrator.register_switch_trigger(AgentSwitchTrigger.CUSTOM, custom_trigger)

        assert AgentSwitchTrigger.CUSTOM in orchestrator.switch_triggers
        assert orchestrator.switch_triggers[AgentSwitchTrigger.CUSTOM] == custom_trigger

    def test_should_switch_agent(self):
        """Test agent switching logic."""
        orchestrator = AgentOrchestrator()

        def trigger(context, agents, current_agent_id):
            return "agent2" if context.get("needs_switch") else None

        orchestrator.register_switch_trigger(AgentSwitchTrigger.CUSTOM, trigger)
        orchestrator.current_agent_id = "agent1"

        # Test no switch
        target = orchestrator.should_switch_agent({})
        assert target is None

        # Test switch
        target = orchestrator.should_switch_agent({"needs_switch": True})
        assert target == "agent2"


class TestFunctionExecutor:
    """Test suite for FunctionExecutor."""

    @pytest.mark.asyncio
    async def test_execute_single_function(self):
        """Test executing a single function call."""
        orchestrator = AgentOrchestrator()

        def test_func(x: int) -> int:
            return x * 2

        agent = Agent(id="agent1", model="gpt-4", functions=[test_func])
        orchestrator.register_agent(agent)

        executor = FunctionExecutor(orchestrator)

        call = RequestFunctionCall(
            name="test_func", arguments={"x": 5}, call_id="call_1", status=ExecutionStatus.PENDING
        )

        results = await executor.execute_function_calls(
            [call], context_variables={}, strategy=FunctionCallStrategy.SEQUENTIAL
        )

        assert len(results) == 1
        assert results[0].status == ExecutionStatus.SUCCESS
        assert results[0].result == 10

    @pytest.mark.asyncio
    async def test_execute_parallel_functions(self):
        """Test executing functions in parallel."""
        orchestrator = AgentOrchestrator()

        async def async_func1(x: int) -> int:
            await asyncio.sleep(0.1)
            return x + 1

        async def async_func2(x: int) -> int:
            await asyncio.sleep(0.1)
            return x * 2

        agent = Agent(id="agent1", model="gpt-4", functions=[async_func1, async_func2])
        orchestrator.register_agent(agent)

        executor = FunctionExecutor(orchestrator)

        calls = [
            RequestFunctionCall(
                name="async_func1",
                arguments={"x": 5},
                call_id="call_1",
                status=ExecutionStatus.PENDING,
            ),
            RequestFunctionCall(
                name="async_func2",
                arguments={"x": 5},
                call_id="call_2",
                status=ExecutionStatus.PENDING,
            ),
        ]

        results = await executor.execute_function_calls(
            calls, context_variables={}, strategy=FunctionCallStrategy.PARALLEL
        )

        assert len(results) == 2
        assert all(r.status == ExecutionStatus.SUCCESS for r in results)
        assert results[0].result == 6
        assert results[1].result == 10

    @pytest.mark.asyncio
    async def test_execute_with_error_handling(self):
        """Test function execution with error handling."""
        orchestrator = AgentOrchestrator()

        def failing_func():
            raise ValueError("Test error")

        agent = Agent(id="agent1", model="gpt-4", functions=[failing_func])
        orchestrator.register_agent(agent)

        executor = FunctionExecutor(orchestrator)

        call = RequestFunctionCall(
            name="failing_func", arguments={}, call_id="call_1", status=ExecutionStatus.PENDING
        )

        results = await executor.execute_function_calls(
            [call], context_variables={}, strategy=FunctionCallStrategy.SEQUENTIAL
        )

        assert len(results) == 1
        assert results[0].status == ExecutionStatus.FAILED
        assert "Test error" in str(results[0].error)

    @pytest.mark.asyncio
    async def test_execute_with_dependencies(self):
        """Test executing functions with dependencies."""
        orchestrator = AgentOrchestrator()

        def func_a(x: int) -> int:
            return x + 1

        def func_b(a_result: int) -> int:
            return a_result * 2

        agent = Agent(id="agent1", model="gpt-4", functions=[func_a, func_b])
        orchestrator.register_agent(agent)

        executor = FunctionExecutor(orchestrator)

        calls = [
            RequestFunctionCall(
                name="func_a",
                arguments={"x": 5},
                call_id="call_a",
                status=ExecutionStatus.PENDING,
            ),
            RequestFunctionCall(
                name="func_b",
                arguments={"a_result": "{call_a.result}"},
                call_id="call_b",
                status=ExecutionStatus.PENDING,
                dependencies=["call_a"],
            ),
        ]

        results = await executor.execute_function_calls(
            calls, context_variables={}, strategy=FunctionCallStrategy.SEQUENTIAL
        )

        assert len(results) == 2
        assert results[0].result == 6  # 5 + 1
        assert results[1].result == 12  # 6 * 2

    @pytest.mark.asyncio
    async def test_execute_non_existent_function(self):
        """Test executing a non-existent function."""
        orchestrator = AgentOrchestrator()
        agent = Agent(id="agent1", model="gpt-4")
        orchestrator.register_agent(agent)

        executor = FunctionExecutor(orchestrator)

        call = RequestFunctionCall(
            name="non_existent", arguments={}, call_id="call_1", status=ExecutionStatus.PENDING
        )

        results = await executor.execute_function_calls(
            [call], context_variables={}, strategy=FunctionCallStrategy.SEQUENTIAL
        )

        assert len(results) == 1
        assert results[0].status == ExecutionStatus.FAILED
        assert "not found" in str(results[0].error).lower()
