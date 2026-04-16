"""Tests for xerxes_agent.executors — FunctionRegistry and AgentOrchestrator."""

from unittest.mock import MagicMock

from xerxes_agent.executors import AgentOrchestrator, FunctionRegistry


def sample_func(x: int) -> int:
    """A sample function."""
    return x * 2


def another_func(text: str) -> str:
    """Another function."""
    return text.upper()


def make_mock_agent(agent_id="agent1", functions=None):
    agent = MagicMock()
    agent.id = agent_id
    agent.functions = functions or []
    return agent


class TestFunctionRegistry:
    def test_register_and_get(self):
        reg = FunctionRegistry()
        reg.register(sample_func, "agent1")
        func, agent_id = reg.get_function("sample_func")
        assert func is sample_func
        assert agent_id == "agent1"

    def test_get_missing(self):
        reg = FunctionRegistry()
        func, agent_id = reg.get_function("nonexistent")
        assert func is None
        assert agent_id is None

    def test_register_with_metadata(self):
        reg = FunctionRegistry()
        reg.register(sample_func, "agent1", metadata={"type": "math"})
        assert reg._function_metadata["sample_func"]["type"] == "math"

    def test_get_functions_by_agent(self):
        reg = FunctionRegistry()
        reg.register(sample_func, "agent1")
        reg.register(another_func, "agent1")
        funcs = reg.get_functions_by_agent("agent1")
        assert len(funcs) == 2

    def test_get_functions_by_agent_empty(self):
        reg = FunctionRegistry()
        reg.register(sample_func, "agent1")
        funcs = reg.get_functions_by_agent("agent2")
        assert len(funcs) == 0


class TestAgentOrchestrator:
    def test_register_agent(self):
        orch = AgentOrchestrator()
        agent = make_mock_agent("agent1", [sample_func])
        orch.register_agent(agent)
        assert "agent1" in orch.agents
        assert orch.current_agent_id == "agent1"

    def test_register_multiple_agents(self):
        orch = AgentOrchestrator()
        a1 = make_mock_agent("a1", [sample_func])
        a2 = make_mock_agent("a2", [another_func])
        orch.register_agent(a1)
        orch.register_agent(a2)
        assert len(orch.agents) == 2
        assert orch.current_agent_id == "a1"

    def test_register_agent_auto_id(self):
        orch = AgentOrchestrator()
        agent = make_mock_agent(None, [])
        orch.register_agent(agent)
        assert agent.id is not None

    def test_should_switch_no_triggers(self):
        orch = AgentOrchestrator()
        result = orch.should_switch_agent({})
        assert result is None

    def test_should_switch_with_trigger(self):
        orch = AgentOrchestrator()
        a1 = make_mock_agent("a1", [])
        a2 = make_mock_agent("a2", [])
        orch.register_agent(a1)
        orch.register_agent(a2)

        from xerxes_agent.types.function_execution_types import AgentSwitchTrigger

        def handler(ctx, agents, current):
            return "a2"

        orch.register_switch_trigger(AgentSwitchTrigger.CUSTOM, handler)
        target = orch.should_switch_agent({})
        assert target == "a2"

    def test_switch_agent(self):
        orch = AgentOrchestrator()
        a1 = make_mock_agent("a1", [])
        a2 = make_mock_agent("a2", [])
        orch.register_agent(a1)
        orch.register_agent(a2)
        orch.switch_agent("a2", reason="test switch")
        assert orch.current_agent_id == "a2"
        assert len(orch.execution_history) == 1
