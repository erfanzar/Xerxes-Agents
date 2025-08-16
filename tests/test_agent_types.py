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

import pytest

from calute import Agent, AgentCapability
from calute.types import AgentSwitchTrigger, FunctionCallStrategy
from calute.types.agent_types import AgentBaseFn, _wrap_static_call


class TestAgentTypes:
    """Test suite for agent types."""

    def test_agent_creation(self):
        """Test basic agent creation."""

        def test_func(x: int) -> int:
            """Test function."""
            return x * 2

        agent = Agent(
            id="test_agent",
            name="Test Agent",
            model="gpt-4",
            instructions="Test instructions",
            functions=[test_func],
            capabilities=[AgentCapability.FUNCTION_CALLING],
        )

        assert agent.id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.model == "gpt-4"
        assert len(agent.functions) == 1
        assert agent.functions[0].__name__ == "test_func"

    def test_agent_with_callable_instructions(self):
        """Test agent with callable instructions."""

        def get_instructions() -> str:
            return "Dynamic instructions"

        agent = Agent(model="gpt-4", instructions=get_instructions)

        # Call the instructions function
        instructions = agent.instructions() if callable(agent.instructions) else agent.instructions
        assert instructions == "Dynamic instructions"

    def test_agent_with_rules(self):
        """Test agent with rules."""

        def get_rules() -> list[str]:
            return ["Rule 1", "Rule 2"]

        agent = Agent(model="gpt-4", rules=get_rules)

        # Call the rules function
        rules = agent.rules() if callable(agent.rules) else agent.rules
        assert rules == ["Rule 1", "Rule 2"]

    def test_agent_function_duplicate_names(self):
        """Test that duplicate function names raise an error."""

        def func1() -> str:
            return "func1"

        def func2() -> str:
            return "func2"

        func2.__name__ = "func1"  # Create duplicate name

        with pytest.raises(ValueError, match="Duplicate function name"):
            Agent(model="gpt-4", functions=[func1, func2])

    def test_agent_base_fn_static_call(self):
        """Test AgentBaseFn with static call."""

        class TestAgentFn(AgentBaseFn):
            @staticmethod
            def static_call(x: int) -> int:
                """Static call test."""
                return x * 3

        wrapped = _wrap_static_call(TestAgentFn)
        assert wrapped.__name__ == "TestAgentFn"
        assert wrapped(5) == 15

    def test_agent_function_call_strategies(self):
        """Test different function call strategies."""
        agent = Agent(
            model="gpt-4",
            function_call_strategy=FunctionCallStrategy.PARALLEL,
            parallel_tool_calls=True,
            function_timeout=60.0,
            max_function_retries=5,
        )

        assert agent.function_call_strategy == FunctionCallStrategy.PARALLEL
        assert agent.parallel_tool_calls is True
        assert agent.function_timeout == 60.0
        assert agent.max_function_retries == 5

    def test_agent_switch_triggers(self):
        """Test agent switch triggers."""
        agent = Agent(
            model="gpt-4",
            switch_triggers=[AgentSwitchTrigger.CAPABILITY_REQUIRED, AgentSwitchTrigger.ERROR_RECOVERY],
            fallback_agent_id="fallback_agent",
        )

        assert len(agent.switch_triggers) == 2
        assert AgentSwitchTrigger.CAPABILITY_REQUIRED in agent.switch_triggers
        assert agent.fallback_agent_id == "fallback_agent"

    def test_agent_model_parameters(self):
        """Test agent model parameters."""
        agent = Agent(
            model="gpt-4",
            top_p=0.9,
            max_tokens=1024,
            temperature=0.5,
            top_k=10,
            min_p=0.1,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            repetition_penalty=1.2,
            stop=["STOP", "END"],
        )

        assert agent.top_p == 0.9
        assert agent.max_tokens == 1024
        assert agent.temperature == 0.5
        assert agent.top_k == 10
        assert agent.min_p == 0.1
        assert agent.presence_penalty == 0.2
        assert agent.frequency_penalty == 0.3
        assert agent.repetition_penalty == 1.2
        assert agent.stop == ["STOP", "END"]

    def test_agent_extra_body(self):
        """Test agent with extra body parameters."""
        extra_params = {"custom_param": "value", "another_param": 123}
        agent = Agent(model="gpt-4", extra_body=extra_params)

        assert agent.extra_body == extra_params
        assert agent.extra_body["custom_param"] == "value"
