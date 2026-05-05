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
"""Tests for xerxes.cortex.task module."""

from unittest.mock import MagicMock

from xerxes.cortex.orchestration.task import (
    ChainLink,
    CortexTask,
    CortexTaskOutput,
    TaskValidationError,
)


def make_mock_agent():
    agent = MagicMock()
    agent.role = "TestAgent"
    agent.goal = "Test goal"
    agent.backstory = "Test backstory"
    return agent


class TestTaskValidationError:
    def test_basic(self):
        e = TaskValidationError("validation failed")
        assert e.message == "validation failed"
        assert str(e) == "validation failed"


class TestChainLink:
    def test_defaults(self):
        link = ChainLink()
        assert link.condition is None
        assert link.next_task is None
        assert link.fallback_task is None

    def test_with_condition(self):
        link = ChainLink(condition=lambda x: "success" in x)
        assert link.condition("success here") is True
        assert link.condition("failure here") is False


class TestCortexTaskOutput:
    def test_basic(self):
        agent = make_mock_agent()
        task = CortexTask(description="Test task", expected_output="Expected")
        output = CortexTaskOutput(task=task, output="Result text", agent=agent)
        assert output.output == "Result text"
        assert output.execution_time == 0.0

    def test_summary_short(self):
        agent = make_mock_agent()
        task = CortexTask(description="Test", expected_output="Expected")
        output = CortexTaskOutput(task=task, output="Short", agent=agent)
        assert output.summary == "Short"

    def test_summary_long(self):
        agent = make_mock_agent()
        task = CortexTask(description="Test", expected_output="Expected")
        output = CortexTaskOutput(task=task, output="x" * 300, agent=agent)
        assert output.summary.endswith("...")
        assert len(output.summary) == 203

    def test_to_dict(self):
        agent = make_mock_agent()
        task = CortexTask(description="Test task", expected_output="Expected")
        output = CortexTaskOutput(task=task, output="Result", agent=agent)
        d = output.to_dict()
        assert d["task_description"] == "Test task"
        assert d["actual_output"] == "Result"
        assert d["agent_role"] == "TestAgent"

    def test_str(self):
        agent = make_mock_agent()
        task = CortexTask(description="Test task", expected_output="Expected")
        output = CortexTaskOutput(task=task, output="Result", agent=agent)
        s = str(output)
        assert "TestAgent" in s

    def test_repr(self):
        agent = make_mock_agent()
        task = CortexTask(description="Test", expected_output="Expected")
        output = CortexTaskOutput(task=task, output="Result", agent=agent)
        r = repr(output)
        assert "CortexTaskOutput" in r


class TestCortexTask:
    def test_basic_creation(self):
        task = CortexTask(description="Analyze data", expected_output="Report")
        assert task.description == "Analyze data"
        assert task.expected_output == "Report"
        assert task.agent is None

    def test_with_agent(self):
        agent = make_mock_agent()
        task = CortexTask(description="Task", expected_output="Output", agent=agent)
        assert task.agent is agent

    def test_with_tools(self):
        task = CortexTask(description="Task", expected_output="Output", tools=[])
        assert task.tools == []

    def test_interpolate_inputs(self):
        task = CortexTask(
            description="Analyze {topic} data",
            expected_output="Report on {topic}",
        )
        task.interpolate_inputs({"topic": "sales"})
        assert "sales" in task.description
        assert "sales" in task.expected_output

    def test_context_default(self):
        task = CortexTask(description="Task", expected_output="Output")
        assert task.context is None

    def test_max_retries_default(self):
        task = CortexTask(description="Task", expected_output="Output")
        assert task.max_retries == 3

    def test_importance_default(self):
        task = CortexTask(description="Task", expected_output="Output")
        assert task.importance == 0.5
