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
"""Tests for xerxes.cortex string_utils, templates, and enums."""

import pytest
from xerxes.cortex.core.enums import ChainType, ProcessType
from xerxes.cortex.core.string_utils import (
    extract_template_variables,
    interpolate_inputs,
    validate_inputs_for_template,
)
from xerxes.cortex.core.templates import PromptTemplate


class TestProcessType:
    def test_values(self):
        assert ProcessType.SEQUENTIAL.value == "sequential"
        assert ProcessType.HIERARCHICAL.value == "hierarchical"
        assert ProcessType.PARALLEL.value == "parallel"
        assert ProcessType.CONSENSUS.value == "consensus"
        assert ProcessType.PLANNED.value == "planned"


class TestChainType:
    def test_values(self):
        assert ChainType.LINEAR.value == "linear"
        assert ChainType.BRANCHING.value == "branching"
        assert ChainType.LOOP.value == "loop"


class TestInterpolateInputs:
    def test_basic(self):
        result = interpolate_inputs("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_vars(self):
        result = interpolate_inputs("{a} and {b}", {"a": "1", "b": "2"})
        assert result == "1 and 2"

    def test_int_value(self):
        result = interpolate_inputs("Count: {n}", {"n": 42})
        assert result == "Count: 42"

    def test_float_value(self):
        result = interpolate_inputs("PI: {pi}", {"pi": 3.14})
        assert result == "PI: 3.14"

    def test_dict_value(self):
        result = interpolate_inputs("Data: {d}", {"d": {"key": "val"}})
        assert '"key"' in result

    def test_list_value(self):
        result = interpolate_inputs("Items: {items}", {"items": [1, 2, 3]})
        assert "[1, 2, 3]" in result

    def test_none_value(self):
        result = interpolate_inputs("Val: {x}", {"x": None})
        assert result == "Val: "

    def test_bool_value(self):
        result = interpolate_inputs("Flag: {f}", {"f": True})
        assert result == "Flag: True"

    def test_empty_string(self):
        assert interpolate_inputs("", {"a": "b"}) == ""

    def test_none_string(self):
        assert interpolate_inputs(None, {"a": "b"}) == ""

    def test_missing_key(self):
        with pytest.raises(KeyError):
            interpolate_inputs("{missing}", {})

    def test_unsupported_type(self):
        with pytest.raises(ValueError):
            interpolate_inputs("{x}", {"x": object()})

    def test_no_placeholders(self):
        result = interpolate_inputs("no vars here", {})
        assert result == "no vars here"


class TestExtractTemplateVariables:
    def test_basic(self):
        avars = extract_template_variables("Hello {name}, year {year}")
        assert avars == {"name", "year"}

    def test_empty(self):
        assert extract_template_variables("") == set()

    def test_none_like(self):
        assert extract_template_variables("no vars") == set()

    def test_underscore_var(self):
        avars = extract_template_variables("{_private}")
        assert "_private" in avars


class TestValidateInputsForTemplate:
    def test_valid(self):
        valid, errors = validate_inputs_for_template("Hello {name}", {"name": "World"})
        assert valid is True
        assert errors == []

    def test_missing(self):
        valid, errors = validate_inputs_for_template("Hello {name}", {})
        assert valid is False
        assert any("name" in e for e in errors)

    def test_extra_allowed(self):
        valid, _errors = validate_inputs_for_template("{a}", {"a": "1", "b": "2"}, allow_extra=True)
        assert valid is True

    def test_extra_not_allowed(self):
        valid, errors = validate_inputs_for_template("{a}", {"a": "1", "b": "2"}, allow_extra=False)
        assert valid is False
        assert any("b" in e for e in errors)


class TestPromptTemplate:
    def test_init(self):
        t = PromptTemplate()
        assert t.use_jinja is True

    def test_render_basic(self):
        t = PromptTemplate()
        result = t.render("Hello {{ name }}!", name="World")
        assert "Hello World!" in result

    def test_render_agent_prompt(self):
        t = PromptTemplate()
        result = t.render_agent_prompt(
            role="Analyst",
            goal="Analyze data",
            backstory="Expert analyst",
        )
        assert "Analyst" in result
        assert "Analyze data" in result

    def test_render_agent_prompt_with_tools(self):
        t = PromptTemplate()
        tools = [{"name": "search", "description": "Search the web"}]
        result = t.render_agent_prompt(
            role="Researcher",
            goal="Research topics",
            backstory="Expert researcher",
            tools=tools,
        )
        assert "search" in result

    def test_render_agent_prompt_with_rules(self):
        t = PromptTemplate()
        result = t.render_agent_prompt(
            role="Writer",
            goal="Write content",
            backstory="Expert writer",
            rules=["Be concise", "Use formal tone"],
        )
        assert "Be concise" in result

    def test_render_task_prompt(self):
        t = PromptTemplate()
        result = t.render_task_prompt(
            description="Analyze Q4 sales",
            expected_output="A detailed report",
        )
        assert "Analyze Q4 sales" in result

    def test_render_task_prompt_with_context(self):
        t = PromptTemplate()
        result = t.render_task_prompt(
            description="Write summary",
            expected_output="Summary",
            context="Previous analysis found X",
        )
        assert "Previous analysis found X" in result

    def test_render_task_prompt_with_constraints(self):
        t = PromptTemplate()
        result = t.render_task_prompt(
            description="task",
            expected_output="output",
            constraints=["Max 500 words"],
        )
        assert "Max 500 words" in result

    def test_render_manager_delegation(self):
        t = PromptTemplate()
        agents = [{"role": "Writer", "goal": "Write"}]
        tasks = [{"description": "Write article", "expected_output": "Article"}]
        result = t.render_manager_delegation(agents, tasks)
        assert "Writer" in result

    def test_render_manager_review(self):
        t = PromptTemplate()
        result = t.render_manager_review(
            agent_role="Writer",
            task_description="Write article",
            output="Here is the article...",
        )
        assert "Writer" in result

    def test_render_consensus(self):
        t = PromptTemplate()
        result = t.render_consensus(
            task_description="Evaluate design",
            agent_outputs={"Designer": "Looks great", "Engineer": "Feasible"},
        )
        assert "Designer" in result
        assert "Engineer" in result

    def test_render_planner(self):
        t = PromptTemplate()
        result = t.render_planner(
            objective="Create campaign",
            agents=[{"role": "Writer", "goal": "Write", "tools": []}],
        )
        assert "Create campaign" in result

    def test_render_step_execution(self):
        t = PromptTemplate()
        result = t.render_step_execution(
            action="analyze",
            description="Analyze data",
            arguments={"source": "data.csv"},
            context="Previous step gathered data",
        )
        assert "analyze" in result
        assert "data.csv" in result

    def test_create_custom_template(self):
        t = PromptTemplate()
        custom = t.create_custom_template("Hello {{ x }}")
        assert custom is not None
        assert "Hello World" in custom.render(x="World")

    def test_get_template_variables(self):
        t = PromptTemplate()
        avars = t.get_template_variables("{{ name }} and {{ age }}")
        assert "name" in avars
        assert "age" in avars

    def test_render_no_jinja_fallback(self):
        t = PromptTemplate()
        t.use_jinja = False
        result = t.render("Hello {name}!", name="World")
        assert "Hello World!" in result

    def test_create_custom_template_no_jinja(self):
        t = PromptTemplate()
        t.use_jinja = False
        assert t.create_custom_template("test") is None

    def test_get_template_variables_no_jinja(self):
        t = PromptTemplate()
        t.use_jinja = False
        assert t.get_template_variables("{{ x }}") == set()
