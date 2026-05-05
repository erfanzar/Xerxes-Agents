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
"""Tests for xerxes.cortex.tool module."""

from xerxes.cortex.core.tool import CortexTool


def sample_function(query: str, limit: int = 10) -> list:
    """Search for items matching the query."""
    return []


class TestCortexTool:
    def test_from_function(self):
        tool = CortexTool.from_function(sample_function)
        assert tool.name == "sample_function"
        assert "Search" in tool.description

    def test_from_function_custom_name(self):
        tool = CortexTool.from_function(sample_function, name="search", description="Custom desc")
        assert tool.name == "search"
        assert tool.description == "Custom desc"

    def test_to_function_json_auto(self):
        tool = CortexTool.from_function(sample_function)
        json_repr = tool.to_function_json()
        assert json_repr["type"] == "function"
        assert "function" in json_repr
        assert json_repr["function"]["name"] == "sample_function"

    def test_to_function_json_manual(self):
        tool = CortexTool(
            name="calc",
            description="Calculate",
            function=lambda: None,
            parameters={"type": "object", "properties": {"x": {"type": "number"}}},
            auto_generate_schema=False,
        )
        json_repr = tool.to_function_json()
        assert json_repr["function"]["name"] == "calc"
        assert "x" in json_repr["function"]["parameters"]["properties"]

    def test_to_function_json_no_schema_no_auto(self):
        tool = CortexTool(
            name="test",
            description="test desc",
            function=lambda: None,
            auto_generate_schema=False,
        )
        json_repr = tool.to_function_json()
        assert json_repr["function"]["parameters"]["type"] == "object"

    def test_from_function_no_docstring(self):
        def no_doc(x: int) -> int:
            return x

        tool = CortexTool.from_function(no_doc)
        assert tool.name == "no_doc"
        assert tool.description == ""
