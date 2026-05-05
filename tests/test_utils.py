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
"""Tests for xerxes.core.utils module."""

from xerxes.core.utils import (
    XerxesBase,
    debug_print,
    estimate_messages_tokens,
    estimate_tokens,
    function_to_json,
    merge_chunk,
    merge_fields,
    run_sync,
)


class TestRunSync:
    def test_basic(self):
        async def coro():
            return 42

        result = run_sync(coro())
        assert result == 42


class TestXerxesBase:
    def test_basic(self):
        class MyModel(XerxesBase):
            name: str
            count: int = 0

        m = MyModel(name="test")
        assert m.name == "test"
        assert m.count == 0


class TestDebugPrint:
    def test_debug_enabled(self, capsys):
        debug_print(True, "hello", "world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_debug_disabled(self, capsys):
        debug_print(False, "should not print")
        captured = capsys.readouterr()
        assert captured.out == ""


class TestMergeFields:
    def test_string_merge(self):
        target = {"text": "Hello"}
        merge_fields(target, {"text": " World"})
        assert target["text"] == "Hello World"

    def test_dict_merge(self):
        target = {"nested": {"key": "val"}}
        merge_fields(target, {"nested": {"key": "2"}})
        assert target["nested"]["key"] == "val2"

    def test_none_value_skipped(self):
        target = {"a": "x"}
        merge_fields(target, {"b": None})
        assert "b" not in target


class TestMergeChunk:
    def test_basic(self):
        response = {"content": "Hello"}
        merge_chunk(response, {"content": " World", "role": "assistant"})
        assert response["content"] == "Hello World"
        assert "role" not in response


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("Hello, world!") > 0

    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_custom_ratio(self):
        result = estimate_tokens("abcdefgh", chars_per_token=2.0)
        assert result == 4


class TestEstimateMessagesTokens:
    def test_basic(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = estimate_messages_tokens(msgs)
        assert result > 0

    def test_empty(self):
        assert estimate_messages_tokens([]) == 0


class TestFunctionToJson:
    def test_basic_function(self):
        def greet(name: str, age: int = 0) -> str:
            """Greet a person.
            name: Person's name
            age: Person's age
            """
            return f"Hello {name}"

        schema = function_to_json(greet)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"
        assert "name" in schema["function"]["parameters"]["properties"]
        assert "age" in schema["function"]["parameters"]["properties"]

    def test_with_context_variables(self):
        def tool(query: str, context_variables: dict | None = None):
            """Search tool."""
            pass

        schema = function_to_json(tool)
        assert "context_variables" not in schema["function"]["parameters"]["properties"]

    def test_with_list_param(self):
        def tool(items: list[str]):
            """Process items."""
            pass

        schema = function_to_json(tool)
        assert schema["function"]["parameters"]["properties"]["items"]["type"] == "array"

    def test_with_optional_param(self):
        def tool(name: str | None = None):
            """Optional param."""
            pass

        schema = function_to_json(tool)
        assert "name" in schema["function"]["parameters"]["properties"]

    def test_with_xerxes_schema(self):
        def tool():
            """Schema test."""
            pass

        tool.__xerxes_schema__ = {
            "name": "custom_tool",
            "description": "A custom tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        schema = function_to_json(tool)
        assert schema["function"]["name"] == "custom_tool"
        assert "Schema test." in schema["function"]["description"]

    def test_no_annotations(self):
        def simple(x):
            """Simple function."""
            pass

        schema = function_to_json(simple)
        assert "x" in schema["function"]["parameters"]["properties"]

    def test_with_future_style_string_annotations(self):
        def tool(name: "str", enabled: "bool", items: "list[str]"):
            """Forward annotation tool.
            name: The tool name.
            enabled: Whether the tool should run.
            items: Input items to process.
            """
            pass

        schema = function_to_json(tool)
        props = schema["function"]["parameters"]["properties"]
        assert props["name"]["type"] == "string"
        assert props["enabled"]["type"] == "boolean"
        assert props["items"]["type"] == "array"
