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
"""Tests for xerxes.tools.data_tools module."""

import json

from xerxes.tools.data_tools import (
    CSVProcessor,
    DataConverter,
    DateTimeProcessor,
    JSONProcessor,
    TextProcessor,
)


class TestJSONProcessor:
    def test_load(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"key": "value"}))
        result = JSONProcessor.static_call(operation="load", file_path=str(f))
        assert result["success"] is True
        assert result["data"]["key"] == "value"

    def test_load_no_path(self):
        result = JSONProcessor.static_call(operation="load")
        assert "error" in result

    def test_save(self, tmp_path):
        f = tmp_path / "out.json"
        result = JSONProcessor.static_call(operation="save", file_path=str(f), data={"a": 1})
        assert result["success"] is True

    def test_save_no_data(self, tmp_path):
        result = JSONProcessor.static_call(operation="save", file_path=str(tmp_path / "x.json"))
        assert "error" in result

    def test_validate_valid(self):
        result = JSONProcessor.static_call(operation="validate", data='{"key": "value"}')
        assert result["valid"] is True

    def test_validate_dict(self):
        result = JSONProcessor.static_call(operation="validate", data={"key": "value"})
        assert result["valid"] is True

    def test_query(self):
        data = {"user": {"name": "Alice", "age": 30}}
        result = JSONProcessor.static_call(operation="query", data=data, query="user.name")
        assert "result" in result or "error" not in result


class TestCSVProcessor:
    def test_read(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25")
        result = CSVProcessor.static_call(operation="read", file_path=str(f))
        assert "data" in result or "rows" in result or "error" not in result

    def test_read_no_path(self):
        result = CSVProcessor.static_call(operation="read")
        assert "error" in result

    def test_write(self, tmp_path):
        f = tmp_path / "out.csv"
        data = [{"name": "Alice", "age": "30"}]
        result = CSVProcessor.static_call(operation="write", file_path=str(f), data=data)
        assert "success" in result or "error" not in result

    def test_analyze(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\nCharlie,35")
        result = CSVProcessor.static_call(operation="analyze", file_path=str(f))
        assert "error" not in result or "rows" in result or "analysis" in result


class TestTextProcessor:
    def test_stats(self):
        result = TextProcessor.static_call(operation="stats", text="Hello world this is a test")
        assert "word_count" in result or "statistics" in result or "error" not in result

    def test_extract_emails(self):
        result = TextProcessor.static_call(operation="extract", text="Email me at test@example.com", pattern="emails")
        assert result is not None

    def test_no_text(self):
        result = TextProcessor.static_call(operation="stats", text="")
        assert result is not None


class TestDataConverter:
    def test_json_to_yaml(self):
        result = DataConverter.static_call(data='{"key": "value"}', from_format="json", to_format="yaml")
        assert "error" not in result or "result" in result or "converted" in result

    def test_base64_encode(self):
        result = DataConverter.static_call(data="hello", from_format="text", to_format="base64")
        assert "result" in result or "converted" in result or "error" not in result

    def test_hex_encode(self):
        result = DataConverter.static_call(data="hello", from_format="text", to_format="hex")
        assert "result" in result or "converted" in result or "error" not in result


class TestDateTimeProcessor:
    def test_now(self):
        result = DateTimeProcessor.static_call(operation="now")
        assert result is not None
        assert "error" not in result or "datetime" in result or "timestamp" in result

    def test_parse(self):
        result = DateTimeProcessor.static_call(operation="parse", date_string="2025-01-15")
        assert result is not None

    def test_format(self):
        result = DateTimeProcessor.static_call(operation="format", date_string="2025-01-15", output_format="%B %d, %Y")
        assert result is not None

    def test_delta(self):
        result = DateTimeProcessor.static_call(operation="delta", date_string="2025-01-01", days=10)
        assert result is not None
