# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.
"""Tests for :mod:`xerxes.runtime.arg_validation`."""

from __future__ import annotations

import pytest
from xerxes.runtime.arg_validation import validate_and_format_error, validate_tool_arguments


@pytest.fixture
def file_read_schema() -> dict:
    return {
        "type": "object",
        "required": ["file_path"],
        "properties": {
            "file_path": {"type": "string"},
            "encoding": {"type": "string"},
        },
    }


@pytest.fixture
def edit_schema() -> dict:
    return {
        "type": "object",
        "required": ["file_path", "old_string", "new_string"],
        "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "edit_mode": {"type": "string", "enum": ["search_replace", "whole_file"]},
        },
        "additionalProperties": False,
    }


class TestValidArguments:
    """Arguments that match the schema pass validation."""

    def test_all_required_present(self, file_read_schema: dict):
        result = validate_tool_arguments("ReadFile", {"file_path": "x.py"}, file_read_schema)
        assert result.ok
        assert result.error == ""

    def test_required_plus_optional(self, file_read_schema: dict):
        result = validate_tool_arguments("ReadFile", {"file_path": "x.py", "encoding": "utf-8"}, file_read_schema)
        assert result.ok

    def test_no_schema_always_passes(self):
        result = validate_tool_arguments("AnyTool", {"anything": 42}, None)
        assert result.ok

    def test_empty_schema_always_passes(self):
        result = validate_tool_arguments("AnyTool", {"anything": 42}, {})
        assert result.ok

    def test_extra_properties_allowed_by_default(self, file_read_schema: dict):
        result = validate_tool_arguments("ReadFile", {"file_path": "x.py", "extra": True}, file_read_schema)
        assert result.ok


class TestMissingRequired:
    """Missing required parameters are caught."""

    def test_missing_required(self, file_read_schema: dict):
        result = validate_tool_arguments("ReadFile", {}, file_read_schema)
        assert not result.ok
        assert "file_path" in result.error
        assert "file_path" in result.missing

    def test_missing_one_of_many(self, edit_schema: dict):
        result = validate_tool_arguments("FileEditTool", {"file_path": "x.py", "old_string": "a"}, edit_schema)
        assert not result.ok
        assert "new_string" in result.missing


class TestTypeChecking:
    """Type mismatches are caught."""

    def test_string_expected_got_int(self, file_read_schema: dict):
        result = validate_tool_arguments("ReadFile", {"file_path": 123}, file_read_schema)
        assert not result.ok
        assert "string" in result.error

    def test_enum_violation(self, edit_schema: dict):
        args = {"file_path": "x.py", "old_string": "a", "new_string": "b", "edit_mode": "invalid"}
        result = validate_tool_arguments("FileEditTool", args, edit_schema)
        assert not result.ok
        assert "edit_mode" in result.error

    def test_integer_not_bool(self):
        schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
        result = validate_tool_arguments("Calc", {"n": True}, schema)
        assert not result.ok

    def test_array_type(self):
        schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array"}}}
        assert validate_tool_arguments("T", {"items": [1, 2]}, schema).ok
        assert not validate_tool_arguments("T", {"items": "not-list"}, schema).ok


class TestAdditionalProperties:
    """additionalProperties=false rejects unknown keys."""

    def test_rejects_unknown_key(self, edit_schema: dict):
        args = {"file_path": "x.py", "old_string": "a", "new_string": "b", "unknown_param": 42}
        result = validate_tool_arguments("FileEditTool", args, edit_schema)
        assert not result.ok
        assert "unknown_param" in result.error


class TestValidateAndFormatError:
    """The convenience wrapper handles JSON-string arguments."""

    def test_valid_returns_none(self, file_read_schema: dict):
        assert validate_and_format_error("ReadFile", {"file_path": "x.py"}, file_read_schema) is None

    def test_invalid_returns_error_string(self, file_read_schema: dict):
        error = validate_and_format_error("ReadFile", {}, file_read_schema)
        assert error is not None
        assert "file_path" in error

    def test_json_string_arguments(self, file_read_schema: dict):
        error = validate_and_format_error("ReadFile", '{"file_path": "x.py"}', file_read_schema)
        assert error is None

    def test_invalid_json_string(self):
        error = validate_and_format_error("ReadFile", "{not valid json", None)
        assert error is not None
        assert "JSON" in error


class TestEdgeCases:
    """Edge cases and error messages."""

    def test_non_dict_arguments(self):
        result = validate_tool_arguments("T", [1, 2, 3], {"type": "object"})
        assert not result.ok
        assert "object" in result.error

    def test_validation_result_is_frozen(self, file_read_schema: dict):
        result = validate_tool_arguments("ReadFile", {"file_path": "x.py"}, file_read_schema)
        with pytest.raises((AttributeError, Exception)):
            result.ok = False
