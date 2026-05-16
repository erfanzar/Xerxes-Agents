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
"""Tests for xerxes.streaming.tool_call_ids."""

from __future__ import annotations

from xerxes.streaming.tool_call_ids import canonicalize_kwargs, deterministic_tool_call_id


class TestCanonicalizeKwargs:
    def test_key_order_independent(self) -> None:
        a = canonicalize_kwargs({"a": 1, "b": 2})
        b = canonicalize_kwargs({"b": 2, "a": 1})
        assert a == b

    def test_non_serializable_falls_back_to_str(self) -> None:
        class X:
            def __repr__(self) -> str:
                return "X"

        # default=str makes this succeed; this just confirms no exception.
        out = canonicalize_kwargs({"x": X()})
        assert "X" in out


class TestDeterministicID:
    def test_stable_across_invocations(self) -> None:
        a = deterministic_tool_call_id("read_file", {"path": "/x"})
        b = deterministic_tool_call_id("read_file", {"path": "/x"})
        assert a == b

    def test_different_name_different_id(self) -> None:
        a = deterministic_tool_call_id("read_file", {"path": "/x"})
        b = deterministic_tool_call_id("write_file", {"path": "/x"})
        assert a != b

    def test_different_kwargs_different_id(self) -> None:
        a = deterministic_tool_call_id("read_file", {"path": "/x"})
        b = deterministic_tool_call_id("read_file", {"path": "/y"})
        assert a != b

    def test_prefix_default_openai_style(self) -> None:
        out = deterministic_tool_call_id("x", {})
        assert out.startswith("call_")

    def test_prefix_custom(self) -> None:
        out = deterministic_tool_call_id("x", {}, prefix="tc_")
        assert out.startswith("tc_")

    def test_length_param(self) -> None:
        out = deterministic_tool_call_id("x", {}, length=8)
        # prefix + 8 hex chars
        assert len(out) == len("call_") + 8

    def test_kwargs_order_doesnt_change_id(self) -> None:
        a = deterministic_tool_call_id("f", {"a": 1, "b": 2})
        b = deterministic_tool_call_id("f", {"b": 2, "a": 1})
        assert a == b
