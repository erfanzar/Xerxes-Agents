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
"""Tests for xerxes.hooks — lifecycle hook system."""

import pytest
from xerxes.extensions.hooks import HOOK_POINTS, HookRunner


class TestHookRunner:
    def test_register_valid_hook(self):
        runner = HookRunner()
        runner.register("before_tool_call", lambda **kw: None)
        assert runner.has_hooks("before_tool_call")

    def test_register_invalid_hook_raises(self):
        runner = HookRunner()
        with pytest.raises(ValueError, match="Unknown hook point"):
            runner.register("invalid_hook", lambda **kw: None)

    def test_all_hook_points_exist(self):
        expected = {
            "before_tool_call",
            "after_tool_call",
            "tool_result_persist",
            "bootstrap_files",
            "on_turn_start",
            "on_turn_end",
            "on_loop_warning",
            "on_error",
        }
        assert HOOK_POINTS == expected

    def test_before_tool_call_mutates_arguments(self):
        runner = HookRunner()

        def add_flag(tool_name, arguments, agent_id):
            args = arguments.copy()
            args["injected"] = True
            return args

        runner.register("before_tool_call", add_flag)
        result = runner.run("before_tool_call", tool_name="search", arguments={"q": "hello"}, agent_id="a1")
        assert result["injected"] is True
        assert result["q"] == "hello"

    def test_mutation_hooks_chain(self):
        runner = HookRunner()

        def hook1(tool_name, result, agent_id):
            return result + " [hook1]"

        def hook2(tool_name, result, agent_id):
            return result + " [hook2]"

        runner.register("after_tool_call", hook1)
        runner.register("after_tool_call", hook2)
        result = runner.run("after_tool_call", tool_name="search", result="original", agent_id="a1")
        assert result == "original [hook1] [hook2]"

    def test_mutation_hook_none_keeps_previous(self):
        runner = HookRunner()

        def noop(tool_name, arguments, agent_id):
            return None  # Don't modify

        def modify(tool_name, arguments, agent_id):
            return {"modified": True}

        runner.register("before_tool_call", noop)
        runner.register("before_tool_call", modify)
        result = runner.run("before_tool_call", tool_name="x", arguments={"original": True}, agent_id="a")
        assert result == {"modified": True}

    def test_observation_hook_collects_results(self):
        runner = HookRunner()
        runner.register("bootstrap_files", lambda agent_id: "file1.md content")
        runner.register("bootstrap_files", lambda agent_id: "file2.md content")
        results = runner.run("bootstrap_files", agent_id="a1")
        assert len(results) == 2
        assert "file1.md content" in results
        assert "file2.md content" in results

    def test_hook_error_logged_not_raised(self):
        runner = HookRunner()

        def bad_hook(**kwargs):
            raise RuntimeError("Hook crashed")

        def good_hook(tool_name, result, agent_id):
            return result + " [good]"

        runner.register("after_tool_call", bad_hook)
        runner.register("after_tool_call", good_hook)
        result = runner.run("after_tool_call", tool_name="x", result="base", agent_id="a")
        assert "[good]" in result

    def test_unregister_hook(self):
        runner = HookRunner()

        def fn(**kw):
            return "x"

        runner.register("on_turn_start", fn)
        assert runner.has_hooks("on_turn_start")
        assert runner.unregister("on_turn_start", fn)
        assert not runner.has_hooks("on_turn_start")

    def test_unregister_nonexistent_returns_false(self):
        runner = HookRunner()
        assert runner.unregister("on_turn_start", lambda: None) is False

    def test_clear_specific_hook(self):
        runner = HookRunner()
        runner.register("on_turn_start", lambda **kw: None)
        runner.register("on_turn_end", lambda **kw: None)
        runner.clear("on_turn_start")
        assert not runner.has_hooks("on_turn_start")
        assert runner.has_hooks("on_turn_end")

    def test_clear_all_hooks(self):
        runner = HookRunner()
        runner.register("on_turn_start", lambda **kw: None)
        runner.register("on_turn_end", lambda **kw: None)
        runner.clear()
        assert not runner.has_hooks("on_turn_start")
        assert not runner.has_hooks("on_turn_end")

    def test_no_hooks_returns_kwarg(self):
        runner = HookRunner()
        result = runner.run("before_tool_call", tool_name="x", arguments={"a": 1}, agent_id="a")
        assert result == {"a": 1}

    def test_tool_result_persist_mutation(self):
        runner = HookRunner()

        def sanitize(tool_name, result, agent_id):
            if isinstance(result, str) and "secret" in result:
                return "[REDACTED]"
            return result

        runner.register("tool_result_persist", sanitize)
        result = runner.run("tool_result_persist", tool_name="x", result="contains secret data", agent_id="a")
        assert result == "[REDACTED]"

        result2 = runner.run("tool_result_persist", tool_name="x", result="normal data", agent_id="a")
        assert result2 == "normal data"
