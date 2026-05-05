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
"""Tests for xerxes.policy — tool policy allow/deny enforcement."""

import pytest
from xerxes.security.policy import PolicyAction, PolicyEngine, ToolPolicy, ToolPolicyViolation


class TestToolPolicy:
    def test_empty_policy_allows_all(self):
        policy = ToolPolicy()
        assert policy.evaluate("anything") == PolicyAction.ALLOW

    def test_allow_list_permits_listed(self):
        policy = ToolPolicy(allow={"search", "read_file"})
        assert policy.evaluate("search") == PolicyAction.ALLOW
        assert policy.evaluate("read_file") == PolicyAction.ALLOW

    def test_allow_list_denies_unlisted(self):
        policy = ToolPolicy(allow={"search"})
        assert policy.evaluate("execute_shell") == PolicyAction.DENY

    def test_deny_list_blocks_listed(self):
        policy = ToolPolicy(deny={"execute_shell", "delete_file"})
        assert policy.evaluate("execute_shell") == PolicyAction.DENY
        assert policy.evaluate("delete_file") == PolicyAction.DENY

    def test_deny_list_allows_unlisted(self):
        policy = ToolPolicy(deny={"execute_shell"})
        assert policy.evaluate("search") == PolicyAction.ALLOW

    def test_allow_takes_precedence_over_deny(self):
        policy = ToolPolicy(allow={"search"}, deny={"search"})
        assert policy.evaluate("search") == PolicyAction.ALLOW

    def test_optional_tools_denied_by_default(self):
        policy = ToolPolicy(optional_tools={"dangerous_tool"})
        assert policy.evaluate("dangerous_tool") == PolicyAction.DENY
        assert policy.evaluate("normal_tool") == PolicyAction.ALLOW

    def test_optional_tools_allowed_via_allow_list(self):
        policy = ToolPolicy(allow={"dangerous_tool"}, optional_tools={"dangerous_tool"})
        assert policy.evaluate("dangerous_tool") == PolicyAction.ALLOW


class TestPolicyEngine:
    def test_global_policy_applies_to_all(self):
        engine = PolicyEngine(global_policy=ToolPolicy(deny={"execute_shell"}))
        assert engine.check("execute_shell") == PolicyAction.DENY
        assert engine.check("execute_shell", agent_id="any") == PolicyAction.DENY
        assert engine.check("search", agent_id="any") == PolicyAction.ALLOW

    def test_agent_policy_overrides_global(self):
        engine = PolicyEngine(
            global_policy=ToolPolicy(deny={"execute_shell"}),
            agent_policies={"coder": ToolPolicy(allow={"execute_shell", "search"})},
        )

        assert engine.check("execute_shell") == PolicyAction.DENY

        assert engine.check("execute_shell", agent_id="coder") == PolicyAction.ALLOW

        assert engine.check("execute_shell", agent_id="reader") == PolicyAction.DENY

    def test_enforce_raises_on_deny(self):
        engine = PolicyEngine(global_policy=ToolPolicy(deny={"execute_shell"}))
        with pytest.raises(ToolPolicyViolation) as exc_info:
            engine.enforce("execute_shell", agent_id="test")
        assert "execute_shell" in str(exc_info.value)

    def test_enforce_passes_on_allow(self):
        engine = PolicyEngine(global_policy=ToolPolicy(allow={"search"}))
        engine.enforce("search")

    def test_set_and_remove_agent_policy(self):
        engine = PolicyEngine()
        engine.set_agent_policy("a1", ToolPolicy(deny={"tool_x"}))
        assert engine.check("tool_x", agent_id="a1") == PolicyAction.DENY
        engine.remove_agent_policy("a1")
        assert engine.check("tool_x", agent_id="a1") == PolicyAction.ALLOW

    def test_listener_called(self):
        events = []
        engine = PolicyEngine(global_policy=ToolPolicy(deny={"x"}))
        engine.add_listener(lambda name, aid, action: events.append((name, aid, action)))
        engine.check("x", agent_id="a")
        engine.check("y", agent_id="b")
        assert len(events) == 2
        assert events[0] == ("x", "a", PolicyAction.DENY)
        assert events[1] == ("y", "b", PolicyAction.ALLOW)
