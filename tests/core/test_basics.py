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
"""Tests for xerxes.core.basics module."""

from xerxes.core.basics import (
    AGENTS_REGISTRY,
    CLIENT_REGISTRY,
    REGISTRY,
    XERXES_REGISTRY,
    _pretty_print,
    basic_registry,
)


class TestPrettyPrint:
    def test_flat_dict(self):
        result = _pretty_print({"key": "value"})
        assert "key:" in result
        assert "value" in result

    def test_nested_dict(self):
        result = _pretty_print({"outer": {"inner": "val"}})
        assert "outer:" in result
        assert "inner:" in result
        assert "val" in result

    def test_empty_dict(self):
        result = _pretty_print({})
        assert result == ""

    def test_indent(self):
        result = _pretty_print({"k": "v"}, indent=4)
        assert result.startswith("    k:")


class TestRegisteries:
    def test_registery_structure(self):
        assert "client" in REGISTRY
        assert "agents" in REGISTRY
        assert "xerxes" in REGISTRY
        assert REGISTRY["client"] is CLIENT_REGISTRY
        assert REGISTRY["agents"] is AGENTS_REGISTRY
        assert REGISTRY["xerxes"] is XERXES_REGISTRY


class TestBasicRegistery:
    def test_register_agents(self):
        @basic_registry("agents", "test_agent_basic")
        class TestAgent:
            def __init__(self):
                self.name = "test"

        assert "test_agent_basic" in AGENTS_REGISTRY
        agent = TestAgent()
        assert agent.to_dict() == {"name": "test"}
        assert "TestAgent" in str(agent)
        assert "TestAgent" in repr(agent)

    def test_register_client(self):
        @basic_registry("client", "test_client_basic")
        class TestClient:
            def __init__(self):
                self.url = "http://test"

        assert "test_client_basic" in CLIENT_REGISTRY

    def test_register_xerxes(self):
        @basic_registry("xerxes", "test_xerxes_basic")
        class TestXerxes:
            pass

        assert "test_xerxes_basic" in XERXES_REGISTRY

    def test_invalid_register_type(self):
        try:

            @basic_registry("invalid", "test")
            class Bad:
                pass

            raise AssertionError("Should have raised AssertionError")
        except AssertionError:
            pass

    def test_to_dict_excludes_private(self):
        @basic_registry("agents", "test_private_basic")
        class TestPrivate:
            def __init__(self):
                self.public = 1
                self._private = 2

        obj = TestPrivate()
        d = obj.to_dict()
        assert "public" in d
        assert "_private" not in d
