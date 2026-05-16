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
"""Tests for the external memory provider framework + plugins."""

from __future__ import annotations

import pytest
from xerxes.memory.plugins._base import ExternalMemoryProviderBase, make_simple_provider
from xerxes.memory.provider import (
    MemoryProvider,
    MemoryToolCall,
    PluginRegistry,
    active,
    register,
    registry,
    set_active,
)


class StubProvider(MemoryProvider):
    name = "stub"

    def __init__(self):
        self._calls = []

    def initialize(self) -> None:
        return None

    def is_available(self) -> bool:
        return True

    def get_tool_schemas(self):
        return [{"name": "stub_test"}]

    def handle_tool_call(self, call):
        self._calls.append(call)
        return {"ok": True}


class TestRegistry:
    def setup_method(self):
        registry()._providers.clear()
        registry()._active = None

    def test_register_and_list(self):
        register(StubProvider())
        assert "stub" in registry().list_names()

    def test_unregister_clears_active(self):
        register(StubProvider())
        set_active("stub")
        registry().unregister("stub")
        assert active() is None

    def test_set_active_unknown_raises(self):
        with pytest.raises(KeyError):
            set_active("ghost")

    def test_active_returns_none_when_unset(self):
        register(StubProvider())
        assert active() is None

    def test_anonymous_provider_rejected(self):
        class _NoName(MemoryProvider):
            name = ""

            def initialize(self):
                pass

            def is_available(self):
                return True

            def get_tool_schemas(self):
                return []

            def handle_tool_call(self, call):
                return {}

        r = PluginRegistry()
        with pytest.raises(ValueError):
            r.register(_NoName())


class TestProviderBase:
    def test_tool_schema_shape(self):
        p = make_simple_provider("testmem")
        schemas = p.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert sorted(names) == ["testmem_add", "testmem_list", "testmem_remove", "testmem_search"]

    def test_add_search_remove_lifecycle(self):
        p = make_simple_provider("testmem")
        p.initialize()
        added = p.handle_tool_call(MemoryToolCall(name="testmem_add", arguments={"content": "user likes tea"}))
        assert added["ok"] is True
        entry_id = added["result"]["id"]
        listed = p.handle_tool_call(MemoryToolCall(name="testmem_list", arguments={}))
        assert any(e["id"] == entry_id for e in listed["result"])
        searched = p.handle_tool_call(MemoryToolCall(name="testmem_search", arguments={"query": "tea"}))
        assert any(e["id"] == entry_id for e in searched["result"])
        removed = p.handle_tool_call(MemoryToolCall(name="testmem_remove", arguments={"entry_id": entry_id}))
        assert removed["ok"] is True
        listed2 = p.handle_tool_call(MemoryToolCall(name="testmem_list", arguments={}))
        assert not any(e["id"] == entry_id for e in listed2["result"])

    def test_unknown_action(self):
        p = make_simple_provider("testmem")
        out = p.handle_tool_call(MemoryToolCall(name="testmem_bogus"))
        assert out["ok"] is False

    def test_unavailable_when_missing_env(self, monkeypatch):
        monkeypatch.delenv("FAKE_KEY", raising=False)

        class _Stub(ExternalMemoryProviderBase):
            name = "fake"
            namespace_label = "fake"
            required_env = ("FAKE_KEY",)

        p = _Stub()
        assert p.is_available() is False

    def test_exception_surfaced_as_error(self):
        def explode(action, args):
            raise RuntimeError("simulated")

        p = make_simple_provider("explody", upstream_caller=explode)
        out = p.handle_tool_call(MemoryToolCall(name="explody_add", arguments={"content": "x"}))
        assert out["ok"] is False
        assert "simulated" in out["error"]


class TestBuiltinPluginsImport:
    def test_all_plugins_importable(self):
        # Re-import the plugin package fresh — registry is cleared by setup.
        import importlib

        plugins = importlib.import_module("xerxes.memory.plugins")
        importlib.reload(plugins)
        names = registry().list_names()
        # At least the local-only one always registers (holographic).
        assert "holographic" in names

    def test_holographic_works_offline(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOLOGRAPHIC_DB_PATH", str(tmp_path / "facts.db"))
        from xerxes.memory.plugins.holographic import HolographicProvider

        p = HolographicProvider()
        assert p.is_available() is True
        p.initialize()
        added = p.handle_tool_call(MemoryToolCall(name="holo_add", arguments={"content": "fact one"}))
        assert added["ok"] is True
        search = p.handle_tool_call(MemoryToolCall(name="holo_search", arguments={"query": "fact"}))
        assert any("fact one" in e["content"] for e in search["result"])
