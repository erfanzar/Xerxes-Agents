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
"""Tests for xerxes.plugins — plugin registration and management."""

import pytest
from xerxes.extensions.plugins import (
    PluginConflictError,
    PluginMeta,
    PluginRegistry,
    PluginType,
)


def _sample_tool(x: str) -> str:
    return f"result: {x}"


def _another_tool(y: int) -> int:
    return y * 2


class TestPluginRegistry:
    def test_register_plugin(self):
        registry = PluginRegistry()
        meta = PluginMeta(name="test_plugin", version="1.0", plugin_type=PluginType.TOOL)
        plugin = registry.register_plugin(meta)
        assert plugin.meta.name == "test_plugin"
        assert "test_plugin" in registry.plugin_names

    def test_duplicate_plugin_raises(self):
        registry = PluginRegistry()
        meta = PluginMeta(name="test_plugin")
        registry.register_plugin(meta)
        with pytest.raises(PluginConflictError):
            registry.register_plugin(meta)

    def test_register_tool(self):
        registry = PluginRegistry()
        meta = PluginMeta(name="my_plugin")
        registry.register_tool("sample", _sample_tool, meta=meta)
        assert registry.get_tool("sample") is _sample_tool

    def test_duplicate_tool_raises(self):
        registry = PluginRegistry()
        meta = PluginMeta(name="p1")
        registry.register_tool("sample", _sample_tool, meta=meta)
        with pytest.raises(PluginConflictError):
            registry.register_tool("sample", _another_tool, meta=PluginMeta(name="p2"))

    def test_register_hook(self):
        registry = PluginRegistry()

        def hook_fn(**kw):
            return None

        registry.register_hook("before_tool_call", hook_fn, plugin_name="test")
        hooks = registry.get_hooks("before_tool_call")
        assert len(hooks) == 1
        assert hooks[0] is hook_fn

    def test_multiple_hooks_same_point(self):
        registry = PluginRegistry()

        def h1(**kw):
            return "h1"

        def h2(**kw):
            return "h2"

        registry.register_hook("after_tool_call", h1, plugin_name="p1")
        registry.register_hook("after_tool_call", h2, plugin_name="p2")
        hooks = registry.get_hooks("after_tool_call")
        assert len(hooks) == 2

    def test_register_provider(self):
        registry = PluginRegistry()
        provider_obj = {"type": "test_provider"}
        meta = PluginMeta(name="prov_plugin", plugin_type=PluginType.PROVIDER)
        registry.register_provider("test_llm", provider_obj, meta=meta)
        assert registry.get_provider("test_llm") == provider_obj

    def test_duplicate_provider_raises(self):
        registry = PluginRegistry()
        registry.register_provider("llm", object(), plugin_name="p1")
        with pytest.raises(PluginConflictError):
            registry.register_provider("llm", object(), plugin_name="p2")

    def test_register_channel(self):
        registry = PluginRegistry()
        chan = object()
        meta = PluginMeta(name="chan_plugin", plugin_type=PluginType.CHANNEL)
        registry.register_channel("telegram", chan, meta=meta)
        assert registry.get_channel("telegram") is chan
        assert registry.get_all_channels() == {"telegram": chan}
        assert registry.get_plugin("chan_plugin").channels == {"telegram": chan}

    def test_duplicate_channel_raises(self):
        registry = PluginRegistry()
        registry.register_channel("slack", object(), plugin_name="p1")
        with pytest.raises(PluginConflictError):
            registry.register_channel("slack", object(), plugin_name="p2")

    def test_unregister_plugin_removes_channels(self):
        registry = PluginRegistry()
        meta = PluginMeta(name="chan_plugin", plugin_type=PluginType.CHANNEL)
        registry.register_channel("discord", object(), meta=meta)
        assert registry.get_channel("discord") is not None
        registry.unregister_plugin("chan_plugin")
        assert registry.get_channel("discord") is None
        assert registry.get_all_channels() == {}

    def test_get_all_tools(self):
        registry = PluginRegistry()
        registry.register_tool("t1", _sample_tool, plugin_name="p1")
        registry.register_tool("t2", _another_tool, plugin_name="p2")
        tools = registry.get_all_tools()
        assert "t1" in tools
        assert "t2" in tools

    def test_unregister_plugin(self):
        registry = PluginRegistry()
        meta = PluginMeta(name="removable")
        registry.register_tool("my_tool", _sample_tool, meta=meta)
        registry.register_hook("before_tool_call", lambda **kw: None, plugin_name="removable")
        assert "removable" in registry.plugin_names
        assert registry.get_tool("my_tool") is not None

        registry.unregister_plugin("removable")
        assert "removable" not in registry.plugin_names
        assert registry.get_tool("my_tool") is None
        assert registry.get_hooks("before_tool_call") == []

    def test_get_nonexistent(self):
        registry = PluginRegistry()
        assert registry.get_tool("nope") is None
        assert registry.get_provider("nope") is None
        assert registry.get_plugin("nope") is None
        assert registry.get_hooks("nonexistent") == []

    def test_discover_from_directory(self, tmp_path):
        # Create a plugin file
        plugin_code = """
from xerxes.extensions.plugins import PluginMeta, PluginType

PLUGIN_META = PluginMeta(name="discovered_plugin", version="1.0", plugin_type=PluginType.TOOL)

def my_discovered_tool(x: str) -> str:
    return f"discovered: {x}"

def register(registry):
    registry.register_tool("discovered_tool", my_discovered_tool, meta=PLUGIN_META)
"""
        (tmp_path / "my_plugin.py").write_text(plugin_code)

        registry = PluginRegistry()
        discovered = registry.discover(tmp_path)
        assert "discovered_plugin" in discovered
        assert registry.get_tool("discovered_tool") is not None

    def test_discover_nonexistent_dir(self):
        registry = PluginRegistry()
        discovered = registry.discover("/nonexistent")
        assert discovered == []

    def test_discover_skips_underscore_files(self, tmp_path):
        (tmp_path / "_private.py").write_text("def register(registry): pass")
        registry = PluginRegistry()
        discovered = registry.discover(tmp_path)
        assert discovered == []


class TestPluginDependencies:
    """Tests for plugin dependency validation and load ordering."""

    def test_version_constraints_satisfied(self):
        registry = PluginRegistry()
        registry.register_plugin(PluginMeta(name="core", version="1.5.0"))
        registry.register_plugin(
            PluginMeta(
                name="extension",
                version="1.0.0",
                version_constraints={"core": ">=1.0,<2.0"},
            )
        )
        errors = registry.validate_dependencies()
        assert errors == []

    def test_missing_dependency(self):
        registry = PluginRegistry()
        registry.register_plugin(
            PluginMeta(
                name="extension",
                version="1.0.0",
                dependencies=["core"],
            )
        )
        errors = registry.validate_dependencies()
        assert len(errors) == 1
        assert "core" in errors[0]
        assert "missing" in errors[0]

    def test_version_conflict(self):
        registry = PluginRegistry()
        registry.register_plugin(PluginMeta(name="core", version="0.5.0"))
        registry.register_plugin(
            PluginMeta(
                name="extension",
                version="1.0.0",
                version_constraints={"core": ">=1.0"},
            )
        )
        errors = registry.validate_dependencies()
        assert len(errors) == 1
        assert "conflict" in errors[0].lower() or "core" in errors[0]

    def test_duplicate_plugin_registration_conflict(self):
        registry = PluginRegistry()
        registry.register_plugin(PluginMeta(name="core", version="1.0.0"))
        with pytest.raises(PluginConflictError):
            registry.register_plugin(PluginMeta(name="core", version="2.0.0"))

    def test_validate_valid_dependency_graph(self):
        registry = PluginRegistry()
        registry.register_plugin(PluginMeta(name="base", version="1.0.0"))
        registry.register_plugin(PluginMeta(name="mid", version="1.0.0", dependencies=["base"]))
        registry.register_plugin(PluginMeta(name="top", version="1.0.0", dependencies=["mid"]))
        errors = registry.validate_dependencies()
        assert errors == []

    def test_validate_missing_deps(self):
        registry = PluginRegistry()
        registry.register_plugin(PluginMeta(name="lonely", version="1.0.0", dependencies=["ghost"]))
        errors = registry.validate_dependencies()
        assert any("ghost" in e for e in errors)

    def test_get_load_order_deterministic(self):
        registry = PluginRegistry()
        registry.register_plugin(PluginMeta(name="base", version="1.0.0"))
        registry.register_plugin(PluginMeta(name="mid", version="1.0.0", dependencies=["base"]))
        registry.register_plugin(PluginMeta(name="top", version="1.0.0", dependencies=["mid"]))
        order = registry.get_load_order()
        assert order.index("base") < order.index("mid")
        assert order.index("mid") < order.index("top")
        # Deterministic
        assert order == registry.get_load_order()

    def test_backward_compat_simple_string_deps(self):
        """Plugins with simple string dependencies (no version) still work."""
        registry = PluginRegistry()
        registry.register_plugin(PluginMeta(name="dep_a", version="1.0.0"))
        registry.register_plugin(PluginMeta(name="consumer", version="1.0.0", dependencies=["dep_a"]))
        errors = registry.validate_dependencies()
        assert errors == []

    def test_backward_compat_string_deps_with_version(self):
        """Legacy dependencies field with version constraint like 'core>=1.0'."""
        registry = PluginRegistry()
        registry.register_plugin(PluginMeta(name="core", version="1.5.0"))
        registry.register_plugin(
            PluginMeta(
                name="consumer",
                version="1.0.0",
                dependencies=["core>=1.0"],
            )
        )
        errors = registry.validate_dependencies()
        assert errors == []

    def test_check_version_conflict(self):
        registry = PluginRegistry()
        registry.register_plugin(
            PluginMeta(
                name="consumer",
                version="1.0.0",
                version_constraints={"core": ">=2.0"},
            )
        )
        conflicts = registry._check_version_conflict("core", "1.0.0")
        assert len(conflicts) == 1

    def test_check_version_no_conflict(self):
        registry = PluginRegistry()
        registry.register_plugin(
            PluginMeta(
                name="consumer",
                version="1.0.0",
                version_constraints={"core": ">=1.0"},
            )
        )
        conflicts = registry._check_version_conflict("core", "2.0.0")
        assert conflicts == []

    def test_get_load_order_no_deps(self):
        registry = PluginRegistry()
        registry.register_plugin(PluginMeta(name="a", version="1.0"))
        registry.register_plugin(PluginMeta(name="b", version="1.0"))
        order = registry.get_load_order()
        assert set(order) == {"a", "b"}

    def test_get_load_order_circular_raises(self):
        from xerxes.extensions.dependency import CircularDependencyError

        registry = PluginRegistry()
        registry.register_plugin(PluginMeta(name="a", version="1.0", dependencies=["b"]))
        registry.register_plugin(PluginMeta(name="b", version="1.0", dependencies=["a"]))
        with pytest.raises(CircularDependencyError):
            registry.get_load_order()
