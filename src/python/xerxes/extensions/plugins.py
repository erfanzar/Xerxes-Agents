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
"""Plugin registry for tools, hooks, providers, and channels.

``PluginRegistry`` discovers, loads, and tracks plugins. Each plugin is
represented by a ``PluginMeta`` and a ``RegisteredPlugin`` that aggregates its
exported capabilities.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import typing as tp
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Categorisation of plugin capabilities."""

    TOOL = "tool"
    HOOK = "hook"
    PROVIDER = "provider"
    CHANNEL = "channel"
    SEARCH = "search"
    SPEECH = "speech"


@dataclass
class PluginMeta:
    """Metadata describing a plugin package.

    Attributes:
        name (str): IN: Unique plugin identifier. OUT: Used as registry key.
        version (str): IN: Semver string. OUT: Used for dependency checks.
        plugin_type (PluginType): IN: Category. OUT: Logged on registration.
        description (str): IN: Human-readable summary. OUT: Optional metadata.
        author (str): IN: Author name. OUT: Optional metadata.
        dependencies (list[str]): IN: Dependency strings. OUT: Parsed during
            validation.
        version_constraints (dict[str, str]): IN: Name-to-constraint map.
            OUT: Parsed during validation.
    """

    name: str
    version: str = "0.1.0"
    plugin_type: PluginType = PluginType.TOOL
    description: str = ""
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    version_constraints: dict[str, str] = field(default_factory=dict)


@dataclass
class RegisteredPlugin:
    """Runtime record of a loaded plugin and its exports.

    Attributes:
        meta (PluginMeta): IN: Plugin metadata. OUT: Stored for introspection.
        tools (dict[str, tp.Callable]): IN: Exported tool functions. OUT:
            Looked up by ``get_tool``.
        hooks (dict[str, tp.Callable]): IN: Exported hook functions. OUT:
            Aggregated by ``get_hooks``.
        provider (tp.Any): IN: Exported provider instance. OUT: Looked up by
            ``get_provider``.
        channels (dict[str, tp.Any]): IN: Exported channel objects. OUT:
            Looked up by ``get_channel``.
    """

    meta: PluginMeta
    tools: dict[str, tp.Callable] = field(default_factory=dict)
    hooks: dict[str, tp.Callable] = field(default_factory=dict)
    provider: tp.Any = None
    channels: dict[str, tp.Any] = field(default_factory=dict)


class PluginConflictError(Exception):
    """Raised when registering a plugin or resource that already exists.

    Args:
        name (str): IN: Conflicting resource name. OUT: Stored.
        existing (str): IN: Name of the existing owner. OUT: Stored.
    """

    def __init__(self, name: str, existing: str) -> None:
        """Initialize with conflict details.

        Args:
            name (str): IN: New resource name. OUT: Stored.
            existing (str): IN: Existing resource owner. OUT: Stored.
        """

        self.name = name
        self.existing = existing
        super().__init__(f"Plugin '{name}' conflicts with existing plugin '{existing}'")


class PluginRegistry:
    """Central registry for plugin metadata and exported capabilities."""

    def __init__(self) -> None:
        """Initialize empty internal indexes."""

        self._plugins: dict[str, RegisteredPlugin] = {}
        self._tools: dict[str, tuple[tp.Callable, str]] = {}
        self._hooks: dict[str, list[tuple[tp.Callable, str]]] = {}
        self._providers: dict[str, tuple[tp.Any, str]] = {}
        self._channels: dict[str, tuple[tp.Any, str]] = {}

    @property
    def plugin_names(self) -> list[str]:
        """List names of all registered plugins.

        Returns:
            list[str]: OUT: Snapshot of ``self._plugins`` keys.
        """

        return list(self._plugins.keys())

    def register_plugin(self, meta: PluginMeta) -> RegisteredPlugin:
        """Create a new plugin entry in the registry.

        Args:
            meta (PluginMeta): IN: Metadata for the plugin. OUT: Stored and
                used as the registry key.

        Returns:
            RegisteredPlugin: OUT: Newly created entry.

        Raises:
            PluginConflictError: OUT: If a plugin with the same name already
                exists.
        """

        if meta.name in self._plugins:
            raise PluginConflictError(meta.name, meta.name)
        plugin = RegisteredPlugin(meta=meta)
        self._plugins[meta.name] = plugin
        logger.info("Registered plugin: %s v%s (%s)", meta.name, meta.version, meta.plugin_type.value)
        return plugin

    def register_tool(
        self,
        tool_name: str,
        func: tp.Callable,
        meta: PluginMeta | None = None,
        plugin_name: str | None = None,
    ) -> None:
        """Register a tool function.

        Args:
            tool_name (str): IN: Unique tool identifier. OUT: Used as lookup
                key.
            func (tp.Callable): IN: Tool implementation. OUT: Stored.
            meta (PluginMeta | None): IN: Plugin metadata (auto-registers the
                plugin if needed). OUT: Used to derive ``plugin_name``.
            plugin_name (str | None): IN: Explicit plugin owner. OUT:
                Defaults to ``meta.name`` or ``"__standalone__"``.

        Raises:
            PluginConflictError: OUT: If ``tool_name`` is already registered.
        """

        pname = plugin_name or (meta.name if meta else "__standalone__")
        if tool_name in self._tools:
            existing_plugin = self._tools[tool_name][1]
            raise PluginConflictError(f"tool:{tool_name}", existing_plugin)

        if meta and pname not in self._plugins:
            self.register_plugin(meta)

        self._tools[tool_name] = (func, pname)
        if pname in self._plugins:
            self._plugins[pname].tools[tool_name] = func

        logger.debug("Registered tool '%s' from plugin '%s'", tool_name, pname)

    def register_hook(
        self,
        hook_name: str,
        func: tp.Callable,
        meta: PluginMeta | None = None,
        plugin_name: str | None = None,
    ) -> None:
        """Register a hook callback.

        Args:
            hook_name (str): IN: Hook point name. OUT: Used as key.
            func (tp.Callable): IN: Callback implementation. OUT: Stored.
            meta (PluginMeta | None): IN: Plugin metadata. OUT: Auto-registers
                plugin if absent.
            plugin_name (str | None): IN: Explicit plugin owner. OUT:
                Defaults to ``meta.name`` or ``"__standalone__"``.
        """

        pname = plugin_name or (meta.name if meta else "__standalone__")
        if meta and pname not in self._plugins:
            self.register_plugin(meta)

        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append((func, pname))
        if pname in self._plugins:
            self._plugins[pname].hooks[hook_name] = func

        logger.debug("Registered hook '%s' from plugin '%s'", hook_name, pname)

    def register_provider(
        self,
        provider_name: str,
        provider: tp.Any,
        meta: PluginMeta | None = None,
        plugin_name: str | None = None,
    ) -> None:
        """Register a provider object.

        Args:
            provider_name (str): IN: Unique provider identifier. OUT: Lookup
                key.
            provider (tp.Any): IN: Provider instance. OUT: Stored.
            meta (PluginMeta | None): IN: Plugin metadata. OUT: Auto-registers
                plugin if absent.
            plugin_name (str | None): IN: Explicit plugin owner. OUT:
                Defaults to ``meta.name`` or ``"__standalone__"``.

        Raises:
            PluginConflictError: OUT: If ``provider_name`` already exists.
        """

        pname = plugin_name or (meta.name if meta else "__standalone__")
        if provider_name in self._providers:
            raise PluginConflictError(f"provider:{provider_name}", self._providers[provider_name][1])

        if meta and pname not in self._plugins:
            self.register_plugin(meta)

        self._providers[provider_name] = (provider, pname)
        if pname in self._plugins:
            self._plugins[pname].provider = provider

    def register_channel(
        self,
        channel_name: str,
        channel: tp.Any,
        meta: PluginMeta | None = None,
        plugin_name: str | None = None,
    ) -> None:
        """Register a channel object.

        Args:
            channel_name (str): IN: Unique channel identifier. OUT: Lookup key.
            channel (tp.Any): IN: Channel instance. OUT: Stored.
            meta (PluginMeta | None): IN: Plugin metadata. OUT: Auto-registers
                plugin if absent.
            plugin_name (str | None): IN: Explicit plugin owner. OUT:
                Defaults to ``meta.name`` or ``"__standalone__"``.

        Raises:
            PluginConflictError: OUT: If ``channel_name`` already exists.
        """

        pname = plugin_name or (meta.name if meta else "__standalone__")
        if channel_name in self._channels:
            raise PluginConflictError(f"channel:{channel_name}", self._channels[channel_name][1])

        if meta and pname not in self._plugins:
            self.register_plugin(meta)

        self._channels[channel_name] = (channel, pname)
        if pname in self._plugins:
            self._plugins[pname].channels[channel_name] = channel

        logger.debug("Registered channel '%s' from plugin '%s'", channel_name, pname)

    def get_channel(self, channel_name: str) -> tp.Any | None:
        """Retrieve a registered channel by name.

        Args:
            channel_name (str): IN: Channel identifier. OUT: Looked up in the
                registry.

        Returns:
            tp.Any | None: OUT: Channel instance or ``None``.
        """

        entry = self._channels.get(channel_name)
        return entry[0] if entry else None

    def get_all_channels(self) -> dict[str, tp.Any]:
        """Return all registered channels.

        Returns:
            dict[str, tp.Any]: OUT: Mapping from channel name to instance.
        """

        return {name: chan for name, (chan, _) in self._channels.items()}

    def get_tool(self, tool_name: str) -> tp.Callable | None:
        """Retrieve a registered tool by name.

        Args:
            tool_name (str): IN: Tool identifier. OUT: Looked up in the
                registry.

        Returns:
            tp.Callable | None: OUT: Tool function or ``None``.
        """

        entry = self._tools.get(tool_name)
        return entry[0] if entry else None

    def get_all_tools(self) -> dict[str, tp.Callable]:
        """Return all registered tools.

        Returns:
            dict[str, tp.Callable]: OUT: Mapping from tool name to function.
        """

        return {name: func for name, (func, _) in self._tools.items()}

    def get_hooks(self, hook_name: str) -> list[tp.Callable]:
        """Return callbacks for a hook point.

        Args:
            hook_name (str): IN: Hook point identifier. OUT: Looked up in the
                registry.

        Returns:
            list[tp.Callable]: OUT: List of registered callbacks.
        """

        return [func for func, _ in self._hooks.get(hook_name, [])]

    def get_provider(self, provider_name: str) -> tp.Any | None:
        """Retrieve a registered provider by name.

        Args:
            provider_name (str): IN: Provider identifier. OUT: Looked up in
                the registry.

        Returns:
            tp.Any | None: OUT: Provider instance or ``None``.
        """

        entry = self._providers.get(provider_name)
        return entry[0] if entry else None

    def get_plugin(self, name: str) -> RegisteredPlugin | None:
        """Retrieve a plugin record by name.

        Args:
            name (str): IN: Plugin identifier. OUT: Looked up in the registry.

        Returns:
            RegisteredPlugin | None: OUT: Plugin record or ``None``.
        """

        return self._plugins.get(name)

    def unregister_plugin(self, name: str) -> None:
        """Remove a plugin and all its exported resources.

        Args:
            name (str): IN: Plugin identifier. OUT: Used to purge entries from
                all internal indexes.

        Returns:
            None: OUT: Plugin and its resources are removed.
        """

        plugin = self._plugins.pop(name, None)
        if not plugin:
            return

        self._tools = {k: v for k, v in self._tools.items() if v[1] != name}
        for hook_name in list(self._hooks.keys()):
            self._hooks[hook_name] = [(f, p) for f, p in self._hooks[hook_name] if p != name]
            if not self._hooks[hook_name]:
                del self._hooks[hook_name]
        self._providers = {k: v for k, v in self._providers.items() if v[1] != name}
        self._channels = {k: v for k, v in self._channels.items() if v[1] != name}
        logger.info("Unregistered plugin: %s", name)

    def validate_dependencies(self) -> list[str]:
        """Validate every plugin's declared dependencies.

        Returns:
            list[str]: OUT: Human-readable error strings for missing or
            conflicting dependencies.
        """

        from xerxes.extensions.dependency import DependencyResolver, parse_dependency

        errors: list[str] = []
        available = {name: p.meta.version for name, p in self._plugins.items()}
        resolver = DependencyResolver()

        for name, plugin in self._plugins.items():
            reqs = []
            for dep_str in plugin.meta.dependencies:
                reqs.append(parse_dependency(dep_str))
            for dep_name, constraint in plugin.meta.version_constraints.items():
                reqs.append(parse_dependency(f"{dep_name}{constraint}"))

            result = resolver.resolve(available, reqs)
            for m in result.missing:
                errors.append(f"Plugin '{name}' requires missing dependency '{m}'")
            for c in result.conflicts:
                errors.append(f"Plugin '{name}' has version conflict: {c}")

        return errors

    def get_load_order(self) -> list[str]:
        """Return plugins in dependency-respecting load order.

        Returns:
            list[str]: OUT: Topologically sorted plugin names.
        """

        from xerxes.extensions.dependency import DependencyResolver, parse_dependency

        resolver = DependencyResolver()
        graph: dict[str, list[str]] = {}
        for name, plugin in self._plugins.items():
            deps: list[str] = []
            for dep_str in plugin.meta.dependencies:
                spec = parse_dependency(dep_str)
                deps.append(spec.name)
            for dep_name in plugin.meta.version_constraints:
                if dep_name not in deps:
                    deps.append(dep_name)
            graph[name] = deps

        return resolver.topological_sort(graph)

    def _check_version_conflict(self, name: str, version: str) -> list[str]:
        """Check whether adding ``name`` at ``version`` would conflict.

        Args:
            name (str): IN: Plugin / package name. OUT: Checked against
                constraints in existing plugins.
            version (str): IN: Proposed version. OUT: Compared with
                ``VersionConstraint``.

        Returns:
            list[str]: OUT: Human-readable conflict messages.
        """

        from xerxes.extensions.dependency import VersionConstraint

        conflicts: list[str] = []
        for pname, plugin in self._plugins.items():
            if name in plugin.meta.version_constraints:
                constraint = plugin.meta.version_constraints[name]
                vc = VersionConstraint(constraint)
                if not vc.satisfies(version):
                    conflicts.append(
                        f"Plugin '{pname}' requires {name}{constraint}, but version {version} would be registered"
                    )
            from xerxes.extensions.dependency import parse_dependency

            for dep_str in plugin.meta.dependencies:
                spec = parse_dependency(dep_str)
                if spec.name == name and spec.version_constraint:
                    vc = VersionConstraint(spec.version_constraint)
                    if not vc.satisfies(version):
                        conflicts.append(
                            f"Plugin '{pname}' requires {dep_str}, but version {version} would be registered"
                        )
        return conflicts

    def discover(self, directory: str | Path) -> list[str]:
        """Load ``*.py`` plugins from a directory.

        Each file is executed as a module; if it exposes a ``register``
        function it is called with ``self``.

        Args:
            directory (str | Path): IN: Directory to scan. OUT: Enumerated for
                ``*.py`` files.

        Returns:
            list[str]: OUT: Names of newly registered plugins.
        """

        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning("Plugin directory not found: %s", dir_path)
            return []

        discovered: list[tp.Any] = []
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = f"xerxes_plugin_{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    if hasattr(module, "register"):
                        before = set(self._plugins.keys())
                        module.register(self)
                        after = set(self._plugins.keys())
                        new_plugins = after - before
                        discovered.extend(new_plugins)
                        logger.info("Loaded plugin from %s: %s", py_file, new_plugins or "(no new plugins)")
            except Exception:
                logger.warning("Failed to load plugin from %s", py_file, exc_info=True)
            finally:
                sys.modules.pop(module_name, None)

        return discovered
