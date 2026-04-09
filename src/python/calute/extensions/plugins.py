# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Plugin system for Calute.

Supports four plugin types:

- **tool**: Provides callable tool functions for agents.
- **hook**: Provides lifecycle hooks (before/after tool, bootstrap, etc.).
- **provider**: Provides LLM provider implementations.
- **channel**: (Future) Provides communication channel integrations.

Plugins are Python modules or classes that register themselves with the
PluginRegistry. Local plugin discovery scans a configured directory for
modules containing a ``register(registry)`` function.

Example plugin module (``my_plugin.py``)::

    from calute.extensions.plugins import PluginMeta, PluginType

    PLUGIN_META = PluginMeta(
        name="my_plugin",
        version="1.0.0",
        plugin_type=PluginType.TOOL,
        description="My custom tools",
    )

    def my_tool(query: str) -> str:
        '''Search for something.'''
        return f"Results for {query}"

    def register(registry):
        registry.register_tool("my_tool", my_tool, meta=PLUGIN_META)
"""

from __future__ import annotations

import importlib
import logging
import sys
import typing as tp
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Enumeration of plugin categories supported by the Calute plugin system.

    Each plugin declares its type via :class:`PluginMeta`, and the
    :class:`PluginRegistry` uses the type to route registrations to the
    correct internal storage.

    Attributes:
        TOOL: Provides callable tool functions for agents.
        HOOK: Provides lifecycle hooks (before/after tool, bootstrap, etc.).
        PROVIDER: Provides LLM provider implementations.
        CHANNEL: Communication channel integrations (reserved for future use).
        SEARCH: Search-engine integrations.
        SPEECH: Speech/TTS integrations.
    """

    TOOL = "tool"
    HOOK = "hook"
    PROVIDER = "provider"
    CHANNEL = "channel"
    SEARCH = "search"
    SPEECH = "speech"


@dataclass
class PluginMeta:
    """Metadata describing a Calute plugin.

    Every plugin must provide a ``PluginMeta`` instance when it registers
    itself.  The metadata is used for dependency resolution, conflict
    detection, and informational logging.

    Attributes:
        name: A unique identifier for the plugin (e.g., ``"my_search_plugin"``).
        version: A semver-style version string for the plugin (default ``"0.1.0"``).
        plugin_type: The category of functionality the plugin provides.
        description: A short human-readable description of the plugin.
        author: The name or handle of the plugin author.
        dependencies: A list of dependency strings (plugin names, optionally
            with version constraints like ``"other_plugin>=1.0"``).
        version_constraints: A mapping of plugin names to version constraint
            expressions (e.g., ``{"core": ">=2.0,<3.0"}``).  These are
            checked in addition to *dependencies*.
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
    """A plugin that has been registered with the :class:`PluginRegistry`.

    Aggregates the plugin metadata together with the tools, hooks, and
    provider that the plugin contributed during registration.

    Attributes:
        meta: The metadata describing the plugin.
        tools: A mapping of tool names to their callable implementations
            registered by this plugin.
        hooks: A mapping of hook point names to their callable
            implementations registered by this plugin.
        provider: An optional LLM provider instance registered by this
            plugin.  ``None`` if the plugin does not provide a provider.
    """

    meta: PluginMeta
    tools: dict[str, tp.Callable] = field(default_factory=dict)
    hooks: dict[str, tp.Callable] = field(default_factory=dict)
    provider: tp.Any = None


class PluginConflictError(Exception):
    """Raised when a plugin or resource name conflicts with an existing registration.

    Attributes:
        name: The name of the plugin or resource that caused the conflict.
        existing: The name of the already-registered plugin that owns the
            conflicting resource.
    """

    def __init__(self, name: str, existing: str) -> None:
        """Initialize with the conflicting and existing names.

        Args:
            name: The identifier of the new plugin or resource (may be
                prefixed with ``"tool:"`` or ``"provider:"`` for sub-resource
                conflicts).
            existing: The identifier of the existing plugin that already
                occupies the conflicting slot.
        """
        self.name = name
        self.existing = existing
        super().__init__(f"Plugin '{name}' conflicts with existing plugin '{existing}'")


class PluginRegistry:
    """Central registry for plugin management, discovery, and dependency validation.

    Plugins register tools, hooks, and providers through this registry.  The
    registry tracks ownership so that all resources belonging to a plugin can
    be removed atomically via :meth:`unregister_plugin`.

    Attributes:
        _plugins: Internal mapping of plugin name to :class:`RegisteredPlugin`.
        _tools: Mapping of tool name to ``(callable, plugin_name)`` tuple.
        _hooks: Mapping of hook name to list of ``(callable, plugin_name)``
            tuples, in registration order.
        _providers: Mapping of provider name to ``(provider, plugin_name)``
            tuple.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register_tool("my_tool", my_func, meta=PluginMeta(name="test"))
        >>> func = registry.get_tool("my_tool")
    """

    def __init__(self) -> None:
        """Initialize the PluginRegistry with empty internal storage for plugins, tools, hooks, and providers."""
        self._plugins: dict[str, RegisteredPlugin] = {}
        self._tools: dict[str, tuple[tp.Callable, str]] = {}
        self._hooks: dict[str, list[tuple[tp.Callable, str]]] = {}
        self._providers: dict[str, tuple[tp.Any, str]] = {}

    @property
    def plugin_names(self) -> list[str]:
        """Return the names of all registered plugins.

        Returns:
            A list of plugin name strings in insertion order.
        """
        return list(self._plugins.keys())

    def register_plugin(self, meta: PluginMeta) -> RegisteredPlugin:
        """Register a plugin by its metadata.

        Creates a new :class:`RegisteredPlugin` entry and stores it in the
        internal registry.  Duplicate names are not allowed.

        Args:
            meta: The :class:`PluginMeta` describing the plugin.

        Returns:
            The newly created :class:`RegisteredPlugin` handle.

        Raises:
            PluginConflictError: If a plugin with the same name is already
                registered.
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
            tool_name: Name for the tool (used for policy checks and LLM schema).
            func: The callable tool function.
            meta: Optional plugin metadata (auto-registers plugin if not already).
            plugin_name: Name of the owning plugin (inferred from meta if provided).
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
        """Register a hook callback function.

        Multiple hooks can be registered for the same hook point; they are
        stored in registration order.  If *meta* is provided and the owning
        plugin is not yet registered, it is auto-registered first.

        Args:
            hook_name: The hook point name (e.g., ``"before_tool_call"``).
            func: The callable to invoke at this hook point.
            meta: Optional plugin metadata.  If provided and the plugin
                is not yet registered, :meth:`register_plugin` is called
                automatically.
            plugin_name: Explicit plugin name override.  If omitted, the
                name is inferred from *meta* or defaults to
                ``"__standalone__"``.
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
        """Register an LLM provider implementation.

        Only one provider can be registered per *provider_name*.  If *meta*
        is provided and the owning plugin is not yet registered, it is
        auto-registered first.

        Args:
            provider_name: A unique name for the provider (e.g., ``"openai"``).
            provider: The provider instance or factory.
            meta: Optional plugin metadata for auto-registration.
            plugin_name: Explicit plugin name override.

        Raises:
            PluginConflictError: If a provider with the same name is already
                registered.
        """
        pname = plugin_name or (meta.name if meta else "__standalone__")
        if provider_name in self._providers:
            raise PluginConflictError(f"provider:{provider_name}", self._providers[provider_name][1])

        if meta and pname not in self._plugins:
            self.register_plugin(meta)

        self._providers[provider_name] = (provider, pname)
        if pname in self._plugins:
            self._plugins[pname].provider = provider

    def get_tool(self, tool_name: str) -> tp.Callable | None:
        """Look up a registered tool by name.

        Args:
            tool_name: The name the tool was registered under.

        Returns:
            The callable tool function, or ``None`` if no tool with the
            given name is registered.
        """
        entry = self._tools.get(tool_name)
        return entry[0] if entry else None

    def get_all_tools(self) -> dict[str, tp.Callable]:
        """Return a mapping of tool name to callable for all registered tools.

        Returns:
            A dictionary mapping each tool name to its callable
            implementation.  The result is a new dict; mutating it does
            not affect the registry.
        """
        return {name: func for name, (func, _) in self._tools.items()}

    def get_hooks(self, hook_name: str) -> list[tp.Callable]:
        """Get all registered hook callbacks for a hook point.

        Args:
            hook_name: The hook point name to query.

        Returns:
            A list of callable hooks in registration order.  Returns an
            empty list if no hooks are registered for the given name.
        """
        return [func for func, _ in self._hooks.get(hook_name, [])]

    def get_provider(self, provider_name: str) -> tp.Any | None:
        """Look up a registered LLM provider by name.

        Args:
            provider_name: The name the provider was registered under.

        Returns:
            The provider instance, or ``None`` if not found.
        """
        entry = self._providers.get(provider_name)
        return entry[0] if entry else None

    def get_plugin(self, name: str) -> RegisteredPlugin | None:
        """Look up a registered plugin by name.

        Args:
            name: The plugin name to look up.

        Returns:
            The :class:`RegisteredPlugin` instance, or ``None`` if no plugin
            with the given name exists.
        """
        return self._plugins.get(name)

    def unregister_plugin(self, name: str) -> None:
        """Remove a plugin and all its associated tools, hooks, and providers.

        If the plugin is not found, this method is a no-op.

        Args:
            name: The name of the plugin to remove.
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
        logger.info("Unregistered plugin: %s", name)

    def validate_dependencies(self) -> list[str]:
        """Validate that all registered plugins have their dependencies met.

        Returns a list of error messages (empty if all dependencies are satisfied).
        """
        from calute.extensions.dependency import DependencyResolver, parse_dependency

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
        """Return plugin names in dependency-safe order (dependencies first).

        Raises:
            calute.extensions.dependency.CircularDependencyError: If circular deps exist.
        """
        from calute.extensions.dependency import DependencyResolver, parse_dependency

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
        """Check if registering a plugin at *version* would violate existing constraints.

        Iterates over all currently registered plugins and checks whether any
        of their declared ``version_constraints`` or ``dependencies`` would
        reject the proposed *version* of *name*.

        Args:
            name: The name of the plugin that would be registered.
            version: The version string of the proposed plugin.

        Returns:
            A list of human-readable conflict descriptions.  An empty list
            indicates no conflicts.
        """
        from calute.extensions.dependency import VersionConstraint

        conflicts: list[str] = []
        for pname, plugin in self._plugins.items():
            if name in plugin.meta.version_constraints:
                constraint = plugin.meta.version_constraints[name]
                vc = VersionConstraint(constraint)
                if not vc.satisfies(version):
                    conflicts.append(
                        f"Plugin '{pname}' requires {name}{constraint}, but version {version} would be registered"
                    )
            from calute.extensions.dependency import parse_dependency

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
        """Discover and load plugins from a directory.

        Scans the given directory for top-level Python files (ignoring those
        starting with ``_``).  Each file is loaded as a temporary module; if
        it exposes a ``register(registry)`` function, that function is called
        with *self* to let the plugin register its resources.

        Modules are removed from ``sys.modules`` after loading to avoid
        polluting the global namespace.

        Args:
            directory: Path to the directory to scan for plugin modules.

        Returns:
            A list of plugin names that were newly registered during
            discovery.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning("Plugin directory not found: %s", dir_path)
            return []

        discovered = []
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = f"calute_plugin_{py_file.stem}"
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
