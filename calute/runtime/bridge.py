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


"""Bridge between the new streaming/runtime infrastructure and existing Calute.

This module connects the new streaming agent loop, execution registry,
query engine, bootstrap, and tool pool to the existing Calute system
(``Calute``, ``FunctionExecutor``, ``AgentFunction``, TUI).

Key functions:

- :func:`build_tool_executor`: Creates a ``tool_executor`` callable that
  bridges to Calute's existing tool system.
- :func:`populate_registry`: Auto-populates the execution registry from
  Calute's tool categories.
- :func:`create_query_engine`: Creates a fully-wired QueryEngine from
  a Calute instance.
- :func:`bootstrap_calute`: Runs the full bootstrap sequence for a
  Calute instance.

Usage::

    from calute.runtime.bridge import populate_registry, build_tool_executor

    registry = populate_registry()
    executor = build_tool_executor(calute_instance, agent)

    from calute.streaming.loop import run
    for event in run("Hello", state, config, system_prompt, tool_executor=executor):
        ...
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def populate_registry(
    registry: Any = None,
    include_web: bool = True,
    include_system: bool = True,
    include_ai: bool = True,
    include_memory: bool = True,
) -> Any:
    """Auto-populate an ExecutionRegistry with all Calute tools.

    Scans ``calute.tools.TOOL_CATEGORIES`` and registers each available
    tool with its category, description, safety flag, and schema.

    Args:
        registry: An :class:`ExecutionRegistry` to populate. If None, creates one.
        include_web: Include web tools (DuckDuckGoSearch, WebScraper, etc.).
        include_system: Include system tools (SystemInfo, ProcessManager, etc.).
        include_ai: Include AI tools (TextEmbedder, TextClassifier, etc.).
        include_memory: Include memory tools.

    Returns:
        The populated :class:`ExecutionRegistry`.
    """
    from calute.runtime.execution_registry import ExecutionRegistry
    from calute.streaming.permissions import SAFE_TOOLS

    if registry is None:
        registry = ExecutionRegistry()

    try:
        import calute.tools as tools_mod
    except ImportError:
        logger.warning("calute.tools not available")
        return registry

    categories = getattr(tools_mod, "TOOL_CATEGORIES", {})

    # Category filter
    skip_categories = set()
    if not include_web:
        skip_categories.add("web")
    if not include_system:
        skip_categories.add("system")
    if not include_ai:
        skip_categories.add("ai")
    if not include_memory:
        skip_categories.add("memory")

    for category, tool_names in categories.items():
        if category in skip_categories:
            continue

        for tool_name in tool_names:
            tool_obj = getattr(tools_mod, tool_name, None)
            if tool_obj is None:
                continue

            # Extract description
            description = ""
            if hasattr(tool_obj, "static_call") and tool_obj.static_call.__doc__:
                description = tool_obj.static_call.__doc__.strip().split("\n")[0]
            elif hasattr(tool_obj, "__doc__") and tool_obj.__doc__:
                description = tool_obj.__doc__.strip().split("\n")[0]

            # Build handler from static_call
            handler = getattr(tool_obj, "static_call", None)

            # Build schema from static_call signature
            schema = _build_tool_schema(tool_name, description, handler)

            is_safe = tool_name in SAFE_TOOLS

            registry.register_tool(
                name=tool_name,
                handler=handler,
                description=description,
                category=category,
                safe=is_safe,
                source_hint=f"calute.tools.{category}",
                schema=schema,
            )

    # Register default slash commands
    for cmd in [
        "help",
        "clear",
        "history",
        "save",
        "load",
        "model",
        "provider",
        "config",
        "cost",
        "context",
        "memory",
        "agents",
        "skills",
        "tools",
        "models",
        "endpoint",
        "apikey",
        "sampling",
        "sessions",
        "profile",
        "power",
        "plans",
    ]:
        registry.register_command(cmd, description=f"/{cmd} command")

    return registry


def _build_tool_schema(name: str, description: str, handler: Any) -> dict[str, Any]:
    """Build an Anthropic-format tool schema from a function's signature."""
    schema: dict[str, Any] = {
        "name": name,
        "description": description or f"Execute {name}",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    if handler is None:
        return schema

    try:
        sig = inspect.signature(handler)
    except (ValueError, TypeError):
        return schema

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "context_variables") or param_name.startswith("_"):
            continue
        if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue

        prop: dict[str, Any] = {}
        annotation = param.annotation

        if annotation is inspect.Parameter.empty or annotation is Any:
            prop["type"] = "string"
        elif annotation is str:
            prop["type"] = "string"
        elif annotation is int:
            prop["type"] = "integer"
        elif annotation is float:
            prop["type"] = "number"
        elif annotation is bool:
            prop["type"] = "boolean"
        elif annotation is list or (hasattr(annotation, "__origin__") and annotation.__origin__ is list):
            prop["type"] = "array"
        else:
            prop["type"] = "string"

        if param.default is not inspect.Parameter.empty:
            if param.default is not None:
                prop["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = prop

    schema["input_schema"]["properties"] = properties
    schema["input_schema"]["required"] = required

    return schema


def build_tool_executor(
    calute_instance: Any = None,
    agent: Any = None,
    registry: Any = None,
) -> Any:
    """Create a tool_executor callable for the streaming agent loop.

    Returns a function ``(tool_name, tool_input) -> result_string`` that
    bridges to Calute's existing tool system.

    The executor tries these strategies in order:
    1. If a registry is provided, execute through the registry.
    2. If a calute_instance and agent are provided, execute through
       the agent's registered functions.
    3. Fall back to calling static_call on the tool class directly.

    Args:
        calute_instance: Optional Calute instance.
        agent: Optional Agent with registered functions.
        registry: Optional ExecutionRegistry.

    Returns:
        A callable ``(tool_name: str, tool_input: dict) -> str``.
    """

    def executor(tool_name: str, tool_input: dict[str, Any]) -> str:
        # Strategy 1: Registry
        if registry is not None:
            result = registry.execute_tool(tool_name, tool_input)
            if result.handled:
                return result.result
            if result.error:
                return f"Error: {result.error}"

        # Strategy 2: Agent functions
        if agent is not None:
            for func in getattr(agent, "functions", []):
                func_name = getattr(func, "name", "") or getattr(func, "__name__", "")
                if func_name == tool_name:
                    callable_fn = getattr(func, "static_call", None) or getattr(func, "callable_func", None) or func
                    try:
                        result = callable_fn(**tool_input)
                        return str(result) if result is not None else ""
                    except Exception as e:
                        return f"Error executing {tool_name}: {e}"

        # Strategy 3: Direct import from calute.tools
        try:
            import calute.tools as tools_mod

            tool_cls = getattr(tools_mod, tool_name, None)
            if tool_cls and hasattr(tool_cls, "static_call"):
                result = tool_cls.static_call(**tool_input)
                return str(result) if result is not None else ""
        except Exception as e:
            return f"Error: {e}"

        return f"Unknown tool: {tool_name}"

    return executor


def create_query_engine(
    calute_instance: Any = None,
    agent: Any = None,
    model: str = "",
    system_prompt: str = "",
    **config_kwargs: Any,
) -> Any:
    """Create a fully-wired QueryEngine from a Calute instance.

    Args:
        calute_instance: Optional Calute instance for tool execution.
        agent: Optional Agent for tool execution.
        model: Model name.
        system_prompt: System prompt.
        **config_kwargs: Additional QueryEngineConfig kwargs.

    Returns:
        A :class:`QueryEngine` with tool executor and registry wired up.
    """
    from calute.runtime.query_engine import QueryEngine

    registry = populate_registry()
    tool_executor = build_tool_executor(
        calute_instance=calute_instance,
        agent=agent,
        registry=registry,
    )
    tool_schemas = registry.tool_schemas()

    engine = QueryEngine.create(
        model=model,
        system_prompt=system_prompt,
        registry=registry,
        **config_kwargs,
    )

    # Bind the tool_executor and schemas as defaults
    engine._default_tool_executor = tool_executor
    engine._default_tool_schemas = tool_schemas

    return engine


def bootstrap_calute(
    calute_instance: Any = None,
    agent: Any = None,
    model: str = "",
    extra_context: str = "",
) -> Any:
    """Run the full bootstrap sequence for a Calute instance.

    Performs environment detection, git info, CLAUDE.md loading,
    tool registration, and system prompt building.

    Args:
        calute_instance: Optional Calute instance.
        agent: Optional Agent with tools.
        model: Model name.
        extra_context: Additional system prompt context.

    Returns:
        A :class:`BootstrapResult`.
    """
    from calute.runtime.bootstrap import bootstrap

    tools = getattr(agent, "functions", []) if agent else []

    result = bootstrap(
        model=model,
        tools=tools,
        extra_context=extra_context,
    )

    # Also populate with all Calute tools
    populate_registry(result.registry)

    return result


__all__ = [
    "bootstrap_calute",
    "build_tool_executor",
    "create_query_engine",
    "populate_registry",
]
