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
"""Bridge module for Xerxes.

Exports:
    - logger
    - populate_registry
    - build_tool_executor
    - create_query_engine
    - bootstrap_xerxes"""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _runtime_context(
    xerxes_instance: Any = None,
    agent: Any = None,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Internal helper to runtime context.

    Args:
        xerxes_instance (Any, optional): IN: xerxes instance. Defaults to None. OUT: Consumed during execution.
        agent (Any, optional): IN: agent. Defaults to None. OUT: Consumed during execution.
        extra_context (dict[str, Any] | None, optional): IN: extra context. Defaults to None. OUT: Consumed during execution.
    Returns:
        dict[str, Any]: OUT: Result of the operation."""

    context = dict(extra_context or {})
    if agent is not None:
        agent_id = getattr(agent, "id", None)
        if agent_id:
            context.setdefault("agent_id", agent_id)
    if xerxes_instance is not None:
        memory_store = getattr(xerxes_instance, "memory_store", None)
        if memory_store is not None:
            context.setdefault("memory_store", memory_store)
    return context


def _call_tool_handler(
    handler: Any,
    tool_input: dict[str, Any],
    *,
    xerxes_instance: Any = None,
    agent: Any = None,
) -> Any:
    """Internal helper to call tool handler.

    Args:
        handler (Any): IN: handler. OUT: Consumed during execution.
        tool_input (dict[str, Any]): IN: tool input. OUT: Consumed during execution.
        xerxes_instance (Any, optional): IN: xerxes instance. Defaults to None. OUT: Consumed during execution.
        agent (Any, optional): IN: agent. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    call_kwargs = dict(tool_input)
    context = _runtime_context(
        xerxes_instance=xerxes_instance,
        agent=agent,
        extra_context=call_kwargs.pop("context_variables", None),
    )

    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        params = signature.parameters
        accepts_context = "context_variables" in params
        accepts_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())

        if accepts_context:
            call_kwargs["context_variables"] = context
        elif accepts_kwargs and context:
            for key, value in context.items():
                call_kwargs.setdefault(key, value)

        call_kwargs = _coerce_argument_types(call_kwargs, signature)

    return handler(**call_kwargs)


def _coerce_argument_types(
    arguments: dict[str, Any],
    signature: inspect.Signature,
) -> dict[str, Any]:
    """Internal helper to coerce argument types.

    Args:
        arguments (dict[str, Any]): IN: arguments. OUT: Consumed during execution.
        signature (inspect.Signature): IN: signature. OUT: Consumed during execution.
    Returns:
        dict[str, Any]: OUT: Result of the operation."""

    import json
    import typing

    coerced = dict(arguments)
    for param_name, param in signature.parameters.items():
        if param_name not in coerced:
            continue
        value = coerced[param_name]
        if not isinstance(value, str):
            continue
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            continue

        origin = getattr(ann, "__origin__", None)
        args_tuple = getattr(ann, "__args__", ())
        is_union = origin is typing.Union or type(ann).__name__ == "UnionType"
        if is_union and any(a is type(None) for a in args_tuple):
            ann = next(a for a in args_tuple if a is not type(None))
            origin = getattr(ann, "__origin__", None)
            args_tuple = getattr(ann, "__args__", ())

        if origin is list and len(args_tuple) == 1:
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    coerced[param_name] = parsed
            except Exception:
                pass
            continue
        if origin is dict and len(args_tuple) == 2:
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    coerced[param_name] = parsed
            except Exception:
                pass
            continue

        if ann is int:
            try:
                coerced[param_name] = int(value)
            except ValueError:
                pass
        elif ann is float:
            try:
                coerced[param_name] = float(value)
            except ValueError:
                pass
        elif ann is bool:
            coerced[param_name] = value.lower() in ("true", "1", "yes", "on")

    return coerced


def populate_registry(
    registry: Any = None,
    include_web: bool = True,
    include_system: bool = True,
    include_ai: bool = True,
    include_memory: bool = True,
) -> Any:
    """Populate registry.

    Args:
        registry (Any, optional): IN: registry. Defaults to None. OUT: Consumed during execution.
        include_web (bool, optional): IN: include web. Defaults to True. OUT: Consumed during execution.
        include_system (bool, optional): IN: include system. Defaults to True. OUT: Consumed during execution.
        include_ai (bool, optional): IN: include ai. Defaults to True. OUT: Consumed during execution.
        include_memory (bool, optional): IN: include memory. Defaults to True. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    from xerxes.runtime.execution_registry import ExecutionRegistry
    from xerxes.streaming.permissions import SAFE_TOOLS

    if registry is None:
        registry = ExecutionRegistry()

    try:
        import xerxes.tools as tools_mod
    except ImportError:
        logger.warning("xerxes.tools not available")
        return registry

    categories = getattr(tools_mod, "TOOL_CATEGORIES", {})

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

            description = ""
            if hasattr(tool_obj, "static_call") and tool_obj.static_call.__doc__:
                description = tool_obj.static_call.__doc__.strip().split("\n")[0]
            elif hasattr(tool_obj, "__doc__") and tool_obj.__doc__:
                description = tool_obj.__doc__.strip().split("\n")[0]

            handler = getattr(tool_obj, "static_call", None)
            if handler is None and callable(tool_obj):
                handler = tool_obj

            schema = _build_tool_schema(tool_name, description, handler)

            is_safe = tool_name in SAFE_TOOLS

            registry.register_tool(
                name=tool_name,
                handler=handler,
                description=description,
                category=category,
                safe=is_safe,
                source_hint=f"xerxes.tools.{category}",
                schema=schema,
            )

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
    """Internal helper to build tool schema.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
        description (str): IN: description. OUT: Consumed during execution.
        handler (Any): IN: handler. OUT: Consumed during execution.
    Returns:
        dict[str, Any]: OUT: Result of the operation."""

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
    xerxes_instance: Any = None,
    agent: Any = None,
    registry: Any = None,
) -> Any:
    """Build tool executor.

    Args:
        xerxes_instance (Any, optional): IN: xerxes instance. Defaults to None. OUT: Consumed during execution.
        agent (Any, optional): IN: agent. Defaults to None. OUT: Consumed during execution.
        registry (Any, optional): IN: registry. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    def executor(tool_name: str, tool_input: dict[str, Any]) -> str:
        """Executor.

        Args:
            tool_name (str): IN: tool name. OUT: Consumed during execution.
            tool_input (dict[str, Any]): IN: tool input. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""
        if registry is not None:
            entry = registry.get_tool(tool_name)
            if entry is not None and entry.handler is not None:
                try:
                    result = _call_tool_handler(
                        entry.handler,
                        tool_input,
                        xerxes_instance=xerxes_instance,
                        agent=agent,
                    )
                    return str(result) if result is not None else ""
                except Exception as e:
                    return f"Error executing {tool_name}: {e}"

        if agent is not None:
            for func in getattr(agent, "functions", []):
                func_name = getattr(func, "name", "") or getattr(func, "__name__", "")
                if func_name == tool_name:
                    callable_fn = getattr(func, "static_call", None) or getattr(func, "callable_func", None) or func
                    try:
                        result = _call_tool_handler(
                            callable_fn,
                            tool_input,
                            xerxes_instance=xerxes_instance,
                            agent=agent,
                        )
                        return str(result) if result is not None else ""
                    except Exception as e:
                        return f"Error executing {tool_name}: {e}"

        try:
            import xerxes.tools as tools_mod

            tool_obj = getattr(tools_mod, tool_name, None)
            if tool_obj is not None:
                handler = getattr(tool_obj, "static_call", None)
                if handler is None and callable(tool_obj):
                    handler = tool_obj
                if handler is not None:
                    result = _call_tool_handler(
                        handler,
                        tool_input,
                        xerxes_instance=xerxes_instance,
                        agent=agent,
                    )
                    return str(result) if result is not None else ""
        except Exception as e:
            return f"Error: {e}"

        return f"Unknown tool: {tool_name}"

    return executor


def create_query_engine(
    xerxes_instance: Any = None,
    agent: Any = None,
    model: str = "",
    system_prompt: str = "",
    **config_kwargs: Any,
) -> Any:
    """Create query engine.

    Args:
        xerxes_instance (Any, optional): IN: xerxes instance. Defaults to None. OUT: Consumed during execution.
        agent (Any, optional): IN: agent. Defaults to None. OUT: Consumed during execution.
        model (str, optional): IN: model. Defaults to ''. OUT: Consumed during execution.
        system_prompt (str, optional): IN: system prompt. Defaults to ''. OUT: Consumed during execution.
        **config_kwargs: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
    Returns:
        Any: OUT: Result of the operation."""

    from xerxes.runtime.query_engine import QueryEngine

    registry = populate_registry()
    tool_executor = build_tool_executor(
        xerxes_instance=xerxes_instance,
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

    setattr(engine, "_default_tool_executor", tool_executor)
    setattr(engine, "_default_tool_schemas", tool_schemas)

    return engine


def bootstrap_xerxes(
    xerxes_instance: Any = None,
    agent: Any = None,
    model: str = "",
    extra_context: str = "",
) -> Any:
    """Bootstrap xerxes.

    Args:
        xerxes_instance (Any, optional): IN: xerxes instance. Defaults to None. OUT: Consumed during execution.
        agent (Any, optional): IN: agent. Defaults to None. OUT: Consumed during execution.
        model (str, optional): IN: model. Defaults to ''. OUT: Consumed during execution.
        extra_context (str, optional): IN: extra context. Defaults to ''. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    from xerxes.runtime.bootstrap import bootstrap

    tools = getattr(agent, "functions", []) if agent else []

    result = bootstrap(
        model=model,
        tools=tools,
        extra_context=extra_context,
    )

    populate_registry(result.registry)

    return result


__all__ = [
    "bootstrap_xerxes",
    "build_tool_executor",
    "create_query_engine",
    "populate_registry",
]
