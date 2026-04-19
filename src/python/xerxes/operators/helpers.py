# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Helpers for exposing Xerxes operator tools.

Contains the :func:`operator_tool` decorator which annotates callables
with the metadata the Xerxes runtime reads when registering operator
tools.
"""

from __future__ import annotations

import typing as tp


def operator_tool(
    name: str,
    *,
    description: str | None = None,
    category: str = "operator",
) -> tp.Callable[[tp.Callable], tp.Callable]:
    """Decorate a callable with a Xerxes public tool schema name.

    The decorator attaches a ``__xerxes_schema__`` dictionary and a
    ``category`` attribute to the wrapped function so the runtime can
    discover and register the tool under a canonical name.

    Args:
        name: Canonical tool name exposed to the LLM (e.g.
            ``"exec_command"`` or ``"web.open"``).
        description: Optional human-readable description included in
            the tool schema.  When ``None``, the function's own
            docstring is used.
        category: Logical tool category label.  Defaults to
            ``"operator"``.

    Returns:
        A decorator that marks the wrapped callable with schema
        metadata and returns it unchanged.

    Example:
        >>> @operator_tool("my_tool", description="Does a thing")
        ... def my_tool(arg: str) -> str:
        ...     return arg
        >>> my_tool.__xerxes_schema__["name"]
        'my_tool'
    """

    def _decorate(func: tp.Callable) -> tp.Callable:
        """Inner decorator that attaches schema metadata to *func*.

        Args:
            func: The callable to annotate.

        Returns:
            The same callable with ``__xerxes_schema__`` and
            ``category`` attributes set.
        """
        schema = dict(getattr(func, "__xerxes_schema__", {}) or {})
        schema["name"] = name
        if description is not None:
            schema["description"] = description
        func.__xerxes_schema__ = schema
        func.category = category
        return func

    return _decorate
