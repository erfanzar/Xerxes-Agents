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
"""Helpers module for Xerxes.

Exports:
    - operator_tool"""

from __future__ import annotations

import typing as tp


def operator_tool(
    name: str,
    *,
    description: str | None = None,
    category: str = "operator",
) -> tp.Callable[[tp.Callable], tp.Callable]:
    """Operator tool.

    Args:
        name (str): IN: name. OUT: Consumed during execution.
        description (str | None, optional): IN: description. Defaults to None. OUT: Consumed during execution.
        category (str, optional): IN: category. Defaults to 'operator'. OUT: Consumed during execution.
    Returns:
        tp.Callable[[tp.Callable], tp.Callable]: OUT: Result of the operation."""

    def _decorate(func: tp.Callable) -> tp.Callable:
        """Internal helper to decorate.

        Args:
            func (tp.Callable): IN: func. OUT: Consumed during execution.
        Returns:
            tp.Callable: OUT: Result of the operation."""

        schema = dict(getattr(func, "__xerxes_schema__", {}) or {})
        schema["name"] = name
        if description is not None:
            schema["description"] = description
        any_func = tp.cast(tp.Any, func)
        any_func.__xerxes_schema__ = schema
        any_func.category = category
        return any_func

    return _decorate
