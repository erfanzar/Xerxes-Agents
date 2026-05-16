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
"""Decorator helpers shared by every operator tool factory.

Provides :func:`operator_tool`, the lightweight annotation that tags a
callable with the metadata the streaming runtime needs to register it as a
tool (canonical name, description, capability category).
"""

from __future__ import annotations

import typing as tp


def operator_tool(
    name: str,
    *,
    description: str | None = None,
    category: str = "operator",
) -> tp.Callable[[tp.Callable], tp.Callable]:
    """Stamp tool metadata onto a function so the runtime can register it.

    The decorator mutates ``__xerxes_schema__`` (creating it if absent) and
    attaches a ``category`` attribute. Existing schema fields are preserved
    unless ``name`` / ``description`` overrides them.

    Args:
        name: Canonical tool name surfaced to the model.
        description: Long-form description shown in the tool catalogue;
            ``None`` keeps whatever description the schema already had.
        category: Capability bucket — defaults to ``"operator"`` so the
            policy engine groups these tools together.
    """

    def _decorate(func: tp.Callable) -> tp.Callable:
        """Attach the schema dict and category attribute to ``func``."""

        schema = dict(getattr(func, "__xerxes_schema__", {}) or {})
        schema["name"] = name
        if description is not None:
            schema["description"] = description
        any_func = tp.cast(tp.Any, func)
        any_func.__xerxes_schema__ = schema
        any_func.category = category
        return any_func

    return _decorate
