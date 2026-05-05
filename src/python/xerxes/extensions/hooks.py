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
"""Hook-point registry and dispatcher for the Xerxes runtime.

Plugins and internal components can register callbacks at predefined hook
points. Mutation hooks (``before_tool_call``, ``after_tool_call``,
``tool_result_persist``) allow returning modified values; observation hooks
simply collect side effects.
"""

from __future__ import annotations

import logging
import typing as tp

logger = logging.getLogger(__name__)

HookCallback = tp.Callable[..., tp.Any]

HOOK_POINTS = frozenset(
    {
        "before_tool_call",
        "after_tool_call",
        "tool_result_persist",
        "bootstrap_files",
        "on_turn_start",
        "on_turn_end",
        "on_loop_warning",
        "on_error",
    }
)

_MUTATION_HOOKS = frozenset({"before_tool_call", "after_tool_call", "tool_result_persist"})


class HookRunner:
    """Stores and executes callbacks for named hook points.

    On instantiation every valid hook point is pre-allocated an empty list.
    """

    def __init__(self) -> None:
        """Initialize empty callback lists for all ``HOOK_POINTS``."""

        self._hooks: dict[str, list[HookCallback]] = {name: [] for name in HOOK_POINTS}

    def register(self, hook_point: str, callback: HookCallback) -> None:
        """Add a callback to a hook point.

        Args:
            hook_point (str): IN: Name from ``HOOK_POINTS``. OUT: Validates
                membership and appends ``callback``.
            callback (HookCallback): IN: Callable to invoke. OUT: Stored in
                the internal list.

        Returns:
            None: OUT: Callback is registered.

        Raises:
            ValueError: OUT: If ``hook_point`` is not in ``HOOK_POINTS``.
        """

        if hook_point not in HOOK_POINTS:
            raise ValueError(f"Unknown hook point '{hook_point}'. Valid: {sorted(HOOK_POINTS)}")
        self._hooks[hook_point].append(callback)
        logger.debug(
            "Registered hook for '%s': %s",
            hook_point,
            callback.__name__ if hasattr(callback, "__name__") else str(callback),
        )

    def unregister(self, hook_point: str, callback: HookCallback) -> bool:
        """Remove a previously registered callback.

        Args:
            hook_point (str): IN: Hook point name. OUT: Looked up in the
                registry.
            callback (HookCallback): IN: Exact callable to remove. OUT:
                Removed by identity from the list.

        Returns:
            bool: OUT: ``True`` if the callback was found and removed.
        """

        if hook_point not in self._hooks:
            return False
        try:
            self._hooks[hook_point].remove(callback)
            return True
        except ValueError:
            return False

    def clear(self, hook_point: str | None = None) -> None:
        """Remove all callbacks, globally or for a specific hook point.

        Args:
            hook_point (str | None): IN: Specific hook to clear, or ``None``
                for all. OUT: Empties the selected list(s).

        Returns:
            None: OUT: Selected callbacks are removed.
        """

        if hook_point:
            self._hooks[hook_point] = []
        else:
            self._hooks = {name: [] for name in HOOK_POINTS}

    def run(self, hook_point: str, **kwargs) -> tp.Any:
        """Execute callbacks for a hook point.

        Args:
            hook_point (str): IN: Name of the hook to fire. OUT: Determines
                mutation vs observation behaviour.
            **kwargs: IN: Keyword arguments forwarded to each callback. OUT:
                May be mutated for mutation hooks.

        Returns:
            tp.Any: OUT: Mutated value for mutation hooks, or list of results
            for observation hooks.
        """

        callbacks = self._hooks.get(hook_point, [])
        if not callbacks:
            return kwargs.get("arguments") if hook_point == "before_tool_call" else kwargs.get("result")

        if hook_point in _MUTATION_HOOKS:
            return self._run_mutation(hook_point, callbacks, **kwargs)
        else:
            return self._run_observation(hook_point, callbacks, **kwargs)

    def _run_mutation(self, hook_point: str, callbacks: list[HookCallback], **kwargs) -> tp.Any:
        """Run mutation hooks, allowing each callback to replace a value.

        Args:
            hook_point (str): IN: Hook name. OUT: Selects whether
                ``arguments`` or ``result`` is the mutable key.
            callbacks (list[HookCallback]): IN: Registered mutators. OUT:
                Invoked sequentially; non-``None`` returns replace the current
                value.
            **kwargs: IN: Arguments / result dict. OUT: Mutated in place.

        Returns:
            tp.Any: OUT: Final mutated value.
        """

        if hook_point == "before_tool_call":
            mutated_key = "arguments"
        else:
            mutated_key = "result"

        current = kwargs.get(mutated_key)
        for cb in callbacks:
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    current = ret
                    kwargs[mutated_key] = current
            except Exception:
                logger.warning("Hook '%s' raised in %s", hook_point, cb, exc_info=True)
        return current

    def _run_observation(self, hook_point: str, callbacks: list[HookCallback], **kwargs) -> list[tp.Any]:
        """Run observation hooks, collecting non-``None`` return values.

        Args:
            hook_point (str): IN: Hook name. OUT: Used only for logging.
            callbacks (list[HookCallback]): IN: Registered observers. OUT:
                Invoked sequentially.
            **kwargs: IN: Forwarded to each callback. OUT: Unchanged.

        Returns:
            list[tp.Any]: OUT: List of non-``None`` results from callbacks.
        """

        results = []
        for cb in callbacks:
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception:
                logger.warning("Hook '%s' raised in %s", hook_point, cb, exc_info=True)
        return results

    def has_hooks(self, hook_point: str) -> bool:
        """Return whether any callbacks are registered for a hook point.

        Args:
            hook_point (str): IN: Hook name to query. OUT: Checked against the
                internal registry.

        Returns:
            bool: OUT: ``True`` if at least one callback exists.
        """

        return bool(self._hooks.get(hook_point))
