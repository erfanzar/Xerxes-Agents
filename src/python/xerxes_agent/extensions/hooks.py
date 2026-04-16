# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Hook system for Xerxes lifecycle events.

Hooks allow external code to observe and mutate agent execution at
well-defined points. Each hook point has documented semantics:

Hook Points:
    - ``before_tool_call(tool_name, arguments, agent_id) -> arguments``
        Called before a tool is executed. Can modify arguments.
        Return modified arguments dict or None to keep original.

    - ``after_tool_call(tool_name, arguments, result, agent_id) -> result``
        Called after a tool executes. Can transform the result.
        Return modified result or None to keep original.

    - ``tool_result_persist(tool_name, result, agent_id) -> result``
        Called before tool result is stored in conversation history.
        Allows sanitizing/transforming results for persistence.

    - ``bootstrap_files(agent_id) -> list[str]``
        Called during prompt assembly. Returns extra content strings
        to inject into the system prompt.

    - ``on_turn_start(agent_id, messages)``
        Called when a new agent turn begins.

    - ``on_turn_end(agent_id, response)``
        Called when an agent turn completes.

    - ``on_error(agent_id, error)``
        Called when an error occurs during execution.

Execution semantics:
    - Hooks run in registration order.
    - A hook that raises is logged and skipped (does not break the chain).
    - Mutation hooks (before_tool_call, after_tool_call, tool_result_persist)
      pass their return value to the next hook in the chain.
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
        "on_error",
    }
)

_MUTATION_HOOKS = frozenset({"before_tool_call", "after_tool_call", "tool_result_persist"})


class HookRunner:
    """Manages registration and execution of lifecycle hooks.

    ``HookRunner`` maintains an ordered list of callback functions for each
    defined hook point.  Hooks are divided into two categories:

    * **Mutation hooks** (``before_tool_call``, ``after_tool_call``,
      ``tool_result_persist``) -- callbacks are chained so each can modify a
      value that is passed to the next callback.
    * **Observation hooks** (``bootstrap_files``, ``on_turn_start``,
      ``on_turn_end``, ``on_error``) -- all callbacks are invoked and their
      non-``None`` return values are collected into a list.

    Callbacks that raise exceptions are logged and skipped without breaking
    the hook chain.

    Attributes:
        _hooks: Internal mapping from hook point names to ordered lists of
            registered callbacks.

    Example:
        >>> runner = HookRunner()
        >>> runner.register("before_tool_call", my_hook_fn)
        >>> modified_args = runner.run("before_tool_call",
        ...     tool_name="search", arguments={"q": "hello"}, agent_id="a1")
    """

    def __init__(self) -> None:
        """Initialize the HookRunner with empty callback lists for all hook points.

        Pre-populates the internal ``_hooks`` dictionary with an empty list
        for every name defined in :data:`HOOK_POINTS`.
        """
        self._hooks: dict[str, list[HookCallback]] = {name: [] for name in HOOK_POINTS}

    def register(self, hook_point: str, callback: HookCallback) -> None:
        """Register a hook callback for a specific hook point.

        Args:
            hook_point: One of the defined HOOK_POINTS.
            callback: The callable to invoke at this hook point.

        Raises:
            ValueError: If hook_point is not recognized.
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
        """Remove a previously registered callback from a hook point.

        Args:
            hook_point: The hook point name to search in.
            callback: The exact callable instance to remove (identity match).

        Returns:
            ``True`` if the callback was found and removed, ``False`` if the
            hook point does not exist or the callback was not registered.
        """
        if hook_point not in self._hooks:
            return False
        try:
            self._hooks[hook_point].remove(callback)
            return True
        except ValueError:
            return False

    def clear(self, hook_point: str | None = None) -> None:
        """Clear all registered callbacks for one or all hook points.

        Args:
            hook_point: If provided, only callbacks for this specific hook
                point are removed.  If ``None`` (the default), callbacks
                for **every** hook point are removed.
        """
        if hook_point:
            self._hooks[hook_point] = []
        else:
            self._hooks = {name: [] for name in HOOK_POINTS}

    def run(self, hook_point: str, **kwargs) -> tp.Any:
        """Execute all hooks registered for a hook point.

        Dispatches to :meth:`_run_mutation` or :meth:`_run_observation`
        depending on the hook category.

        For mutation hooks (``before_tool_call``, ``after_tool_call``,
        ``tool_result_persist``):
            - The return value from each hook is passed as updated kwargs
              to the next hook in the chain.
            - For ``before_tool_call``: the return value replaces
              ``'arguments'``.
            - For ``after_tool_call`` / ``tool_result_persist``: the return
              value replaces ``'result'``.
            - Returns the final mutated value.

        For observation hooks (``bootstrap_files``, ``on_turn_start``, etc.):
            - All callbacks are invoked; return values are collected.
            - Returns a list of non-``None`` return values.

        If no callbacks are registered the method returns the relevant
        default value (``arguments`` or ``result`` from *kwargs*).

        Args:
            hook_point: The hook point name to execute.
            **kwargs: Keyword arguments forwarded to every callback.  The
                exact keys depend on the hook point (see module docstring).

        Returns:
            The final mutated value for mutation hooks, or a list of
            collected results for observation hooks.
        """
        callbacks = self._hooks.get(hook_point, [])
        if not callbacks:
            return kwargs.get("arguments") if hook_point == "before_tool_call" else kwargs.get("result")

        if hook_point in _MUTATION_HOOKS:
            return self._run_mutation(hook_point, callbacks, **kwargs)
        else:
            return self._run_observation(hook_point, callbacks, **kwargs)

    def _run_mutation(self, hook_point: str, callbacks: list[HookCallback], **kwargs) -> tp.Any:
        """Run mutation hooks, chaining non-``None`` return values.

        Each callback receives the current *kwargs*.  When a callback returns
        a non-``None`` value the corresponding mutable key (``"arguments"``
        for ``before_tool_call``, ``"result"`` for others) is updated in
        *kwargs* before the next callback is invoked.

        Args:
            hook_point: The mutation hook point name.
            callbacks: Ordered list of callbacks to invoke.
            **kwargs: The keyword arguments passed through the chain.

        Returns:
            The final value of the mutated key after all callbacks have run.
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

        All callbacks are invoked regardless of individual return values.
        Exceptions are logged and do not prevent subsequent callbacks from
        executing.

        Args:
            hook_point: The observation hook point name.
            callbacks: Ordered list of callbacks to invoke.
            **kwargs: The keyword arguments forwarded to every callback.

        Returns:
            A list of non-``None`` values returned by the callbacks.
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
        """Check whether any callbacks are registered for a hook point.

        Args:
            hook_point: The hook point name to query.

        Returns:
            ``True`` if at least one callback is registered, ``False``
            otherwise (including when *hook_point* is not a recognized name).
        """
        return bool(self._hooks.get(hook_point))
