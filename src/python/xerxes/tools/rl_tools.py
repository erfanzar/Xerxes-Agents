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

"""Reinforcement-learning control-plane agent tools.

Ten ``rl_*`` tools that let the agent kick off, inspect, and tear
down RL training runs without leaving the chat.

The tools delegate every action to a pluggable :class:`RLBackend` —
the default :class:`InMemoryRLBackend` is enough for tests and demos,
while production deployments register their own backend (e.g. the
EasyDeL eSurge controller) via :func:`set_rl_backend`.

Tools:

- :class:`rl_list_environments`
- :class:`rl_select_environment`
- :class:`rl_get_current_config`
- :class:`rl_edit_config`
- :class:`rl_start_training`
- :class:`rl_stop_training`
- :class:`rl_check_status`
- :class:`rl_get_results`
- :class:`rl_list_runs`
- :class:`rl_test_inference`
"""

from __future__ import annotations

import logging
import threading
import time
import typing as tp
import uuid
from dataclasses import dataclass, field

from ..types import AgentBaseFn

logger = logging.getLogger(__name__)


@dataclass
class RLEnvironment:
    """Static description of a registered training environment."""

    name: str
    description: str = ""
    config: dict[str, tp.Any] = field(default_factory=dict)


@dataclass
class RLRun:
    """Lifecycle record for a single training run."""

    run_id: str
    environment: str
    config: dict[str, tp.Any]
    status: str = "queued"
    metrics: dict[str, tp.Any] = field(default_factory=dict)
    results: dict[str, tp.Any] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0


class RLBackend(tp.Protocol):
    """Pluggable backend the ``rl_*`` tools dispatch to."""

    def list_environments(self) -> list[RLEnvironment]:
        """Return every registered environment known to the backend."""
        ...

    def select_environment(self, name: str) -> RLEnvironment | None:
        """Activate *name* and return it, or ``None`` when unregistered."""
        ...

    def get_current_config(self) -> dict[str, tp.Any]:
        """Return ``{"environment", "config"}`` for the currently selected env."""
        ...

    def edit_config(self, updates: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Merge *updates* into the active environment's config and return the result."""
        ...

    def start(self) -> RLRun:
        """Kick off a training run using the selected environment's config."""
        ...

    def stop(self, run_id: str) -> RLRun | None:
        """Stop the run with ``run_id`` and return its final record, or ``None``."""
        ...

    def status(self, run_id: str) -> RLRun | None:
        """Return the live :class:`RLRun` for ``run_id``, or ``None`` if unknown."""
        ...

    def results(self, run_id: str) -> dict[str, tp.Any]:
        """Return the final metrics/results payload for a completed run."""
        ...

    def list_runs(self) -> list[RLRun]:
        """Return every run the backend has tracked (running or finished)."""
        ...

    def test_inference(self, prompt: str, run_id: str | None = None) -> dict[str, tp.Any]:
        """Sample the model (optionally from *run_id*) with *prompt* for quick checks."""
        ...


class InMemoryRLBackend:
    """Reference RL backend kept entirely in memory.

    Useful for tests, demos, and offline development. Real production
    code should swap in something talking to EasyDeL / vLLM / etc.

    Example:
        >>> b = InMemoryRLBackend()
        >>> b.register("cartpole-v1", {"lr": 1e-3, "steps": 100})
        >>> set_rl_backend(b)
    """

    def __init__(self) -> None:
        """Initialise empty environment registry and run store."""
        self._lock = threading.Lock()
        self._environments: dict[str, RLEnvironment] = {}
        self._selected: str | None = None
        self._config: dict[str, tp.Any] = {}
        self._runs: dict[str, RLRun] = {}

    def register(
        self,
        name: str,
        config: dict[str, tp.Any] | None = None,
        description: str = "",
    ) -> RLEnvironment:
        """Register a new environment with *name* / *config* and return it."""
        env = RLEnvironment(name=name, description=description, config=dict(config or {}))
        with self._lock:
            self._environments[name] = env
        return env

    def list_environments(self) -> list[RLEnvironment]:
        """Return a snapshot of every registered environment."""
        with self._lock:
            return list(self._environments.values())

    def select_environment(self, name: str) -> RLEnvironment | None:
        """Activate *name*, seeding the current config from its defaults."""
        with self._lock:
            env = self._environments.get(name)
            if env is None:
                return None
            self._selected = name
            self._config = dict(env.config)
            return env

    def get_current_config(self) -> dict[str, tp.Any]:
        """Return a snapshot of the active environment and its running config."""
        with self._lock:
            return {"environment": self._selected, "config": dict(self._config)}

    def edit_config(self, updates: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Merge *updates* into the current config and return the new state."""
        with self._lock:
            self._config.update(updates)
            return {"environment": self._selected, "config": dict(self._config)}

    def start(self) -> RLRun:
        """Start a new in-memory training run for the currently selected environment."""
        with self._lock:
            if self._selected is None:
                raise RuntimeError("no environment selected")
            run_id = uuid.uuid4().hex[:12]
            run = RLRun(
                run_id=run_id,
                environment=self._selected,
                config=dict(self._config),
                status="running",
                started_at=time.time(),
                metrics={"step": 0, "reward": 0.0},
            )
            self._runs[run_id] = run
            return run

    def stop(self, run_id: str) -> RLRun | None:
        """Mark a running run as stopped, recording the end timestamp."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            if run.status == "running":
                run.status = "stopped"
                run.ended_at = time.time()
            return run

    def status(self, run_id: str) -> RLRun | None:
        """Return the :class:`RLRun` for *run_id*, or ``None`` if unknown."""
        with self._lock:
            return self._runs.get(run_id)

    def results(self, run_id: str) -> dict[str, tp.Any]:
        """Return final (or partial, while running) metrics for *run_id*."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return {"error": "not_found"}
            if run.status == "running":
                return {"status": "running", "partial_metrics": dict(run.metrics)}
            return {
                "status": run.status,
                "metrics": dict(run.metrics),
                "results": dict(run.results),
                "duration_s": (run.ended_at or time.time()) - run.started_at,
            }

    def list_runs(self) -> list[RLRun]:
        """Return every tracked run (running or finished)."""
        with self._lock:
            return list(self._runs.values())

    def test_inference(self, prompt: str, run_id: str | None = None) -> dict[str, tp.Any]:
        """Return a mock completion for *prompt*; real backends should sample the model."""
        with self._lock:
            return {
                "run_id": run_id or self._selected or "ad-hoc",
                "prompt": prompt[:200],
                "completion": f"[mock] {prompt[:80]}",
            }


_backend_lock = threading.Lock()
_backend: RLBackend = InMemoryRLBackend()


def set_rl_backend(backend: RLBackend) -> None:
    """Install a custom :class:`RLBackend` for all ``rl_*`` tools."""
    global _backend
    with _backend_lock:
        _backend = backend


def get_rl_backend() -> RLBackend:
    """Return the currently installed RL backend (default: in-memory)."""
    with _backend_lock:
        return _backend


def reset_rl_backend() -> None:
    """Restore the default in-memory backend (mainly for tests)."""
    set_rl_backend(InMemoryRLBackend())


def _run_to_dict(run: RLRun) -> dict[str, tp.Any]:
    """Serialise *run* to a JSON-friendly dict for tool responses."""
    return {
        "run_id": run.run_id,
        "environment": run.environment,
        "status": run.status,
        "metrics": dict(run.metrics),
        "started_at": run.started_at,
        "ended_at": run.ended_at,
    }


class rl_list_environments(AgentBaseFn):
    """List every registered RL environment."""

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return the catalog of environments the active backend exposes.

        Returns:
            ``{"count": int, "environments": [{name, description, config}]}``.
        """
        envs = get_rl_backend().list_environments()
        return {
            "count": len(envs),
            "environments": [{"name": e.name, "description": e.description, "config": dict(e.config)} for e in envs],
        }


class rl_select_environment(AgentBaseFn):
    """Pick an environment to operate on."""

    @staticmethod
    def static_call(name: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Activate ``name`` so subsequent ``rl_*`` calls use its config.

        Args:
            name: Environment name returned by ``rl_list_environments``.

        Returns:
            The selected environment metadata or ``{"error": "not_found"}``.
        """
        env = get_rl_backend().select_environment(name)
        if env is None:
            return {"error": "not_found", "name": name}
        return {"name": env.name, "description": env.description, "config": dict(env.config)}


class rl_get_current_config(AgentBaseFn):
    """Read the active environment's current configuration."""

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return ``{"environment": <name|None>, "config": {...}}``."""
        return get_rl_backend().get_current_config()


class rl_edit_config(AgentBaseFn):
    """Patch fields on the active environment's config."""

    @staticmethod
    def static_call(
        updates: dict[str, tp.Any],
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Merge ``updates`` into the active config and return the new state.

        Args:
            updates: Dict of fields to set/replace, e.g. ``{"lr": 3e-4}``.
        """
        return get_rl_backend().edit_config(dict(updates or {}))


class rl_start_training(AgentBaseFn):
    """Kick off a training run with the current config."""

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Begin a new run; returns ``{"run_id", "status", ...}``.

        Use this only after :class:`rl_select_environment` (and
        optionally :class:`rl_edit_config`) have been called.
        """
        try:
            run = get_rl_backend().start()
        except Exception as exc:
            return {"error": str(exc)}
        return _run_to_dict(run)


class rl_stop_training(AgentBaseFn):
    """Terminate a running training job."""

    @staticmethod
    def static_call(run_id: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Stop ``run_id`` and return the post-stop status."""
        run = get_rl_backend().stop(run_id)
        if run is None:
            return {"error": "not_found", "run_id": run_id}
        return _run_to_dict(run)


class rl_check_status(AgentBaseFn):
    """Poll the live status + metrics of a run."""

    @staticmethod
    def static_call(run_id: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return ``{"run_id", "status", "metrics", ...}`` for the run."""
        run = get_rl_backend().status(run_id)
        if run is None:
            return {"error": "not_found", "run_id": run_id}
        return _run_to_dict(run)


class rl_get_results(AgentBaseFn):
    """Read final results for a completed (or partial for live) run."""

    @staticmethod
    def static_call(run_id: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return the run's metrics + summary blob.

        For runs still in ``running`` state the response is
        ``{"status": "running", "partial_metrics": {...}}``.
        """
        return get_rl_backend().results(run_id)


class rl_list_runs(AgentBaseFn):
    """List every training run the backend remembers."""

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return ``{"count", "runs": [...]}`` sorted by start time desc."""
        runs = sorted(
            get_rl_backend().list_runs(),
            key=lambda r: r.started_at,
            reverse=True,
        )
        return {"count": len(runs), "runs": [_run_to_dict(r) for r in runs]}


class rl_test_inference(AgentBaseFn):
    """Run a single inference against a trained policy."""

    @staticmethod
    def static_call(
        prompt: str,
        run_id: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Execute one inference against ``run_id`` (default: active env).

        Args:
            prompt: Free-text prompt the agent wants to evaluate.
            run_id: Optional specific run; defaults to whichever
                checkpoint the backend considers current.
        """
        return get_rl_backend().test_inference(prompt, run_id=run_id)


__all__ = [
    "InMemoryRLBackend",
    "RLBackend",
    "RLEnvironment",
    "RLRun",
    "get_rl_backend",
    "reset_rl_backend",
    "rl_check_status",
    "rl_edit_config",
    "rl_get_current_config",
    "rl_get_results",
    "rl_list_environments",
    "rl_list_runs",
    "rl_select_environment",
    "rl_start_training",
    "rl_stop_training",
    "rl_test_inference",
    "set_rl_backend",
]
