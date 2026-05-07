# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the license.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reinforcement learning tools for environment management, training control, and experiment tracking.

This module provides tools for managing RL environments, controlling training runs,
and tracking experiment results. Useful for building agents that interact with
RL training systems.

Example:
    >>> from xerxes.tools.rl_tools import rl_list_environments, rl_start_training
    >>> envs = rl_list_environments.static_call()
    >>> rl_start_training.static_call()
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
    """Representation of a reinforcement learning environment.

    Attributes:
        name: Unique identifier for the environment.
        description: Human-readable description.
        config: Environment configuration dictionary.
    """

    name: str
    description: str = ""
    config: dict[str, tp.Any] = field(default_factory=dict)


@dataclass
class RLRun:
    """Representation of an RL training run.

    Attributes:
        run_id: Unique identifier for the run.
        environment: Name of the environment.
        config: Run configuration.
        status: Current status (queued, running, completed, failed).
        metrics: Dictionary of training metrics.
        results: Final results when run completes.
        started_at: Unix timestamp when run started.
        ended_at: Unix timestamp when run ended.
    """

    run_id: str
    environment: str
    config: dict[str, tp.Any]
    status: str = "queued"
    metrics: dict[str, tp.Any] = field(default_factory=dict)
    results: dict[str, tp.Any] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0


class RLBackend(tp.Protocol):
    """Protocol defining the interface for RL backends.

    Implement this protocol to create custom RL backend integrations.
    """

    def list_environments(self) -> list[RLEnvironment]:
        """List available environments.

        Returns:
            List of available RLEnvironment objects.
        """
        ...

    def select_environment(self, name: str) -> RLEnvironment | None:
        """Select an environment for training.

        Args:
            name: Name of the environment to select.

        Returns:
            The selected RLEnvironment, or None if not found.
        """
        ...

    def get_current_config(self) -> dict[str, tp.Any]:
        """Get current backend configuration.

        Returns:
            Dictionary of current configuration settings.
        """
        ...

    def edit_config(self, updates: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Update backend configuration.

        Args:
            updates: Dictionary of configuration changes.

        Returns:
            Updated configuration dictionary.
        """
        ...

    def start(self) -> RLRun:
        """Start a new training run.

        Returns:
            RLRun object representing the started run.
        """
        ...

    def stop(self, run_id: str) -> RLRun | None:
        """Stop a running or queued training run.

        Args:
            run_id: ID of the run to stop.

        Returns:
            The stopped RLRun, or None if not found.
        """
        ...

    def status(self, run_id: str) -> RLRun | None:
        """Get the status of a training run.

        Args:
            run_id: ID of the run to query.

        Returns:
            RLRun object with current status, or None if not found.
        """
        ...

    def results(self, run_id: str) -> dict[str, tp.Any]:
        """Get results for a completed run.

        Args:
            run_id: ID of the run.

        Returns:
            Dictionary containing run results and metrics.
        """
        ...

    def list_runs(self) -> list[RLRun]:
        """List all training runs.

        Returns:
            List of all RLRun objects.
        """
        ...

    def test_inference(self, prompt: str, run_id: str | None = None) -> dict[str, tp.Any]:
        """Test inference with a trained model.

        Args:
            prompt: Input prompt for inference.
            run_id: Optional run ID to use specific model.

        Returns:
            Dictionary containing inference results.
        """
        ...


class InMemoryRLBackend:
    """In-memory implementation of RLBackend for testing and development.

    This backend stores all state in memory and is useful for testing
    without requiring an actual RL framework.

    Example:
        >>> backend = InMemoryRLBackend()
        >>> backend.register("cartpole", config={"threshold": 200})
        >>> backend.select_environment("cartpole")
    """

    def __init__(self) -> None:
        """Initialize the in-memory backend with empty state."""
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
        """Register a new environment with the backend.

        Args:
            name: Unique name for the environment.
            config: Configuration dictionary for the environment.
            description: Human-readable description.

        Returns:
            The registered RLEnvironment.
        """
        env = RLEnvironment(name=name, description=description, config=dict(config or {}))
        with self._lock:
            self._environments[name] = env
        return env

    def list_environments(self) -> list[RLEnvironment]:
        """List all registered environments.

        Returns:
            List of all registered RLEnvironment objects.
        """
        with self._lock:
            return list(self._environments.values())

    def select_environment(self, name: str) -> RLEnvironment | None:
        """Select an environment for training.

        Args:
            name: Name of the environment to select.

        Returns:
            The selected RLEnvironment, or None if not found.
        """
        with self._lock:
            env = self._environments.get(name)
            if env is None:
                return None
            self._selected = name
            self._config = dict(env.config)
            return env

    def get_current_config(self) -> dict[str, tp.Any]:
        """Get current configuration.

        Returns:
            Dictionary with selected environment and config.
        """
        with self._lock:
            return {"environment": self._selected, "config": dict(self._config)}

    def edit_config(self, updates: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Update the current configuration.

        Args:
            updates: Dictionary of configuration changes.

        Returns:
            Updated configuration.
        """
        with self._lock:
            self._config.update(updates)
            return {"environment": self._selected, "config": dict(self._config)}

    def start(self) -> RLRun:
        """Start a new training run.

        Returns:
            The created RLRun object.

        Raises:
            RuntimeError: If no environment is selected.
        """
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
        """Stop a training run.

        Args:
            run_id: ID of the run to stop.

        Returns:
            The stopped RLRun, or None if not found.
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            if run.status == "running":
                run.status = "stopped"
                run.ended_at = time.time()
            return run

    def status(self, run_id: str) -> RLRun | None:
        """Get status of a run.

        Args:
            run_id: ID of the run.

        Returns:
            RLRun with current status, or None if not found.
        """
        with self._lock:
            return self._runs.get(run_id)

    def results(self, run_id: str) -> dict[str, tp.Any]:
        """Get results for a run.

        Args:
            run_id: ID of the run.

        Returns:
            Dictionary with run results and metrics.
        """
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
        """List all runs.

        Returns:
            List of all RLRun objects.
        """
        with self._lock:
            return list(self._runs.values())

    def test_inference(self, prompt: str, run_id: str | None = None) -> dict[str, tp.Any]:
        """Test inference (mock implementation).

        Args:
            prompt: Input prompt.
            run_id: Optional run ID.

        Returns:
            Mock inference results.
        """
        with self._lock:
            return {
                "run_id": run_id or self._selected or "ad-hoc",
                "prompt": prompt[:200],
                "completion": f"[mock] {prompt[:80]}",
            }


_backend_lock = threading.Lock()
_backend: RLBackend = InMemoryRLBackend()


def set_rl_backend(backend: RLBackend) -> None:
    """Set the global RL backend.

    Args:
        backend: RLBackend implementation to use globally.
    """
    global _backend
    with _backend_lock:
        _backend = backend


def get_rl_backend() -> RLBackend:
    """Get the global RL backend.

    Returns:
        The current RLBackend instance.
    """
    with _backend_lock:
        return _backend


def reset_rl_backend() -> None:
    """Reset the global backend to a fresh InMemoryRLBackend."""
    set_rl_backend(InMemoryRLBackend())


def _run_to_dict(run: RLRun) -> dict[str, tp.Any]:
    """Convert RLRun to dictionary for serialization.

    Args:
        run: RLRun object to convert.

    Returns:
        Dictionary representation of the run.
    """
    return {
        "run_id": run.run_id,
        "environment": run.environment,
        "status": run.status,
        "metrics": dict(run.metrics),
        "started_at": run.started_at,
        "ended_at": run.ended_at,
    }


class rl_list_environments(AgentBaseFn):
    """List all available RL environments.

    Example:
        >>> rl_list_environments.static_call()
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """List available RL environments.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with count and environments list.
        """
        envs = get_rl_backend().list_environments()
        return {
            "count": len(envs),
            "environments": [{"name": e.name, "description": e.description, "config": dict(e.config)} for e in envs],
        }


class rl_select_environment(AgentBaseFn):
    """Select an environment for training.

    Example:
        >>> rl_select_environment.static_call(name="cartpole")
    """

    @staticmethod
    def static_call(name: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Select an RL environment.

        Args:
            name: Name of the environment to select.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with environment details, or error if not found.
        """
        env = get_rl_backend().select_environment(name)
        if env is None:
            return {"error": "not_found", "name": name}
        return {"name": env.name, "description": env.description, "config": dict(env.config)}


class rl_get_current_config(AgentBaseFn):
    """Get current backend configuration.

    Example:
        >>> rl_get_current_config.static_call()
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Get current configuration.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with environment and config.
        """
        return get_rl_backend().get_current_config()


class rl_edit_config(AgentBaseFn):
    """Update backend configuration.

    Example:
        >>> rl_edit_config.static_call(updates={"learning_rate": 0.001})
    """

    @staticmethod
    def static_call(
        updates: dict[str, tp.Any],
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Update configuration.

        Args:
            updates: Dictionary of configuration changes.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Updated configuration.
        """
        return get_rl_backend().edit_config(dict(updates or {}))


class rl_start_training(AgentBaseFn):
    """Start a new training run.

    Example:
        >>> rl_start_training.static_call()
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Start training run.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with run details, or error if no environment selected.
        """
        try:
            run = get_rl_backend().start()
        except Exception as exc:
            return {"error": str(exc)}
        return _run_to_dict(run)


class rl_stop_training(AgentBaseFn):
    """Stop a running training run.

    Example:
        >>> rl_stop_training.static_call(run_id="abc123")
    """

    @staticmethod
    def static_call(run_id: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Stop a training run.

        Args:
            run_id: ID of the run to stop.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with run details, or error if not found.
        """
        run = get_rl_backend().stop(run_id)
        if run is None:
            return {"error": "not_found", "run_id": run_id}
        return _run_to_dict(run)


class rl_check_status(AgentBaseFn):
    """Check the status of a training run.

    Example:
        >>> rl_check_status.static_call(run_id="abc123")
    """

    @staticmethod
    def static_call(run_id: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Check run status.

        Args:
            run_id: ID of the run to check.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with run status details.
        """
        run = get_rl_backend().status(run_id)
        if run is None:
            return {"error": "not_found", "run_id": run_id}
        return _run_to_dict(run)


class rl_get_results(AgentBaseFn):
    """Get results for a completed run.

    Example:
        >>> rl_get_results.static_call(run_id="abc123")
    """

    @staticmethod
    def static_call(run_id: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Get run results.

        Args:
            run_id: ID of the run.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with run results and metrics.
        """
        return get_rl_backend().results(run_id)


class rl_list_runs(AgentBaseFn):
    """List all training runs.

    Example:
        >>> rl_list_runs.static_call()
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """List all training runs.

        Args:
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with count and runs list.
        """
        runs = sorted(
            get_rl_backend().list_runs(),
            key=lambda r: r.started_at,
            reverse=True,
        )
        return {"count": len(runs), "runs": [_run_to_dict(r) for r in runs]}


class rl_test_inference(AgentBaseFn):
    """Test inference with a trained model.

    Example:
        >>> rl_test_inference.static_call(prompt="What action should I take?")
    """

    @staticmethod
    def static_call(
        prompt: str,
        run_id: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Test inference.

        Args:
            prompt: Input prompt for inference.
            run_id: Optional run ID to use specific model.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary with inference results.
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
