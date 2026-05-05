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
"""Rl tools module for Xerxes.

Exports:
    - logger
    - RLEnvironment
    - RLRun
    - RLBackend
    - InMemoryRLBackend
    - set_rl_backend
    - get_rl_backend
    - reset_rl_backend
    - rl_list_environments
    - rl_select_environment
    - ... and 8 more."""

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
    """Rlenvironment.

    Attributes:
        name (str): name.
        description (str): description.
        config (dict[str, tp.Any]): config."""

    name: str
    description: str = ""
    config: dict[str, tp.Any] = field(default_factory=dict)


@dataclass
class RLRun:
    """Rlrun.

    Attributes:
        run_id (str): run id.
        environment (str): environment.
        config (dict[str, tp.Any]): config.
        status (str): status.
        metrics (dict[str, tp.Any]): metrics.
        results (dict[str, tp.Any]): results.
        started_at (float): started at.
        ended_at (float): ended at."""

    run_id: str
    environment: str
    config: dict[str, tp.Any]
    status: str = "queued"
    metrics: dict[str, tp.Any] = field(default_factory=dict)
    results: dict[str, tp.Any] = field(default_factory=dict)
    started_at: float = 0.0
    ended_at: float = 0.0


class RLBackend(tp.Protocol):
    """Rlbackend.

    Inherits from: tp.Protocol
    """

    def list_environments(self) -> list[RLEnvironment]:
        """List environments.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[RLEnvironment]: OUT: Result of the operation."""

        ...

    def select_environment(self, name: str) -> RLEnvironment | None:
        """Select environment.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
        Returns:
            RLEnvironment | None: OUT: Result of the operation."""

        ...

    def get_current_config(self) -> dict[str, tp.Any]:
        """Retrieve the current config.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        ...

    def edit_config(self, updates: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Edit config.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            updates (dict[str, tp.Any]): IN: updates. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        ...

    def start(self) -> RLRun:
        """Start.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            RLRun: OUT: Result of the operation."""

        ...

    def stop(self, run_id: str) -> RLRun | None:
        """Stop.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            run_id (str): IN: run id. OUT: Consumed during execution.
        Returns:
            RLRun | None: OUT: Result of the operation."""

        ...

    def status(self, run_id: str) -> RLRun | None:
        """Status.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            run_id (str): IN: run id. OUT: Consumed during execution.
        Returns:
            RLRun | None: OUT: Result of the operation."""

        ...

    def results(self, run_id: str) -> dict[str, tp.Any]:
        """Results.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            run_id (str): IN: run id. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        ...

    def list_runs(self) -> list[RLRun]:
        """List runs.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[RLRun]: OUT: Result of the operation."""

        ...

    def test_inference(self, prompt: str, run_id: str | None = None) -> dict[str, tp.Any]:
        """Test inference.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str): IN: prompt. OUT: Consumed during execution.
            run_id (str | None, optional): IN: run id. Defaults to None. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        ...


class InMemoryRLBackend:
    """In memory rlbackend."""

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Register.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            config (dict[str, tp.Any] | None, optional): IN: config. Defaults to None. OUT: Consumed during execution.
            description (str, optional): IN: description. Defaults to ''. OUT: Consumed during execution.
        Returns:
            RLEnvironment: OUT: Result of the operation."""

        env = RLEnvironment(name=name, description=description, config=dict(config or {}))
        with self._lock:
            self._environments[name] = env
        return env

    def list_environments(self) -> list[RLEnvironment]:
        """List environments.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[RLEnvironment]: OUT: Result of the operation."""

        with self._lock:
            return list(self._environments.values())

    def select_environment(self, name: str) -> RLEnvironment | None:
        """Select environment.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
        Returns:
            RLEnvironment | None: OUT: Result of the operation."""

        with self._lock:
            env = self._environments.get(name)
            if env is None:
                return None
            self._selected = name
            self._config = dict(env.config)
            return env

    def get_current_config(self) -> dict[str, tp.Any]:
        """Retrieve the current config.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        with self._lock:
            return {"environment": self._selected, "config": dict(self._config)}

    def edit_config(self, updates: dict[str, tp.Any]) -> dict[str, tp.Any]:
        """Edit config.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            updates (dict[str, tp.Any]): IN: updates. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        with self._lock:
            self._config.update(updates)
            return {"environment": self._selected, "config": dict(self._config)}

    def start(self) -> RLRun:
        """Start.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            RLRun: OUT: Result of the operation."""

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
        """Stop.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            run_id (str): IN: run id. OUT: Consumed during execution.
        Returns:
            RLRun | None: OUT: Result of the operation."""

        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            if run.status == "running":
                run.status = "stopped"
                run.ended_at = time.time()
            return run

    def status(self, run_id: str) -> RLRun | None:
        """Status.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            run_id (str): IN: run id. OUT: Consumed during execution.
        Returns:
            RLRun | None: OUT: Result of the operation."""

        with self._lock:
            return self._runs.get(run_id)

    def results(self, run_id: str) -> dict[str, tp.Any]:
        """Results.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            run_id (str): IN: run id. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
        """List runs.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[RLRun]: OUT: Result of the operation."""

        with self._lock:
            return list(self._runs.values())

    def test_inference(self, prompt: str, run_id: str | None = None) -> dict[str, tp.Any]:
        """Test inference.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            prompt (str): IN: prompt. OUT: Consumed during execution.
            run_id (str | None, optional): IN: run id. Defaults to None. OUT: Consumed during execution.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        with self._lock:
            return {
                "run_id": run_id or self._selected or "ad-hoc",
                "prompt": prompt[:200],
                "completion": f"[mock] {prompt[:80]}",
            }


_backend_lock = threading.Lock()
_backend: RLBackend = InMemoryRLBackend()


def set_rl_backend(backend: RLBackend) -> None:
    """Set the rl backend.

    Args:
        backend (RLBackend): IN: backend. OUT: Consumed during execution."""

    global _backend
    with _backend_lock:
        _backend = backend


def get_rl_backend() -> RLBackend:
    """Retrieve the rl backend.

    Returns:
        RLBackend: OUT: Result of the operation."""

    with _backend_lock:
        return _backend


def reset_rl_backend() -> None:
    """Reset rl backend."""

    set_rl_backend(InMemoryRLBackend())


def _run_to_dict(run: RLRun) -> dict[str, tp.Any]:
    """Internal helper to run to dict.

    Args:
        run (RLRun): IN: run. OUT: Consumed during execution.
    Returns:
        dict[str, tp.Any]: OUT: Result of the operation."""

    return {
        "run_id": run.run_id,
        "environment": run.environment,
        "status": run.status,
        "metrics": dict(run.metrics),
        "started_at": run.started_at,
        "ended_at": run.ended_at,
    }


class rl_list_environments(AgentBaseFn):
    """Rl list environments.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        envs = get_rl_backend().list_environments()
        return {
            "count": len(envs),
            "environments": [{"name": e.name, "description": e.description, "config": dict(e.config)} for e in envs],
        }


class rl_select_environment(AgentBaseFn):
    """Rl select environment.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(name: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            name (str): IN: name. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        env = get_rl_backend().select_environment(name)
        if env is None:
            return {"error": "not_found", "name": name}
        return {"name": env.name, "description": env.description, "config": dict(env.config)}


class rl_get_current_config(AgentBaseFn):
    """Rl get current config.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return get_rl_backend().get_current_config()


class rl_edit_config(AgentBaseFn):
    """Rl edit config.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        updates: dict[str, tp.Any],
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            updates (dict[str, tp.Any]): IN: updates. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return get_rl_backend().edit_config(dict(updates or {}))


class rl_start_training(AgentBaseFn):
    """Rl start training.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        try:
            run = get_rl_backend().start()
        except Exception as exc:
            return {"error": str(exc)}
        return _run_to_dict(run)


class rl_stop_training(AgentBaseFn):
    """Rl stop training.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(run_id: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            run_id (str): IN: run id. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        run = get_rl_backend().stop(run_id)
        if run is None:
            return {"error": "not_found", "run_id": run_id}
        return _run_to_dict(run)


class rl_check_status(AgentBaseFn):
    """Rl check status.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(run_id: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            run_id (str): IN: run id. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        run = get_rl_backend().status(run_id)
        if run is None:
            return {"error": "not_found", "run_id": run_id}
        return _run_to_dict(run)


class rl_get_results(AgentBaseFn):
    """Rl get results.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(run_id: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            run_id (str): IN: run id. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        return get_rl_backend().results(run_id)


class rl_list_runs(AgentBaseFn):
    """Rl list runs.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        runs = sorted(
            get_rl_backend().list_runs(),
            key=lambda r: r.started_at,
            reverse=True,
        )
        return {"count": len(runs), "runs": [_run_to_dict(r) for r in runs]}


class rl_test_inference(AgentBaseFn):
    """Rl test inference.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        prompt: str,
        run_id: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            prompt (str): IN: prompt. OUT: Consumed during execution.
            run_id (str | None, optional): IN: run id. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
