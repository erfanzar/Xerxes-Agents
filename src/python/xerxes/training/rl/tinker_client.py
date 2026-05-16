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
"""Tinker (Nous Research) RL service client wrapper.

Tinker is a hosted RL training orchestration service; its SDK is an
optional extra (``xerxes-agent[rl]``). This wrapper exposes
``start`` / ``status`` / ``cancel`` and accepts test doubles so
unit tests don't need credentials.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .status import RLRunState, RLRunStatus


@dataclass
class TinkerRunConfig:
    """Hyperparameters for one Tinker RL training run.

    Attributes:
        model: base model identifier to fine-tune.
        env: RL environment name (matches an :class:`RLEnvironment`).
        learning_rate: optimizer learning rate.
        batch_size: rollout batch size per iteration.
        steps: total training iterations.
        extra: provider-specific overrides merged into the payload.
    """

    model: str
    env: str
    learning_rate: float = 1e-5
    batch_size: int = 8
    steps: int = 100
    extra: dict[str, Any] = None  # type: ignore[assignment]

    def to_payload(self) -> dict[str, Any]:
        """Render as a JSON-safe dict for the Tinker run-create call."""
        payload: dict[str, Any] = {
            "model": self.model,
            "env": self.env,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "steps": self.steps,
        }
        if self.extra:
            payload.update(self.extra)
        return payload


class TinkerClient:
    """Tinker SDK wrapper with three injectable backend callables.

    Tests supply ``backend_*`` directly; production code calls
    :meth:`from_env` to build the SDK-backed client from
    ``TINKER_API_KEY``.
    """

    def __init__(
        self,
        *,
        backend_start: Callable[[dict[str, Any]], str] | None = None,
        backend_status: Callable[[str], dict[str, Any]] | None = None,
        backend_cancel: Callable[[str], bool] | None = None,
    ) -> None:
        """Bind explicit backend callables (or leave them ``None`` for fallbacks)."""
        self._start_fn = backend_start
        self._status_fn = backend_status
        self._cancel_fn = backend_cancel

    @classmethod
    def from_env(cls) -> TinkerClient:
        """Build a client from the installed Tinker SDK and ``TINKER_API_KEY``.

        Raises ``RuntimeError`` if the SDK or the API key is missing.
        """
        if importlib.util.find_spec("tinker") is None:
            raise RuntimeError("Tinker SDK not installed; install xerxes-agent[rl]")
        token = os.environ.get("TINKER_API_KEY")
        if not token:
            raise RuntimeError("TINKER_API_KEY not set")
        import tinker  # type: ignore

        sdk_client = tinker.Client(api_key=token)  # type: ignore[attr-defined]

        def start_fn(payload):
            return sdk_client.runs.create(**payload).id  # type: ignore[no-any-return]

        def status_fn(run_id):
            return sdk_client.runs.get(run_id).model_dump()  # type: ignore[no-any-return]

        def cancel_fn(run_id):
            sdk_client.runs.cancel(run_id)
            return True

        return cls(backend_start=start_fn, backend_status=status_fn, backend_cancel=cancel_fn)

    def start(self, config: TinkerRunConfig) -> str:
        """Submit a new run; return the Tinker-assigned ``run_id``."""
        if self._start_fn is None:
            raise RuntimeError("Tinker backend not configured")
        return self._start_fn(config.to_payload())

    def status(self, run_id: str) -> RLRunState:
        """Fetch and translate the current run status into :class:`RLRunState`."""
        if self._status_fn is None:
            return RLRunState(run_id=run_id, error="status backend not configured", status=RLRunStatus.FAILED)
        raw = self._status_fn(run_id)
        return RLRunState(
            run_id=run_id,
            status=_map_status(raw.get("status", "pending")),
            iteration=int(raw.get("iteration", 0)),
            reward=raw.get("reward"),
            loss=raw.get("loss"),
            tokens_seen=int(raw.get("tokens_seen", 0)),
            wandb_url=raw.get("wandb_url", ""),
            error=raw.get("error", ""),
        )

    def cancel(self, run_id: str) -> bool:
        """Cancel ``run_id``; return ``False`` when no cancel backend exists."""
        if self._cancel_fn is None:
            return False
        return self._cancel_fn(run_id)


def _map_status(s: str) -> RLRunStatus:
    """Map a vendor status string to :class:`RLRunStatus`."""
    s = s.lower()
    if s in ("pending", "queued"):
        return RLRunStatus.PENDING
    if s in ("running", "active"):
        return RLRunStatus.RUNNING
    if s in ("succeeded", "completed", "success"):
        return RLRunStatus.SUCCEEDED
    if s in ("cancelled", "canceled"):
        return RLRunStatus.CANCELLED
    return RLRunStatus.FAILED


__all__ = ["TinkerClient", "TinkerRunConfig"]
