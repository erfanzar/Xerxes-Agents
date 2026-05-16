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
"""State machine for tracking an RL training run.

``RLRunStatus`` is the set of legal states; ``_TRANSITIONS`` encodes
the allowed moves between them, exposed via :func:`can_transition`.
:class:`RLRunState` is the dataclass snapshot the daemon broadcasts.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class RLRunStatus(enum.Enum):
    """Lifecycle states of an RL training run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RLRunState:
    """Current observable state of one RL training run.

    Attributes:
        run_id: stable run identifier.
        status: current :class:`RLRunStatus`.
        iteration: current training iteration counter.
        reward: latest reward value (mean over last batch).
        loss: latest loss value.
        tokens_seen: cumulative tokens consumed so far.
        wandb_url: live dashboard URL when WandB is wired.
        error: failure message when ``status == FAILED``.
        metadata: free-form annotations.
    """

    run_id: str
    status: RLRunStatus = RLRunStatus.PENDING
    iteration: int = 0
    reward: float | None = None
    loss: float | None = None
    tokens_seen: int = 0
    wandb_url: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_TRANSITIONS = {
    RLRunStatus.PENDING: {RLRunStatus.RUNNING, RLRunStatus.CANCELLED, RLRunStatus.FAILED},
    RLRunStatus.RUNNING: {RLRunStatus.SUCCEEDED, RLRunStatus.FAILED, RLRunStatus.CANCELLED},
    RLRunStatus.SUCCEEDED: set(),
    RLRunStatus.FAILED: set(),
    RLRunStatus.CANCELLED: set(),
}


def can_transition(from_status: RLRunStatus, to_status: RLRunStatus) -> bool:
    """Return ``True`` when ``from_status -> to_status`` is allowed."""
    return to_status in _TRANSITIONS[from_status]


__all__ = ["RLRunState", "RLRunStatus", "can_transition"]
