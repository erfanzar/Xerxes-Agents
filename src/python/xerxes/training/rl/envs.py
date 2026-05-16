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
"""RL environment registry plus a few built-in environment definitions.

Each env carries metadata and an optional ``reward_fn`` callable
that maps a trajectory dict to a scalar reward. Real RL workloads
(SWE-bench, terminal-test, agentic-OPD) live in ``/environments/``
outside the package; this module is the discovery surface Xerxes
exposes to RL drivers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

RewardFn = Callable[[dict[str, Any]], float]


@dataclass
class RLEnvironment:
    """Metadata describing one RL training environment.

    Attributes:
        name: unique slug used for registration and CLI lookup.
        description: short human-readable summary.
        config_path: optional path to an env-specific YAML/JSON.
        backend: ``"local"`` / ``"modal"`` / ``"docker"`` / etc.
        tags: free-form labels for filtering.
        reward_fn: trajectory -> reward callable; absent for envs
            scored entirely by the backend.
    """

    name: str
    description: str
    config_path: str = ""
    backend: str = "local"
    tags: list[str] = field(default_factory=list)
    reward_fn: RewardFn | None = None


class RLEnvironmentRegistry:
    """In-process registry mapping env name to :class:`RLEnvironment`."""

    def __init__(self) -> None:
        """Start with an empty registry."""
        self._envs: dict[str, RLEnvironment] = {}

    def register(self, env: RLEnvironment) -> None:
        """Register ``env``; raises ``ValueError`` if its name is empty."""
        if not env.name:
            raise ValueError("env name required")
        self._envs[env.name] = env

    def list_envs(self) -> list[RLEnvironment]:
        """Return all registered environments sorted by name."""
        return sorted(self._envs.values(), key=lambda e: e.name)

    def get(self, name: str) -> RLEnvironment | None:
        """Look up an environment by name, or ``None`` if absent."""
        return self._envs.get(name)


def builtin_envs() -> RLEnvironmentRegistry:
    """Return a registry pre-populated with the bundled environments."""
    reg = RLEnvironmentRegistry()
    reg.register(
        RLEnvironment(
            name="xerxes-terminal-test",
            description="Synthetic 4-task curriculum (greeting.txt, count.txt, answer.txt, eval-arithmetic)",
            backend="local",
            tags=["coding", "tooling"],
            reward_fn=lambda traj: 1.0 if traj.get("passed_all_tasks") else 0.0,
        )
    )
    reg.register(
        RLEnvironment(
            name="xerxes-swe-bench",
            description="SWE-bench style env: write code → tests verify reward.",
            backend="modal",
            tags=["coding", "swe"],
            reward_fn=lambda traj: float(traj.get("tests_passed", 0)) / max(1.0, float(traj.get("tests_total", 1))),
        )
    )
    reg.register(
        RLEnvironment(
            name="agentic-opd",
            description="Objective Policy Distribution: open-ended objective scoring.",
            backend="modal",
            tags=["research"],
        )
    )
    return reg


__all__ = ["RLEnvironment", "RLEnvironmentRegistry", "RewardFn", "builtin_envs"]
