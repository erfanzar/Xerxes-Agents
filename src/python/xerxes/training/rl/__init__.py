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
"""Reinforcement-learning training plumbing.

External dependencies (gated on credentials):

* Tinker — Nous Research's RL training service consumed via
  :class:`TinkerClient` / :class:`TinkerRunConfig`.
* Atropos — environment submodule providing the actual RL envs
  registered via :class:`RLEnvironmentRegistry`.
* Weights & Biases — metrics destination wrapped by :class:`WandBHook`.

The :class:`RLRunStatus` / :class:`RLRunState` types implement a
small state machine the daemon uses to report training progress.
"""

from .envs import RLEnvironment, RLEnvironmentRegistry, builtin_envs
from .status import RLRunState, RLRunStatus
from .tinker_client import TinkerClient, TinkerRunConfig
from .wandb_hook import WandBHook

__all__ = [
    "RLEnvironment",
    "RLEnvironmentRegistry",
    "RLRunState",
    "RLRunStatus",
    "TinkerClient",
    "TinkerRunConfig",
    "WandBHook",
    "builtin_envs",
]
