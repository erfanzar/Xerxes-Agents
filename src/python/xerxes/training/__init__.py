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
"""Training-data tooling.

Three subsystems:

* :class:`BatchRunner` — concurrent offline inference with resumable
  JSONL output and content-hash deduplication, used to build
  evaluation sets or distillation corpora.
* :class:`TrajectoryCompressor` — token-budgeted summarisation of
  recorded agent trajectories for RL replay.
* :mod:`xerxes.training.rl` — Tinker RL client wrapper, environment
  helpers, status reporters, and a Weights & Biases hook.
"""

from .batch_runner import BatchRecord, BatchRunner, BatchSummary
from .trajectory_compressor import TrajectoryCompressor

__all__ = ["BatchRecord", "BatchRunner", "BatchSummary", "TrajectoryCompressor"]
