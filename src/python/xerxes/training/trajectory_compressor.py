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
"""Offline trajectory compressor for training-data pipelines.

Wraps :class:`ContextCompressor` and adds:

* JSONL-in / JSONL-out IO with content-hash resume.
* Per-trajectory :class:`TrajectoryMetrics` capturing before/after
  token counts, ratios, and strategy labels.
* Parallel execution via :class:`concurrent.futures.ThreadPoolExecutor`.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..context.compressor import CompressionResult, ContextCompressor, naive_summarizer

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMetrics:
    """Per-trajectory compression report.

    Attributes:
        trajectory_id: input id or derived content hash.
        tokens_before: token count of the original messages.
        tokens_after: token count after compression.
        ratio: ``tokens_after / tokens_before`` (0 when before is 0).
        compressed_count: middle messages folded into a summary.
        pruned_tool_results: tool messages shrunk by pre-pruning.
        strategy: label of the strategy that produced the output.
    """

    trajectory_id: str
    tokens_before: int
    tokens_after: int
    ratio: float
    compressed_count: int
    pruned_tool_results: int
    strategy: str


@dataclass
class CompressionRun:
    """Aggregate result of one :meth:`TrajectoryCompressor.run`.

    Attributes:
        processed: trajectories successfully compressed.
        skipped: trajectories matched against ``already_done``.
        metrics: per-trajectory :class:`TrajectoryMetrics`.
        errors: ``(trajectory_id, exception_str)`` for failures.
    """

    processed: int = 0
    skipped: int = 0
    metrics: list[TrajectoryMetrics] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


class TrajectoryCompressor:
    """Compress trajectories in bulk with a thread-pool fanout."""

    def __init__(
        self,
        *,
        compressor: ContextCompressor | None = None,
        workers: int = 4,
    ) -> None:
        """Bind an inner :class:`ContextCompressor` and the worker count."""
        self._compressor = compressor or ContextCompressor(
            threshold=0.5, context_window=200_000, summarizer=naive_summarizer
        )
        self._workers = max(1, int(workers))

    @staticmethod
    def _hash_trajectory(traj: dict[str, Any]) -> str:
        """Return a 16-char hash of the trajectory messages (resume key)."""
        body = json.dumps(traj.get("messages", []), sort_keys=True, default=str)
        return hashlib.sha1(body.encode()).hexdigest()[:16]

    def compress_one(self, traj: dict[str, Any]) -> tuple[dict[str, Any], TrajectoryMetrics]:
        """Compress one trajectory and return ``(new_traj, metrics)``."""
        messages = traj.get("messages", [])
        result: CompressionResult = self._compressor.compress(list(messages))
        new_traj = dict(traj)
        new_traj["messages"] = result.messages
        new_traj["compression"] = {
            "strategy": result.metadata.get("strategy", "unknown"),
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "summary_tokens": result.summary_tokens,
        }
        metrics = TrajectoryMetrics(
            trajectory_id=str(traj.get("id") or self._hash_trajectory(traj)),
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            ratio=(result.tokens_after / result.tokens_before) if result.tokens_before else 0.0,
            compressed_count=result.compressed_count,
            pruned_tool_results=result.pruned_tool_results,
            strategy=result.metadata.get("strategy", "unknown"),
        )
        return new_traj, metrics

    def run(
        self,
        trajectories: Iterable[dict[str, Any]],
        *,
        out_path: Path | None = None,
        metrics_path: Path | None = None,
        already_done: set[str] | None = None,
    ) -> CompressionRun:
        """Compress ``trajectories`` and (optionally) write outputs and metrics.

        ``out_path`` receives one compressed trajectory per JSONL line;
        ``metrics_path`` receives a pretty-printed JSON array of every
        :class:`TrajectoryMetrics`.
        """
        run = CompressionRun()
        done = set(already_done or ())
        out_handle = open(out_path, "a", encoding="utf-8") if out_path else None
        try:
            with cf.ThreadPoolExecutor(max_workers=self._workers) as pool:
                futures: list[tuple[str, cf.Future]] = []
                for traj in trajectories:
                    tid = str(traj.get("id") or self._hash_trajectory(traj))
                    if tid in done:
                        run.skipped += 1
                        continue
                    futures.append((tid, pool.submit(self.compress_one, traj)))
                for tid, fut in futures:
                    try:
                        out_traj, metrics = fut.result()
                    except Exception as exc:
                        run.errors.append((tid, str(exc)))
                        continue
                    run.processed += 1
                    run.metrics.append(metrics)
                    if out_handle is not None:
                        out_handle.write(json.dumps(out_traj, default=str) + "\n")
        finally:
            if out_handle is not None:
                out_handle.close()
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps([m.__dict__ for m in run.metrics], indent=2),
                encoding="utf-8",
            )
        return run


__all__ = ["CompressionRun", "TrajectoryCompressor", "TrajectoryMetrics"]
