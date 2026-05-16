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
"""Concurrent bulk inference for offline eval and dataset generation.

Each :class:`BatchRecord` describes one prompt; the runner invokes
an injected :data:`RunnerFn` callable (the production version wraps
``run_agent_loop``), captures per-record stats, and supports
content-hash deduplication when resuming a partial JSONL output.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BatchRecord:
    """One row of input for :class:`BatchRunner`.

    Attributes:
        id: stable identifier, written to the output JSONL.
        prompt: raw text fed to the runner callable.
        metadata: free-form fields kept for downstream consumers.
    """

    id: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """One row of output, with usage and (optionally) an error message.

    Attributes:
        id: matches the originating :class:`BatchRecord` id.
        response: assistant output text.
        tool_calls: count of tool invocations observed.
        input_tokens: prompt tokens consumed.
        output_tokens: completion tokens emitted.
        cost_usd: estimated USD cost (see :func:`llms.calc_cost`).
        finish_reason: model stop reason or ``"error"``.
        error: error string when the run failed, else ``None``.
        metadata: free-form fields propagated by the runner.
    """

    id: str
    response: str
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = "stop"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchSummary:
    """Aggregate counters returned by :meth:`BatchRunner.run`.

    Attributes:
        total: records seen (including dedup hits).
        succeeded: runs that returned without an error.
        failed: runs that raised or returned an error.
        skipped: records skipped due to dedup.
        total_input_tokens: sum of successful input token counts.
        total_output_tokens: sum of successful output token counts.
        total_cost_usd: sum of successful per-run cost estimates.
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0


RunnerFn = Callable[[BatchRecord], BatchResult]


def content_hash(record: BatchRecord) -> str:
    """Return a stable 16-char dedup hash of ``prompt + sorted metadata``."""
    body = record.prompt + "|" + json.dumps(record.metadata, sort_keys=True, default=str)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[BatchRecord]:
    """Load a JSONL file of input records (one per line)."""
    out: list[BatchRecord] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        out.append(
            BatchRecord(id=str(data.get("id") or "auto"), prompt=data["prompt"], metadata=data.get("metadata", {}))
        )
    return out


def load_completed_ids(path: Path, *, dedup_field: str = "id") -> set[str]:
    """Scan ``path`` (a results JSONL) and return previously written ids."""
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if dedup_field in data:
            out.add(str(data[dedup_field]))
    return out


class BatchRunner:
    """Fan out ``runner_fn`` across many records on a thread pool."""

    def __init__(
        self,
        runner_fn: RunnerFn,
        *,
        workers: int = 4,
    ) -> None:
        """Configure the executor with ``workers`` concurrent threads."""
        self._runner = runner_fn
        self._workers = max(1, int(workers))

    def run(
        self,
        records: Iterable[BatchRecord],
        *,
        out_path: Path | None = None,
        resume_ids: set[str] | None = None,
        dedup_by: str = "id",
    ) -> BatchSummary:
        """Execute ``records`` concurrently and append results to ``out_path``.

        ``resume_ids`` carries previously seen ids; ``dedup_by`` picks
        between record ``"id"`` and :func:`content_hash` as the key.
        """
        summary = BatchSummary()
        seen = set(resume_ids or ())
        handle = open(out_path, "a", encoding="utf-8") if out_path else None
        try:
            with cf.ThreadPoolExecutor(max_workers=self._workers) as pool:
                futures = []
                for rec in records:
                    key = rec.id if dedup_by == "id" else content_hash(rec)
                    summary.total += 1
                    if key in seen:
                        summary.skipped += 1
                        continue
                    seen.add(key)
                    futures.append((rec, pool.submit(self._safe_run, rec)))
                for _rec, fut in futures:
                    res = fut.result()
                    if handle is not None:
                        handle.write(json.dumps(asdict(res), default=str) + "\n")
                    if res.error:
                        summary.failed += 1
                    else:
                        summary.succeeded += 1
                        summary.total_input_tokens += res.input_tokens
                        summary.total_output_tokens += res.output_tokens
                        summary.total_cost_usd += res.cost_usd
        finally:
            if handle is not None:
                handle.close()
        return summary

    def _safe_run(self, rec: BatchRecord) -> BatchResult:
        """Invoke ``runner_fn`` and capture any exception as ``BatchResult.error``."""
        try:
            return self._runner(rec)
        except Exception as exc:
            return BatchResult(id=rec.id, response="", error=str(exc), finish_reason="error")


__all__ = [
    "BatchRecord",
    "BatchResult",
    "BatchRunner",
    "BatchSummary",
    "RunnerFn",
    "content_hash",
    "load_completed_ids",
    "read_jsonl",
]
