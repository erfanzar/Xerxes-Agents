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
"""Tests for training.trajectory_compressor + training.batch_runner."""

from __future__ import annotations

import json

from xerxes.context.compressor import ContextCompressor
from xerxes.training.batch_runner import (
    BatchRecord,
    BatchResult,
    BatchRunner,
    content_hash,
    load_completed_ids,
    read_jsonl,
)
from xerxes.training.trajectory_compressor import TrajectoryCompressor

# ---------------------------- trajectory compressor -------------------------


class TestTrajectoryCompressor:
    def _msgs(self, n: int):
        return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"} for i in range(n)]

    def test_compress_one_returns_metrics(self):
        c = TrajectoryCompressor(
            compressor=ContextCompressor(
                threshold=0.01,
                context_window=200,
                protect_first=1,
                protect_last=1,
                summarizer=lambda m, b: "SUMMARY",
            )
        )
        traj = {"id": "t1", "messages": self._msgs(6)}
        out, metrics = c.compress_one(traj)
        assert metrics.trajectory_id == "t1"
        assert metrics.tokens_before > 0
        assert len(out["messages"]) < 6 or metrics.strategy == "no-middle"

    def test_run_processes_and_skips(self, tmp_path):
        c = TrajectoryCompressor(
            compressor=ContextCompressor(
                threshold=0.01,
                context_window=200,
                protect_first=1,
                protect_last=1,
                summarizer=lambda m, b: "S",
            )
        )
        run = c.run(
            [
                {"id": "a", "messages": self._msgs(5)},
                {"id": "b", "messages": self._msgs(5)},
            ],
            out_path=tmp_path / "out.jsonl",
            already_done={"b"},
            metrics_path=tmp_path / "metrics.json",
        )
        assert run.processed == 1
        assert run.skipped == 1
        assert (tmp_path / "out.jsonl").exists()
        assert (tmp_path / "metrics.json").exists()

    def test_run_records_errors(self, tmp_path):
        # Override compress_one to fail.
        c = TrajectoryCompressor()

        def boom(_traj):
            raise RuntimeError("synthetic")

        c.compress_one = boom  # type: ignore[assignment]
        run = c.run([{"id": "a", "messages": self._msgs(2)}])
        assert run.errors and run.errors[0][1] == "synthetic"


# ---------------------------- batch runner ---------------------------------


class TestBatchRunner:
    def test_run_each_record(self):
        seen: list[str] = []

        def runner(rec):
            seen.append(rec.id)
            return BatchResult(id=rec.id, response="ok", input_tokens=10, output_tokens=5)

        br = BatchRunner(runner)
        summary = br.run([BatchRecord(id="a", prompt="x"), BatchRecord(id="b", prompt="y")])
        assert summary.total == 2
        assert summary.succeeded == 2
        assert set(seen) == {"a", "b"}

    def test_resume_skips_completed(self):
        def runner(rec):
            return BatchResult(id=rec.id, response="ok")

        br = BatchRunner(runner)
        summary = br.run([BatchRecord(id="a", prompt="x"), BatchRecord(id="b", prompt="y")], resume_ids={"a"})
        assert summary.skipped == 1
        assert summary.succeeded == 1

    def test_writes_results_to_jsonl(self, tmp_path):
        def runner(rec):
            return BatchResult(id=rec.id, response="ok", input_tokens=5, output_tokens=3, cost_usd=0.01)

        br = BatchRunner(runner)
        out = tmp_path / "out.jsonl"
        br.run([BatchRecord(id="a", prompt="x")], out_path=out)
        text = out.read_text(encoding="utf-8")
        assert "ok" in text
        assert "a" in text

    def test_run_captures_errors(self):
        def runner(rec):
            raise RuntimeError("kaboom")

        br = BatchRunner(runner)
        summary = br.run([BatchRecord(id="a", prompt="x")])
        assert summary.failed == 1

    def test_content_hash_stable(self):
        rec_a = BatchRecord(id="x", prompt="hi", metadata={"k": 1})
        rec_b = BatchRecord(id="y", prompt="hi", metadata={"k": 1})
        assert content_hash(rec_a) == content_hash(rec_b)

    def test_read_jsonl(self, tmp_path):
        path = tmp_path / "in.jsonl"
        path.write_text(json.dumps({"id": "a", "prompt": "hi"}) + "\n" + json.dumps({"id": "b", "prompt": "yo"}) + "\n")
        records = read_jsonl(path)
        assert [r.id for r in records] == ["a", "b"]

    def test_load_completed_ids(self, tmp_path):
        path = tmp_path / "out.jsonl"
        path.write_text(json.dumps({"id": "a", "response": "ok"}) + "\n")
        assert load_completed_ids(path) == {"a"}
