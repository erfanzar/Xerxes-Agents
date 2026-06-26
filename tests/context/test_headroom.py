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

from __future__ import annotations

import json

from xerxes.context.headroom import compress_tool_result


def test_headroom_compresses_json_but_keeps_errors() -> None:
    payload = [{"id": index, "status": "ok", "payload": "x" * 120} for index in range(50)]
    payload.append({"id": 99, "error": "boom", "traceback": "frame\n" * 80})

    result = compress_tool_result("JSONProcessor", json.dumps(payload), max_chars=900)

    assert result.content_type == "json"
    assert result.original_chars > result.compressed_chars
    assert "boom" in result.compressed
    assert "omitted" in result.compressed
    assert len(result.compressed) <= 900


def test_headroom_compresses_unified_diff_by_files_and_hunks() -> None:
    chunks: list[str] = []
    for file_index in range(14):
        chunks.extend(
            [
                f"diff --git a/src/file_{file_index}.py b/src/file_{file_index}.py",
                f"--- a/src/file_{file_index}.py",
                f"+++ b/src/file_{file_index}.py",
            ]
        )
        for hunk_index in range(8):
            chunks.extend(
                [
                    f"@@ -{hunk_index * 10},6 +{hunk_index * 10},6 @@",
                    " context before",
                    f"-old value {file_index}-{hunk_index}",
                    f"+new value {file_index}-{hunk_index}",
                    " context after",
                ]
            )
    text = "\n".join(chunks)

    result = compress_tool_result("exec_command", text, max_chars=1800)

    assert result.content_type == "diff"
    assert "Xerxes headroom diff preview" in result.compressed
    assert "diff --git a/src/file_0.py b/src/file_0.py" in result.compressed
    assert "hunks omitted" in result.compressed
    assert len(result.compressed) <= 1800


def test_headroom_groups_search_results_by_file() -> None:
    lines = [
        f"src/module_{file_index}.py:{line_index}:def function_{line_index}(): pass"
        for file_index in range(5)
        for line_index in range(12)
    ]

    result = compress_tool_result("GrepTool", "\n".join(lines), max_chars=1200)

    assert result.content_type == "search"
    assert "Xerxes headroom search preview" in result.compressed
    assert "src/module_0.py" in result.compressed
    assert "more matches in this file" in result.compressed
    assert len(result.compressed) <= 1200


def test_headroom_log_preview_keeps_errors_and_summaries() -> None:
    lines = [f"INFO build line {index}" for index in range(120)]
    lines[70] = "ERROR tests/test_runtime.py::test_case failed"
    lines[71] = "Traceback (most recent call last):"
    lines[72] = "  File 'tests/test_runtime.py', line 10, in test_case"
    lines.append("FAILED tests/test_runtime.py::test_case - AssertionError")
    lines.append("1 failed, 200 passed in 12.34s")

    result = compress_tool_result("exec_command", "\n".join(lines), max_chars=1100)

    assert result.content_type == "log"
    assert "ERROR tests/test_runtime.py::test_case failed" in result.compressed
    assert "1 failed, 200 passed" in result.compressed
    assert "log lines omitted" in result.compressed
    assert len(result.compressed) <= 1100
