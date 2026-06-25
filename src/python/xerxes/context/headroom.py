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
"""Native Headroom-style compression for model-visible tool results.

Xerxes keeps the full original in project memory, then sends the model a
bounded preview. This module provides deterministic content routing and
specialized previews for the high-noise outputs agents produce most often:
JSON, unified diffs, grep/ripgrep results, logs, and generic text.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, TypeVar

DEFAULT_HEADROOM_PREVIEW_CHARS = 4_000
_IMPORTANT_RE = re.compile(
    r"\b(error|failed?|failure|fatal|exception|traceback|warning|warn|panic|timeout|denied|segmentation|assert)\b",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"(=+ .* =+|short test summary|tests? passed|tests? failed|passed in|failed in|npm err|cargo test|all checks passed)",
    re.IGNORECASE,
)
_T = TypeVar("_T")


@dataclass(frozen=True)
class HeadroomResult:
    """Compressed tool-output preview and metadata."""

    content_type: str
    original_chars: int
    compressed_chars: int
    original_lines: int
    compressed_lines: int
    compressed: str
    notes: tuple[str, ...] = ()

    @property
    def kept_ratio(self) -> float:
        """Return compressed/original character ratio."""
        if self.original_chars <= 0:
            return 1.0
        return min(1.0, self.compressed_chars / self.original_chars)

    def metadata_line(self) -> str:
        """Return a compact human-readable compression summary."""
        percent = self.kept_ratio * 100
        return (
            f"{self.content_type} preview: {self.original_chars:,} -> "
            f"{self.compressed_chars:,} chars ({percent:.1f}% kept)"
        )


def compress_tool_result(
    tool_name: str,
    content: Any,
    *,
    max_chars: int = DEFAULT_HEADROOM_PREVIEW_CHARS,
) -> HeadroomResult:
    """Compress a tool result into a bounded preview.

    Args:
        tool_name: Tool that produced the output. Used only as a hint.
        content: Raw tool output.
        max_chars: Maximum preview size before a final fit pass.

    Returns:
        A routed, bounded preview with metadata.
    """
    text = _stringify(content)
    if not text:
        return _result("empty", text, "", notes=("empty tool result",))
    if _looks_binary(text):
        preview = "[Binary-like tool result omitted. Read the stored project-memory artifact for bytes.]"
        return _result("binary", text, preview, notes=("binary-like content",))

    bounded_chars = max(512, int(max_chars))
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        compressed = _compress_json(text, bounded_chars)
        if compressed is not None:
            return _result("json", text, compressed)
    if _looks_like_diff(text):
        return _result("diff", text, _compress_diff(text, bounded_chars))
    if _looks_like_search_output(text):
        return _result("search", text, _compress_search(text, bounded_chars))
    if _looks_like_log(text, tool_name=tool_name):
        return _result("log", text, _compress_log(text, bounded_chars))
    return _result("text", text, _compress_text(text, bounded_chars))


def _result(content_type: str, original: str, compressed: str, *, notes: tuple[str, ...] = ()) -> HeadroomResult:
    fitted = compressed.rstrip()
    return HeadroomResult(
        content_type=content_type,
        original_chars=len(original),
        compressed_chars=len(fitted),
        original_lines=len(original.splitlines()),
        compressed_lines=len(fitted.splitlines()),
        compressed=fitted,
        notes=notes,
    )


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return str(content)


def _looks_binary(text: str) -> bool:
    if "\x00" in text:
        return True
    sample = text[:4096]
    if not sample:
        return False
    control_count = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\n\r\t")
    return control_count / len(sample) > 0.05


def _fit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.rstrip()
    marker = f"\n\n[... {len(text) - max_chars:,} chars omitted by Xerxes headroom ...]\n\n"
    remaining = max(256, max_chars - len(marker))
    head = remaining // 2
    tail = remaining - head
    return f"{text[:head].rstrip()}{marker}{text[-tail:].lstrip()}".rstrip()


def _compress_json(text: str, max_chars: int) -> str | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None

    minified = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(minified) <= max_chars:
        return minified

    shrunk = _shrink_json(value)
    rendered = json.dumps(shrunk, ensure_ascii=False, indent=2, sort_keys=True)
    return _fit_text(rendered, max_chars)


def _shrink_json(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return _json_atom_preview(value)
    if isinstance(value, dict):
        items = list(value.items())
        if len(items) > 24:
            important = [(key, item) for key, item in items if _json_key_is_important(key)]
            head = items[:10]
            tail = items[-5:]
            selected = _dedupe_pairs([*important[:8], *head, *tail])
            omitted = len(items) - len(selected)
        else:
            selected = items
            omitted = 0
        out: dict[str, Any] = {str(key): _shrink_json(item, depth=depth + 1) for key, item in selected}
        if omitted > 0:
            out["..."] = f"{omitted} keys omitted"
        return out
    if isinstance(value, list):
        important = [item for item in value if _json_value_is_important(item)]
        sample = _dedupe_json_items([*important[:5], *value[:3], *value[-2:]])
        omitted = len(value) - len(sample)
        return {
            "type": "array",
            "items": len(value),
            "sample": [_shrink_json(item, depth=depth + 1) for item in sample],
            "omitted": max(0, omitted),
        }
    return _json_atom_preview(value)


def _json_key_is_important(key: Any) -> bool:
    return bool(_IMPORTANT_RE.search(str(key)))


def _json_value_is_important(value: Any) -> bool:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        rendered = str(value)
    return bool(_IMPORTANT_RE.search(rendered))


def _json_atom_preview(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 240:
        return f"{value[:160]} ... [{len(value) - 220} chars omitted] ... {value[-60:]}"
    return value


def _dedupe_pairs(items: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    seen: set[str] = set()
    selected: list[tuple[Any, Any]] = []
    for key, value in items:
        marker = str(key)
        if marker not in seen:
            seen.add(marker)
            selected.append((key, value))
    return selected


def _dedupe_json_items(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    selected: list[Any] = []
    for item in items:
        try:
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            marker = repr(item)
        if marker not in seen:
            seen.add(marker)
            selected.append(item)
    return selected


def _looks_like_diff(text: str) -> bool:
    return text.startswith("diff --git ") or "\ndiff --git " in text or bool(re.search(r"^@@ -\d+", text, re.MULTILINE))


def _compress_diff(text: str, max_chars: int) -> str:
    files = _split_diff_files(text)
    if not files:
        return _compress_text(text, max_chars)

    max_files = 12
    total_files = len(files)
    selected_files = files[:max_files]
    out: list[str] = [
        f"[Xerxes headroom diff preview: {total_files} files, showing {len(selected_files)}]",
    ]
    for file_index, file_text in enumerate(selected_files, start=1):
        reduced = _compress_diff_file(file_text)
        out.extend(["", f"# File {file_index}", reduced.rstrip()])
    if total_files > len(selected_files):
        out.append(f"\n[... {total_files - len(selected_files)} diff files omitted ...]")
    return _fit_text("\n".join(out), max_chars)


def _split_diff_files(text: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"^diff --git ", text, re.MULTILINE)]
    if not starts:
        return [text] if "@@ -" in text else []
    starts.append(len(text))
    return [text[starts[index] : starts[index + 1]].rstrip() for index in range(len(starts) - 1)]


def _compress_diff_file(file_text: str) -> str:
    lines = file_text.splitlines()
    hunk_indices = [index for index, line in enumerate(lines) if line.startswith("@@ ")]
    if not hunk_indices:
        return "\n".join(lines[:40])

    first_hunk = hunk_indices[0]
    header = lines[:first_hunk]
    hunk_bounds = [*hunk_indices, len(lines)]
    hunks = [lines[hunk_bounds[index] : hunk_bounds[index + 1]] for index in range(len(hunk_bounds) - 1)]
    selected = hunks[:4]
    if len(hunks) > 6:
        selected.extend(hunks[-2:])
    elif len(hunks) > 4:
        selected.extend(hunks[4:])

    out = [*header[:12]]
    if len(header) > 12:
        out.append(f"[... {len(header) - 12} header lines omitted ...]")
    for hunk in selected:
        out.extend(_trim_diff_hunk(hunk))
    if len(hunks) > len(selected):
        out.append(f"[... {len(hunks) - len(selected)} hunks omitted ...]")
    return "\n".join(out)


def _trim_diff_hunk(hunk: list[str], *, context: int = 2) -> list[str]:
    if not hunk:
        return []
    keep = {0}
    for index, line in enumerate(hunk):
        if index == 0:
            continue
        if (line.startswith("+") and not line.startswith("+++")) or (
            line.startswith("-") and not line.startswith("---")
        ):
            for kept_index in range(max(1, index - context), min(len(hunk), index + context + 1)):
                keep.add(kept_index)
    out: list[str] = []
    previous = -1
    for index in sorted(keep):
        if previous != -1 and index > previous + 1:
            out.append(f"[... {index - previous - 1} context lines omitted ...]")
        out.append(hunk[index])
        previous = index
    return out


def _looks_like_search_output(text: str) -> bool:
    lines = [line for line in text.splitlines()[:80] if line.strip()]
    if len(lines) < 6:
        return False
    parsed = sum(1 for line in lines if _parse_search_line(line) is not None)
    return parsed >= max(4, len(lines) // 3)


def _parse_search_line(line: str) -> tuple[str, int, str] | None:
    for match in re.finditer(r"[:\-](\d+)[:\-]", line):
        path = line[: match.start()]
        if not path:
            continue
        try:
            number = int(match.group(1))
        except ValueError:
            continue
        content = line[match.end() :]
        return path, number, content
    return None


def _compress_search(text: str, max_chars: int) -> str:
    grouped: OrderedDict[str, list[tuple[int, str]]] = OrderedDict()
    unparsed = 0
    total_matches = 0
    for line in text.splitlines():
        parsed = _parse_search_line(line)
        if parsed is None:
            unparsed += 1
            continue
        path, number, content = parsed
        grouped.setdefault(path, []).append((number, content))
        total_matches += 1

    max_files = 20
    max_per_file = 6
    out = [f"[Xerxes headroom search preview: {total_matches} matches across {len(grouped)} files]"]
    for path, matches in list(grouped.items())[:max_files]:
        shown = _select_first_last(matches, max_per_file)
        out.extend(["", path])
        for number, content in shown:
            out.append(f"  {number}: {content}")
        omitted = len(matches) - len(shown)
        if omitted > 0:
            out.append(f"  [... {omitted} more matches in this file ...]")
    dropped_files = max(0, len(grouped) - max_files)
    if dropped_files:
        out.append(f"\n[... {dropped_files} files omitted ...]")
    if unparsed:
        out.append(f"\n[... {unparsed} non-search lines omitted ...]")
    return _fit_text("\n".join(out), max_chars)


def _select_first_last(items: list[_T], limit: int) -> list[_T]:
    if len(items) <= limit:
        return list(items)
    head_count = max(1, limit // 2)
    tail_count = max(1, limit - head_count)
    return [*items[:head_count], *items[-tail_count:]]


def _looks_like_log(text: str, *, tool_name: str) -> bool:
    lowered_name = tool_name.lower()
    if any(hint in lowered_name for hint in ("exec", "shell", "test", "pytest", "build", "npm", "cargo")):
        return True
    lines = text.splitlines()
    if len(lines) < 40:
        return False
    marked = sum(1 for line in lines[:200] if _IMPORTANT_RE.search(line) or _SUMMARY_RE.search(line))
    return marked > 0


def _compress_log(text: str, max_chars: int) -> str:
    lines = text.splitlines()
    important = {
        index
        for index, line in enumerate(lines)
        if _IMPORTANT_RE.search(line) or _SUMMARY_RE.search(line) or line.startswith(("FAILED ", "ERROR "))
    }
    keep = set(range(min(8, len(lines))))
    keep.update(range(max(0, len(lines) - 10), len(lines)))
    for index in important:
        for kept_index in range(max(0, index - 2), min(len(lines), index + 3)):
            keep.add(kept_index)

    out = [f"[Xerxes headroom log preview: kept {len(keep)} of {len(lines)} lines]"]
    previous = -1
    for index in sorted(keep):
        if previous != -1 and index > previous + 1:
            out.append(f"[... {index - previous - 1} log lines omitted ...]")
        out.append(f"L{index + 1}: {lines[index]}")
        previous = index
    return _fit_text("\n".join(out), max_chars)


def _compress_text(text: str, max_chars: int) -> str:
    lines = text.splitlines()
    if len(lines) <= 60:
        return _fit_text(text, max_chars)
    selected = _select_first_last(lines, 40)
    omitted = len(lines) - len(selected)
    head_count = min(20, len(selected))
    out = [
        f"[Xerxes headroom text preview: {len(lines)} lines, {omitted} omitted]",
        *selected[:head_count],
    ]
    tail = selected[head_count:]
    if tail:
        out.extend([f"[... {omitted} middle lines omitted ...]", *tail])
    return _fit_text("\n".join(out), max_chars)
