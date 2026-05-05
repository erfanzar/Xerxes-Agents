#!/usr/bin/env python3
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
"""Remove docstrings and comments from Python source files."""

import ast
import io
import sys
import tokenize
from pathlib import Path


def get_docstring_positions(source: str) -> set[tuple[int, int]]:
    """Return a set of (start, end) character positions for all docstrings."""
    positions = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return positions

    # Module-level docstring
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        node = tree.body[0]
        positions.add((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef)):
            # Also remove attribute docstrings (standalone string literals)
            for _i, stmt in enumerate(node.body):
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    positions.add((stmt.lineno, stmt.col_offset, stmt.end_lineno, stmt.end_col_offset))
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if docstring is None:
            continue
        if node.body and isinstance(node.body[0], ast.Expr):
            expr = node.body[0]
            positions.add((expr.lineno, expr.col_offset, expr.end_lineno, expr.end_col_offset))
    return positions


def remove_docstrings_and_comments(source: str) -> str:
    """Return source with docstrings and comments removed."""
    docstring_positions = get_docstring_positions(source)

    tokens = []
    try:
        readline = io.StringIO(source).readline
        for tok in tokenize.generate_tokens(readline):
            tokens.append(tok)
    except tokenize.TokenError:
        pass

    # Build a list of character intervals to remove.
    intervals = []

    for tok in tokens:
        t_type = tok.type
        t_start = tok.start
        t_end = tok.end

        if t_type == tokenize.COMMENT:
            intervals.append((*t_start, *t_end))
            continue

        if t_type == tokenize.STRING:
            # Check if this string is a docstring (match start position only,
            # because AST and tokenizer may disagree on end column in Python 3.13+)
            for dl, dc, _el, _ec in docstring_positions:
                if t_start == (dl, dc):
                    intervals.append((*t_start, *t_end))
                    break

    if not intervals:
        return source

    # Sort intervals by position
    intervals.sort(key=lambda x: (x[0], x[1]))

    # Merge overlapping or adjacent intervals
    merged = [list(intervals[0])]
    for current in intervals[1:]:
        last = merged[-1]
        # Check if current overlaps or is adjacent to last
        if (current[0] < last[2]) or (current[0] == last[2] and current[1] <= last[3]):
            # Extend last if current ends later
            if (current[2] > last[2]) or (current[2] == last[2] and current[3] > last[3]):
                last[2] = current[2]
                last[3] = current[3]
        else:
            merged.append(list(current))

    # Now reconstruct the source by removing merged intervals.
    lines = source.splitlines(keepends=True)

    if not lines:
        return source

    # Map intervals to lines
    line_intervals = {i: [] for i in range(1, len(lines) + 1)}

    for sl, sc, el, ec in merged:
        for line_no in range(sl, el + 1):
            start_c = sc if line_no == sl else 0
            end_c = ec if line_no == el else None  # None means end of line
            line_intervals[line_no].append((start_c, end_c))

    result_lines = []
    for i, line in enumerate(lines, start=1):
        intervals_on_line = line_intervals[i]
        if not intervals_on_line:
            result_lines.append(line)
            continue

        # Sort and merge intervals on this line
        intervals_on_line.sort()
        merged_line = [list(intervals_on_line[0])]
        for cur in intervals_on_line[1:]:
            lst = merged_line[-1]
            if cur[0] <= lst[1] or lst[1] is None:
                if lst[1] is None or (cur[1] is not None and cur[1] > lst[1]):
                    lst[1] = cur[1]
            else:
                merged_line.append(list(cur))

        # Build the line without intervals
        new_line_parts = []
        pos = 0
        line_content = line.rstrip("\n\r")
        newline = line[len(line_content) :]

        for start, end in merged_line:
            if start > pos:
                new_line_parts.append(line_content[pos:start])
            pos = end if end is not None else len(line_content)

        if pos < len(line_content):
            new_line_parts.append(line_content[pos:])

        new_line = "".join(new_line_parts).rstrip()
        if new_line or (not new_line and i < len(lines)):
            result_lines.append(new_line + newline)
        else:
            result_lines.append(newline)

    result = "".join(result_lines)
    # Clean up: collapse multiple blank lines
    cleaned_lines = []
    for line in result.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "":
            # Only keep blank line if previous line wasn't blank
            if cleaned_lines and cleaned_lines[-1].strip() == "":
                continue
        cleaned_lines.append(line)

    # Remove trailing blank lines
    while cleaned_lines and cleaned_lines[-1].strip() == "":
        cleaned_lines.pop()

    final = "".join(cleaned_lines)
    if source.endswith("\n") and not final.endswith("\n"):
        final += "\n"
    return final


def process_file(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    cleaned = remove_docstrings_and_comments(source)
    if cleaned != source:
        path.write_text(cleaned, encoding="utf-8")
        print(f"Processed: {path}")
    else:
        print(f"No changes: {path}")


def main():
    files = sys.argv[1:]
    if not files:
        print("Usage: python remove_docstrings.py <file1.py> [file2.py] ...")
        sys.exit(1)
    for f in files:
        process_file(Path(f))


if __name__ == "__main__":
    main()
