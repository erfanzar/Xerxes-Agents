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
"""Remove duplicate/attribute docstrings from class and module bodies."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def remove_extra_docstrings(source: str) -> str:
    """Remove standalone string literals that are not the first statement in a class/module body."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    removals = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef)):
            continue

        body = node.body
        if not body:
            continue

        # Check if first statement is a docstring
        first_is_docstring = (
            isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )

        if not first_is_docstring:
            # Remove ALL standalone string literals in the body (except we keep first if it's docstring)
            pass

        for i in range(len(body)):
            stmt = body[i]
            if not isinstance(stmt, ast.Expr):
                continue
            if not isinstance(stmt.value, ast.Constant):
                continue
            if not isinstance(stmt.value.value, str):
                continue

            if i == 0 and first_is_docstring:
                continue  # Keep the official docstring

            # This is an extra string literal - remove it
            # Find the line range for this statement
            start_line = stmt.lineno - 1  # 0-indexed
            end_line = stmt.end_lineno - 1

            # Also remove trailing blank lines after the string literal
            while end_line + 1 < len(lines) and lines[end_line + 1].strip() == "":
                end_line += 1

            removals.append((start_line, end_line))

    if not removals:
        return source

    # Sort and merge removals
    removals.sort(reverse=True)

    for start, end in removals:
        # Remove lines[start:end+1]
        del lines[start : end + 1]

    return "".join(lines)


def process_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    cleaned = remove_extra_docstrings(source)

    if cleaned == source:
        return False

    # Verify syntax
    try:
        ast.parse(cleaned)
    except SyntaxError as e:
        print(f"  SKIP (would break syntax): {path} — {e}")
        return False

    path.write_text(cleaned, encoding="utf-8")
    print(f"  CLEANED: {path}")
    return True


def main():
    files = sys.argv[1:]
    if not files:
        print("Usage: python cleanup_duplicate_docstrings.py <file1.py> [file2.py] ...")
        sys.exit(1)

    total = 0
    for f in files:
        if process_file(Path(f)):
            total += 1

    print(f"\nTotal files cleaned: {total}")


if __name__ == "__main__":
    main()
