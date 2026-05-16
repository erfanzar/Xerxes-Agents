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
"""Normalise Apache 2.0 license headers across the repo.

Targets ``.py``, ``.sh``, ``.yml``, ``.yaml``, ``Dockerfile``. Two fixes:

1.  Replace ``Copyright 2026 Xerxes-Agents Author`` with
    ``Copyright 2026 Xerxes-Agents Author`` (drops the "The" article).
2.  When a file has the opening license lines but is missing the trailing
    "Unless required by applicable law…" block, splice the block in
    immediately after the ``LICENSE-2.0`` line.

Idempotent — running it a second time is a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

WRONG_COPYRIGHT = "Copyright 2026 Xerxes-Agents Author"
RIGHT_COPYRIGHT = "Copyright 2026 Xerxes-Agents Author"

LICENSE_URL_LINE = "#     https://www.apache.org/licenses/LICENSE-2.0"

TRAILER = (
    "#\n"
    "# Unless required by applicable law or agreed to in writing, software\n"
    '# distributed under the License is distributed on an "AS IS" BASIS,\n'
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "# See the License for the specific language governing permissions and\n"
    "# limitations under the License."
)

TRAILER_MARKER = "Unless required by applicable law"

EXTS = {".py", ".sh", ".yml", ".yaml"}
FILE_NAMES = {"Dockerfile"}

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "build", "dist", ".ruff_cache", ".pytest_cache"}


def should_visit(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIRS:
        return False
    if path.suffix in EXTS:
        return True
    if path.name in FILE_NAMES:
        return True
    return False


def fix_text(text: str) -> tuple[str, list[str]]:
    """Return (new_text, list_of_fixes_applied)."""
    fixes: list[str] = []
    new_text = text

    if WRONG_COPYRIGHT in new_text:
        new_text = new_text.replace(WRONG_COPYRIGHT, RIGHT_COPYRIGHT)
        fixes.append("the-article")

    if LICENSE_URL_LINE in new_text and TRAILER_MARKER not in new_text:
        # Splice the trailer in directly after the LICENSE_URL_LINE. We do this
        # only once per file (first occurrence), which is what we want for
        # any sane header.
        idx = new_text.index(LICENSE_URL_LINE)
        end_of_line = new_text.index("\n", idx)
        new_text = new_text[: end_of_line + 1] + TRAILER + "\n" + new_text[end_of_line + 1 :]
        fixes.append("trailer")

    return new_text, fixes


def main() -> int:
    changed: list[tuple[Path, list[str]]] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or not should_visit(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_text, fixes = fix_text(text)
        if not fixes or new_text == text:
            continue
        path.write_text(new_text, encoding="utf-8")
        changed.append((path, fixes))

    print(f"Files changed: {len(changed)}")
    for path, fixes in sorted(changed, key=lambda p: str(p[0])):
        rel = path.relative_to(REPO_ROOT)
        print(f"  {rel}  [{', '.join(fixes)}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
