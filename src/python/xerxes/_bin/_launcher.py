# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Thin launcher that execs the bundled TypeScript/Ink CLI via Node.

Registered as the `xerxes` console_script entry point so that `xerxes`
works after `pip install`. The bundle is produced from
`src/typescript/xerxes-cli/` via `bun build --target=node` at install
time (see `hatch_build.py`).

Requires Node.js (≥20) on the user's PATH at runtime.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

BIN_DIR = Path(__file__).parent
BUNDLE = BIN_DIR / "xerxes.mjs"


def _find_node() -> str | None:
    """Locate a Node.js binary — prefer `node`, fall back to common install paths."""
    node = shutil.which("node")
    if node:
        return node
    candidates = [
        Path.home() / ".nvm" / "versions" / "node",
        Path("/usr/local/bin/node"),
        Path("/opt/homebrew/bin/node"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
        if c.is_dir():
            versions = sorted(c.glob("v*/bin/node"), reverse=True)
            if versions:
                return str(versions[0])
    return None


def launch() -> None:
    """Launch the native Xerxes CLI."""
    if not BUNDLE.exists():
        print(
            f"Xerxes CLI bundle not found at {BUNDLE}.\n"
            "Install Bun (https://bun.sh) and reinstall with `uv pip install -e .`.",
            file=sys.stderr,
        )
        sys.exit(1)

    node = _find_node()
    if not node:
        print(
            "Node.js not found. Xerxes's CLI requires Node ≥20 at runtime.\n"
            "Install from https://nodejs.org or `brew install node`.",
            file=sys.stderr,
        )
        sys.exit(1)

    args = [str(BUNDLE), *sys.argv[1:]]
    os.execv(node, [node, *args])
