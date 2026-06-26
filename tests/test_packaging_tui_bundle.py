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

import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUI_BUNDLE = ROOT / "src" / "ui-tui" / "dist" / "entry.js"
TUI_BUNDLE_REL = "src/ui-tui/dist/entry.js"
PACKAGE_BUNDLE_PATH = "xerxes/_tui_dist/entry.js"


def test_hatch_force_includes_built_tui_bundle() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include[TUI_BUNDLE_REL] == PACKAGE_BUNDLE_PATH
    assert TUI_BUNDLE.is_file(), "run `npm run build` in src/ui-tui before packaging"
    assert TUI_BUNDLE.stat().st_size > 100_000


def test_built_tui_bundle_is_not_gitignored() -> None:
    if shutil.which("git") is None:
        return

    proc = subprocess.run(
        ["git", "check-ignore", "-v", TUI_BUNDLE_REL],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "!dist/entry.js" in proc.stdout
