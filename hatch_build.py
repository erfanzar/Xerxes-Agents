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

"""Custom hatch build hook that bundles the TypeScript CLI during install.

When you run `pip install -e .` or `uv pip install -e .`, this hook:
1. Looks for the pre-built CLI bundle at `src/python/xerxes/_bin/xerxes.mjs`
2. If missing, tries to build it with `bun` from `src/typescript/xerxes-cli/`
3. Stages the bundle and launcher into the wheel
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class TSBuildHook(BuildHookInterface):
    """Hatch build hook that stages the TypeScript CLI bundle."""

    PLUGIN_NAME = "ts-build"

    def initialize(self, version: str, build_data: dict) -> None:
        """Called before the build — ensure bundle exists and stage it."""
        root = Path(self.root)
        ts_dir = root / "src" / "typescript" / "xerxes-cli"
        bin_dir = root / "src" / "python" / "xerxes" / "_bin"
        bundle = bin_dir / "xerxes.mjs"
        launcher = bin_dir / "_launcher.py"

        # If bundle is missing, try to build it.
        if not bundle.exists():
            self._try_build_bundle(ts_dir, bundle)

        if not bundle.exists():
            print("[ts-build] WARNING: xerxes.mjs bundle not found — CLI will not work.")
            print("[ts-build] Install Bun (https://bun.sh) and rebuild.")
            return

        bin_dir.mkdir(parents=True, exist_ok=True)

        # Ensure __init__.py exists.
        init = bin_dir / "__init__.py"
        if not init.exists():
            init.write_text(
                '"""CLI launcher and bundled TypeScript bundle for Xerxes."""\n\n'
                "import pathlib\n\n"
                "BIN_DIR = pathlib.Path(__file__).parent\n"
            )

        # Include _bin/ in the wheel.
        build_data.setdefault("force_include", {})
        build_data["force_include"][str(bundle)] = "xerxes/_bin/xerxes.mjs"
        build_data["force_include"][str(launcher)] = "xerxes/_bin/_launcher.py"
        build_data["force_include"][str(init)] = "xerxes/_bin/__init__.py"
        print(f"[ts-build] Staged CLI bundle ({bundle.stat().st_size / 1024:.0f} KB) for install.")

    def _try_build_bundle(self, ts_dir: Path, bundle: Path) -> None:
        """Attempt to build the TypeScript bundle with bun."""
        bun = shutil.which("bun")
        if not bun:
            # Try common fallback paths.
            for candidate in [
                Path.home() / ".bun" / "bin" / "bun",
                Path("/usr/local/bin/bun"),
                Path("/opt/homebrew/bin/bun"),
            ]:
                if candidate.exists():
                    bun = str(candidate)
                    break

        if not bun:
            print("[ts-build] bun not found — skipping TypeScript build.")
            return

        if not ts_dir.exists():
            print(f"[ts-build] TypeScript source dir not found: {ts_dir}")
            return

        print(f"[ts-build] Building TypeScript CLI with bun ...")

        # Install dependencies if needed.
        if not (ts_dir / "node_modules").exists():
            try:
                subprocess.run([bun, "install"], cwd=str(ts_dir), check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[ts-build] bun install failed (exit {exc.returncode}).")
                return

        try:
            subprocess.run(
                [
                    bun,
                    "build",
                    "src/index.tsx",
                    "--target=node",
                    "--minify",
                    f"--outfile={bundle}",
                ],
                cwd=str(ts_dir),
                check=True,
            )
            print(f"[ts-build] Bundle built → {bundle}")
        except subprocess.CalledProcessError as exc:
            print(f"[ts-build] Bundle build failed (exit {exc.returncode}).")
