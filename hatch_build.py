# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Custom hatch build hook that compiles Rust binaries during install.

When you run `pip install -e .` or `uv pip install -e .`, this hook:
1. Runs `cargo build --release` in the Rust workspace
2. Copies the resulting binaries into `src/python/calute/_bin/`
3. Registers them as package data so they're accessible after install

The binaries are also exposed as console_scripts entry points via thin
Python wrappers, but the native binaries in _bin/ can be called directly.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class RustBuildHook(BuildHookInterface):
    """Hatch build hook that compiles the Rust workspace."""

    PLUGIN_NAME = "rust-build"

    def initialize(self, version: str, build_data: dict) -> None:
        """Called before the build — compile Rust and stage binaries."""
        root = Path(self.root)
        rust_dir = root / "src" / "rust"
        bin_dir = root / "src" / "python" / "calute" / "_bin"

        if not rust_dir.exists():
            print("[rust-build] No src/rust/ directory found, skipping Rust build.")
            return

        # Check for cargo.
        cargo = shutil.which("cargo")
        if not cargo:
            print("[rust-build] WARNING: cargo not found — skipping Rust build.")
            print("[rust-build] Install Rust via https://rustup.rs to build native binaries.")
            return

        print(f"[rust-build] Building Rust workspace in {rust_dir} ...")

        try:
            subprocess.run(
                [cargo, "build", "--release", "--workspace"],
                cwd=str(rust_dir),
                check=True,
                env={**os.environ, "CARGO_TERM_COLOR": "always"},
            )
        except subprocess.CalledProcessError as exc:
            print(f"[rust-build] WARNING: Rust build failed (exit {exc.returncode}).")
            print("[rust-build] Python package will install without native binaries.")
            return

        # Determine binary extension.
        ext = ".exe" if platform.system() == "Windows" else ""

        # Expected binaries.
        target_dir = rust_dir / "target" / "release"
        binaries = {f"calute{ext}": "calute-cli"}

        # Copy binaries into the package.
        bin_dir.mkdir(parents=True, exist_ok=True)

        copied = []
        for binary_name, _crate_name in binaries.items():
            src = target_dir / binary_name
            if src.exists():
                dst = bin_dir / binary_name
                shutil.copy2(str(src), str(dst))
                # Ensure executable on Unix.
                if platform.system() != "Windows":
                    dst.chmod(0o755)
                copied.append(binary_name)
                print(f"[rust-build] Copied {binary_name} → {dst}")
            else:
                print(f"[rust-build] WARNING: Expected binary not found: {src}")

        if copied:
            # Write a __init__.py so the _bin dir is a package (for importlib.resources).
            init = bin_dir / "__init__.py"
            if not init.exists():
                init.write_text(
                    '"""Native Rust binaries for Calute."""\n\n'
                    "import pathlib\n\n"
                    "BIN_DIR = pathlib.Path(__file__).parent\n"
                )

            # Include _bin/ in the wheel.
            build_data.setdefault("force_include", {})
            for name in copied:
                rel = f"calute/_bin/{name}"
                build_data["force_include"][str(bin_dir / name)] = rel
            build_data["force_include"][str(init)] = "calute/_bin/__init__.py"

            print(f"[rust-build] {len(copied)} binaries staged for install.")
        else:
            print("[rust-build] No binaries were produced.")
