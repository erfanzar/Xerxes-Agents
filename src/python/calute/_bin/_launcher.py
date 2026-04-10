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

"""Thin launcher that execs the native Rust binary.

Registered as the `calute` console_script entry point so that
`calute` works after `pip install`.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

BIN_DIR = Path(__file__).parent


def launch() -> None:
    """Launch the native Rust CLI (calute)."""
    ext = ".exe" if platform.system() == "Windows" else ""
    binary = BIN_DIR / f"calute{ext}"

    if not binary.exists():
        print(
            f"Native binary not found at {binary}.\n"
            f"Run 'cd src/rust && cargo build --release' or reinstall with 'pip install -e .'",
            file=sys.stderr,
        )
        sys.exit(1)

    args = [str(binary), *sys.argv[1:]]

    if platform.system() == "Windows":
        import subprocess

        sys.exit(subprocess.call(args))
    else:
        os.execv(str(binary), args)
