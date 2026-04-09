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

    args = [str(binary)] + sys.argv[1:]

    if platform.system() == "Windows":
        import subprocess

        sys.exit(subprocess.call(args))
    else:
        os.execv(str(binary), args)
