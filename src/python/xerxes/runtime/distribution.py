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
"""Distribution and packaging helpers for ``xerxes`` install/release tooling.

Detects whether the current host is Termux/Android or WSL2, renders a ready-to-commit
Homebrew formula, strips extras that lack Android wheels, and provides shell
and PowerShell install snippets used by the docs and release scripts.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass


@dataclass
class PlatformInfo:
    """Snapshot of the running host used for distribution decisions.

    Attributes:
        system: ``platform.system()`` value (``"Darwin"``, ``"Linux"``,
            ``"Windows"``).
        release: Kernel/release string as returned by ``platform.release()``.
        is_termux: ``True`` when Xerxes is running under Termux/Android.
        is_wsl: ``True`` when the host is Windows Subsystem for Linux.
        python: Python version (``"3.11.7"``-style).
    """

    system: str
    release: str
    is_termux: bool
    is_wsl: bool
    python: str

    @property
    def is_darwin(self) -> bool:
        """``True`` when running on macOS."""
        return self.system == "Darwin"

    @property
    def is_linux(self) -> bool:
        """``True`` when running on Linux (including WSL/Termux)."""
        return self.system == "Linux"

    @property
    def is_windows(self) -> bool:
        """``True`` when running on native Windows."""
        return self.system == "Windows"


def detect_platform() -> PlatformInfo:
    """Return ``PlatformInfo`` describing the current host."""
    sysname = platform.system()
    release = platform.release()
    is_termux = bool(os.environ.get("PREFIX", "").startswith("/data/data/com.termux/")) or "termux" in release.lower()
    is_wsl = "microsoft" in release.lower() or "WSL_DISTRO_NAME" in os.environ
    return PlatformInfo(
        system=sysname,
        release=release,
        is_termux=is_termux,
        is_wsl=is_wsl,
        python=sys.version.split()[0],
    )


HOMEBREW_FORMULA_TEMPLATE = """\
class XerxesAgent < Formula
  include Language::Python::Virtualenv

  desc "Multi-agent orchestration framework + terminal coding agent"
  homepage "{homepage}"
  url "{tarball_url}"
  sha256 "{sha256}"
  license "Apache-2.0"

  depends_on "python@3.11"

  def install
    virtualenv_install_with_resources
  end

  test do
    system "#{{bin}}/xerxes", "--help"
  end
end
"""


def render_homebrew_formula(
    *, version: str, tarball_url: str, sha256: str, homepage: str = "https://github.com/erfanzar/Xerxes-Agents"
) -> str:
    """Render a ready-to-commit Homebrew formula for the given release.

    Args:
        version: Released version, embedded into the formula header.
        tarball_url: URL pointing at the source tarball.
        sha256: SHA-256 of the tarball.
        homepage: Project homepage; defaults to the GitHub repo.
    """
    return HOMEBREW_FORMULA_TEMPLATE.format(version=version, tarball_url=tarball_url, sha256=sha256, homepage=homepage)


TERMUX_EXTRA_EXCLUDED = frozenset(
    {
        "faster_whisper",
        "sentence_transformers",
        "playwright",
        "edge_tts",
        "ctranslate2",
        "onnxruntime",
        "elevenlabs",
        "sounddevice",
    }
)


def termux_filter_extras(extras: dict[str, list[str]]) -> dict[str, list[str]]:
    """Strip dependencies that lack Android wheels from a Termux build.

    Args:
        extras: Mapping of extra name → dependency list (e.g. as parsed from
            ``pyproject.toml``).

    Returns:
        A copy of ``extras`` with any dependency matching
        :data:`TERMUX_EXTRA_EXCLUDED` removed. Comparison is case-insensitive
        and ignores ``-``/``_`` so ``faster-whisper`` matches the
        ``faster_whisper`` blocklist entry.
    """

    def _normalize(s: str) -> str:
        """Lower-case ``s`` and collapse ``-`` into ``_`` for matching."""
        return s.lower().replace("-", "_")

    out: dict[str, list[str]] = {}
    for extra_name, deps in extras.items():
        kept = [d for d in deps if not any(forbidden in _normalize(d) for forbidden in TERMUX_EXTRA_EXCLUDED)]
        out[extra_name] = kept
    return out


SHELL_INSTALL_SNIPPET = """\
#!/usr/bin/env bash
set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv tool install xerxes-agent
echo "✓ xerxes installed. Run 'xerxes' to start."
"""


POWERSHELL_INSTALL_SNIPPET = """\
$ErrorActionPreference = 'Stop'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv..."
  Invoke-Expression "& $((Invoke-WebRequest -UseBasicParsing https://astral.sh/uv/install.ps1).Content)"
}
uv tool install xerxes-agent
Write-Host "✓ xerxes installed."
"""


__all__ = [
    "HOMEBREW_FORMULA_TEMPLATE",
    "POWERSHELL_INSTALL_SNIPPET",
    "SHELL_INSTALL_SNIPPET",
    "TERMUX_EXTRA_EXCLUDED",
    "PlatformInfo",
    "detect_platform",
    "render_homebrew_formula",
    "termux_filter_extras",
]
