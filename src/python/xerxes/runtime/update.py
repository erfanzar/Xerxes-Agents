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
"""``xerxes update`` — detect install mode and run the matching upgrade.

Provides pure helpers that classify how the current interpreter received its
``xerxes-agent`` distribution (uv tool, pip user, pip system, or editable
checkout), query PyPI for the latest published version, and finally exec the
appropriate upgrade command. The CLI relies on these helpers so it never
shells out blindly; consumers may also call them directly to display update
banners.
"""

from __future__ import annotations

import enum
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx


class InstallMode(enum.Enum):
    """Enumerates how the running Xerxes distribution was installed.

    Attributes:
        UV_TOOL: Installed via ``uv tool install xerxes-agent``.
        PIP_USER: Installed with ``pip install --user``.
        PIP_SYSTEM: Installed into a system or virtualenv ``site-packages``.
        EDITABLE: Installed as ``pip install -e .`` from a local checkout.
        UNKNOWN: Detection failed; the caller should fall back to manual
            upgrade instructions.
    """

    UV_TOOL = "uv_tool"
    PIP_USER = "pip_user"
    PIP_SYSTEM = "pip_system"
    EDITABLE = "editable"
    UNKNOWN = "unknown"


@dataclass
class UpdateAvailable:
    """Result returned when a newer PyPI release of Xerxes is available.

    Attributes:
        installed_version: Version string of the currently imported package.
        latest_version: Highest version currently published on PyPI.
        mode: Detected install mode, used to pick the upgrade command.
    """

    installed_version: str
    latest_version: str
    mode: InstallMode


def _installed_version() -> str:
    """Return the installed ``xerxes-agent`` version, or ``"0.0.0"`` on failure."""
    try:
        from importlib.metadata import version

        return version("xerxes-agent")
    except Exception:
        return "0.0.0"


def detect_install_mode() -> InstallMode:
    """Heuristically detect how Xerxes was installed.

    Inspects ``sys.executable`` and the ``xerxes-agent`` distribution metadata
    to distinguish uv tool installs (which live under ``~/.local/share/uv/tools/``),
    editable checkouts (package imported from ``src/python/xerxes``), and
    regular pip user/system installs. Returns :class:`InstallMode.UNKNOWN`
    when none of the heuristics match.
    """
    exe = Path(sys.executable).resolve()
    exe_str = str(exe)
    if "/uv/tools/" in exe_str or "\\uv\\tools\\" in exe_str:
        return InstallMode.UV_TOOL
    # Check for editable install marker.
    try:
        from importlib.metadata import distribution

        dist = distribution("xerxes-agent")
        files = dist.files or []
        if any("xerxes" in str(f) and (".egg-link" in str(f) or "PKG-INFO" not in str(f)) for f in files):
            # We're editable if the package is loaded from a path containing 'src/python/xerxes'.
            import xerxes as _xer  # type: ignore

            module_path = getattr(_xer, "__file__", "") or ""
            if "src/python/xerxes" in module_path.replace("\\", "/"):
                return InstallMode.EDITABLE
    except Exception:
        pass
    # Pip-user vs pip-system: user-site contains os.path.expanduser("~").
    try:
        import site

        user_site = site.getusersitepackages()
        if user_site and user_site in exe_str:
            return InstallMode.PIP_USER
        return InstallMode.PIP_SYSTEM
    except Exception:
        return InstallMode.UNKNOWN


def latest_pypi_version(
    *,
    package: str = "xerxes-agent",
    client: httpx.Client | None = None,
    timeout: float = 10.0,
) -> str | None:
    """Return the newest published version of ``package`` from PyPI.

    Args:
        package: PyPI distribution name to query.
        client: Optional pre-built ``httpx.Client``; when omitted a one-shot
            client is created and closed before returning.
        timeout: Per-request timeout in seconds for the implicit client.

    Returns:
        The latest version string, or ``None`` if PyPI is unreachable or the
        response body cannot be parsed.
    """
    own = client is None
    c = client or httpx.Client(timeout=timeout)
    try:
        try:
            resp = c.get(f"https://pypi.org/pypi/{package}/json")
            resp.raise_for_status()
            return resp.json()["info"]["version"]  # type: ignore[no-any-return]
        except (httpx.HTTPError, KeyError, ValueError):
            return None
    finally:
        if own:
            c.close()


def check_for_update(*, client: httpx.Client | None = None) -> UpdateAvailable | None:
    """Compare installed and PyPI versions and report when an upgrade exists.

    Returns ``None`` when PyPI is unreachable or the installed build is
    already current; otherwise returns an :class:`UpdateAvailable` carrying
    the detected install mode so callers can prompt with the right command.
    """
    latest = latest_pypi_version(client=client)
    if latest is None:
        return None
    current = _installed_version()
    if _semver_gt(latest, current):
        return UpdateAvailable(installed_version=current, latest_version=latest, mode=detect_install_mode())
    return None


def _semver_gt(a: str, b: str) -> bool:
    """Return ``True`` when ``a`` is strictly newer than ``b`` (best-effort)."""

    def _parts(v: str) -> tuple[int, ...]:
        """Split ``v`` into up to three integer components, defaulting to zeros."""
        try:
            return tuple(int(p) for p in v.split(".")[:3])
        except ValueError:
            return (0, 0, 0)

    return _parts(a) > _parts(b)


def apply_update(*, dry_run: bool = False) -> dict[str, object]:
    """Run the upgrade command appropriate to the detected install mode.

    Args:
        dry_run: When ``True``, return the resolved argv without executing it.

    Returns:
        A dict with ``ok`` (bool), ``mode`` (the install mode string), and
        either ``argv``/``stdout``/``stderr`` from the subprocess, or
        ``error`` when the mode could not be determined.
    """
    mode = detect_install_mode()
    argv: list[str]
    if mode is InstallMode.UV_TOOL and shutil.which("uv"):
        argv = ["uv", "tool", "upgrade", "xerxes-agent"]
    elif mode is InstallMode.EDITABLE:
        argv = ["pip", "install", "-e", "."]
    elif mode in (InstallMode.PIP_USER, InstallMode.PIP_SYSTEM):
        argv = [sys.executable, "-m", "pip", "install", "--upgrade", "xerxes-agent"]
        if mode is InstallMode.PIP_USER:
            argv.append("--user")
    else:
        return {"ok": False, "mode": mode.value, "error": "unknown install mode; update manually"}
    if dry_run:
        return {"ok": True, "mode": mode.value, "argv": argv, "dry_run": True}
    proc = subprocess.run(argv, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0,
        "mode": mode.value,
        "argv": argv,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


__all__ = [
    "InstallMode",
    "UpdateAvailable",
    "apply_update",
    "check_for_update",
    "detect_install_mode",
    "latest_pypi_version",
]
