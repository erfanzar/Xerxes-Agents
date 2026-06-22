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
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from subprocess import TimeoutExpired

import httpx

logger = logging.getLogger(__name__)

MANAGED_VENV_ENV = "XERXES_VENV"
MANAGED_VENV_SOURCE_ENV = "XERXES_UPDATE_SOURCE"
MANAGED_VENV_SOURCE_FILE = ".xerxes-source"
DEFAULT_MANAGED_VENV = "~/.xerxes-venv"
DEFAULT_UPDATE_SPEC = "xerxes-agent @ git+https://github.com/erfanzar/Xerxes-Agents.git"


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


@dataclass(frozen=True)
class GitUpdateStatus:
    """Local git checkout status compared with its upstream ref.

    ``ahead_count`` is the number of local commits not present upstream.
    ``behind_count`` is the number of upstream commits not present in local
    ``HEAD``; it is the "updates available ahead" count shown in the TUI.
    """

    is_git: bool
    branch: str = ""
    upstream: str = ""
    head_hash: str = ""
    upstream_hash: str = ""
    ahead_count: int = 0
    behind_count: int = 0
    error: str = ""

    @property
    def updates_ahead_available(self) -> int:
        """Return how many upstream commits are available ahead of ``HEAD``."""
        return self.behind_count


def installed_version() -> str:
    """Return the installed ``xerxes-agent`` version, or ``"0.0.0"`` on failure."""
    return _installed_version()


def managed_venv_path() -> Path:
    """Return the Xerxes-managed venv path used by ``scripts/install.sh``."""
    return Path(os.environ.get(MANAGED_VENV_ENV, DEFAULT_MANAGED_VENV)).expanduser()


def _venv_python_path(venv: Path) -> Path:
    """Return the platform-specific Python executable path inside ``venv``."""
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def managed_venv_python() -> Path | None:
    """Return the managed venv Python when ``~/.xerxes-venv`` is available."""
    python = _venv_python_path(managed_venv_path())
    return python if python.exists() else None


def _managed_source_from_env() -> str:
    """Build the default managed-venv update requirement from env overrides."""
    source = os.environ.get(MANAGED_VENV_SOURCE_ENV, "").strip()
    if source:
        return source
    version = os.environ.get("XERXES_VERSION", "").strip()
    ref = os.environ.get("XERXES_REF", "").strip()
    if version:
        source = f"xerxes-agent=={version}"
    elif ref:
        source = f"{DEFAULT_UPDATE_SPEC}@{ref}"
    else:
        source = DEFAULT_UPDATE_SPEC
    extras = os.environ.get("XERXES_INSTALL_EXTRAS", "").strip()
    if not extras:
        return source
    if source.startswith("xerxes-agent @ "):
        return f"xerxes-agent[{extras}] {source.removeprefix('xerxes-agent ')}"
    if source.startswith("xerxes-agent=="):
        return f"xerxes-agent[{extras}]=={source.removeprefix('xerxes-agent==')}"
    if source == "xerxes-agent":
        return f"xerxes-agent[{extras}]"
    return source


def git_update_source() -> str:
    """Return the Git requirement used by ``xerxes update --git``.

    This intentionally ignores ``XERXES_VERSION`` and saved managed-venv
    source files: ``--git`` means install from the default Git branch head.
    """
    extras = os.environ.get("XERXES_INSTALL_EXTRAS", "").strip()
    if not extras:
        return DEFAULT_UPDATE_SPEC
    return f"xerxes-agent[{extras}] @ {DEFAULT_UPDATE_SPEC.removeprefix('xerxes-agent @ ')}"


def managed_venv_update_source() -> str:
    """Return the requirement used to update the managed Xerxes venv."""
    source_file = managed_venv_path() / MANAGED_VENV_SOURCE_FILE
    try:
        source = source_file.read_text(encoding="utf-8").strip()
    except OSError:
        source = ""
    return source or _managed_source_from_env()


def _installed_version() -> str:
    """Return the installed ``xerxes-agent`` version, or ``"0.0.0"`` on failure."""
    try:
        from importlib.metadata import version

        return version("xerxes-agent")
    except Exception:
        # Package metadata unavailable (not installed / running from source) — not an error.
        logger.debug("xerxes-agent version lookup failed; defaulting to 0.0.0", exc_info=True)
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
        # Heuristic probing of distribution metadata; failure just falls through.
        logger.debug("install-mode heuristic (editable) failed", exc_info=True)
    # Pip-user vs pip-system: user-site contains os.path.expanduser("~").
    try:
        import site

        user_site = site.getusersitepackages()
        if user_site and user_site in exe_str:
            return InstallMode.PIP_USER
        return InstallMode.PIP_SYSTEM
    except Exception:
        # Site module probing failed; cannot classify further.
        logger.debug("install-mode heuristic (pip site) failed", exc_info=True)
        return InstallMode.UNKNOWN


def _git_output(args: list[str], *, cwd: Path, timeout: float) -> str:
    """Run ``git`` with ``args`` and return stripped stdout."""
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
    ).strip()


def _fallback_upstream(branch: str, *, cwd: Path, timeout: float) -> str:
    """Return a usable local remote-tracking ref when ``@{u}`` is unset."""
    candidates: list[str] = []
    if branch and branch != "HEAD":
        candidates.append(f"origin/{branch}")
    candidates.extend(["origin/main", "origin/master"])
    for ref in candidates:
        try:
            _git_output(["rev-parse", "--verify", ref], cwd=cwd, timeout=timeout)
            return ref
        except (subprocess.CalledProcessError, TimeoutExpired):
            continue
    return ""


def git_update_status(
    *,
    cwd: str | Path | None = None,
    fetch: bool = False,
    timeout: float = 1.0,
) -> GitUpdateStatus:
    """Compare the current git ``HEAD`` with its upstream tracking ref.

    Args:
        cwd: Directory inside the git checkout. Defaults to ``Path.cwd()``.
        fetch: When ``True``, run a best-effort ``git fetch`` first so the
            upstream ref reflects remote state. The TUI banner passes
            ``False`` to avoid network work during startup.
        timeout: Per-git-command timeout in seconds.

    Returns:
        A :class:`GitUpdateStatus` with ``is_git=False`` outside a checkout.
    """
    workdir = Path.cwd() if cwd is None else Path(cwd)
    try:
        if _git_output(["rev-parse", "--is-inside-work-tree"], cwd=workdir, timeout=timeout) != "true":
            return GitUpdateStatus(is_git=False)
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutExpired):
        return GitUpdateStatus(is_git=False)

    try:
        branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd=workdir, timeout=timeout)
        head_hash = _git_output(["rev-parse", "--short=12", "HEAD"], cwd=workdir, timeout=timeout)
    except (subprocess.CalledProcessError, TimeoutExpired) as exc:
        return GitUpdateStatus(is_git=True, error=str(exc))

    try:
        upstream = _git_output(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=workdir,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, TimeoutExpired):
        upstream = _fallback_upstream(branch, cwd=workdir, timeout=timeout)

    if not upstream:
        return GitUpdateStatus(is_git=True, branch=branch, head_hash=head_hash, error="no upstream ref")

    if fetch:
        remote = upstream.split("/", 1)[0]
        try:
            _git_output(["fetch", "--quiet", "--no-tags", remote], cwd=workdir, timeout=max(timeout, 10.0))
        except (subprocess.CalledProcessError, FileNotFoundError, TimeoutExpired) as exc:
            logger.debug("git fetch for update status failed", exc_info=True)
            fetch_error = f"fetch failed: {exc}"
        else:
            fetch_error = ""
    else:
        fetch_error = ""

    try:
        counts = _git_output(["rev-list", "--left-right", "--count", f"HEAD...{upstream}"], cwd=workdir, timeout=timeout)
        ahead_raw, behind_raw = counts.split()
        upstream_hash = _git_output(["rev-parse", "--short=12", upstream], cwd=workdir, timeout=timeout)
        return GitUpdateStatus(
            is_git=True,
            branch=branch,
            upstream=upstream,
            head_hash=head_hash,
            upstream_hash=upstream_hash,
            ahead_count=int(ahead_raw),
            behind_count=int(behind_raw),
            error=fetch_error,
        )
    except (subprocess.CalledProcessError, TimeoutExpired, ValueError) as exc:
        return GitUpdateStatus(
            is_git=True,
            branch=branch,
            upstream=upstream,
            head_hash=head_hash,
            error=fetch_error or str(exc),
        )


def format_git_update_status(status: GitUpdateStatus) -> str:
    """Return a compact human-readable git update status line."""
    if not status.is_git:
        return "not a git checkout"

    head = f"HEAD {status.head_hash}" if status.head_hash else "HEAD unknown"
    if status.updates_ahead_available > 0:
        upstream = status.upstream or "upstream"
        upstream_hash = f" {status.upstream_hash}" if status.upstream_hash else ""
        return f"{status.updates_ahead_available} ahead available ({upstream}{upstream_hash}; {head})"
    if status.ahead_count > 0:
        return f"current upstream; local {status.ahead_count} ahead ({head})"
    if status.error:
        return f"unknown ({status.error}; {head})"
    return f"current ({head})"


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
        """Split ``v`` into integer components, defaulting to zeros."""
        try:
            return tuple(int(p) for p in v.split("."))
        except ValueError:
            return (0, 0, 0)

    return _parts(a) > _parts(b)


def apply_update(*, dry_run: bool = False, git: bool = False) -> dict[str, object]:
    """Run the upgrade command appropriate to the detected install mode.

    Args:
        dry_run: When ``True``, return the resolved argv without executing it.
        git: When ``True``, install from the Git repository head instead of
            the detected PyPI/editable/saved source.

    Returns:
        A dict with ``ok`` (bool), ``mode`` (the install mode string), and
        either ``argv``/``stdout``/``stderr`` from the subprocess, or
        ``error`` when the mode could not be determined.
    """
    managed_python = managed_venv_python()
    if managed_python is not None:
        source = git_update_source() if git else managed_venv_update_source()
        if shutil.which("uv"):
            managed_argv = ["uv", "pip", "install", "--python", str(managed_python), "--upgrade"]
            if git:
                managed_argv.extend(["--reinstall-package", "xerxes-agent", "--refresh-package", "xerxes-agent"])
            managed_argv.append(source)
        else:
            managed_argv = [str(managed_python), "-m", "pip", "install", "--upgrade"]
            if git:
                managed_argv.append("--force-reinstall")
            managed_argv.append(source)
        if dry_run:
            return {"ok": True, "mode": "managed_venv", "argv": managed_argv, "dry_run": True}
        try:
            proc = subprocess.run(managed_argv, capture_output=True, text=True)
        except FileNotFoundError as exc:
            return {"ok": False, "mode": "managed_venv", "argv": managed_argv, "error": str(exc)}
        return {
            "ok": proc.returncode == 0,
            "mode": "managed_venv",
            "argv": managed_argv,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    mode = detect_install_mode()
    argv: list[str]
    if git:
        source = git_update_source()
        if mode is InstallMode.UV_TOOL and shutil.which("uv"):
            argv = ["uv", "tool", "install", "--force", "--refresh-package", "xerxes-agent", source]
        elif mode in (InstallMode.EDITABLE, InstallMode.PIP_USER, InstallMode.PIP_SYSTEM):
            argv = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", source]
            if mode is InstallMode.PIP_USER:
                argv.append("--user")
        else:
            return {"ok": False, "mode": mode.value, "error": "unknown install mode; update manually"}
    elif mode is InstallMode.UV_TOOL and shutil.which("uv"):
        argv = ["uv", "tool", "upgrade", "xerxes-agent"]
    elif mode is InstallMode.EDITABLE:
        if shutil.which("uv"):
            argv = ["uv", "pip", "install", "-e", "."]
        else:
            argv = [sys.executable, "-m", "pip", "install", "-e", "."]
    elif mode in (InstallMode.PIP_USER, InstallMode.PIP_SYSTEM):
        argv = [sys.executable, "-m", "pip", "install", "--upgrade", "xerxes-agent"]
        if mode is InstallMode.PIP_USER:
            argv.append("--user")
    else:
        return {"ok": False, "mode": mode.value, "error": "unknown install mode; update manually"}
    if dry_run:
        return {"ok": True, "mode": mode.value, "argv": argv, "dry_run": True}
    try:
        proc = subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return {"ok": False, "mode": mode.value, "argv": argv, "error": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "mode": mode.value,
        "argv": argv,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


__all__ = [
    "GitUpdateStatus",
    "InstallMode",
    "UpdateAvailable",
    "apply_update",
    "check_for_update",
    "detect_install_mode",
    "format_git_update_status",
    "git_update_source",
    "git_update_status",
    "installed_version",
    "latest_pypi_version",
    "managed_venv_path",
    "managed_venv_python",
    "managed_venv_update_source",
]
