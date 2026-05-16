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
"""``xerxes doctor`` — diagnose configuration and dependencies.

Each check is a callable returning a :class:`Diagnosis` whose ``severity`` is
``"ok"``, ``"warn"``, or ``"fail"``. :func:`run_all_checks` returns the full
report; :func:`run_minimal` runs only the fast subset used by
``xerxes update --check`` and similar fast paths.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Severity = Literal["ok", "warn", "fail"]


@dataclass
class Diagnosis:
    """One doctor check result.

    Attributes:
        name: Short identifier shown next to the icon in the report.
        severity: ``"ok"``, ``"warn"``, or ``"fail"``.
        message: Human-readable one-line summary.
        fix_hint: Optional remediation suggestion shown when not OK.
    """

    name: str
    severity: Severity
    message: str
    fix_hint: str = ""


CheckFn = Callable[[], Diagnosis]
"""Callable signature for a doctor check: takes no args, returns :class:`Diagnosis`."""


# ---------------------------- individual checks ----------------------------


def check_python_version() -> Diagnosis:
    """Verify the interpreter is Python 3.11 or newer."""
    v = sys.version_info
    if v >= (3, 11):
        return Diagnosis("python", "ok", f"Python {v.major}.{v.minor}.{v.micro}")
    return Diagnosis(
        "python",
        "fail",
        f"Python 3.11+ required (have {v.major}.{v.minor})",
        fix_hint="Install Python 3.11 or newer.",
    )


def check_xerxes_on_path() -> Diagnosis:
    """Verify the ``xerxes`` CLI entry point is discoverable on ``PATH``."""
    found = shutil.which("xerxes")
    if found:
        return Diagnosis("xerxes-on-path", "ok", f"xerxes binary at {found}")
    return Diagnosis(
        "xerxes-on-path",
        "warn",
        "`xerxes` not on PATH",
        fix_hint="Add the install bin dir to PATH or reinstall with uv.",
    )


def check_provider_keys() -> Diagnosis:
    """Verify at least one well-known LLM provider API key is set in the env."""
    known = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")
    present = [k for k in known if os.environ.get(k)]
    if present:
        return Diagnosis("provider-keys", "ok", f"providers configured via env: {', '.join(present)}")
    return Diagnosis(
        "provider-keys",
        "warn",
        "No provider API key in environment",
        fix_hint="Run `xerxes /provider` or set OPENAI_API_KEY / ANTHROPIC_API_KEY.",
    )


def check_xerxes_home() -> Diagnosis:
    """Verify the Xerxes home directory exists (or warn that it will be created)."""
    home = os.environ.get("XERXES_HOME") or os.path.expanduser("~/.xerxes")
    if os.path.isdir(home):
        return Diagnosis("xerxes-home", "ok", f"XERXES_HOME present at {home}")
    return Diagnosis(
        "xerxes-home",
        "warn",
        f"XERXES_HOME not yet created at {home}",
        fix_hint="Will be created on first run — usually safe to ignore.",
    )


def check_required_imports() -> Diagnosis:
    """Verify hard runtime dependencies (``pydantic``, ``httpx``, ``openai``, ``rich``) are importable."""
    must = ("pydantic", "httpx", "openai", "rich")
    missing = [m for m in must if importlib.util.find_spec(m) is None]
    if not missing:
        return Diagnosis("imports", "ok", "core dependencies importable")
    return Diagnosis(
        "imports",
        "fail",
        f"missing imports: {', '.join(missing)}",
        fix_hint="`uv pip install -e .` or `pip install xerxes-agent`.",
    )


def check_optional_imports() -> Diagnosis:
    """Report which optional integrations (browser, MCP, TTS, STT) are usable."""
    optional = {
        "playwright": "browser tools",
        "mcp": "MCP integration",
        "edge_tts": "voice / TTS",
        "faster_whisper": "STT (transcription)",
    }
    missing = [name for name in optional if importlib.util.find_spec(name) is None]
    if not missing:
        return Diagnosis("optional-imports", "ok", "every optional dep importable")
    return Diagnosis(
        "optional-imports",
        "warn",
        f"optional deps absent: {', '.join(missing)}",
        fix_hint="Install matching extras (e.g. `xerxes-agent[voice]`).",
    )


def check_platform() -> Diagnosis:
    """Warn when running natively on Windows (WSL2 is required)."""
    sysname = platform.system()
    if sysname == "Windows":
        return Diagnosis(
            "platform",
            "warn",
            "Native Windows is unsupported; run in WSL2.",
            fix_hint="Install WSL2 and run xerxes there.",
        )
    return Diagnosis("platform", "ok", f"{sysname} {platform.release()}")


DEFAULT_CHECKS: tuple[CheckFn, ...] = (
    check_python_version,
    check_platform,
    check_required_imports,
    check_optional_imports,
    check_xerxes_on_path,
    check_provider_keys,
    check_xerxes_home,
)

MINIMAL_CHECKS: tuple[CheckFn, ...] = (
    check_python_version,
    check_required_imports,
    check_xerxes_on_path,
)


def run_all_checks(checks: tuple[CheckFn, ...] | None = None) -> list[Diagnosis]:
    """Run every check in ``checks`` (defaults to :data:`DEFAULT_CHECKS`)."""
    return [c() for c in (checks or DEFAULT_CHECKS)]


def run_minimal() -> list[Diagnosis]:
    """Run only :data:`MINIMAL_CHECKS` — fast subset for routine update probes."""
    return run_all_checks(MINIMAL_CHECKS)


def has_failures(report: list[Diagnosis]) -> bool:
    """Return ``True`` when any diagnosis in ``report`` has ``severity == "fail"``."""
    return any(d.severity == "fail" for d in report)


def format_report(report: list[Diagnosis]) -> str:
    """Render a doctor report for terminals using check/warn/cross icons."""
    icons = {"ok": "✓", "warn": "!", "fail": "✗"}
    lines = []
    for d in report:
        line = f"{icons[d.severity]} {d.name}: {d.message}"
        if d.fix_hint and d.severity != "ok":
            line += f"\n    → {d.fix_hint}"
        lines.append(line)
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CHECKS",
    "MINIMAL_CHECKS",
    "CheckFn",
    "Diagnosis",
    "check_optional_imports",
    "check_platform",
    "check_provider_keys",
    "check_python_version",
    "check_required_imports",
    "check_xerxes_home",
    "check_xerxes_on_path",
    "format_report",
    "has_failures",
    "run_all_checks",
    "run_minimal",
]
