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
"""Workspace-change guardrails for model-driven edits.

The model can claim a change is safe while the working tree tells a different
story. This module classifies the current Git status and recent tool audit
records so daemon/bridge frontends can surface high-risk edits at turn end.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

ChangeSeverity = Literal["info", "warning", "error"]

BUILD_CONFIG_PATHS = {
    ".github/workflows/ci.yml",
    ".github/workflows/ci.yaml",
    "Dockerfile",
    "Makefile",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "uv.lock",
}
CRITICAL_SOURCE_PREFIXES = (
    "src/python/xerxes/bridge/",
    "src/python/xerxes/daemon/",
    "src/python/xerxes/runtime/",
    "src/python/xerxes/security/",
    "src/python/xerxes/streaming/",
    "src/python/xerxes/tools/",
)
VERIFICATION_MARKERS = (
    "pytest",
    "ruff check",
    "ruff format",
    "mypy",
    "pre-commit",
    "git diff --check",
    "uv lock --check",
    "docker build",
)
RECENT_TOOL_EXECUTIONS = 50


@dataclass(frozen=True)
class WorkspaceChange:
    """One parsed ``git status --porcelain`` row."""

    status: str
    path: str
    old_path: str = ""

    @property
    def deleted(self) -> bool:
        """Return true when Git marks this path as deleted."""
        return "D" in self.status and not self.untracked

    @property
    def tracked(self) -> bool:
        """Return true for any tracked status row."""
        return self.status != "??"

    @property
    def untracked(self) -> bool:
        """Return true for untracked files."""
        return self.status == "??"


@dataclass(frozen=True)
class ChangeGuardFinding:
    """One risk detected in the working tree."""

    severity: ChangeSeverity
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ChangeGuardReport:
    """Risk classification plus recent verification evidence."""

    findings: tuple[ChangeGuardFinding, ...]
    verification_commands: tuple[str, ...]
    status_available: bool = True

    @property
    def severity(self) -> ChangeSeverity:
        """Return the highest severity in the report."""
        if any(f.severity == "error" for f in self.findings):
            return "error"
        if any(f.severity == "warning" for f in self.findings):
            return "warning"
        return "info"

    @property
    def should_notify(self) -> bool:
        """Return true when the frontend should show the guard report."""
        if not self.findings:
            return False
        if self.severity == "error":
            return True
        return not self.verification_commands

    @property
    def fingerprint(self) -> str:
        """Stable fingerprint used to suppress duplicate notifications."""
        payload = {
            "findings": [f.__dict__ for f in self.findings],
            "verification_commands": self.verification_commands,
            "status_available": self.status_available,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()


def analyze_workspace_changes(cwd: Path, tool_executions: list[dict[str, Any]] | None = None) -> ChangeGuardReport:
    """Inspect ``cwd`` with Git and classify risky model-made changes.

    Args:
        cwd: Project directory to inspect.
        tool_executions: Session audit records from recent model tool calls.

    Returns:
        A report suitable for UI notification. Non-Git directories return an
        empty, status-unavailable report instead of raising.
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ChangeGuardReport(findings=(), verification_commands=(), status_available=False)
    if proc.returncode != 0:
        return ChangeGuardReport(findings=(), verification_commands=(), status_available=False)
    return analyze_status_lines(proc.stdout.splitlines(), tool_executions or [])


def analyze_status_lines(
    lines: list[str],
    tool_executions: list[dict[str, Any]] | None = None,
) -> ChangeGuardReport:
    """Classify already-collected porcelain status lines."""
    changes = tuple(parse_porcelain_status(lines))
    findings = tuple(_findings_for_changes(changes))
    verification_commands = tuple(_recent_verification_commands(tool_executions or []))
    return ChangeGuardReport(findings=findings, verification_commands=verification_commands)


def parse_porcelain_status(lines: list[str]) -> list[WorkspaceChange]:
    """Parse ``git status --porcelain=v1`` output."""
    changes: list[WorkspaceChange] = []
    for raw in lines:
        if len(raw) < 4:
            continue
        status = raw[:2]
        path_text = raw[3:]
        old_path = ""
        path = path_text
        if " -> " in path_text:
            old_path, path = path_text.split(" -> ", 1)
        changes.append(
            WorkspaceChange(status=status, path=_normalize_git_path(path), old_path=_normalize_git_path(old_path))
        )
    return changes


def format_change_guard_notification(report: ChangeGuardReport) -> str:
    """Render a concise human-readable notification body."""
    if not report.findings:
        return ""
    lines = ["Risky workspace changes detected:"]
    for finding in report.findings:
        path = f" [{finding.path}]" if finding.path else ""
        lines.append(f"- {finding.message}{path}")
    if report.verification_commands:
        lines.append("")
        lines.append("Recent verification:")
        for command in report.verification_commands[:3]:
            lines.append(f"- {command}")
    else:
        lines.append("")
        lines.append("No recent pytest/ruff/mypy/pre-commit/git diff --check command was found in this session.")
    return "\n".join(lines)


def _findings_for_changes(changes: tuple[WorkspaceChange, ...]) -> list[ChangeGuardFinding]:
    findings: list[ChangeGuardFinding] = []

    deleted_tests = sorted(c.path for c in changes if c.deleted and _is_test_file(c.path))
    if deleted_tests:
        findings.append(
            ChangeGuardFinding(
                severity="error",
                code="deleted-tests",
                path=_sample_paths(deleted_tests),
                message=f"{len(deleted_tests)} tracked test file(s) were deleted",
            )
        )

    deleted_sources = sorted(c.path for c in changes if c.deleted and _is_source_file(c.path))
    if deleted_sources:
        findings.append(
            ChangeGuardFinding(
                severity="warning",
                code="deleted-source",
                path=_sample_paths(deleted_sources),
                message=f"{len(deleted_sources)} source file(s) were deleted",
            )
        )

    build_config = sorted(c.path for c in changes if c.tracked and _is_build_config(c.path))
    if build_config:
        findings.append(
            ChangeGuardFinding(
                severity="warning",
                code="build-config-changed",
                path=_sample_paths(build_config),
                message="build, install, lockfile, or CI configuration changed",
            )
        )

    critical_sources = sorted(c.path for c in changes if c.tracked and _is_critical_source(c.path) and not c.deleted)
    if critical_sources:
        findings.append(
            ChangeGuardFinding(
                severity="warning",
                code="runtime-critical-changed",
                path=_sample_paths(critical_sources),
                message="runtime, daemon, bridge, security, streaming, or tool code changed",
            )
        )

    return findings


def _recent_verification_commands(tool_executions: list[dict[str, Any]]) -> list[str]:
    commands: list[str] = []
    for execution in tool_executions[-RECENT_TOOL_EXECUTIONS:]:
        command = _tool_execution_command(execution)
        if not command:
            continue
        normalized = " ".join(command.split())
        lowered = normalized.lower()
        if any(marker in lowered for marker in VERIFICATION_MARKERS):
            commands.append(_truncate(normalized, 180))
    return commands[-5:]


def _tool_execution_command(execution: dict[str, Any]) -> str:
    name = str(execution.get("name", "")).lower()
    inputs = execution.get("inputs")
    if not isinstance(inputs, dict):
        return ""
    for key in ("cmd", "command", "shell_command"):
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if "shell" in name or "exec" in name or "bash" in name:
        for value in inputs.values():
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _normalize_git_path(path: str) -> str:
    return path.strip().strip('"').replace("\\", "/")


def _is_test_file(path: str) -> bool:
    pure = PurePosixPath(path)
    return bool(pure.parts and pure.parts[0] == "tests" and pure.name.startswith("test_") and pure.suffix == ".py")


def _is_source_file(path: str) -> bool:
    pure = PurePosixPath(path)
    return bool(path.startswith("src/python/xerxes/") and pure.suffix == ".py")


def _is_build_config(path: str) -> bool:
    return path in BUILD_CONFIG_PATHS or path.startswith(".github/workflows/")


def _is_critical_source(path: str) -> bool:
    return _is_source_file(path) and any(path.startswith(prefix) for prefix in CRITICAL_SOURCE_PREFIXES)


def _sample_paths(paths: list[str], *, limit: int = 5) -> str:
    sample = paths[:limit]
    suffix = f", +{len(paths) - limit} more" if len(paths) > limit else ""
    return ", ".join(sample) + suffix


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
