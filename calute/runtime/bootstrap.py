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


"""Bootstrap graph and system initialization for the Calute runtime.

Manages the startup sequence: environment detection, tool registration,
context building, and system prompt assembly.

Inspired by the claw-code ``bootstrap_graph``, ``setup``, and
``system_init`` patterns.

Usage::

    from calute.runtime.bootstrap import bootstrap, BootstrapResult

    result = bootstrap(model="gpt-4o", cwd="/path/to/project")
    print(result.system_prompt)
    print(result.registry.summary())
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .execution_registry import ExecutionRegistry


@dataclass
class BootstrapStage:
    """A single stage in the bootstrap sequence.

    Attributes:
        name: Stage name.
        status: ``"ok"``, ``"skipped"``, ``"failed"``.
        detail: Description of what happened.
        duration_ms: Time taken in milliseconds.
    """

    name: str
    status: str = "ok"
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class BootstrapResult:
    """Result of the full bootstrap sequence.

    Attributes:
        stages: List of completed stages.
        registry: The assembled execution registry.
        system_prompt: The built system initialization prompt.
        context: Runtime context information.
    """

    stages: list[BootstrapStage] = field(default_factory=list)
    registry: ExecutionRegistry = field(default_factory=ExecutionRegistry)
    system_prompt: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(s.status != "failed" for s in self.stages)

    def as_markdown(self) -> str:
        lines = [
            "# Bootstrap Report",
            "",
            f"Status: {'OK' if self.ok else 'FAILED'}",
            f"Stages: {len(self.stages)}",
            "",
        ]
        for stage in self.stages:
            icon = {"ok": "+", "skipped": "~", "failed": "!"}[stage.status]
            lines.append(f"- [{icon}] {stage.name}: {stage.detail} ({stage.duration_ms:.1f}ms)")
        return "\n".join(lines)


def bootstrap(
    model: str = "",
    cwd: str | Path | None = None,
    tools: list[Any] | None = None,
    commands: dict[str, Any] | None = None,
    include_git_info: bool = True,
    include_claude_md: bool = True,
    extra_context: str = "",
) -> BootstrapResult:
    """Run the full bootstrap sequence.

    Stages:
    1. Detect environment (Python, platform, CWD).
    2. Detect git repository info.
    3. Load CLAUDE.md project context.
    4. Register built-in commands.
    5. Register tools.
    6. Build system initialization prompt.

    Args:
        model: Active model name.
        cwd: Working directory (defaults to current).
        tools: Optional list of tools to register.
        commands: Optional dict of command_name -> handler.
        include_git_info: Whether to include git info in context.
        include_claude_md: Whether to load CLAUDE.md files.
        extra_context: Additional context to include in system prompt.

    Returns:
        :class:`BootstrapResult` with registry and system prompt.
    """
    import time

    result = BootstrapResult()
    working_dir = Path(cwd) if cwd else Path.cwd()

    # ── Stage 1: Environment ──────────────────────────────────────────
    t0 = time.monotonic()
    result.context = {
        "cwd": str(working_dir),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "model": model,
        "date": datetime.now().strftime("%Y-%m-%d %A"),
    }
    result.stages.append(
        BootstrapStage(
            name="environment",
            detail=f"Python {result.context['python_version']} on {platform.system()}",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
    )

    # ── Stage 2: Git info ─────────────────────────────────────────────
    t0 = time.monotonic()
    git_info = ""
    if include_git_info:
        git_info = _get_git_info(working_dir)
        result.context["git_info"] = git_info
    result.stages.append(
        BootstrapStage(
            name="git_info",
            status="ok" if git_info else "skipped",
            detail=git_info[:80] if git_info else "Not a git repository",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
    )

    # ── Stage 3: CLAUDE.md ────────────────────────────────────────────
    t0 = time.monotonic()
    claude_md = ""
    if include_claude_md:
        claude_md = _load_claude_md(working_dir)
        result.context["claude_md"] = claude_md
    result.stages.append(
        BootstrapStage(
            name="claude_md",
            status="ok" if claude_md else "skipped",
            detail=f"{len(claude_md)} chars" if claude_md else "No CLAUDE.md found",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
    )

    # ── Stage 4: Register commands ────────────────────────────────────
    t0 = time.monotonic()
    if commands:
        for name, handler in commands.items():
            result.registry.register_command(name, handler=handler)
    # Register default slash commands
    for cmd_name in [
        "help",
        "clear",
        "history",
        "save",
        "load",
        "model",
        "config",
        "cost",
        "context",
        "memory",
        "agents",
        "skills",
    ]:
        if not result.registry.get_command(cmd_name):
            result.registry.register_command(cmd_name, description=f"/{cmd_name} command")
    result.stages.append(
        BootstrapStage(
            name="commands",
            detail=f"{result.registry.command_count} commands registered",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
    )

    # ── Stage 5: Register tools ───────────────────────────────────────
    t0 = time.monotonic()
    if tools:
        result.registry.register_from_agent_functions(tools)
    result.stages.append(
        BootstrapStage(
            name="tools",
            detail=f"{result.registry.tool_count} tools registered",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
    )

    # ── Stage 6: Build system prompt ──────────────────────────────────
    t0 = time.monotonic()
    result.system_prompt = _build_system_prompt(result.context, extra_context)
    result.stages.append(
        BootstrapStage(
            name="system_prompt",
            detail=f"{len(result.system_prompt)} chars",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
    )

    return result


# ── Internal helpers ───────────────────────────────────────────────────────


def _get_git_info(cwd: Path) -> str:
    """Get git branch, status, and recent commits."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-5"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        parts = [f"Branch: {branch}"]
        if status:
            parts.append(f"Status:\n{status}")
        if log:
            parts.append(f"Recent commits:\n{log}")
        return "\n".join(parts)
    except Exception:
        return ""


def _load_claude_md(cwd: Path) -> str:
    """Load CLAUDE.md from cwd/parents and ~/.claude/CLAUDE.md."""
    parts = []

    # Global
    global_md = Path.home() / ".claude" / "CLAUDE.md"
    if global_md.exists():
        try:
            parts.append(f"[Global CLAUDE.md]\n{global_md.read_text()}")
        except Exception:
            pass

    # Project (walk up)
    p = cwd
    for _ in range(10):
        candidate = p / "CLAUDE.md"
        if candidate.exists():
            try:
                parts.append(f"[Project CLAUDE.md: {candidate}]\n{candidate.read_text()}")
            except Exception:
                pass
            break
        parent = p.parent
        if parent == p:
            break
        p = parent

    return "\n\n".join(parts)


def _build_system_prompt(context: dict[str, Any], extra: str = "") -> str:
    """Build the system initialization prompt."""
    parts = [
        "You are Calute, an AI coding assistant with access to tools via function calling.",
        "",
        "# Tools",
        "ReadFile, WriteFile, FileEditTool, AppendFile, ListDir, GlobTool, GrepTool,",
        "ExecuteShell, ExecutePythonCode, DuckDuckGoSearch, WebScraper,",
        "AgentTool, SendMessageTool, TodoWriteTool, Calculator, JSONProcessor, CSVProcessor",
        "",
        "# How to decide",
        "1. Can you answer from knowledge alone? → Reply directly.",
        "2. Need to read a file? → ReadFile(file_path=...)",
        "3. Need to write a file? → WriteFile(file_path=..., content=...)",
        "4. Need to edit a file? → FileEditTool(file_path=..., old_string=..., new_string=...)",
        "5. Need to run a command? → ExecuteShell(command=...)",
        "6. Need to find files? → GlobTool(pattern=...)",
        "7. Need to search code? → GrepTool(pattern=...)",
        "8. Need web info? → DuckDuckGoSearch(query=...)",
        "9. Need to list a directory? → ListDir(directory_path=...)",
        "",
        "# Critical",
        "- Call tools via the function calling API. NEVER write <tool_call> XML in your text.",
        "- Be concise and direct.",
        "- Read files before editing them.",
        "- Use absolute paths for file operations.",
        "",
        "# Environment",
        f"- Date: {context.get('date', '')}",
        f"- CWD: {context.get('cwd', '')}",
        f"- Platform: {context.get('platform', '')}",
        f"- Model: {context.get('model', '')}",
    ]

    git_info = context.get("git_info", "")
    if git_info:
        parts.extend(["", "# Git", git_info])

    claude_md = context.get("claude_md", "")
    if claude_md:
        parts.extend(["", "# Project Context", claude_md])

    if extra:
        parts.extend(["", extra])

    return "\n".join(parts)


__all__ = [
    "BootstrapResult",
    "BootstrapStage",
    "bootstrap",
]
