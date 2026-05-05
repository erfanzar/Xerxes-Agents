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
"""Bootstrap module for Xerxes.

Exports:
    - BootstrapStage
    - BootstrapResult
    - bootstrap"""

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
    """Bootstrap stage.

    Attributes:
        name (str): name.
        status (str): status.
        detail (str): detail.
        duration_ms (float): duration ms."""

    name: str
    status: str = "ok"
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class BootstrapResult:
    """Bootstrap result.

    Attributes:
        stages (list[BootstrapStage]): stages.
        registry (ExecutionRegistry): registry.
        system_prompt (str): system prompt.
        context (dict[str, Any]): context."""

    stages: list[BootstrapStage] = field(default_factory=list)
    registry: ExecutionRegistry = field(default_factory=ExecutionRegistry)
    system_prompt: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Return Ok.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            bool: OUT: Result of the operation."""
        return all(s.status != "failed" for s in self.stages)

    def as_markdown(self) -> str:
        """As markdown.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""
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
    include_xerxes_md: bool = True,
    extra_context: str = "",
) -> BootstrapResult:
    """Bootstrap.

    Args:
        model (str, optional): IN: model. Defaults to ''. OUT: Consumed during execution.
        cwd (str | Path | None, optional): IN: cwd. Defaults to None. OUT: Consumed during execution.
        tools (list[Any] | None, optional): IN: tools. Defaults to None. OUT: Consumed during execution.
        commands (dict[str, Any] | None, optional): IN: commands. Defaults to None. OUT: Consumed during execution.
        include_git_info (bool, optional): IN: include git info. Defaults to True. OUT: Consumed during execution.
        include_xerxes_md (bool, optional): IN: include xerxes md. Defaults to True. OUT: Consumed during execution.
        extra_context (str, optional): IN: extra context. Defaults to ''. OUT: Consumed during execution.
    Returns:
        BootstrapResult: OUT: Result of the operation."""

    import time

    result = BootstrapResult()
    working_dir = Path(cwd) if cwd else Path.cwd()

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

    t0 = time.monotonic()
    xerxes_md = ""
    if include_xerxes_md:
        xerxes_md = _load_xerxes_md(working_dir)
        result.context["xerxes_md"] = xerxes_md
    result.stages.append(
        BootstrapStage(
            name="xerxes_md",
            status="ok" if xerxes_md else "skipped",
            detail=f"{len(xerxes_md)} chars" if xerxes_md else "No XERXES.md found",
            duration_ms=(time.monotonic() - t0) * 1000,
        )
    )

    t0 = time.monotonic()
    if commands:
        for name, handler in commands.items():
            result.registry.register_command(name, handler=handler)
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


def _get_git_info(cwd: Path) -> str:
    """Internal helper to get git info.

    Args:
        cwd (Path): IN: cwd. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

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


def _load_xerxes_md(cwd: Path) -> str:
    """Internal helper to load xerxes md.

    Args:
        cwd (Path): IN: cwd. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    from xerxes.core.paths import xerxes_subdir
    from xerxes.security.prompt_scanner import scan_context_content

    parts = []

    global_md = xerxes_subdir("XERXES.md")
    if global_md.exists():
        try:
            raw = global_md.read_text(encoding="utf-8")
            safe = scan_context_content(raw, filename="Global XERXES.md")
            parts.append(f"[Global XERXES.md]\n{safe}")
        except Exception:
            pass

    p = cwd
    for _ in range(10):
        candidate = p / "XERXES.md"
        if candidate.exists():
            try:
                raw = candidate.read_text(encoding="utf-8")
                safe = scan_context_content(raw, filename=f"Project XERXES.md: {candidate}")
                parts.append(f"[Project XERXES.md: {candidate}]\n{safe}")
            except Exception:
                pass
            break
        parent = p.parent
        if parent == p:
            break
        p = parent

    return "\n\n".join(parts)


def _build_system_prompt(context: dict[str, Any], extra: str = "") -> str:
    """Internal helper to build system prompt.

    Args:
        context (dict[str, Any]): IN: context. OUT: Consumed during execution.
        extra (str, optional): IN: extra. Defaults to ''. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""

    parts = [
        "You are Xerxes, an AI coding assistant with access to tools via function calling.",
        "",
        "# Tools",
        "ReadFile, WriteFile, FileEditTool, AppendFile, ListDir, GlobTool, GrepTool,",
        "ExecuteShell, ExecutePythonCode, GoogleSearch, DuckDuckGoSearch, WebScraper,",
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
        "8. Need web info? → GoogleSearch(query=...)  (preferred — uses Google CSE API when GOOGLE_API_KEY+GOOGLE_CSE_ID are set; otherwise scrapes google.com).",
        "   Or → DuckDuckGoSearch(query=...) when Google blocks the scrape and no API key is set.",
        "9. Need to list a directory? → ListDir(directory_path=...)",
        "",
        "# Web search via curl (when GoogleSearch returns 0 results)",
        "Google blocks naked scrapes from datacenter IPs. When GoogleSearch comes back empty,",
        "fall back to ExecuteShell with a real browser User-Agent.",
        "",
        "RECIPE 1 — Google raw HTML:",
        "  curl -sSL --compressed -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15' -H 'Accept-Language: en-US,en;q=0.9' 'https://www.google.com/search?q=YOUR+QUERY&num=10&hl=en'",
        "",
        "RECIPE 2 — Google + extract URLs in one pipe:",
        "  curl -sSL --compressed -A 'Mozilla/5.0' 'https://www.google.com/search?q=YOUR+QUERY&num=10' | grep -oE 'href=\"https?://[^\"]+' | grep -vE 'google\\.|youtube\\.|webcache' | sort -u | head -20",
        "",
        "RECIPE 3 — DuckDuckGo HTML (never blocks; use when Google returns the JS bot stub):",
        "  curl -sSL --compressed -A 'Mozilla/5.0' 'https://html.duckduckgo.com/html/?q=YOUR+QUERY'",
        "",
        "Tips:",
        "  - URL-encode spaces as '+' (or %20)",
        "  - Add '&num=N' to control result count (max 30)",
        "  - Add '&tbs=qdr:d|w|m|y' for day/week/month/year recency",
        "  - For site-specific: prepend 'site:example.com' to the query",
        "  - If Google returns ~90KB of JS with no <h3> tags, that's the bot stub — retry RECIPE 3",
        "",
        "# Multi-Agent Orchestration (CRITICAL)",
        "You have access to sub-agent spawning tools. Use them AGGRESSIVELY.",
        "- Default to PARALLEL execution. Sequential work is a last resort.",
        "- If a task has 2+ independent parts → SpawnAgents with wait=true.",
        "- If a task has 3+ parts → SpawnAgents is MANDATORY, not optional.",
        "- Research + implementation + review → 3 parallel agents, not 1 monolithic response.",
        "- Writing code in multiple files → SpawnAgents with coder subagents + worktree isolation.",
        "- Do NOT write all the code yourself. Delegate to coder sub-agents.",
        "- AgentTool: spawn one agent. SpawnAgents: spawn many in parallel.",
        "- TaskCreateTool: fire-and-forget background work.",
        "- Always name your agents descriptively so you can reference them later.",
        "",
        "# Critical",
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

    xerxes_md = context.get("xerxes_md", "")
    if xerxes_md:
        parts.extend(["", "# Project Context", xerxes_md])

    if extra:
        parts.extend(["", extra])

    return "\n".join(parts)


__all__ = [
    "BootstrapResult",
    "BootstrapStage",
    "bootstrap",
]
