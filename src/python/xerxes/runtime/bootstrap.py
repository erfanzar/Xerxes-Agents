# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Bootstrap graph and system initialization for the Xerxes runtime.

Manages the startup sequence: environment detection, tool registration,
context building, and system prompt assembly.

Inspired by the claw-code ``bootstrap_graph``, ``setup``, and
``system_init`` patterns.

Usage::

    from xerxes.runtime.bootstrap import bootstrap, BootstrapResult

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
    include_xerxes_md: bool = True,
    extra_context: str = "",
) -> BootstrapResult:
    """Run the full bootstrap sequence.

    Stages:
    1. Detect environment (Python, platform, CWD).
    2. Detect git repository info.
    3. Load XERXES.md project context.
    4. Register built-in commands.
    5. Register tools.
    6. Build system initialization prompt.

    Args:
        model: Active model name.
        cwd: Working directory (defaults to current).
        tools: Optional list of tools to register.
        commands: Optional dict of command_name -> handler.
        include_git_info: Whether to include git info in context.
        include_xerxes_md: Whether to load XERXES.md files.
        extra_context: Additional context to include in system prompt.

    Returns:
        :class:`BootstrapResult` with registry and system prompt.
    """
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


def _load_xerxes_md(cwd: Path) -> str:
    """Load XERXES.md from cwd/parents and ~/.xerxes/XERXES.md.

    Each file is scanned for prompt-injection threats before being
    included in the system prompt.
    """
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
    """Build the system initialization prompt."""
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
