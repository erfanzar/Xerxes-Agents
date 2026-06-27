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
"""Runtime, session, workspace, and turn execution managers for the daemon.

This module is the daemon's core: :class:`RuntimeManager` owns provider
bootstrap, the shared tool executor, and the skill registry;
:class:`WorkspaceManager` creates per-agent Markdown workspaces under
``$XERXES_HOME/agents``; :class:`SessionManager` maps stable session keys to
:class:`DaemonSession` records persisted under ``$XERXES_HOME/sessions``;
and :class:`TurnRunner` runs one turn at a time on a thread pool, converting
internal streaming events into the daemon's wire-protocol payloads
(``tool_call``, ``tool_result``, ``approval_request``, ``status_update``, ...).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from ..bridge import profiles
from ..channels.workspace import MarkdownAgentWorkspace
from ..context.window_usage import estimate_context_tokens
from ..core.paths import xerxes_subdir
from ..extensions.skills import SkillRegistry, default_skill_discovery_dirs, get_active_skills
from ..operators import OperatorRuntimeConfig, OperatorState
from ..runtime.agent_memory import AgentMemory
from ..runtime.bootstrap import bootstrap
from ..runtime.bridge import build_tool_executor, populate_registry, register_operator_tools
from ..runtime.change_guard import analyze_workspace_changes, format_change_guard_notification
from ..runtime.config_context import set_config as set_global_config
from ..runtime.interaction_modes import mode_switch_hint, normalize_interaction_mode
from ..runtime.project_workspace import load_project_agent_workspace
from ..streaming.events import (
    AgentState,
    PermissionRequest,
    ProviderRetry,
    SkillSuggestion,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
)
from ..streaming.loop import run as run_agent_loop
from ..tools.agent_memory_tool import set_active_memory
from ..tools.agent_meta_tools import set_skill_registry
from .config import DaemonConfig

_DAEMON_TERMINAL_OPERATOR_TOOLS = {
    "exec_command",
    "write_stdin",
    "list_terminal_sessions",
    "close_terminal_session",
}

EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]


def _format_agent_event(evt: dict[str, Any]) -> str:
    """Render one mailbox event as a single human-readable line.

    Used to splice sub-agent activity into the main agent's conversation
    between tool iterations so a passive main agent still notices when its
    children spawn, finish, or change tools.
    """
    agent = evt.get("agent") or evt.get("task_id", "?")
    etype = evt.get("type", "?")
    data = evt.get("data") or {}
    if etype == "spawn":
        return f"[agent {agent}] spawned ({data.get('agent_type') or 'general'})"
    if etype == "tool_start":
        tool = data.get("tool", "?")
        preview = data.get("input_preview", "")
        return f"[agent {agent}] → {tool}({preview})" if preview else f"[agent {agent}] → {tool}()"
    if etype == "tool_end":
        tool = data.get("tool", "?")
        ms = data.get("duration_ms")
        suffix = f" in {ms:.0f}ms" if isinstance(ms, (int, float)) else ""
        return f"[agent {agent}] ← {tool}{suffix}"
    if etype == "coordination":
        path = data.get("path", "?")
        writer = data.get("writer", "?")
        return f"[agent {agent}] coordination: {path} changed by {writer}; re-read before continuing"
    if etype == "file_write":
        path = data.get("path", "?")
        readers = data.get("readers_notified", 0)
        return f"[agent {agent}] wrote {path}; notified {readers} stale reader(s)"
    if etype == "text_burst":
        chars = data.get("chars", 0)
        return f"[agent {agent}] +{chars} chars"
    if etype == "done":
        status = data.get("status", "?")
        preview = (data.get("result_preview") or "").strip().splitlines()
        first = preview[0][:120] if preview else ""
        return f"[agent {agent}] {status}" + (f" — {first}" if first else "")
    if etype == "cancelled":
        return f"[agent {agent}] cancelled ({data.get('reason', 'unspecified')})"
    return f"[agent {agent}] {etype}"


@dataclass
class RuntimeManager:
    """Provider/runtime singleton shared by every session.

    Holds the resolved runtime config dict, the bootstrapped system prompt,
    the tool executor + JSON schemas, and the active skill registry. The
    agent's two-tier persistent memory (global + per-project) is attached
    here so every tool call sees the same scopes.

    Attributes:
        config: The static :class:`DaemonConfig` the runtime was built from.
        runtime_config: Live merged runtime settings (model, sampling, mode).
        system_prompt: Resolved system prompt from :func:`bootstrap`.
        tool_executor: Callable used by the streaming loop to dispatch tools.
        tool_schemas: JSON schemas of registered tools, sent to the model.
        skill_registry: Discovered :class:`SkillRegistry`.
        skills_dir: Writable user-skills directory.
    """

    config: DaemonConfig
    runtime_config: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    tool_executor: Any = None
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    operator_state: OperatorState | None = None
    skill_registry: SkillRegistry = field(default_factory=SkillRegistry)
    skills_dir: Path = field(default_factory=lambda: xerxes_subdir("skills"))

    @property
    def model(self) -> str:
        """The active model id, or empty string when nothing's loaded."""
        return str(self.runtime_config.get("model", ""))

    def reload(self, overrides: dict[str, Any] | None = None) -> None:
        """Re-resolve provider settings, rebuild tools, and reinstall agent memory.

        Merges ``overrides`` on top of the daemon config and active profile,
        then re-bootstraps the runtime, rebuilds the tool executor, refreshes
        the skill registry, and rebinds the agent's persistent memory to the
        current working directory.
        """
        runtime = self.config.resolved_runtime()
        clean_overrides = {k: v for k, v in (overrides or {}).items() if v not in (None, "")}
        runtime.update(clean_overrides)

        profile = profiles.get_active_profile()
        if profile:
            # Backfill provider settings from the active profile for any field the
            # caller didn't override. Critically, ``base_url``/``api_key`` are filled
            # even when ``model`` IS provided (e.g. ``/model <id>`` switches within
            # the same provider) — otherwise the override would drop the creds and
            # break the next turn.
            if not runtime.get("model"):
                runtime["model"] = profile.get("model", "")
            if not runtime.get("base_url"):
                runtime["base_url"] = profile.get("base_url", "")
            if not runtime.get("api_key"):
                runtime["api_key"] = profile.get("api_key", "")
            if profile.get("provider"):
                runtime["provider"] = profile.get("provider", "")
            else:
                runtime.pop("provider", None)
            if profile.get("provider") == "claude-code":
                runtime.pop("api_key", None)
            for _k, _v in profile.get("sampling", {}).items():
                runtime.setdefault(_k, _v)

        runtime.setdefault("permission_mode", "accept-all")
        self._normalize_reasoning_config(runtime)
        cwd = Path(str(clean_overrides.get("project_dir") or self.config.project_dir or os.getcwd())).expanduser()
        try:
            cwd = cwd.resolve()
        except OSError:
            cwd = cwd.absolute()
        self.config.project_dir = str(cwd)
        runtime["project_dir"] = str(cwd)
        set_global_config(runtime)

        boot = bootstrap(model=str(runtime.get("model", "")), cwd=cwd)
        registry = populate_registry()
        self.operator_state = OperatorState(
            OperatorRuntimeConfig(
                enabled=True,
                power_tools_enabled=True,
                shell_default_workdir=str(cwd),
                allowed_tool_names=set(_DAEMON_TERMINAL_OPERATOR_TOOLS),
            )
        )
        register_operator_tools(registry, self.operator_state, set(_DAEMON_TERMINAL_OPERATOR_TOOLS))
        self.discover_skills()

        # Initialise the agent's persistent two-tier memory: global (cross-
        # project) + project-scoped, both rooted under $XERXES_HOME. The
        # tool wrappers in xerxes.tools.agent_memory_tool point at this
        # instance so every agent_memory_* call sees the right scopes.
        self.agent_memory = AgentMemory(project_root=cwd)
        self.agent_memory.ensure()
        set_active_memory(self.agent_memory)

        self.runtime_config = runtime
        self.system_prompt = boot.system_prompt
        self.tool_executor = build_tool_executor(registry=registry)
        self.tool_schemas = registry.tool_schemas()
        set_skill_registry(self.skill_registry)

    def status(self) -> dict[str, Any]:
        """Return a short summary used by ``runtime.status`` and ``/status``."""
        return {
            "ok": bool(self.model),
            "model": self.model,
            "base_url": self.runtime_config.get("base_url", ""),
            "permission_mode": self.runtime_config.get("permission_mode", "accept-all"),
            "tools": len(self.tool_schemas),
            "skills": len(self.skill_registry.skill_names),
            "reasoning_effort": self.reasoning_state()["effort"],
        }

    @property
    def permission_mode(self) -> str:
        """Current permission strategy (``auto`` | ``manual`` | ``accept-all``)."""
        return str(self.runtime_config.get("permission_mode", "accept-all"))

    def set_permission_mode(self, mode: str) -> str:
        """Set the active permission mode and re-publish the global config.

        Valid modes are ``auto``, ``manual``, and ``accept-all``. Unknown
        values silently fall back to ``auto`` so a stray slash command can't
        break the daemon.
        """
        valid = ("auto", "manual", "accept-all")
        clean = mode.strip().lower()
        if clean not in valid:
            clean = "auto"
        self.runtime_config["permission_mode"] = clean
        set_global_config(self.runtime_config)
        return clean

    def toggle_yolo(self) -> str:
        """Flip between ``auto`` and ``accept-all`` and return the new mode."""
        return self.set_permission_mode("auto" if self.permission_mode == "accept-all" else "accept-all")

    def toggle_flag(self, name: str) -> bool:
        """Flip a boolean runtime flag (``debug``, ``verbose``, ``thinking``) and republish config."""
        current = bool(self.runtime_config.get(name, False))
        new = not current
        self.runtime_config[name] = new
        set_global_config(self.runtime_config)
        return new

    # Reasoning/"thinking" effort levels. Each level maps to an Anthropic
    # ``budget_tokens`` preset and an OpenAI-compatible ``reasoning_effort``
    # string (OpenAI o-series, MiniMax, …). ``off`` disables reasoning.
    REASONING_LEVELS: ClassVar[dict[str, int]] = {"off": 0, "low": 2048, "medium": 8192, "high": 24576}

    def reasoning_state(self) -> dict[str, Any]:
        """Return the current reasoning effort: ``{effort, thinking, budget_tokens}``."""
        thinking = bool(self.runtime_config.get("thinking", False))
        effort = "off" if not thinking else str(self.runtime_config.get("reasoning_effort") or "medium")
        return {
            "effort": effort,
            "thinking": thinking,
            "budget_tokens": int(self.runtime_config.get("thinking_budget", 0) or 0),
        }

    def set_reasoning_effort(self, level: str) -> dict[str, Any]:
        """Set the reasoning effort to one of ``off``/``low``/``medium``/``high``.

        Updates ``thinking`` (bool), ``reasoning_effort`` (the level string sent
        to OpenAI-compatible providers), and ``thinking_budget`` (Anthropic
        ``budget_tokens``) in the live config, then republishes it. Raises
        ``ValueError`` for an unrecognised level.
        """
        clean = level.strip().lower()
        aliases = {"none": "off", "no": "off", "false": "off", "0": "off", "disable": "off", "med": "medium"}
        clean = aliases.get(clean, clean)
        if clean not in self.REASONING_LEVELS:
            raise ValueError(f"Unknown reasoning level '{level}'. Use one of: {', '.join(self.REASONING_LEVELS)}.")
        if clean == "off":
            self.runtime_config["thinking"] = False
            self.runtime_config["thinking_budget"] = 0
            self.runtime_config.pop("reasoning_effort", None)
        else:
            self.runtime_config["thinking"] = True
            self.runtime_config["reasoning_effort"] = clean
            self.runtime_config["thinking_budget"] = self.REASONING_LEVELS[clean]
        profile = profiles.get_active_profile()
        if profile:
            profiles.update_sampling(
                profile["name"],
                {
                    "thinking": self.runtime_config.get("thinking", False),
                    "reasoning_effort": self.runtime_config.get("reasoning_effort"),
                    "thinking_budget": self.runtime_config.get("thinking_budget", 0),
                },
            )
        set_global_config(self.runtime_config)
        return self.reasoning_state()

    def _normalize_reasoning_config(self, runtime: dict[str, Any]) -> None:
        """Normalize thinking/reasoning knobs before a runtime config becomes active."""
        provider = str(runtime.get("provider", "")).strip().lower()
        model = str(runtime.get("model", "")).strip()
        raw_effort = runtime.get("reasoning_effort")
        if isinstance(raw_effort, str) and raw_effort.strip():
            effort = raw_effort.strip().lower()
            effort = {"none": "off", "no": "off", "false": "off", "0": "off", "disable": "off", "med": "medium"}.get(
                effort, effort
            )
            if effort == "off" or runtime.get("thinking") is False:
                runtime["thinking"] = False
                runtime["thinking_budget"] = 0
                runtime.pop("reasoning_effort", None)
                return
            if effort in self.REASONING_LEVELS:
                runtime["thinking"] = True
                runtime["reasoning_effort"] = effort
                runtime["thinking_budget"] = self.REASONING_LEVELS[effort]
                return

        if "thinking" in runtime:
            if bool(runtime.get("thinking")):
                effort = str(runtime.get("reasoning_effort") or "medium").strip().lower()
                if effort not in self.REASONING_LEVELS or effort == "off":
                    effort = "medium"
                runtime["reasoning_effort"] = effort
                runtime["thinking_budget"] = int(runtime.get("thinking_budget") or self.REASONING_LEVELS[effort])
            else:
                runtime["thinking_budget"] = 0
                runtime.pop("reasoning_effort", None)
            return

        if provider == "claude-code" or model.startswith("claude-code/"):
            runtime["thinking"] = True
            runtime["reasoning_effort"] = "medium"
            runtime["thinking_budget"] = self.REASONING_LEVELS["medium"]

    def discover_skills(self) -> list[str]:
        """Re-scan bundled, user, and cwd skill directories; return sorted ids.

        Returned ids include both root skill names and ``name:subcommand``
        forms for skills that declare sub-commands.
        """
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.skill_registry = SkillRegistry()
        self.skill_registry.discover(
            *default_skill_discovery_dirs(user_skills_dir=self.skills_dir, cwd=self.config.project_dir or os.getcwd())
        )
        set_skill_registry(self.skill_registry)
        return sorted(self.skill_names_with_subs())

    def skill_names_with_subs(self) -> list[str]:
        """Return every invokable skill identifier including ``name:subcommand`` forms.

        Used by the TUI slash-completer and the ``/skills`` listing.
        """
        out: list[str] = []
        for skill in self.skill_registry.get_all():
            out.append(skill.name)
            for sub in skill.metadata.subcommands:
                out.append(f"{skill.name}:{sub}")
        return out

    def skills_list_text(self) -> str:
        """Render the ``/skills`` slash output, including sub-commands and tags."""
        self.discover_skills()
        skills = self.skill_registry.get_all()
        if not skills:
            return f"No skills found.\n  Skills directory: {self.skills_dir}\n  Create one with /skill-create"
        total_with_subs = sum(1 + len(s.metadata.subcommands) for s in skills)
        lines = [f"Skills ({len(skills)} root, {total_with_subs} including sub-commands):"]
        for skill in sorted(skills, key=lambda item: item.name):
            tags = f" [{', '.join(skill.metadata.tags)}]" if skill.metadata.tags else ""
            lines.append(f"  /{skill.name}{tags} - {skill.metadata.description or 'No description'}")
            for sub in skill.metadata.subcommands:
                lines.append(f"    /{skill.name}:{sub}")
        lines.append("\nUse /skill <name>[:<sub>] to invoke a skill, or /<skill-name>[:<sub>] for shorthand.")
        return "\n".join(lines)

    def active_skill_prompt(self) -> str:
        """Concatenate every active skill's prompt section, separated by blank lines."""
        sections: list[str] = []
        for name in get_active_skills():
            skill = self.skill_registry.get(name)
            if skill is not None:
                sections.append(skill.to_prompt_section())
        return "\n\n".join(sections)


@dataclass
class DaemonSession:
    """One persistent agent conversation tied to a session key.

    Attributes:
        id: Short hex session id; matches the on-disk filename.
        key: Stable client-supplied key (e.g. ``"tui:default"`` or a session id).
        agent_id: Name of the agent definition driving the session.
        workspace: Markdown workspace at ``$XERXES_HOME/agents/<agent_id>``.
        project_dir: User project directory this session operates on.
        state: Cumulative streaming-loop state (messages, tokens, tools).
        lock: Asyncio lock serialising turns on this session.
        cancel_requested: Set true to tell the streaming loop to stop early.
        active_turn_id: Current turn id while one is running.
        interaction_mode: Active session-scoped mode reported to clients.
        plan_mode: Whether ``interaction_mode`` is currently plan mode.
        pending_steers: Thread-safe queue of ``/steer`` strings drained by
            the streaming loop between tool iterations.
    """

    id: str
    key: str
    agent_id: str
    workspace: MarkdownAgentWorkspace
    project_dir: Path
    state: AgentState = field(default_factory=AgentState)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancel_requested: bool = False
    active_turn_id: str = ""
    interaction_mode: str = "code"
    plan_mode: bool = False
    runtime_config: dict[str, Any] = field(default_factory=dict)
    pending_steers: queue.Queue[str] = field(default_factory=queue.Queue)

    def drain_steers(self) -> list[str]:
        """Pop every queued steer string and return them in arrival order."""
        out: list[str] = []
        while True:
            try:
                out.append(self.pending_steers.get_nowait())
            except queue.Empty:
                break
        return out

    def status(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot used by ``session.status`` and the daemon listing."""
        return {
            "id": self.id,
            "key": self.key,
            "agent_id": self.agent_id,
            "workspace": str(self.workspace.path),
            "cwd": str(self.project_dir),
            "active_turn_id": self.active_turn_id,
            "mode": self.interaction_mode,
            "plan_mode": self.plan_mode,
            "model": str(self.runtime_config.get("model", "")),
            "messages": len(self.state.messages),
            "turn_count": self.state.turn_count,
            "input_tokens": self.state.total_input_tokens,
            "output_tokens": self.state.total_output_tokens,
            "cancel_requested": self.cancel_requested,
        }


def render_session_system_prompt(
    runtime: RuntimeManager,
    session: DaemonSession,
    *,
    mode: str,
    tolerate_errors: bool = False,
) -> str:
    """Build the provider-visible system prompt for a daemon session."""
    try:
        workspace_context = session.workspace.load_context()
    except Exception:
        if not tolerate_errors:
            raise
        workspace_prompt = ""
    else:
        workspace_prompt = workspace_context.prompt

    try:
        memory_section = runtime.agent_memory.to_prompt_section()
    except Exception:
        memory_section = ""
    try:
        project_workspace_section = load_project_agent_workspace(session.project_dir).prompt
    except Exception:
        project_workspace_section = ""
    base_system_prompt = runtime.system_prompt.rstrip()
    live_project_workspace_section = (
        project_workspace_section
        if project_workspace_section and project_workspace_section not in base_system_prompt
        else ""
    )
    return "\n\n".join(
        part
        for part in (
            base_system_prompt,
            live_project_workspace_section,
            workspace_prompt,
            memory_section,
            runtime.active_skill_prompt(),
            mode_switch_hint(mode),
        )
        if part
    )


class WorkspaceManager:
    """Materialise per-agent Markdown workspaces under ``$XERXES_HOME/agents``."""

    def __init__(self, config: DaemonConfig) -> None:
        """Resolve the workspace root and default agent id from ``config``."""
        self.config = config
        self.root = Path(config.workspace.get("root", "") or "").expanduser()
        if not str(self.root):
            from ..core.paths import xerxes_subdir

            self.root = xerxes_subdir("agents")
        self.default_agent_id = str(config.workspace.get("default_agent_id", "default") or "default")

    def workspace_for(self, agent_id: str | None = None) -> MarkdownAgentWorkspace:
        """Return a ready-to-use workspace for ``agent_id`` (creating directories as needed)."""
        agent = (agent_id or self.default_agent_id or "default").strip() or "default"
        workspace = MarkdownAgentWorkspace(self.root / agent)
        workspace.ensure()
        return workspace


class SessionManager:
    """Maps client session keys to persistent :class:`DaemonSession` state.

    Each session is persisted atomically to ``$XERXES_HOME/sessions/<id>.json``
    so ``xerxes -r <id>`` truly rehydrates the conversation. Message context
    compaction happens inside the model runtime before provider calls; the
    session manager only caps append-only side buffers.
    """

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        *,
        keep_messages: int = 80,
        store_dir: Path | None = None,
    ) -> None:
        """Configure the storage location and the per-session message cap."""
        self.workspace_manager = workspace_manager
        self.keep_messages = max(4, keep_messages)
        self._sessions: dict[str, DaemonSession] = {}
        self._store_dir = store_dir or xerxes_subdir("sessions")
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Return the on-disk JSON path for ``session_id``."""
        return self._store_dir / f"{session_id}.json"

    def _load_state(self, session_id: str) -> dict[str, Any] | None:
        """Read the persisted record for ``session_id`` or return ``None`` if absent/corrupt."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _archive_path(self, session_id: str) -> Path:
        """Return the sidecar path where compacted-out messages are appended."""
        return self._store_dir / f"{session_id}.archive.jsonl"

    def _archive_evicted(self, session: DaemonSession, evicted: list[dict[str, Any]]) -> None:
        """Append compacted-out messages to a sidecar so history isn't lost unrecoverably.

        Best-effort: a failure here must never break the turn.
        """
        if not evicted:
            return
        try:
            with self._archive_path(session.id).open("a", encoding="utf-8") as fh:
                for m in evicted:
                    fh.write(json.dumps(m, default=str, ensure_ascii=False) + "\n")
        except OSError:
            pass

    @staticmethod
    def _repair_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Back-fill synthetic results for orphaned tool calls before reuse.

        The session is persisted at ``TurnDone`` — which fires after the assistant
        message (with its ``tool_calls``) is appended but before the tool results
        are. A crash in that window leaves a ``tool_use`` with no matching
        ``tool_result`` on disk, which makes Anthropic 400 on resume. Insert a
        synthetic error result for every unanswered call so ``xerxes -r`` always
        rehydrates into a valid conversation.
        """
        repaired: list[dict[str, Any]] = []
        i = 0
        n = len(messages)
        while i < n:
            m = messages[i]
            repaired.append(m)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                answered: set[str | None] = set()
                j = i + 1
                while j < n and messages[j].get("role") == "tool":
                    answered.add(messages[j].get("tool_call_id"))
                    repaired.append(messages[j])
                    j += 1
                for tc in m["tool_calls"]:
                    cid = tc.get("id")
                    if cid and cid not in answered:
                        repaired.append(
                            {
                                "role": "tool",
                                "tool_call_id": cid,
                                "name": tc.get("name", ""),
                                "content": "[interrupted: no result recorded — the previous turn ended before this tool finished]",
                                "is_error": True,
                            }
                        )
                i = j
            else:
                i += 1
        return repaired

    @staticmethod
    def _message_text(content: Any) -> str:
        """Return readable text from a persisted message ``content`` field."""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            value = content.get("text") or content.get("content") or ""
            return str(value)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    value = item.get("text") or item.get("content") or ""
                    if value:
                        parts.append(str(value))
            return "\n".join(parts)
        return str(content or "")

    def _current_project_dir(self) -> Path:
        """Return the active project directory for newly created sessions."""
        raw = self.workspace_manager.config.project_dir or os.getcwd()
        path = Path(str(raw)).expanduser()
        try:
            return path.resolve()
        except OSError:
            return path.absolute()

    def _record_project_dir(self, record: dict[str, Any]) -> Path:
        """Return the persisted project dir, migrating old workspace-cwd records."""
        raw = record.get("cwd") or record.get("project_dir") or ""
        if raw:
            path = Path(str(raw)).expanduser()
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path.absolute()
            try:
                workspace_root = self.workspace_manager.root.resolve()
            except OSError:
                workspace_root = self.workspace_manager.root.absolute()
            if resolved != workspace_root and workspace_root not in resolved.parents:
                return resolved
        return self._current_project_dir()

    @classmethod
    def _derive_title_from_messages(cls, messages: Any) -> str:
        """Use the first persisted user prompt as a compact session title."""
        if not isinstance(messages, list):
            return ""
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            text = " ".join(cls._message_text(message.get("content")).split())
            if text:
                return text[:77] + "..." if len(text) > 80 else text
        return ""

    @staticmethod
    def _record_has_history(record: dict[str, Any]) -> bool:
        """Return ``True`` when a persisted record has resumable content."""
        messages = record.get("messages", [])
        message_count = len(messages) if isinstance(messages, list) else 0
        try:
            turn_count = int(record.get("turn_count", 0) or 0)
        except (TypeError, ValueError):
            turn_count = 0
        return message_count > 0 or turn_count > 0

    @staticmethod
    def _session_has_history(session: DaemonSession) -> bool:
        """Return ``True`` when ``session`` should be persisted for resume."""
        try:
            turn_count = int(session.state.turn_count or 0)
        except (TypeError, ValueError):
            turn_count = 0
        return bool(session.state.messages) or turn_count > 0

    def save(self, session: DaemonSession) -> None:
        """Persist ``session`` to disk via a tempfile + ``os.replace`` swap.

        Atomic so the JSON file is never partially overwritten if the daemon
        crashes mid-write.
        """
        path = self._session_path(session.id)
        if not self._session_has_history(session):
            path.unlink(missing_ok=True)
            return

        metadata = dict(session.state.metadata or {})
        if not str(metadata.get("title") or "").strip():
            title = self._derive_title_from_messages(session.state.messages)
            if title:
                metadata["title"] = title
        session.state.metadata = metadata
        record = {
            "session_id": session.id,
            "key": session.key,
            "agent_id": session.agent_id,
            "cwd": str(session.project_dir),
            "workspace": str(session.workspace.path),
            "updated_at": datetime.now(UTC).isoformat(),
            "messages": session.state.messages,
            "turn_count": session.state.turn_count,
            "interaction_mode": session.interaction_mode,
            "plan_mode": session.plan_mode,
            "total_input_tokens": session.state.total_input_tokens,
            "total_output_tokens": session.state.total_output_tokens,
            "metadata": metadata,
            "thinking_content": session.state.thinking_content,
            "tool_executions": session.state.tool_executions,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile

        fd, tmp_path = tempfile.mkstemp(prefix=f".{session.id}.", suffix=".json", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, default=str, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def open(self, session_key: str = "default", agent_id: str | None = None) -> DaemonSession:
        """Return (or create) the live :class:`DaemonSession` for ``session_key``.

        Resolution order is:
            1. In-memory session already bound to this key.
            2. On-disk record whose id matches ``session_key`` (set by
               ``xerxes -r <id>``).
            3. A brand-new session with no prior history.

        Rehydration is strict and id-keyed: launching ``xerxes`` without
        ``-r`` always produces a clean session — slot keys like
        ``tui:default`` never silently restore the most recent prior chat.
        """
        key = session_key or "default"
        if key in self._sessions:
            return self._sessions[key]
        agent = agent_id or self.workspace_manager.default_agent_id
        workspace = self.workspace_manager.workspace_for(agent)
        project_dir = self._current_project_dir()

        # Only rehydrate when the key is itself a valid session id that
        # has a saved record on disk. ``tui:default`` and other slot
        # keys never load anything — they always create fresh sessions.
        loaded: dict[str, Any] | None = None
        if _looks_like_id(key):
            loaded = self._load_state(key)

        if loaded is not None:
            session_id = str(loaded.get("session_id") or key)
            session = DaemonSession(
                id=session_id,
                key=key,
                agent_id=str(loaded.get("agent_id") or agent),
                workspace=workspace,
                project_dir=self._record_project_dir(loaded),
                interaction_mode=normalize_interaction_mode(
                    loaded.get("interaction_mode") or loaded.get("mode") or "code",
                    plan_mode=bool(loaded.get("plan_mode", False)),
                ),
                plan_mode=bool(loaded.get("plan_mode", False)),
            )
            session.plan_mode = session.interaction_mode == "plan"
            state = session.state
            # Repair any orphaned tool calls from a mid-turn crash so resume
            # doesn't 400 (see _repair_tool_pairs).
            state.messages = self._repair_tool_pairs(list(loaded.get("messages", [])))
            state.turn_count = int(loaded.get("turn_count", 0))
            state.total_input_tokens = int(loaded.get("total_input_tokens", 0))
            state.total_output_tokens = int(loaded.get("total_output_tokens", 0))
            metadata = loaded.get("metadata", {})
            state.metadata = dict(metadata) if isinstance(metadata, dict) else {}
            state.thinking_content = list(loaded.get("thinking_content", []))
            state.tool_executions = list(loaded.get("tool_executions", []))
        else:
            # Brand-new session — use the key as the id if it looks like a
            # short uuid (`-r <id>`), otherwise generate one.
            session_id = key if _looks_like_id(key) else uuid.uuid4().hex[:12]
            session = DaemonSession(
                id=session_id,
                key=key,
                agent_id=agent,
                workspace=workspace,
                project_dir=project_dir,
            )
        self._sessions[key] = session
        return session

    def _find_latest_by_key(self, key: str) -> dict[str, Any] | None:
        """Return the newest on-disk record whose ``"key"`` field matches ``key``."""
        latest: tuple[float, dict[str, Any]] | None = None
        for path in self._store_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("key") != key:
                continue
            mtime = path.stat().st_mtime
            if latest is None or mtime > latest[0]:
                latest = (mtime, record)
        return latest[1] if latest else None

    def get(self, session_key: str = "default") -> DaemonSession | None:
        """Return the live session bound to ``session_key``, or ``None``."""
        return self._sessions.get(session_key or "default")

    def list_saved(self) -> list[dict[str, Any]]:
        """Return persisted session records, newest first."""
        records: list[dict[str, Any]] = []
        for path in self._store_dir.glob("*.json"):
            if path.name.startswith("."):
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                mtime = path.stat().st_mtime
            except (OSError, json.JSONDecodeError):
                continue
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if not self._record_has_history(record):
                continue
            session_id = str(record.get("session_id") or path.stem)
            messages = record.get("messages", [])
            title = str(metadata.get("title") or "").strip() or self._derive_title_from_messages(messages)
            records.append(
                {
                    "id": session_id,
                    "session_id": session_id,
                    "key": str(record.get("key") or ""),
                    "title": title,
                    "agent_id": str(record.get("agent_id") or ""),
                    "updated_at": str(record.get("updated_at") or ""),
                    "turn_count": int(record.get("turn_count", 0) or 0),
                    "messages": len(messages) if isinstance(messages, list) else 0,
                    "path": str(path),
                    "_mtime": mtime,
                }
            )
        records.sort(key=lambda item: (str(item.get("updated_at") or ""), float(item.get("_mtime", 0.0))), reverse=True)
        for record in records:
            record.pop("_mtime", None)
        return records

    def find_saved(self, query: str) -> list[dict[str, Any]]:
        """Return saved sessions matching ``query`` by id, key, or title."""
        needle = query.strip()
        if not needle:
            return []
        needle_lower = needle.lower()
        records = self.list_saved()
        exact = [
            record
            for record in records
            if needle == str(record.get("id") or "")
            or needle == str(record.get("session_id") or "")
            or needle == str(record.get("key") or "")
            or needle_lower == str(record.get("title") or "").lower()
        ]
        if exact:
            return exact
        return [
            record
            for record in records
            if str(record.get("id") or "").startswith(needle)
            or str(record.get("session_id") or "").startswith(needle)
            or str(record.get("title") or "").lower().startswith(needle_lower)
        ]

    def evict(self, session_key: str) -> None:
        """Drop the in-memory session bound to ``session_key`` (no-op if absent).

        Used when the TUI re-connects without an explicit resume id and a
        stale slot still points at the previous chat. The on-disk record is
        untouched — a later ``-r <id>`` can still load it.
        """
        self._sessions.pop(session_key, None)

    def list(self) -> list[dict[str, Any]]:
        """Return :meth:`DaemonSession.status` for every live session."""
        return [session.status() for session in self._sessions.values()]

    def cancel(self, session_key: str = "default") -> bool:
        """Request cancellation of the live turn on ``session_key`` (no-op if absent)."""
        session = self.get(session_key)
        if not session:
            return False
        session.cancel_requested = True
        return True

    def cancel_all(self) -> int:
        """Request cancellation on every live session; return how many were marked."""
        count = 0
        for session in self._sessions.values():
            session.cancel_requested = True
            count += 1
        return count

    # ``thinking_content`` and ``tool_executions`` are append-only side
    # buffers. The runtime provisioner owns message compaction, while these
    # side buffers are bounded here during persistence.
    _THINKING_KEEP = 32
    _TOOL_EXEC_KEEP = 200

    def compact_if_needed(self, session: DaemonSession) -> bool:
        """Trim side buffers that are not sent back to the model.

        Returns ``True`` when a side buffer was trimmed.
        """
        state = session.state
        compacted = False

        if len(state.thinking_content) > self._THINKING_KEEP:
            state.thinking_content = state.thinking_content[-self._THINKING_KEEP :]
            compacted = True

        if len(state.tool_executions) > self._TOOL_EXEC_KEEP:
            state.tool_executions = state.tool_executions[-self._TOOL_EXEC_KEEP :]
            compacted = True

        return compacted


_ID_RE = __import__("re").compile(r"^[0-9a-fA-F]{8,32}$")


def _looks_like_id(text: str) -> bool:
    """True when ``text`` matches the short hex form ``xerxes -r`` passes."""
    return bool(_ID_RE.match(text))


class TurnRunner:
    """Drive one agent turn on a worker thread and bridge events to clients.

    The runner owns the turn thread pool, the per-request permission queues,
    session-scoped approvals, and per-task sub-agent buffers used to fold the
    chatty streaming events of background sub-agents into compact preview
    notifications.
    """

    def __init__(
        self,
        runtime: RuntimeManager,
        sessions: SessionManager,
        *,
        max_workers: int = 8,
    ) -> None:
        """Build the worker pool and the bookkeeping dicts used during turns."""
        self.runtime = runtime
        self.sessions = sessions
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._permission_lock = threading.Lock()
        self._permission_waiters: dict[str, queue.Queue[str]] = {}
        # Mirror of ``_permission_waiters`` for the interactive
        # ``AskUserQuestionTool``. The tool runs synchronously inside the
        # worker thread, so we need a thread-safe ``Queue`` to park it on
        # while the asyncio dispatcher routes the TUI's reply back.
        self._question_lock = threading.Lock()
        self._question_waiters: dict[str, queue.Queue[str]] = {}
        self._session_approvals: dict[str, set[str]] = {}
        self._subagent_buffer_lock = threading.Lock()
        self._subagent_parent_tool: dict[str, str] = {}
        self._subagent_tool_id_fifo: dict[str, list[str]] = {}
        self._subagent_tool_counts: dict[str, int] = {}
        self._subagent_text_buffers: dict[str, str] = {}
        self._subagent_thinking_buffers: dict[str, str] = {}
        self._current_tool_call_id = ""
        self._event_sink: Callable[[str, dict[str, Any]], None] | None = None
        # Install a session_search backend over the persisted transcripts so the
        # session_search tool actually works on the daemon path. Previously no
        # searcher was ever installed here and every call returned an error,
        # which silently broke cross-session recall.
        try:
            from ..tools.agent_meta_tools import set_session_searcher

            set_session_searcher(self._build_session_searcher())
        except Exception:
            pass

    def _build_session_searcher(self) -> Callable[..., dict[str, Any]]:
        """Return a ``session_search`` backend over the daemon's persisted transcripts.

        Scans the :class:`SessionManager` store (one JSON record per session) for
        messages whose text contains the query — a best-effort, case-insensitive
        substring match, newest sessions first. Enough to make cross-session
        recall work without a separate FTS index the daemon never populates.
        """
        store = self.sessions

        def _search(
            query: str,
            limit: int = 5,
            agent_id: str | None = None,
            session_id: str | None = None,
        ) -> dict[str, Any]:
            q = (query or "").strip().lower()
            if not q:
                return {"hits": []}
            try:
                paths = sorted(store._store_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            except OSError:
                return {"hits": []}
            hits: list[dict[str, Any]] = []
            for path in paths:
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(record, dict):
                    continue
                if session_id and str(record.get("session_id")) != session_id:
                    continue
                if agent_id and str(record.get("agent_id")) != agent_id:
                    continue
                for msg in record.get("messages", []):
                    if not isinstance(msg, dict):
                        continue  # tolerate legacy/hand-edited records
                    content = msg.get("content")
                    text = content if isinstance(content, str) else json.dumps(content, default=str)
                    pos = text.lower().find(q)
                    if pos == -1:
                        continue
                    flat = text.strip().replace("\n", " ")
                    start = max(0, flat.lower().find(q) - 80)
                    hits.append(
                        {
                            "session_id": record.get("session_id"),
                            "agent_id": record.get("agent_id"),
                            "role": msg.get("role"),
                            "snippet": flat[start : start + 240],
                        }
                    )
                    if len(hits) >= limit:
                        return {"hits": hits}
            return {"hits": hits}

        return _search

    def close(self) -> None:
        """Shut the worker pool down without waiting for in-flight tasks."""
        self._pool.shutdown(wait=False)
        # Detach the process-global session searcher this runner installed so it
        # can't outlive the runner (avoids cross-runner/test-isolation bleed).
        try:
            from ..tools.agent_meta_tools import set_session_searcher

            set_session_searcher(None)
        except Exception:
            pass

    def _set_session_mode(
        self,
        session: DaemonSession,
        mode: Any,
        plan_mode: bool = False,
        *,
        publish: bool = False,
    ) -> tuple[str, bool]:
        """Update the session-scoped interaction mode and optionally publish it globally."""
        normalized = normalize_interaction_mode(mode, plan_mode=plan_mode)
        session.interaction_mode = normalized
        session.plan_mode = normalized == "plan"
        if publish:
            session.runtime_config = dict(session.runtime_config or self.runtime.runtime_config)
            session.runtime_config["mode"] = normalized
            session.runtime_config["plan_mode"] = session.plan_mode
        return session.interaction_mode, session.plan_mode

    def _sync_session_mode_from_global(self, session: DaemonSession) -> tuple[str, bool]:
        """Return the session-scoped mode after model tools mutate the session."""
        return session.interaction_mode, session.plan_mode

    def set_event_sink(self, sink: Callable[[str, dict[str, Any]], None] | None) -> None:
        """Install the daemon-level event sink used by background sub-agents."""
        self._event_sink = sink

    def handle_agent_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Receive a sub-agent runtime event and forward it through the sink."""
        sink = self._event_sink
        if sink is not None:
            self._emit_subagent_summary(event_type, data, sink)

    def ask_user_question(self, question: str) -> str:
        """Blocking ``AskUserQuestionTool`` callback — drives the TUI question panel.

        Runs on the worker thread that's executing the tool. Generates a
        request id, emits a ``question_request`` wire event through the
        installed sink (the daemon broadcasts it to every connected
        client), then parks on a thread-safe queue until the asyncio
        dispatcher delivers the answer via :meth:`respond_question`. The
        active session's cancel flag is polled so a ``/cancel`` mid-question
        unblocks us with ``[cancelled]`` instead of stranding the turn.

        Returns the user's answer (multiple answers joined by newlines) or
        a sentinel string when no client is connected / cancel was
        requested. Never raises — failure modes degrade to the same
        non-interactive fallback as the original tool.
        """
        from ..runtime.session_context import get_active_session

        # ``set_event_sink`` is called during ``DaemonServer.run()`` before
        # any turn can dispatch a tool, so the sink is reliably present
        # whenever this method runs from the live daemon. The previous
        # "no sink → fake answer" fallback masked real misconfiguration;
        # if the sink were ever ``None`` here in production we'd rather
        # see the resulting NoneType crash and a stack trace than a
        # silent hallucinated answer.
        sink = self._event_sink
        if sink is None:
            raise RuntimeError("TurnRunner.ask_user_question called before event sink installed")

        request_id = uuid.uuid4().hex[:12]
        waiter: queue.Queue[str] = queue.Queue()
        with self._question_lock:
            self._question_waiters[request_id] = waiter
        try:
            sink(
                "question_request",
                {
                    "id": request_id,
                    "tool_call_id": self._current_tool_call_id,
                    "questions": [
                        {
                            "id": "q",
                            "question": question,
                            "options": [],
                            "allow_free_form": True,
                        }
                    ],
                },
            )
            session = get_active_session()
            # Poll with a short timeout so a turn cancel can unblock us
            # without leaving a dangling waiter for hours.
            while True:
                if session is not None and getattr(session, "cancel_requested", False):
                    return "[cancelled]"
                try:
                    return waiter.get(timeout=0.5)
                except queue.Empty:
                    continue
        finally:
            with self._question_lock:
                self._question_waiters.pop(request_id, None)

    async def respond_question(self, request_id: str, answers: dict[str, str] | str | None) -> bool:
        """Deliver the TUI's answer to a parked ``ask_user_question`` waiter.

        Accepts the per-question-id dict the TUI's ``QuestionRequestPanel``
        sends, a bare string for free-form answers, or ``None`` (treated
        as an empty answer). Returns ``False`` when the request id is
        unknown — typically because the user answered after the turn was
        cancelled, in which case the wait has already returned.
        """
        with self._question_lock:
            waiter = self._question_waiters.get(request_id)
        if waiter is None:
            return False
        if isinstance(answers, dict):
            joined = "\n".join(str(v) for v in answers.values() if v is not None)
        else:
            joined = str(answers or "")
        waiter.put_nowait(joined)
        return True

    async def respond_permission(self, request_id: str, response: str) -> bool:
        """Resolve a pending permission prompt with the TUI's answer."""
        with self._permission_lock:
            waiter = self._permission_waiters.get(request_id)
        if waiter is None:
            return False
        waiter.put_nowait(response.strip().lower())
        return True

    async def run_turn(
        self,
        session: DaemonSession,
        text: str,
        emit: EmitFn,
        *,
        mode: str = "code",
        plan_mode: bool = False,
    ) -> str:
        """Run one turn on ``session`` and stream events through ``emit``.

        The blocking streaming loop runs in the worker pool; events are pushed
        into an asyncio queue and re-emitted to the caller. The function
        returns once the worker finishes (or cancellation is observed),
        yielding the concatenated assistant text.
        """
        turn_id = uuid.uuid4().hex[:12]
        output_parts: list[str] = []
        async with session.lock:
            mode, plan_mode = self._set_session_mode(session, mode, plan_mode, publish=True)
            session.cancel_requested = False
            session.active_turn_id = turn_id
            await emit("turn_begin", {"user_input": text})
            await emit("step_begin", {"n": session.state.turn_count + 1})
            await emit("status_update", self._status_payload(session, mode=mode, plan_mode=plan_mode))

            queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def push(event_type: str, payload: dict[str, Any]) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, (event_type, payload))

            future = loop.run_in_executor(self._pool, self._run_sync, session, text, mode, plan_mode, push, output_parts)
            try:
                while True:
                    item_task = asyncio.create_task(queue.get())
                    wait_set: set[asyncio.Future[Any]] = {item_task, future}
                    done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                    if item_task in done:
                        item = item_task.result()
                        if item is not None:
                            await emit(item[0], item[1])
                        if future.done() and queue.empty():
                            break
                    else:
                        item_task.cancel()
                        if queue.empty():
                            break
                await future
                while not queue.empty():
                    item = queue.get_nowait()
                    if item is not None:
                        await emit(item[0], item[1])
            except Exception as exc:
                await emit(
                    "notification",
                    {
                        "id": uuid.uuid4().hex[:12],
                        "category": "daemon",
                        "type": "turn_error",
                        "severity": "error",
                        "title": "Turn failed",
                        "body": str(exc),
                        "payload": {},
                    },
                )
            finally:
                late_steers = session.drain_steers()
                if late_steers:
                    joined = "\n\n".join(late_steers)
                    session.state.messages.append(
                        {"role": "user", "content": f"[steer from user saved for next turn]\n{joined}"}
                    )
                    await emit(
                        "notification",
                        {
                            "id": uuid.uuid4().hex[:12],
                            "category": "slash",
                            "type": "result",
                            "severity": "info",
                            "title": "",
                            "body": (
                                "Steer saved for next turn: "
                                f"{late_steers[0][:80]}{'…' if len(late_steers[0]) > 80 else ''}"
                            ),
                            "payload": {},
                        },
                    )
                session.active_turn_id = ""
                self.sessions.compact_if_needed(session)
                if session.cancel_requested:
                    await emit("step_interrupted", {})
                await emit("turn_end", {})
                await emit("status_update", self._status_payload(session, mode=mode, plan_mode=plan_mode))

        return "".join(output_parts)

    def _run_sync(
        self,
        session: DaemonSession,
        text: str,
        mode: str,
        plan_mode: bool,
        push: Callable[[str, dict[str, Any]], None],
        output_parts: list[str],
    ) -> None:
        """Blocking worker-thread body of one turn.

        Builds the system prompt from the runtime prompt + workspace context
        + agent memory + active skills, then iterates the streaming loop and
        translates each event (text/thinking/tool/permission/done/suggestion)
        into a daemon wire event via ``push``. Persists the session at
        :class:`TurnDone`. Binds ``session`` to a thread-local so tools that
        need session-scoped state (``AwaitAgents`` to read ``pending_steers``
        and the cancel flag, ``CheckAgentMessages`` for the mailbox cursor)
        can fetch it without plumbing.
        """
        from ..agents.subagent_manager import SubAgentManager
        from ..runtime.config_context import set_active_config
        from ..runtime.session_context import set_active_session
        from ..tools.claude_tools import _get_agent_manager

        system_prompt = render_session_system_prompt(self.runtime, session, mode=mode)
        config = dict(session.runtime_config or self.runtime.runtime_config)
        config["mode"] = mode
        config["plan_mode"] = plan_mode
        session.runtime_config = dict(config)

        # Bind the session to this worker thread so tools can find it. The
        # cursor tracks the last mailbox seq we auto-injected so the drain
        # only emits *new* sub-agent events between iterations.
        set_active_session(session)
        mgr: SubAgentManager | None
        try:
            mgr = _get_agent_manager()
            # Hand the manager the daemon's tool plumbing so subagents can
            # actually call tools when ``_run_streaming_loop`` runs them.
            if mgr._tool_executor is None:
                mgr._tool_executor = self.runtime.tool_executor
            if mgr._tool_schemas is None:
                mgr._tool_schemas = self.runtime.tool_schemas
        except Exception:
            mgr = None

        cursor = {"seq": mgr.latest_seq() if mgr is not None else 0}

        def _drain_agent_events() -> list[str]:
            """Drain new mailbox events as compact one-line summaries."""
            if mgr is None:
                return []
            new_events = mgr.drain_mailbox(since_seq=cursor["seq"])
            if not new_events:
                return []
            cursor["seq"] = new_events[-1]["seq"]
            return [_format_agent_event(evt) for evt in new_events]

        try:
            self._run_event_loop(
                session=session,
                config=config,
                mode=mode,
                plan_mode=plan_mode,
                push=push,
                output_parts=output_parts,
                user_message=text,
                system_prompt=system_prompt,
                drain_agent_events=_drain_agent_events,
            )
        finally:
            # Always unbind the session — tools called after the turn (in
            # tests, for example) should not pick up a stale handle.
            set_active_config(None)
            set_active_session(None)

    @staticmethod
    def _diff_display_blocks(result: Any) -> list[dict[str, Any]]:
        """Surface an embedded unified diff from a tool result as a ``diff`` display block.

        File-mutating tools (``FileEditTool``, ``write_file``) return their
        unified diff inline in the result string. We lift it into a structured
        ``diff`` block so the TUI can render additions/removals instead of
        collapsing the whole diff into a single 40-char summary line. The probe
        is tool-agnostic: any result whose lines contain an adjacent
        ``--- ``/``+++ `` header pair is treated as carrying a diff (the same
        ``git diff`` a shell command might print also renders nicely). Stringified
        tool payloads with escaped ``\\n`` never split into real lines, so they
        cannot false-trigger.
        """
        if not isinstance(result, str) or "\n" not in result:
            return []
        lines = result.split("\n")
        for i in range(len(lines) - 1):
            if lines[i].startswith("--- ") and lines[i + 1].startswith("+++ "):
                diff = "\n".join(lines[i:]).strip("\n")
                return [{"type": "diff", "diff": diff, "language": ""}] if diff else []
        return []

    _TODO_STATUS_BY_MARK: ClassVar[dict[str, str]] = {
        " ": "pending",
        "~": "in_progress",
        "x": "completed",
        "X": "completed",
    }
    _TODO_LINE_RE: ClassVar[re.Pattern[str]] = re.compile(r"\s*\d+\.\s*\[(.)\]\s*(.*)$")

    @classmethod
    def _todo_display_blocks(cls, result: Any) -> list[dict[str, Any]]:
        """Surface a ``TodoWriteTool`` result as a structured ``todo`` display block.

        The tool returns a ``# Todo List`` summary whose rows carry ``[ ]`` /
        ``[~]`` / ``[x]`` status markers. We parse them back into
        ``{content, status}`` items so the TUI can drive (and live-update) its
        pinned todo panel instead of collapsing the whole list into a one-line
        tool summary. Keyed on the ``# Todo List`` header so ordinary tool
        output can't false-trigger.
        """
        if not isinstance(result, str) or not result.lstrip().startswith("# Todo List"):
            return []
        items: list[dict[str, str]] = []
        for line in result.splitlines():
            match = cls._TODO_LINE_RE.match(line)
            if match:
                items.append(
                    {
                        "content": match.group(2).strip(),
                        "status": cls._TODO_STATUS_BY_MARK.get(match.group(1), "pending"),
                    }
                )
        return [{"type": "todo", "items": items}] if items else []

    def _run_event_loop(
        self,
        *,
        session: DaemonSession,
        config: dict[str, Any],
        mode: str,
        plan_mode: bool,
        push: Callable[[str, dict[str, Any]], None],
        output_parts: list[str],
        user_message: str,
        system_prompt: str,
        drain_agent_events: Callable[[], list[str]],
    ) -> None:
        """Drive the streaming-loop iterator and translate events into wire payloads."""
        for event in run_agent_loop(
            user_message=user_message,
            state=session.state,
            config=config,
            system_prompt=system_prompt,
            tool_executor=self.runtime.tool_executor,
            tool_schemas=self.runtime.tool_schemas,
            cancel_check=lambda: session.cancel_requested,
            steer_drain=session.drain_steers,
            agent_event_drain=drain_agent_events,
        ):
            if isinstance(event, TextChunk):
                output_parts.append(event.text)
                push("text_part", {"text": event.text})
            elif isinstance(event, ProviderRetry):
                retry_body = (
                    f"{event.error}\nUse /retry-connection to retry the last prompt."
                    if event.final
                    else f"{event.error}\nRetrying provider connection in {event.delay}s "
                    f"({event.attempt}/{event.max_attempts})."
                )
                push(
                    "notification",
                    {
                        "id": f"{session.id}:provider-connection",
                        "category": "provider_connection",
                        "type": "failed" if event.final else "retrying",
                        "severity": "error" if event.final else "warning",
                        "title": "Provider connection",
                        "body": retry_body,
                        "payload": {
                            "attempt": event.attempt,
                            "max_attempts": event.max_attempts,
                            "delay": event.delay,
                            "final": event.final,
                        },
                    },
                )
            elif isinstance(event, ThinkingChunk):
                push("think_part", {"think": event.text})
            elif isinstance(event, ToolStart):
                self._current_tool_call_id = event.tool_call_id
                push(
                    "tool_call",
                    {
                        "id": event.tool_call_id,
                        "name": event.name,
                        "arguments": json.dumps(event.inputs, ensure_ascii=False, default=str),
                    },
                )
            elif isinstance(event, ToolEnd):
                self._sync_session_mode_from_global(session)
                display_blocks = self._diff_display_blocks(event.result) + self._todo_display_blocks(event.result)
                push(
                    "tool_result",
                    {
                        "tool_call_id": event.tool_call_id,
                        "return_value": event.result,
                        "duration_ms": event.duration_ms,
                        "display_blocks": display_blocks,
                    },
                )
                push("status_update", self._status_payload(session, mode=mode, plan_mode=plan_mode))
                if self._current_tool_call_id == event.tool_call_id:
                    self._current_tool_call_id = ""
            elif isinstance(event, PermissionRequest):
                if self._is_session_approved(session.key, event.tool_name):
                    event.granted = True
                    continue

                request_id = uuid.uuid4().hex[:12]
                waiter: queue.Queue[str] = queue.Queue()
                with self._permission_lock:
                    self._permission_waiters[request_id] = waiter
                try:
                    push(
                        "approval_request",
                        {
                            "id": request_id,
                            "tool_call_id": "",
                            "action": event.tool_name,
                            "description": event.description,
                        },
                    )
                    response = self._wait_for_permission_response(session, request_id, waiter)
                    if response == "approve_for_session":
                        self._approve_for_session(session.key, event.tool_name)
                    event.granted = response in {"approve", "approve_for_session"}
                finally:
                    with self._permission_lock:
                        self._permission_waiters.pop(request_id, None)
            elif isinstance(event, TurnDone):
                self._sync_session_mode_from_global(session)
                push("status_update", self._status_payload(session, mode=mode, plan_mode=plan_mode))
                self._emit_change_guard_if_needed(session, push)
                # Persist the session so /resume + `xerxes -r <id>` actually
                # rehydrate this conversation on the next launch.
                try:
                    self.sessions.save(session)
                except Exception:
                    pass
            elif isinstance(event, SkillSuggestion):
                push(
                    "notification",
                    {
                        "id": uuid.uuid4().hex[:12],
                        "category": "skill",
                        "type": "suggestion",
                        "severity": "info",
                        "title": f"Skill suggested: {event.skill_name}",
                        "body": event.description,
                        "payload": {"source_path": event.source_path},
                    },
                )

    def _emit_change_guard_if_needed(
        self,
        session: DaemonSession,
        push: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Notify once when the current Git diff contains high-risk edits."""
        cwd = Path(self.runtime.config.project_dir or os.getcwd()).expanduser()
        report = analyze_workspace_changes(cwd, session.state.tool_executions)
        if not report.should_notify:
            return

        metadata = dict(session.state.metadata or {})
        if metadata.get("last_change_guard_fingerprint") == report.fingerprint:
            return
        metadata["last_change_guard_fingerprint"] = report.fingerprint
        session.state.metadata = metadata

        push(
            "notification",
            {
                "id": uuid.uuid4().hex[:12],
                "category": "workspace_guard",
                "type": "risky_changes",
                "severity": report.severity,
                "title": "Workspace guard",
                "body": format_change_guard_notification(report),
                "payload": {
                    "fingerprint": report.fingerprint,
                    "findings": [finding.__dict__ for finding in report.findings],
                    "verification_commands": list(report.verification_commands),
                },
            },
        )

    SUBAGENT_PREVIEW_CHARS = 100

    def _emit_subagent_summary(
        self,
        event_type: str,
        data: dict[str, Any],
        push: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Fold an ``agent_*`` sub-agent event into a per-agent lane update.

        Each emission carries structured lane fields (``status``, ``count``,
        ``action``, ``result``) on the ``subagent_stream`` notification so the
        TUI can render a live per-agent dashboard. The per-task call counter is
        what makes 12 agents by thousands of calls cheap: we count, we never ship
        the calls themselves."""
        if not event_type.startswith("agent_"):
            return

        agent_name = data.get("agent_name") or data.get("agent_type") or "subagent"
        agent_type = data.get("agent_type") or ""
        task_id = str(data.get("task_id", ""))
        short_id = (task_id[:8] + "...") if len(task_id) > 8 else task_id
        prefix = f"{agent_name}#{short_id}" if short_id else agent_name

        if event_type == "agent_spawn":
            if task_id:
                with self._subagent_buffer_lock:
                    self._subagent_tool_counts[task_id] = 0
                    if self._current_tool_call_id:
                        self._subagent_parent_tool[task_id] = self._current_tool_call_id
            self._emit_subagent_stream(
                push,
                task_id,
                prefix,
                f"spawned: {str(data.get('prompt', ''))[:120]}",
                agent_type=agent_type,
                status="running",
                action="spawned",
            )
            return

        if event_type == "agent_text":
            self._stream_subagent_chunk(
                push, task_id, prefix, data.get("text") or "", kind="text", agent_type=agent_type
            )
            return

        if event_type == "agent_thinking":
            self._stream_subagent_chunk(
                push, task_id, prefix, data.get("text") or "", kind="thinking", agent_type=agent_type
            )
            return

        if event_type == "agent_tool_start":
            self._emit_subagent_tool_event(push, task_id, agent_type, data, kind="start")
            with self._subagent_buffer_lock:
                self._subagent_tool_counts[task_id] = self._subagent_tool_counts.get(task_id, 0) + 1
            inputs = data.get("inputs") or {}
            key = next(iter(inputs.values()), "") if isinstance(inputs, dict) else ""
            tool_name = data.get("tool_name", "tool")
            self._emit_subagent_stream(
                push,
                task_id,
                prefix,
                f"o {tool_name}({str(key)[:80]})",
                agent_type=agent_type,
                status="running",
                action=f"{tool_name} {str(key)[:48]}".strip(),
            )
            return

        if event_type == "agent_tool_end":
            self._emit_subagent_tool_event(push, task_id, agent_type, data, kind="end")
            mark = "OK" if data.get("permitted", True) else "DENIED"
            tool_name = data.get("tool_name", "tool")
            self._emit_subagent_stream(
                push,
                task_id,
                prefix,
                f"{mark} {tool_name} - {float(data.get('duration_ms', 0) or 0):.0f}ms",
                agent_type=agent_type,
                status="running",
                action=str(tool_name),
            )
            return

        if event_type == "agent_done":
            status = str(data.get("status") or "completed")
            result = " ".join(str(data.get("result") or "").split())[:160]
            # Emit the final lane state (status + result) *before* dropping the
            # counter so the dashboard shows where each agent landed; the lane is
            # cleared by the TUI when the turn ends.
            self._emit_subagent_stream(
                push,
                task_id,
                prefix,
                result or status,
                agent_type=agent_type,
                status=status,
                action=result or status,
                result=result,
            )
            with self._subagent_buffer_lock:
                self._subagent_parent_tool.pop(task_id, None)
                self._subagent_tool_id_fifo.pop(task_id, None)
                self._subagent_text_buffers.pop(task_id, None)
                self._subagent_thinking_buffers.pop(task_id, None)
                self._subagent_tool_counts.pop(task_id, None)

    def _stream_subagent_chunk(
        self,
        push: Callable[[str, dict[str, Any]], None],
        task_id: str,
        prefix: str,
        text: str,
        *,
        kind: str,
        agent_type: str = "",
    ) -> None:
        """Update the per-agent lane's current-action line from a text/thinking chunk."""
        if not text or not task_id:
            return
        buffers = self._subagent_text_buffers if kind == "text" else self._subagent_thinking_buffers
        cap = self.SUBAGENT_PREVIEW_CHARS
        with self._subagent_buffer_lock:
            merged = buffers.get(task_id, "") + text
            if len(merged) > cap * 2:
                merged = merged[-cap * 2 :]
            buffers[task_id] = merged
            tail = " ".join(merged.split())
        if not tail:
            return
        if len(tail) > cap:
            tail = "..." + tail[-cap:]
        label_suffix = " (thinking)" if kind == "thinking" else ""
        self._emit_subagent_stream(
            push, task_id, f"{prefix}{label_suffix}", tail, agent_type=agent_type, status="running", action=tail
        )

    def _emit_subagent_stream(
        self,
        push: Callable[[str, dict[str, Any]], None],
        task_id: str,
        label: str,
        body: str,
        *,
        agent_type: str = "",
        status: str = "running",
        action: str = "",
        result: str = "",
    ) -> None:
        """Emit a ``subagent_stream`` notification carrying structured per-agent lane state.

        The ``count`` is read from the live per-task counter so every lane update
        ships the current call total without the caller threading it through."""
        count = self._subagent_tool_counts.get(task_id, 0)
        push(
            "notification",
            {
                "id": uuid.uuid4().hex[:12],
                "category": "subagent_stream",
                "type": "subagent_stream",
                "severity": "info",
                "title": "",
                "body": body,
                "payload": {
                    "task_id": task_id,
                    "label": label,
                    "agent_type": agent_type,
                    "status": status,
                    "count": count,
                    "action": action or body,
                    "result": result,
                    # The spawn tool-call this agent belongs to, so the TUI can
                    # scope/collapse each spawn's dashboard independently.
                    "parent": self._subagent_parent_tool.get(task_id, ""),
                },
            },
        )

    def _emit_subagent_tool_event(
        self,
        push: Callable[[str, dict[str, Any]], None],
        task_id: str,
        agent_type: str,
        data: dict[str, Any],
        *,
        kind: str,
    ) -> bool:
        """Wrap a sub-agent inner tool call/result in a ``subagent_event`` payload.

        Returns ``True`` when the nested event was emitted with a known
        parent tool-call id (so callers can suppress the flat fallback). Falls
        through to ``False`` if either id is missing.
        """
        with self._subagent_buffer_lock:
            parent_id = self._subagent_parent_tool.get(task_id) or self._current_tool_call_id
            raw_inner = data.get("tool_call_id")
            if kind == "start":
                inner_id = str(raw_inner) if raw_inner else f"sub_{uuid.uuid4().hex[:12]}"
                self._subagent_tool_id_fifo.setdefault(task_id, []).append(inner_id)
            else:
                if raw_inner:
                    inner_id = str(raw_inner)
                    fifo = self._subagent_tool_id_fifo.get(task_id) or []
                    if inner_id in fifo:
                        fifo.remove(inner_id)
                else:
                    fifo = self._subagent_tool_id_fifo.get(task_id) or []
                    inner_id = fifo.pop(0) if fifo else ""
        if not parent_id or not inner_id:
            return False
        if kind == "start":
            inputs = data.get("inputs") or {}
            try:
                arguments = json.dumps(inputs, ensure_ascii=False, default=str)
            except Exception:
                arguments = ""
            inner = {
                "type": "ToolCall",
                "payload": {
                    "id": inner_id,
                    "name": data.get("tool_name", ""),
                    "arguments": arguments,
                },
            }
        else:
            inner = {
                "type": "ToolResult",
                "payload": {
                    "tool_call_id": inner_id,
                    "return_value": data.get("result") or "",
                    "duration_ms": float(data.get("duration_ms", 0) or 0),
                    "display_blocks": [],
                },
            }
        push(
            "subagent_event",
            {
                "parent_tool_call_id": parent_id,
                "agent_id": task_id,
                "subagent_type": agent_type,
                "event": inner,
            },
        )
        return True

    def _wait_for_permission_response(
        self,
        session: DaemonSession,
        request_id: str,
        waiter: queue.Queue[str],
    ) -> str:
        """Block the worker thread until the TUI responds or the session is cancelled."""
        while not session.cancel_requested:
            with self._permission_lock:
                if request_id not in self._permission_waiters:
                    return "reject"
            try:
                return waiter.get(timeout=0.1)
            except queue.Empty:
                continue
        return "reject"

    def _is_session_approved(self, session_key: str, tool_name: str) -> bool:
        """True when ``tool_name`` has been approve-for-session on ``session_key``."""
        with self._permission_lock:
            return tool_name in self._session_approvals.get(session_key, set())

    def _approve_for_session(self, session_key: str, tool_name: str) -> None:
        """Remember a per-session approve-all decision for ``tool_name``."""
        with self._permission_lock:
            self._session_approvals.setdefault(session_key, set()).add(tool_name)

    def _status_payload(self, session: DaemonSession, *, mode: str, plan_mode: bool) -> dict[str, Any]:
        """Build the ``status_update`` payload for ``session``."""
        status_mode = str(getattr(session, "interaction_mode", mode) or mode)
        status_plan_mode = bool(getattr(session, "plan_mode", plan_mode))
        runtime_config = getattr(session, "runtime_config", None) or self.runtime.runtime_config
        model = str(runtime_config.get("model", "")) or self.runtime.model
        reasoning_effort = "off"
        if bool(runtime_config.get("thinking", False)):
            reasoning_effort = str(runtime_config.get("reasoning_effort") or "medium")
        system_prompt = render_session_system_prompt(
            self.runtime,
            session,
            mode=status_mode,
            tolerate_errors=True,
        )
        return {
            "context_tokens": estimate_context_tokens(
                session.state.messages,
                model=model,
                system_prompt=system_prompt,
                tool_schemas=self.runtime.tool_schemas,
            ),
            "max_context": self._resolve_context_limit(runtime_config),
            "mcp_status": {},
            "plan_mode": status_plan_mode,
            "reasoning_effort": reasoning_effort,
            "mode": status_mode,
        }

    def _resolve_context_limit(self, runtime_config: dict[str, Any] | None = None) -> int:
        """Resolve the model's context window for the status bar.

        Prefers an explicit ``context_limit`` / ``max_context`` from runtime
        config (so users can pin custom values for self-hosted models), then
        falls back to :func:`xerxes.llms.registry.get_context_limit` which
        knows the published windows for every shipped provider. The status
        bar used to render ``0/0`` when no override was set; this guarantees
        a useful denominator even on fresh installs.
        """
        cfg = runtime_config or self.runtime.runtime_config
        explicit = cfg.get("context_limit", cfg.get("max_context", 0)) or 0
        if explicit:
            try:
                return int(explicit)
            except (TypeError, ValueError):
                pass
        model = cfg.get("model", "") or self.runtime.model
        if not model:
            return 0
        try:
            from xerxes.llms.registry import get_context_limit

            return int(get_context_limit(model))
        except Exception:
            return 0


__all__ = ["DaemonSession", "RuntimeManager", "SessionManager", "TurnRunner", "WorkspaceManager"]
