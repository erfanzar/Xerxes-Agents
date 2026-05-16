# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import threading
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..bridge import profiles
from ..channels.workspace import MarkdownAgentWorkspace
from ..core.paths import xerxes_subdir
from ..extensions.skills import SkillRegistry, get_active_skills
from ..runtime.agent_memory import AgentMemory
from ..runtime.bootstrap import bootstrap
from ..runtime.bridge import build_tool_executor, populate_registry
from ..runtime.config_context import set_config as set_global_config
from ..streaming.events import (
    AgentState,
    PermissionRequest,
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
        runtime.update({k: v for k, v in (overrides or {}).items() if v not in (None, "")})

        profile = profiles.get_active_profile()
        if profile and not runtime.get("model"):
            runtime.update(
                {
                    "model": profile.get("model", ""),
                    "base_url": profile.get("base_url", ""),
                    "api_key": profile.get("api_key", ""),
                    **profile.get("sampling", {}),
                }
            )

        if not runtime.get("model"):
            raise RuntimeError("No provider profile or daemon runtime model configured.")

        runtime.setdefault("permission_mode", "accept-all")
        set_global_config(runtime)

        cwd = Path(self.config.project_dir or os.getcwd()).expanduser()
        boot = bootstrap(model=str(runtime.get("model", "")), cwd=cwd)
        registry = populate_registry()
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
            "permission_mode": self.runtime_config.get("permission_mode", "auto"),
            "tools": len(self.tool_schemas),
            "skills": len(self.skill_registry.skill_names),
        }

    @property
    def permission_mode(self) -> str:
        """Current permission strategy (``auto`` | ``manual`` | ``accept-all``)."""
        return str(self.runtime_config.get("permission_mode", "auto"))

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

    def discover_skills(self) -> list[str]:
        """Re-scan bundled, user, and cwd skill directories; return sorted ids.

        Returned ids include both root skill names and ``name:subcommand``
        forms for skills that declare sub-commands.
        """
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        import xerxes as _xerxes_pkg

        bundled = Path(_xerxes_pkg.__file__).parent / "skills"
        discover_dirs = [str(self.skills_dir), str(Path.cwd() / "skills")]
        if bundled.is_dir():
            discover_dirs.insert(0, str(bundled))
        self.skill_registry.discover(*discover_dirs)
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
        state: Cumulative streaming-loop state (messages, tokens, tools).
        lock: Asyncio lock serialising turns on this session.
        cancel_requested: Set true to tell the streaming loop to stop early.
        active_turn_id: Current turn id while one is running.
        pending_steers: Thread-safe queue of ``/steer`` strings drained by
            the streaming loop between tool iterations.
    """

    id: str
    key: str
    agent_id: str
    workspace: MarkdownAgentWorkspace
    state: AgentState = field(default_factory=AgentState)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancel_requested: bool = False
    active_turn_id: str = ""
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
            "active_turn_id": self.active_turn_id,
            "messages": len(self.state.messages),
            "turn_count": self.state.turn_count,
            "input_tokens": self.state.total_input_tokens,
            "output_tokens": self.state.total_output_tokens,
            "cancel_requested": self.cancel_requested,
        }


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
    so ``xerxes -r <id>`` truly rehydrates the conversation. The ``keep_messages``
    bound caps in-memory history length; older messages are compacted out by
    :meth:`compact_if_needed` and a marker is appended to the agent's daily note.
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

    # ---------------------------- persistence helpers -----------------------

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

    def save(self, session: DaemonSession) -> None:
        """Persist ``session`` to disk via a tempfile + ``os.replace`` swap.

        Atomic so the JSON file is never partially overwritten if the daemon
        crashes mid-write.
        """
        record = {
            "session_id": session.id,
            "key": session.key,
            "agent_id": session.agent_id,
            "cwd": str(session.workspace.path),
            "updated_at": datetime.now(UTC).isoformat(),
            "messages": session.state.messages,
            "turn_count": session.state.turn_count,
            "total_input_tokens": session.state.total_input_tokens,
            "total_output_tokens": session.state.total_output_tokens,
            "thinking_content": session.state.thinking_content,
            "tool_executions": session.state.tool_executions,
        }
        path = self._session_path(session.id)
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

    # ---------------------------- main API ----------------------------------

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
            )
            state = session.state
            state.messages = list(loaded.get("messages", []))
            state.turn_count = int(loaded.get("turn_count", 0))
            state.total_input_tokens = int(loaded.get("total_input_tokens", 0))
            state.total_output_tokens = int(loaded.get("total_output_tokens", 0))
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

    # Parallel trim limits applied at the same time we compact ``messages``.
    # ``thinking_content`` and ``tool_executions`` are append-only side
    # buffers — they accumulate one entry per turn / tool call and were
    # never trimmed, so a long-running session would silently hold tens
    # of MB of stale reasoning text and audit records.
    _THINKING_KEEP = 32
    _TOOL_EXEC_KEEP = 200

    def compact_if_needed(self, session: DaemonSession) -> bool:
        """Trim ``session`` history to ``keep_messages`` (plus parallel buffers).

        Returns ``True`` when the session was actually compacted.
        """
        state = session.state
        compacted = False

        if len(state.messages) > self.keep_messages:
            removed = len(state.messages) - self.keep_messages
            session.workspace.append_daily_note(
                f"[session:{session.key}] compacted {removed} old messages; kept last {self.keep_messages}."
            )
            state.messages = state.messages[-self.keep_messages :]
            compacted = True

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
        self._subagent_text_buffers: dict[str, str] = {}
        self._subagent_thinking_buffers: dict[str, str] = {}
        self._current_tool_call_id = ""
        self._event_sink: Callable[[str, dict[str, Any]], None] | None = None

    def close(self) -> None:
        """Shut the worker pool down without waiting for in-flight tasks."""
        self._pool.shutdown(wait=False)

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
                    done, _pending = await asyncio.wait({item_task, future}, return_when=asyncio.FIRST_COMPLETED)
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
        from ..runtime.session_context import set_active_session
        from ..tools.claude_tools import _get_agent_manager

        workspace_context = session.workspace.load_context()
        # Inject the agent's persistent memory (global + project) so the
        # model can read its own notes at every turn. The agent updates
        # these files via agent_memory_* tools.
        try:
            memory_section = self.runtime.agent_memory.to_prompt_section()
        except Exception:
            memory_section = ""
        system_prompt = "\n\n".join(
            part
            for part in (
                self.runtime.system_prompt.rstrip(),
                workspace_context.prompt,
                memory_section,
                self.runtime.active_skill_prompt(),
            )
            if part
        )
        config = dict(self.runtime.runtime_config)
        config["mode"] = mode
        config["plan_mode"] = plan_mode

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
            set_active_session(None)

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
                push(
                    "tool_result",
                    {
                        "tool_call_id": event.tool_call_id,
                        "return_value": event.result,
                        "duration_ms": event.duration_ms,
                        "display_blocks": [],
                    },
                )
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
                push(
                    "status_update",
                    {
                        "context_tokens": session.state.total_input_tokens + session.state.total_output_tokens,
                        "max_context": self._resolve_context_limit(),
                        "mcp_status": {},
                        "plan_mode": plan_mode,
                        "mode": mode,
                    },
                )
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

    SUBAGENT_PREVIEW_CHARS = 100

    def _emit_subagent_summary(
        self,
        event_type: str,
        data: dict[str, Any],
        push: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Fold an ``agent_*`` sub-agent event into compact preview/tool notifications."""
        if not event_type.startswith("agent_"):
            return

        agent_name = data.get("agent_name") or data.get("agent_type") or "subagent"
        agent_type = data.get("agent_type") or ""
        task_id = str(data.get("task_id", ""))
        short_id = (task_id[:8] + "...") if len(task_id) > 8 else task_id
        prefix = f"{agent_name}#{short_id}" if short_id else agent_name

        if event_type == "agent_spawn":
            if task_id and self._current_tool_call_id:
                with self._subagent_buffer_lock:
                    self._subagent_parent_tool[task_id] = self._current_tool_call_id
            self._emit_subagent_stream(push, task_id, prefix, f"spawned: {str(data.get('prompt', ''))[:120]}")
            return

        if event_type == "agent_text":
            self._stream_subagent_chunk(push, task_id, prefix, data.get("text") or "", kind="text")
            return

        if event_type == "agent_thinking":
            self._stream_subagent_chunk(push, task_id, prefix, data.get("text") or "", kind="thinking")
            return

        if event_type == "agent_tool_start":
            self._emit_subagent_tool_event(push, task_id, agent_type, data, kind="start")
            inputs = data.get("inputs") or {}
            key = next(iter(inputs.values()), "") if isinstance(inputs, dict) else ""
            self._emit_subagent_stream(push, task_id, prefix, f"o {data.get('tool_name', 'tool')}({str(key)[:80]})")
            return

        if event_type == "agent_tool_end":
            self._emit_subagent_tool_event(push, task_id, agent_type, data, kind="end")
            mark = "OK" if data.get("permitted", True) else "DENIED"
            self._emit_subagent_stream(
                push,
                task_id,
                prefix,
                f"{mark} {data.get('tool_name', 'tool')} - {float(data.get('duration_ms', 0) or 0):.0f}ms",
            )
            return

        if event_type == "agent_done":
            with self._subagent_buffer_lock:
                self._subagent_parent_tool.pop(task_id, None)
                self._subagent_tool_id_fifo.pop(task_id, None)
                self._subagent_text_buffers.pop(task_id, None)
                self._subagent_thinking_buffers.pop(task_id, None)
            self._emit_subagent_stream(push, task_id, prefix, "")

    def _stream_subagent_chunk(
        self,
        push: Callable[[str, dict[str, Any]], None],
        task_id: str,
        prefix: str,
        text: str,
        *,
        kind: str,
    ) -> None:
        """Update the rolling preview line for one sub-agent text or thinking chunk."""
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
        self._emit_subagent_stream(push, task_id, f"{prefix}{label_suffix}", tail)

    @staticmethod
    def _emit_subagent_stream(
        push: Callable[[str, dict[str, Any]], None],
        task_id: str,
        label: str,
        body: str,
    ) -> None:
        """Emit a transient ``subagent_stream`` preview notification (empty ``body`` clears)."""
        push(
            "notification",
            {
                "id": uuid.uuid4().hex[:12],
                "category": "subagent_stream",
                "type": "subagent_stream",
                "severity": "info",
                "title": "",
                "body": body,
                "payload": {"task_id": task_id, "label": label},
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
        return {
            "context_tokens": session.state.total_input_tokens + session.state.total_output_tokens,
            "max_context": self._resolve_context_limit(),
            "mcp_status": {},
            "plan_mode": plan_mode,
            "mode": mode,
        }

    def _resolve_context_limit(self) -> int:
        """Resolve the model's context window for the status bar.

        Prefers an explicit ``context_limit`` / ``max_context`` from runtime
        config (so users can pin custom values for self-hosted models), then
        falls back to :func:`xerxes.llms.registry.get_context_limit` which
        knows the published windows for every shipped provider. The status
        bar used to render ``0/0`` when no override was set; this guarantees
        a useful denominator even on fresh installs.
        """
        cfg = self.runtime.runtime_config
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
