# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Daemon JSON-RPC dispatcher and lifecycle owner.

:class:`DaemonServer` is the single process that the TUI, remote websocket
clients, and configured channels (Telegram, Slack, ...) all talk to. It
multiplexes three listening surfaces — a Unix socket (``SocketChannel``), a
WebSocket gateway (``WebSocketGateway``), and optional channel webhooks
(``ChannelWebhookServer``) — onto one async event loop, and dispatches each
inbound JSON-RPC method to the correct handler. The protocol version exposed
to clients is :data:`DAEMON_PROTOCOL_VERSION` (currently ``14``); legacy
``task.*`` methods now respond with :data:`MIGRATED_ERROR` to nudge callers
to the new ``session.*`` / ``turn.*`` surface.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from ..bridge import profiles
from ..channels.types import ChannelMessage, MessageDirection
from ..extensions.skills import activate_skill, inject_skill_config, skill_matches_platform
from ..runtime.config_context import get_event_callback, set_event_callback
from .channels import ChannelManager, ChannelWebhookServer
from .config import DaemonConfig, load_config
from .gateway import EmitFn, WebSocketGateway
from .log import DaemonLogger
from .runtime import DaemonSession, RuntimeManager, SessionManager, TurnRunner, WorkspaceManager
from .socket_channel import SocketChannel

MIGRATED_ERROR = (
    "Old daemon task API was removed; use session.open, turn.submit, turn.cancel, session.list, and runtime.status."
)
DAEMON_PROTOCOL_VERSION = (
    35  # bumped: AskUserQuestionTool now drives the TUI panel via TurnRunner.ask_user_question
)


# Sentinel labels used by the ``/provider`` interactive panel. Centralised
# here so the dispatcher (``_advance_provider_flow``) and the panel emitter
# (``_emit_provider_main_panel``) stay in sync without stringly-typed magic.
_PROVIDER_ADD_LABEL = "+ Add new profile…"
_PROVIDER_EDIT_LABEL = "✎ Edit existing profile…"
_PROVIDER_REMOVE_LABEL = "✗ Remove existing profile…"
_PROVIDER_CANCEL_LABEL = "Cancel"

# Inference wire-format tags accepted by :func:`bridge.profiles.save_profile`.
# Mirrors the values ``_guess_provider`` returns; ``auto`` defers to that
# heuristic. Surfacing them as a picker lets users override the guess
# (essential for self-hosted vLLM/SGLang/Ollama endpoints).
_PROVIDER_TYPE_OPTIONS: tuple[str, ...] = (
    "auto",
    "openai",
    "anthropic",
    "ollama",
    "gemini",
    "deepseek",
    "groq",
    "together",
    "kimi",  # general Kimi chat (api.moonshot.cn)
    "kimi-code",  # coding-specialised endpoint (api.kimi.com/coding/v1)
    "minimax",
    "local",
    "custom",
)


# Ordered (key, question) pairs that the ``/skill-create`` interview walks
# through. ``name`` is filled in by the slug prompt (or the inline argument)
# *before* the interview proper begins, so the loop here covers the four
# scope questions that shape the SKILL.md body. ``pitfalls`` accepts an empty
# answer (just press Enter to skip) — the others must be non-empty.
#
# Type ``auto`` at *any* question to stop being asked and let the model fill
# the remaining fields itself from session context.
_SKILL_CREATE_STEPS: tuple[tuple[str, str, bool], ...] = (
    (
        "what",
        "What should this skill do? One or two sentences. "
        "Type `auto` to let me infer everything from this session, or `/cancel` to abort.",
        True,
    ),
    (
        "when",
        "When should a future session activate this skill? Describe the trigger. Type `auto` to let me decide.",
        True,
    ),
    (
        "tools",
        "Which tools or commands does the procedure use? List them, comma-separated. Type `auto` to let me decide.",
        True,
    ),
    (
        "pitfalls",
        "Any pitfalls or things that went wrong worth recording? Press Enter to skip, or type `auto` to let me decide.",
        False,  # empty answer is OK
    ),
)

# Sentinel a user types to delegate an interview field to the model.
_SKILL_CREATE_AUTO = "<<auto>>"


class DaemonServer:
    """Single daemon process owning sessions, turns, channels, and transports.

    Composes the :class:`RuntimeManager` (provider + tools), :class:`SessionManager`,
    :class:`TurnRunner`, :class:`ChannelManager`, and the three listening
    surfaces (websocket gateway, Unix socket, optional channel webhook
    server). Holds the current TUI session key, interaction mode, and plan-mode
    state so single-client commands like ``prompt`` and ``set_mode`` don't
    have to pass them every call.
    """

    def __init__(self, config: DaemonConfig) -> None:
        """Build managers and transports; nothing is started until :meth:`run`."""
        self.config = config
        self.logger = DaemonLogger(config.log_dir)
        self.runtime = RuntimeManager(config)
        self.workspaces = WorkspaceManager(config)
        self.sessions = SessionManager(
            self.workspaces,
            keep_messages=int(config.workspace.get("session_keep_messages", 80)),
        )
        self.turns = TurnRunner(self.runtime, self.sessions, max_workers=config.max_concurrent_turns)
        self.channels = ChannelManager(config)
        self._webhooks: ChannelWebhookServer | None = None
        self._gateway = WebSocketGateway(config.ws_host, config.ws_port, auth_token=config.auth_token or None)
        self._socket = SocketChannel(config.socket_path)
        self._shutdown = False
        self._current_session_key = "tui:default"
        self._current_mode = "code"
        self._current_plan_mode = False
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._agent_event_callback: Any = None
        # When a slash command needs an argument it couldn't parse from the
        # original input, it parks itself here as ``(command_name, session_key)``
        # and the next plain user message is consumed as the missing argument
        # rather than dispatched as a chat turn. ``/cancel`` clears the parked
        # state.
        self._pending_slash_arg: tuple[str, str] | None = None
        # Multi-step ``/skill-create`` interview state. Each entry from
        # ``_SKILL_CREATE_STEPS`` is asked one at a time; once every key is
        # filled the synthesized draft turn is submitted to the agent.
        self._pending_skill_create: dict[str, Any] | None = None
        # Active ``/provider`` interactive panel state (``main``/``add``/
        # ``edit``/``remove`` step). Resolved by ``question_response``.
        self._provider_flow: dict[str, Any] | None = None

        # Hook ``AskUserQuestionTool`` to the live TurnRunner so a tool
        # call from inside any agent turn lights up the TUI's interactive
        # question panel instead of dumping the question text as
        # non-interactive fallback. The callback runs on the tool's worker
        # thread; the TUI's eventual ``question_response`` RPC unblocks
        # it via :meth:`TurnRunner.respond_question`.
        from ..tools.claude_tools import set_ask_user_question_callback

        set_ask_user_question_callback(self.turns.ask_user_question)

    async def run(self) -> None:
        """Start every transport and block until :meth:`shutdown` is invoked.

        Performs ``chdir`` into ``project_dir``, reloads the runtime, hooks
        signal handlers, starts the gateway / socket / channels (and webhook
        server when any channel exposes ``handle_webhook``), then idles on a
        polling sleep until ``_shutdown`` is set.
        """
        if self.config.project_dir:
            os.chdir(Path(self.config.project_dir).expanduser())
        try:
            self.runtime.reload()
        except Exception as exc:
            self.logger.error(str(exc))
            raise SystemExit(1) from exc

        self.channels.load()
        self._write_pid()

        loop = asyncio.get_running_loop()

        def emit_daemon_event(event_type: str, payload: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(self._broadcast_event, event_type, payload)

        self.turns.set_event_sink(emit_daemon_event)
        self._agent_event_callback = self.turns.handle_agent_event
        set_event_callback(self._agent_event_callback)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        await self._gateway.start(self._handle_rpc)
        await self._socket.start(self._handle_rpc)
        await self.channels.start_all(self._handle_channel_message)
        if self._has_enabled_webhook_channels():
            self._webhooks = ChannelWebhookServer(
                self.channels,
                host=str(self.config.control.get("webhook_host", self.config.ws_host)),
                port=int(self.config.control.get("webhook_port", 11997)),
            )
            await self._webhooks.start()

        self.logger.info(
            "Daemon running",
            ws=f"ws://{self.config.ws_host}:{self.config.ws_port}",
            socket=self.config.socket_path,
            model=self.runtime.model,
        )
        while not self._shutdown:
            await asyncio.sleep(0.25)

    async def shutdown(self) -> None:
        """Stop every transport, cancel sessions and background tasks, remove the pid file."""
        if self._shutdown:
            return
        self._shutdown = True
        if self._agent_event_callback is not None and get_event_callback() is self._agent_event_callback:
            set_event_callback(None)
        self._agent_event_callback = None
        self.turns.set_event_sink(None)
        self.sessions.cancel_all()
        if self._webhooks is not None:
            await self._webhooks.stop()
        await self.channels.stop_all()
        await self._socket.stop()
        await self._gateway.stop()
        self.turns.close()
        current = asyncio.current_task()
        for task in list(self._background_tasks):
            if task is not current:
                task.cancel()
        self._remove_pid()
        self.logger.close()

    async def _handle_rpc(self, method: str, params: dict[str, Any], emit: EmitFn) -> dict[str, Any]:
        """Dispatch one JSON-RPC request from a websocket or socket client.

        Recognised methods cover initialization, prompt submission, session and
        turn management, runtime status/reload, channel toggles, slash commands,
        permission and question responses, provider profile CRUD, and shutdown.
        Legacy ``task.*`` and bare ``submit``/``list``/``status`` calls return
        :data:`MIGRATED_ERROR`. ``emit`` is the per-connection event sender so
        handlers can stream side-effects back to the originating client.
        """
        if method in {"task.submit", "task.cancel", "task.list", "task.status", "submit", "list", "status"}:
            return {"ok": False, "error": MIGRATED_ERROR}

        if method == "initialize":
            return await self._initialize(params, emit)
        if method == "prompt":
            return await self._submit_turn(
                {
                    "session_key": self._current_session_key,
                    "text": params.get("user_input", ""),
                    "mode": params.get("mode", self._current_mode),
                    "plan_mode": params.get("plan_mode", self._current_plan_mode),
                },
                emit,
            )
        if method == "session.open":
            session = self.sessions.open(
                str(params.get("session_key") or params.get("key") or "default"),
                str(params.get("agent_id") or self.workspaces.default_agent_id),
            )
            return {"ok": True, "session": session.status()}
        if method == "session.list":
            return {"ok": True, "sessions": self.sessions.list()}
        if method == "session.status":
            session = self.sessions.get(str(params.get("session_key") or self._current_session_key))
            return {"ok": bool(session), "session": session.status() if session else None}
        if method == "turn.submit":
            return await self._submit_turn(params, emit)
        if method in {"turn.cancel", "cancel"}:
            return {"ok": self.sessions.cancel(str(params.get("session_key") or self._current_session_key))}
        if method == "cancel_all":
            return {"ok": True, "cancelled": self.sessions.cancel_all()}
        if method == "turn.steer" or method == "steer":
            await emit("steer_input", {"content": str(params.get("content", ""))})
            return {"ok": True}
        if method == "runtime.status":
            return {
                "ok": True,
                **self.runtime.status(),
                "pid": os.getpid(),
                "daemon_protocol": DAEMON_PROTOCOL_VERSION,
                "channels": self.channels.list(),
            }
        if method == "runtime.reload":
            self.runtime.reload(params)
            await self._emit_status(emit)
            return {"ok": True, **self.runtime.status()}
        if method == "channel.list":
            return {"ok": True, "channels": self.channels.list()}
        if method == "channel.enable":
            return {"ok": self.channels.enable(str(params.get("name", "")))}
        if method == "channel.disable":
            return {"ok": self.channels.disable(str(params.get("name", "")))}
        if method == "slash":
            await self._handle_slash(str(params.get("command", "")), emit)
            return {"ok": True}
        if method == "set_plan_mode":
            self._current_plan_mode = bool(params.get("enabled", params.get("plan_mode", False)))
            await self._emit_status(emit)
            return {"ok": True}
        if method == "set_mode":
            self._current_mode = str(params.get("mode", self._current_mode) or self._current_mode)
            self._current_plan_mode = self._current_mode == "plan"
            await self._emit_status(emit)
            return {"ok": True}
        if method == "permission_response":
            return {
                "ok": await self.turns.respond_permission(
                    str(params.get("request_id", "")),
                    str(params.get("response", "reject")),
                )
            }
        if method == "question_response":
            rid = str(params.get("request_id", ""))
            answers = dict(params.get("answers", {}) or {})
            flow = self._provider_flow
            if flow is not None and flow.get("request_id") == rid:
                await self._advance_provider_flow(answers, emit)
            else:
                # Fall through to the general-purpose waiter used by the
                # ``AskUserQuestionTool`` callback. ``False`` means the
                # request id was unknown (e.g. the turn was already
                # cancelled) — fine, drop silently.
                await self.turns.respond_question(rid, answers)
            return {"ok": True}
        if method == "fetch_models":
            return self._fetch_models(params)
        if method == "provider_save":
            profile = profiles.save_profile(
                str(params.get("name", "")),
                str(params.get("base_url", "")),
                str(params.get("api_key", "")),
                str(params.get("model", "")),
                str(params.get("provider", "")),
            )
            self.runtime.reload()
            await self._emit_init_done(emit)
            return {"ok": True, "profile": profile}
        if method == "provider_list":
            return {"ok": True, "profiles": profiles.list_profiles()}
        if method == "provider_select":
            ok = profiles.set_active(str(params.get("name", "")))
            if ok:
                self.runtime.reload()
                await self._emit_init_done(emit)
            return {"ok": ok}
        if method == "provider_delete":
            removed = profiles.delete_profile(str(params.get("name", "")))
            if removed:
                # ``delete_profile`` may unset the active profile; reload so
                # the runtime picks up whichever profile is now active (or
                # falls back to env vars / nothing).
                try:
                    self.runtime.reload()
                except Exception:
                    pass
                await self._emit_init_done(emit)
            return {"ok": removed}
        if method == "shutdown":
            self._track_task(self.shutdown())
            return {"ok": True}
        return {"ok": False, "error": f"Unknown method: {method}"}

    async def _initialize(self, params: dict[str, Any], emit: EmitFn) -> dict[str, Any]:
        """Handle ``initialize`` — bind the client to a session, emit ``init_done``.

        Applies any ``model``/``base_url``/``api_key``/``permission_mode``
        overrides, opens (or rehydrates) the session named by
        ``resume_session_id``, emits the ``init_done`` and a ``status_update``
        event, and replays prior turns as inline scrollback notifications.
        """
        overrides = {
            key: params.get(key) for key in ("model", "base_url", "api_key", "permission_mode") if params.get(key)
        }
        if overrides:
            self.runtime.reload(overrides)
        resume_id = str(params.get("resume_session_id") or "").strip()
        self._current_session_key = resume_id or "tui:default"
        # Daemon is long-lived; the in-memory ``SessionManager._sessions``
        # slot for ``tui:default`` outlives any single TUI connection.
        # When the user launches ``xerxes`` *without* ``-r`` they expect a
        # fresh chat — but the cached slot would otherwise hand back the
        # previous session's messages. Evict the slot first so ``open``
        # synthesises a new one.
        if not resume_id:
            self.sessions.evict("tui:default")
        session = self.sessions.open(self._current_session_key, self.workspaces.default_agent_id)
        await emit(
            "init_done",
            {
                "model": self.runtime.model,
                "session_id": session.id,
                "cwd": str(Path.cwd()),
                "git_branch": self._git_branch(),
                "context_limit": self._resolve_context_limit(),
                "agent_name": session.agent_id,
                "skills": self.runtime.discover_skills(),
            },
        )
        await self._emit_status(emit)
        # Replay prior turns so the user actually SEES their resumed history.
        # The session manager loads messages into state on open(), but without
        # this replay the TUI scrollback stays blank.
        if session.state.messages:
            await self._replay_session_history(session, emit)
        return {
            "ok": True,
            "session": session.status(),
            "daemon_protocol": DAEMON_PROTOCOL_VERSION,
            **self.runtime.status(),
        }

    async def _submit_turn(self, params: dict[str, Any], emit: EmitFn) -> dict[str, Any]:
        """Queue a new turn on the resolved session and stream events to ``emit``."""
        # Don't strip yet — empty/whitespace input is a *valid* answer for the
        # optional ``pitfalls`` step of the /skill-create interview ("Press
        # Enter to skip"). We re-strip per-branch below.
        raw_text = str(params.get("text") or params.get("prompt") or params.get("user_input") or "")
        text = raw_text.strip()
        session_key = str(params.get("session_key") or self._current_session_key)

        # Skip interception for synthetic prompts the daemon itself submits
        # (otherwise we'd consume our own draft request as an interview answer).
        is_internal = params.get("_internal_slash", False)

        async def _finish_intercepted(consumed_for: str, *, cancelled: bool = False) -> dict[str, Any]:
            # The TUI cleared its turn_done_event when it called ``prompt`` and
            # now waits on it. Without a matching ``turn_end`` the spinner sits
            # forever and subsequent inputs get queued. Emit the bookended pair
            # so the TUI re-enters the idle state.
            await emit("turn_begin", {})
            await emit("turn_end", {})
            return {"ok": True, "consumed_for": consumed_for, "cancelled": cancelled}

        if not is_internal:
            # /skill-create scope interview — multi-step, owns its own state.
            if self._pending_skill_create is not None and self._pending_skill_create["session_key"] == session_key:
                if text.lower() in {"/cancel", "cancel"}:
                    self._pending_skill_create = None
                    await self._emit_slash(emit, "Cancelled `/skill-create`.")
                    return await _finish_intercepted("skill-create", cancelled=True)
                # Pass the raw text — ``_advance_skill_create`` distinguishes
                # required vs optional fields itself (e.g. ``pitfalls`` accepts
                # empty input, so the daemon-level "not text" early-bail
                # below would have eaten the user's "Enter to skip").
                await self._advance_skill_create(raw_text, emit)
                return await _finish_intercepted("skill-create")

            # Generic single-arg slash continuation (e.g. the slug prompt).
            if self._pending_slash_arg is not None:
                parked_cmd, parked_session = self._pending_slash_arg
                if parked_session == session_key:
                    self._pending_slash_arg = None
                    if text.lower() in {"/cancel", "cancel"}:
                        await self._emit_slash(emit, f"Cancelled `/{parked_cmd}`.")
                        return await _finish_intercepted(parked_cmd, cancelled=True)
                    # For /skill-create specifically, the parked argument is the
                    # slug — pivot directly into the scope interview rather than
                    # re-dispatching (which would just re-park for the same slug).
                    if parked_cmd == "skill-create":
                        safe_name = "".join(ch for ch in text.lower() if ch.isalnum() or ch in {"-", "_"}).strip("-_")
                        if not safe_name:
                            await self._emit_slash(
                                emit,
                                "That doesn't look like a valid slug. "
                                "Use kebab-case letters/digits, e.g. `commit-helper`. `/cancel` to abort.",
                            )
                            self._pending_slash_arg = ("skill-create", session_key)
                            return await _finish_intercepted("skill-create")
                        await self._start_skill_create_interview(safe_name, emit)
                        return await _finish_intercepted("skill-create")
                    # Otherwise re-dispatch the parked command with the answer.
                    await self._handle_slash(f"/{parked_cmd} {text}", emit)
                    return await _finish_intercepted(parked_cmd)

        # No active intercept — a normal chat turn requires non-empty text.
        if not text:
            return {"ok": False, "error": "Empty prompt"}
        agent_id = str(params.get("agent_id") or self.workspaces.default_agent_id)
        mode = str(params.get("mode") or self._current_mode or "code")
        plan_mode = bool(params.get("plan_mode", self._current_plan_mode or mode == "plan"))
        self._current_session_key = session_key
        self._current_mode = mode
        self._current_plan_mode = plan_mode
        session = self.sessions.open(session_key, agent_id)
        turn_task = self._track_task(self.turns.run_turn(session, text, emit, mode=mode, plan_mode=plan_mode))
        return {"ok": True, "session": session.status(), "turn_task": turn_task}

    async def _handle_channel_message(self, message: ChannelMessage) -> None:
        """Run a one-shot turn for an inbound channel message and reply on the same channel."""
        if not message.text.strip():
            return
        session_key = self._channel_session_key(message)
        session = self.sessions.open(session_key, self.workspaces.default_agent_id)
        output_parts: list[str] = []

        async def emit(event_type: str, payload: dict[str, Any]) -> None:
            if event_type == "text_part":
                output_parts.append(str(payload.get("text", "")))

        await self.turns.run_turn(session, self._format_channel_prompt(message), emit, mode="code", plan_mode=False)
        channel = self.channels.channels.get(message.channel)
        if channel and channel.instance:
            await channel.instance.send(
                ChannelMessage(
                    text="".join(output_parts).strip() or "(no response)",
                    channel=message.channel,
                    channel_user_id=message.channel_user_id,
                    room_id=message.room_id,
                    reply_to=message.platform_message_id,
                    direction=MessageDirection.OUTBOUND,
                )
            )

    async def _emit_status(self, emit: EmitFn) -> None:
        """Push a ``status_update`` reflecting the current session's token usage and mode."""
        session = self.sessions.open(self._current_session_key, self.workspaces.default_agent_id)
        await emit(
            "status_update",
            {
                "context_tokens": session.state.total_input_tokens + session.state.total_output_tokens,
                "max_context": self._resolve_context_limit(),
                "mcp_status": {},
                "plan_mode": self._current_plan_mode,
                "mode": self._current_mode,
            },
        )

    @staticmethod
    def _fetch_models(params: dict[str, Any]) -> dict[str, Any]:
        """Look up the active provider's models, falling back to the active profile."""
        base_url = str(params.get("base_url", ""))
        api_key = str(params.get("api_key", ""))
        if not base_url:
            profile = profiles.get_active_profile()
            if profile:
                base_url = profile.get("base_url", "")
                api_key = api_key or profile.get("api_key", "")
        try:
            models = (
                [{"id": model, "name": model} for model in profiles.fetch_models(base_url, api_key)] if base_url else []
            )
            return {"ok": True, "models": models}
        except Exception as exc:
            return {"ok": False, "models": [], "error": str(exc)}

    @staticmethod
    def _channel_session_key(message: ChannelMessage) -> str:
        """Derive a stable per-conversation session key from a channel message.

        Group/supergroup/channel chats are scoped by room + thread; private
        chats by per-user id so each DM gets its own session.
        """
        meta = message.metadata or {}
        chat_type = str(meta.get("chat_type", "")).lower()
        thread_id = str(meta.get("thread_id", "") or "main")
        if chat_type in {"group", "supergroup", "channel"}:
            return f"{message.channel}:chat:{message.room_id or ''}:thread:{thread_id}"
        return f"{message.channel}:private:{message.channel_user_id or message.room_id or ''}"

    @staticmethod
    def _format_channel_prompt(message: ChannelMessage) -> str:
        """Render a channel message as a prompt with origin metadata."""
        meta = message.metadata or {}
        return (
            f"[{message.channel} message]\n"
            f"room_id: {message.room_id or ''}\n"
            f"from_user_id: {message.channel_user_id or ''}\n"
            f"thread_id: {meta.get('thread_id', '')}\n\n"
            f"{message.text}"
        )

    @staticmethod
    def _git_branch() -> str:
        """Return the current git branch name, or empty string if unavailable."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=Path.cwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1,
                check=False,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _write_pid(self) -> None:
        """Write the current PID to ``config.pid_file``."""
        pid_path = Path(self.config.pid_file).expanduser()
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def _remove_pid(self) -> None:
        """Delete the PID file if it exists."""
        Path(self.config.pid_file).expanduser().unlink(missing_ok=True)

    def _track_task(self, coro: Any) -> asyncio.Task[Any]:
        """Schedule a coroutine and keep a reference so the GC won't drop it.

        Returns the created task so callers can chain ``add_done_callback``
        or ``await`` it directly when they need to know when the underlying
        work finishes.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _has_enabled_webhook_channels(self) -> bool:
        """True when at least one enabled channel implements ``handle_webhook``."""
        for channel in self.channels.channels.values():
            if channel.enabled and channel.instance is not None and hasattr(channel.instance, "handle_webhook"):
                return True
        return False

    def _broadcast_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Fan an event out to every connected socket and websocket client."""
        self._socket.broadcast(event_type, payload)
        self._gateway.broadcast(event_type, payload)

    def _on_background_task_done(self, task: asyncio.Task[Any]) -> None:
        """Drop completed tasks and log unexpected exceptions."""
        self._background_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.logger.error("Background task failed", error=str(exc))

    async def _handle_slash(self, command: str, emit: EmitFn) -> None:
        """Dispatch a ``/cmd`` string to plugins, built-ins, or skill shorthand.

        Resolution order: slash-plugin registry, built-in commands (``/help``,
        ``/skills``, ``/yolo``, ``/permissions``, ``/context``, ...), then the
        skill registry treating ``cmd`` (or ``cmd:sub``) as a skill name.
        """
        stripped = command.strip()
        if not stripped.startswith("/"):
            return
        raw = stripped[1:]
        cmd, _, args = raw.partition(" ")
        cmd = cmd.strip().lower()
        args = args.strip()

        # Resolve aliases against the canonical registry so /q, /quit, /reset, etc. all work.
        from ..bridge.commands import resolve_command
        from ..extensions.slash_plugins import resolve_slash

        original_cmd = cmd  # preserve typed name for user-facing responses
        resolved = resolve_command(cmd)
        if resolved is not None and resolved.name != cmd:
            cmd = resolved.name

        # Plugin-registered slash commands.
        plugin = resolve_slash(cmd)
        if plugin is not None:
            try:
                out = plugin.handler(args) if args else plugin.handler()
            except Exception as exc:
                await self._emit_slash(emit, f"Plugin /{cmd} failed: {exc}")
                return
            await self._emit_slash(emit, str(out) if out is not None else f"Plugin /{cmd} ran.")
            return

        if cmd == "skills":
            await self._emit_slash(emit, self.runtime.skills_list_text())
            await emit("init_done", {"skills": self.runtime.discover_skills()})
            return

        if cmd == "skill":
            await self._slash_skill(args, emit)
            return

        if cmd == "skill-create":
            await self._slash_skill_create(args, emit)
            return

        # ---- permission / mode toggles -----------------------------------

        if cmd == "yolo":
            new_mode = self.runtime.toggle_yolo()
            label = "ON (accept-all)" if new_mode == "accept-all" else "OFF (auto)"
            await self._emit_slash(emit, f"YOLO mode {label}")
            return

        if cmd == "permissions":
            if args:
                new_mode = self.runtime.set_permission_mode(args)
                await self._emit_slash(emit, f"Permission mode: {new_mode}")
            else:
                # Cycle through valid modes (matches bridge/server behavior).
                modes = ["auto", "accept-all", "manual"]
                current = self.runtime.permission_mode
                idx = modes.index(current) if current in modes else 0
                new_mode = self.runtime.set_permission_mode(modes[(idx + 1) % len(modes)])
                await self._emit_slash(emit, f"Permission mode: {new_mode}")
            return

        # ``/thinking`` is the alias for canonical ``/reasoning``; accept either.
        if cmd in {"thinking", "reasoning", "verbose", "debug"}:
            # Persist under canonical key, but echo whatever the user typed.
            new_value = self.runtime.toggle_flag(cmd)
            await self._emit_slash(emit, f"{original_cmd.title()}: {new_value}")
            return

        # ---- info commands ------------------------------------------------

        if cmd == "help":
            await self._emit_slash(emit, self._help_text())
            return

        if cmd == "commands":
            from ..bridge.commands import COMMAND_REGISTRY

            lines = [f"  /{c.name:<14} {c.description}" for c in COMMAND_REGISTRY]
            await self._emit_slash(emit, "Commands:\n" + "\n".join(lines))
            return

        if cmd == "context":
            status = self.runtime.status()
            await self._emit_slash(
                emit,
                "\n".join(
                    [
                        f"Model:           {status['model']}",
                        f"Permission mode: {status['permission_mode']}",
                        f"Tools:           {status['tools']}",
                        f"Skills:          {status['skills']}",
                        f"Workspace:       {self.config.project_dir or os.getcwd()}",
                    ]
                ),
            )
            return

        if cmd == "status":
            await self._emit_slash(emit, json.dumps(self.runtime.status(), indent=2))
            return

        if cmd == "provider":
            await self._slash_provider(args, emit)
            return

        if cmd in {"exit", "quit", "q"}:
            await self._emit_slash(emit, "(use Ctrl+D or close the terminal to exit)")
            return

        if cmd == "clear":
            # Clear is a per-session concern; the TUI handles its own scrollback,
            # so we just acknowledge.
            await self._emit_slash(emit, "Cleared.")
            return

        # ---- omnibus handler for the rest of COMMAND_REGISTRY ----------------
        # Dispatched via ``_BULK_SLASH_HANDLERS`` so adding a command means one
        # line in the registry + one line in the dispatch table below.
        handler = _BULK_SLASH_HANDLERS.get(cmd)
        if handler is not None:
            await handler(self, args, emit)
            return

        # ---- skill-name shorthand (supports ``/skill:sub``) --------------

        # ``/autoresearch:fix`` should resolve to the autoresearch skill with
        # the ``fix`` subcommand. We split before looking up the skill so a
        # skill named "autoresearch" handles "fix" as a sub-command, not the
        # whole token.
        skill_name = cmd
        subcommand = ""
        if ":" in cmd:
            skill_name, subcommand = cmd.split(":", 1)

        skill = self.runtime.skill_registry.get(skill_name)
        if skill is not None:
            # Validate subcommand against the skill's declared sub-commands.
            declared_subs = skill.metadata.subcommands
            if subcommand and declared_subs and subcommand not in declared_subs:
                await self._emit_slash(
                    emit,
                    f"Skill /{skill_name} has no sub-command '{subcommand}'. "
                    f"Available: {', '.join('/' + skill_name + ':' + s for s in declared_subs)}",
                )
                return
            # Build the args string the existing dispatcher understands:
            #   "<name>"            — no subcommand, no args
            #   "<name>:<sub>"      — subcommand only
            #   "<name>:<sub> <a>"  — subcommand + free-form args
            if subcommand and args:
                composed = f"{skill_name}:{subcommand} {args}"
            elif subcommand:
                composed = f"{skill_name}:{subcommand}"
            elif args:
                composed = f"{skill_name}:{args}"
            else:
                composed = skill_name
            await self._slash_skill(composed, emit, run_now=True)
            return

        # If the command is in the registry but slipped through every branch
        # above, fail honestly so the gap can be caught by a grep rather than
        # disguised as "unknown command".
        if resolved is not None:
            await self._emit_slash(
                emit,
                f"`/{cmd}` is registered but not yet wired in the daemon. "
                "Open an issue with the command name and what you expected.",
            )
            return
        await self._emit_slash(emit, f"Unknown command: /{cmd} (type /help)")

    def _help_text(self) -> str:
        """Render the ``/help`` output grouped by command category."""
        from ..bridge.commands import CATEGORIES, COMMAND_REGISTRY

        # Group by category for readable help output.
        by_cat: dict[str, list[Any]] = {c: [] for c in CATEGORIES}
        for cmd in COMMAND_REGISTRY:
            by_cat.setdefault(cmd.category, []).append(cmd)
        lines = ["Slash commands (run `/commands` for the flat list):"]
        for cat in CATEGORIES:
            cmds = by_cat.get(cat, [])
            if not cmds:
                continue
            lines.append(f"\n  [{cat}]")
            for c in cmds:
                # Wrap each ``/cmd <hint>`` in a backtick code span so the
                # Markdown renderer the TUI applies to slash notifications
                # doesn't silently strip angle-bracketed placeholders as
                # HTML tags.
                hint = f" {c.args_hint}" if c.args_hint else ""
                lines.append(f"    `/{c.name}{hint}`  — {c.description}")
        return "\n".join(lines)

    async def _slash_provider(self, args: str, emit: EmitFn) -> None:
        """Open the interactive provider panel (or quick-switch via ``/provider <name>``).

        Forms:
          * ``/provider``        — pops a TUI question panel listing every
            profile plus ``Add``/``Edit``/``Remove``/``Cancel`` actions.
          * ``/provider <name>`` — quick switch by name (no panel).

        The panel is built on top of the standard ``question_request`` wire
        event, so the same arrow-key + Enter UX as agent ``ask_user`` prompts
        applies. Multi-step actions (add/edit/remove) are batched into a
        single follow-up question_request with one entry per field.
        """
        from ..bridge import profiles

        target = args.strip()
        plist = profiles.list_profiles()

        # Inline quick-switch — bypass the panel entirely.
        if target:
            if not any(p.get("name") == target for p in plist):
                names = ", ".join(f"`{p.get('name', '')}`" for p in plist if p.get("name"))
                msg = f"No profile named `{target}`."
                if names:
                    msg += f" Available: {names}."
                await self._emit_slash(emit, msg)
                return
            if not profiles.set_active(target):
                await self._emit_slash(emit, f"Failed to switch to `{target}`.")
                return
            try:
                self.runtime.reload({})
            except Exception:
                pass
            switched = profiles.get_active_profile() or {}
            await self._emit_slash(
                emit,
                f"Switched to `{target}` (model: `{switched.get('model', '?')}`).",
            )
            await self._emit_init_done(emit)
            return

        # No profiles + bare /provider — fall back to env-var guidance instead
        # of popping an empty panel.
        if not plist:
            await self._emit_slash(
                emit,
                "No provider profiles configured.\n"
                "Set `XERXES_BASE_URL`, `XERXES_API_KEY`, and `XERXES_MODEL` in the "
                "environment to start, or run `/provider` again after adding one via "
                "the `provider_save` JSON-RPC method.\n"
                "(An add-profile flow inside the panel is also available — type "
                "`/provider` once at least one profile exists.)",
            )
            return

        await self._emit_provider_main_panel(emit)

    async def _emit_provider_main_panel(self, emit: EmitFn) -> None:
        """Send the top-level provider action question to the TUI."""
        from ..bridge import profiles

        plist = profiles.list_profiles()
        active_name = (profiles.get_active_profile() or {}).get("name", "")

        options: list[str] = []
        # Existing profiles come first — selecting one switches to it.
        for p in plist:
            name = p.get("name", "")
            marker = "  ← active" if name == active_name else ""
            model = p.get("model", "?")
            base = p.get("base_url", "")
            options.append(f"{name}  ({model} @ {base}){marker}")
        options.append(_PROVIDER_ADD_LABEL)
        if plist:
            options.append(_PROVIDER_EDIT_LABEL)
            options.append(_PROVIDER_REMOVE_LABEL)
        options.append(_PROVIDER_CANCEL_LABEL)

        rid = uuid.uuid4().hex
        self._provider_flow = {"step": "main", "request_id": rid}
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "action",
                        "question": "Provider profiles — pick a profile to switch, or choose an action:",
                        "options": options,
                        "allow_free_form": False,
                    }
                ],
            },
        )

    async def _emit_provider_add_panel(self, emit: EmitFn) -> None:
        """Stage 1 of the Add flow — ask for name + provider type.

        Stage 2 (``_emit_provider_credentials_panel``) follows once the
        provider type is known so its base URL and default model can be
        pre-filled from :data:`xerxes.llms.registry.PROVIDERS`. Without that
        split the user would have to retype URLs we already know.
        """
        rid = uuid.uuid4().hex
        self._provider_flow = {"step": "add_meta", "request_id": rid}
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "name",
                        "question": "New profile name (short slug, e.g. `kimi-code` or `openai-prod`):",
                        "options": [],
                        "allow_free_form": True,
                    },
                    {
                        "id": "provider_type",
                        "question": "Inference provider type:",
                        "options": list(_PROVIDER_TYPE_OPTIONS),
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _emit_provider_credentials_panel(self, emit: EmitFn, name: str, provider_type: str) -> None:
        """Stage 2 of the Add flow — ask base URL + API key.

        Pre-fills the base URL from :data:`xerxes.llms.registry.PROVIDERS` so
        users picking a well-known provider can usually just hit Enter and
        paste an API key. The model is collected separately in stage 3 after
        we try to enumerate ``/models`` from the actual endpoint.
        """
        from ..llms.registry import PROVIDERS

        default_url = ""
        default_model = ""
        if provider_type and provider_type not in {"auto", "custom"}:
            prov_cfg = PROVIDERS.get(provider_type)
            if prov_cfg is not None:
                # ``base_url`` is optional on the dataclass (anthropic and a
                # few others rely on the SDK default). Treat ``None`` as "no
                # suggestion" rather than letting it crash the f-string.
                default_url = prov_cfg.base_url or ""
                default_model = prov_cfg.models[0] if prov_cfg.models else ""

        url_question = "Base URL"
        if default_url:
            url_question += f" (press Enter for `{default_url}`)"
        url_question += ":"

        rid = uuid.uuid4().hex
        self._provider_flow = {
            "step": "add_creds",
            "request_id": rid,
            "name": name,
            "provider_type": provider_type,
            "default_url": default_url,
            "default_model": default_model,
        }
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "base_url",
                        "question": url_question,
                        "options": [],
                        "allow_free_form": True,
                    },
                    {
                        "id": "api_key",
                        "question": "API key (blank uses env var when available):",
                        "options": [],
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _emit_provider_model_panel(
        self,
        emit: EmitFn,
        *,
        name: str,
        provider_type: str,
        base_url: str,
        api_key: str,
        default_model: str,
    ) -> None:
        """Stage 3 of the Add flow — try ``GET /models``, then ask the user.

        Network call runs on a worker thread (so the daemon's event loop
        stays responsive) with a hard 3s timeout. Every failure path falls
        back to a free-form question with the registry default as a hint —
        the Add flow never gets stuck because of a flaky provider.
        """
        # Sentinel that lets the user open the free-text mode even when a
        # list of options is offered. Detected in ``_advance_provider_flow``.
        type_sentinel = "— Type a custom model id —"

        # Pull the model catalogue. ``profiles.fetch_models`` is synchronous
        # httpx; offload to a thread so we don't block other RPC handlers.
        models: list[str] = []
        fetch_error: str = ""
        try:
            models = await asyncio.to_thread(profiles.fetch_models, base_url, api_key)
        except Exception as exc:
            fetch_error = str(exc)
        # Order: registry default first (if it's in the list), then alphabetical.
        if default_model and default_model in models:
            models = [default_model] + [m for m in models if m != default_model]

        question_text = "Pick a model"
        if fetch_error:
            question_text += " (couldn't reach `/models` — type one manually)"
        elif not models:
            question_text += " (the endpoint returned no catalogue — type one)"
        elif default_model:
            question_text += f" (first option = registry default `{default_model}`)"
        question_text += ":"

        rid = uuid.uuid4().hex
        self._provider_flow = {
            "step": "add_model",
            "request_id": rid,
            "name": name,
            "provider_type": provider_type,
            "base_url": base_url,
            "api_key": api_key,
            "default_model": default_model,
            "type_sentinel": type_sentinel,
        }
        # Always allow free-form so users can paste a model id even when the
        # provider returns a list. Append the sentinel after real options so
        # picking it triggers the custom-text follow-up.
        options = list(models)
        if options:
            options.append(type_sentinel)
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "model",
                        "question": question_text,
                        "options": options,
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _finalize_provider_add(
        self,
        emit: EmitFn,
        name: str,
        provider_type: str,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        """Save the new profile, reload the runtime, and confirm to the user.

        Called from every terminal branch of the Add flow (model picker,
        free-text fallback). Handles the same provider-tag resolution and
        ``init_done`` refresh the old single-shot Add path used to do.
        """
        self._provider_flow = None
        saved_type = "" if provider_type == "auto" else provider_type
        if not model:
            await self._emit_slash(
                emit,
                f"Add cancelled — no model id given and `{provider_type}` has no default.",
            )
            return
        try:
            profiles.save_profile(name, base_url, api_key, model, saved_type)
        except Exception as exc:
            await self._emit_slash(emit, f"Failed to save profile: `{exc}`")
            return
        try:
            self.runtime.reload({})
        except Exception:
            pass
        saved = next(
            (p for p in profiles.list_profiles() if p.get("name") == name),
            {},
        )
        resolved_type = saved.get("provider") or saved_type or "custom"
        await self._emit_slash(
            emit,
            f"Added profile `{name}` (type `{resolved_type}`, model `{model}` @ `{base_url}`) and switched to it.",
        )
        await self._emit_init_done(emit)

    async def _emit_provider_custom_model_panel(
        self, emit: EmitFn, name: str, provider_type: str, base_url: str, api_key: str, default_model: str
    ) -> None:
        """Fallback panel asking for a free-text model id (used after the user
        picks the "type custom" sentinel on the model picker)."""
        rid = uuid.uuid4().hex
        self._provider_flow = {
            "step": "add_model_text",
            "request_id": rid,
            "name": name,
            "provider_type": provider_type,
            "base_url": base_url,
            "api_key": api_key,
            "default_model": default_model,
        }
        question_text = "Model id"
        if default_model:
            question_text += f" (press Enter for `{default_model}`)"
        question_text += " — e.g. `gpt-4o`, `kimi-for-coding`:"
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "model",
                        "question": question_text,
                        "options": [],
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _emit_provider_edit_panel(self, emit: EmitFn) -> None:
        """Ask which profile + field to edit + the new value (batched)."""
        from ..bridge import profiles

        names = [p.get("name", "") for p in profiles.list_profiles() if p.get("name")]
        rid = uuid.uuid4().hex
        self._provider_flow = {"step": "edit", "request_id": rid}
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "profile",
                        "question": "Which profile to edit?",
                        "options": names,
                        "allow_free_form": False,
                    },
                    {
                        "id": "field",
                        "question": "Which field?",
                        "options": ["base_url", "api_key", "model", "name", "provider_type"],
                        "allow_free_form": False,
                    },
                    {
                        "id": "value",
                        "question": "New value:",
                        "options": [],
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _emit_provider_remove_panel(self, emit: EmitFn) -> None:
        """Ask which profile to remove + confirm."""
        from ..bridge import profiles

        names = [p.get("name", "") for p in profiles.list_profiles() if p.get("name")]
        rid = uuid.uuid4().hex
        self._provider_flow = {"step": "remove", "request_id": rid}
        await emit(
            "question_request",
            {
                "id": rid,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "profile",
                        "question": "Which profile to remove?",
                        "options": names,
                        "allow_free_form": False,
                    },
                    {
                        "id": "confirm",
                        "question": "Type `yes` to confirm deletion:",
                        "options": ["yes", "no"],
                        "allow_free_form": True,
                    },
                ],
            },
        )

    async def _advance_provider_flow(self, answers: dict[str, str], emit: EmitFn) -> None:
        """State machine that consumes ``question_response`` answers."""
        from ..bridge import profiles

        flow = self._provider_flow
        if flow is None:
            return
        step = flow.get("step", "")

        # Cancel sentinel everywhere — bail without touching disk.
        for v in answers.values():
            if v == _PROVIDER_CANCEL_LABEL:
                self._provider_flow = None
                await self._emit_slash(emit, "Cancelled.")
                return

        if step == "main":
            choice = answers.get("action", "")
            if choice == _PROVIDER_ADD_LABEL:
                await self._emit_provider_add_panel(emit)
                return
            if choice == _PROVIDER_EDIT_LABEL:
                await self._emit_provider_edit_panel(emit)
                return
            if choice == _PROVIDER_REMOVE_LABEL:
                await self._emit_provider_remove_panel(emit)
                return
            # Otherwise the user picked an existing profile row — pull the
            # name off the front of the rendered label.
            name = choice.split("  ", 1)[0].strip()
            self._provider_flow = None
            if not name:
                await self._emit_slash(emit, "No profile selected.")
                return
            if not profiles.set_active(name):
                await self._emit_slash(emit, f"Failed to switch to `{name}`.")
                return
            try:
                self.runtime.reload({})
            except Exception:
                pass
            switched = profiles.get_active_profile() or {}
            await self._emit_slash(
                emit,
                f"Switched to `{name}` (model: `{switched.get('model', '?')}`).",
            )
            await self._emit_init_done(emit)
            return

        if step == "add_meta":
            # Stage 1 → 2: collect name + provider type, then pop the
            # credentials panel with provider-aware defaults.
            name = (answers.get("name") or "").strip()
            provider_type = (answers.get("provider_type") or "").strip() or "auto"
            if not name:
                self._provider_flow = None
                await self._emit_slash(emit, "Add cancelled — profile name is required.")
                return
            await self._emit_provider_credentials_panel(emit, name, provider_type)
            return

        if step == "add_creds":
            # Stage 2 → 3: capture base_url + api_key, then try to enumerate
            # models from the endpoint and present the result as a picker.
            name = flow.get("name", "")
            provider_type = flow.get("provider_type", "auto")
            default_url = flow.get("default_url", "")
            default_model = flow.get("default_model", "")
            base_url = (answers.get("base_url") or "").strip() or default_url
            api_key = (answers.get("api_key") or "").strip()
            if not base_url:
                self._provider_flow = None
                await self._emit_slash(
                    emit,
                    f"Add cancelled — base_url is required for `{provider_type}` (no registry default to fall back to).",
                )
                return
            await self._emit_provider_model_panel(
                emit,
                name=name,
                provider_type=provider_type,
                base_url=base_url,
                api_key=api_key,
                default_model=default_model,
            )
            return

        if step == "add_model":
            # Stage 3 (picker). The user either picked a real model, picked
            # the "type custom" sentinel, or typed a free-form id.
            picked = (answers.get("model") or "").strip()
            type_sentinel = flow.get("type_sentinel", "")
            if picked == type_sentinel:
                await self._emit_provider_custom_model_panel(
                    emit,
                    flow.get("name", ""),
                    flow.get("provider_type", "auto"),
                    flow.get("base_url", ""),
                    flow.get("api_key", ""),
                    flow.get("default_model", ""),
                )
                return
            # Empty picker answer = accept registry default.
            model = picked or flow.get("default_model", "")
            await self._finalize_provider_add(
                emit,
                flow.get("name", ""),
                flow.get("provider_type", "auto"),
                flow.get("base_url", ""),
                flow.get("api_key", ""),
                model,
            )
            return

        if step == "add_model_text":
            # Free-text fallback (user picked "type custom" or /models failed).
            model = (answers.get("model") or "").strip() or flow.get("default_model", "")
            await self._finalize_provider_add(
                emit,
                flow.get("name", ""),
                flow.get("provider_type", "auto"),
                flow.get("base_url", ""),
                flow.get("api_key", ""),
                model,
            )
            return

        if step == "edit":
            self._provider_flow = None
            target = (answers.get("profile") or "").strip()
            field = (answers.get("field") or "").strip()
            value = (answers.get("value") or "").strip()
            if not target or not field:
                await self._emit_slash(emit, "Edit cancelled — no profile or field selected.")
                return
            existing = next(
                (p for p in profiles.list_profiles() if p.get("name") == target),
                None,
            )
            if existing is None:
                await self._emit_slash(emit, f"No profile named `{target}` (it may have been removed).")
                return
            merged = dict(existing)
            # The user-facing field "provider_type" maps to the underlying
            # "provider" column on the profile dict.
            store_key = "provider" if field == "provider_type" else field
            if field == "name":
                # Rename: save under the new key and remove the old.
                new_name = value
                if not new_name:
                    await self._emit_slash(emit, "Edit cancelled — new name is empty.")
                    return
                profiles.save_profile(
                    new_name,
                    merged.get("base_url", ""),
                    merged.get("api_key", ""),
                    merged.get("model", ""),
                    merged.get("provider", ""),
                )
                profiles.delete_profile(target)
                target = new_name
            else:
                # ``auto`` for provider_type means "let the URL heuristic
                # pick" — store as empty so save_profile triggers the guess.
                stored_value = "" if field == "provider_type" and value == "auto" else value
                merged[store_key] = stored_value
                profiles.save_profile(
                    merged.get("name", ""),
                    merged.get("base_url", ""),
                    merged.get("api_key", ""),
                    merged.get("model", ""),
                    merged.get("provider", ""),
                )
            try:
                self.runtime.reload({})
            except Exception:
                pass
            shown = "***redacted***" if field == "api_key" else value
            await self._emit_slash(emit, f"Updated `{target}`: `{field}` = `{shown}`.")
            await self._emit_init_done(emit)
            return

        if step == "remove":
            self._provider_flow = None
            target = (answers.get("profile") or "").strip()
            confirm = (answers.get("confirm") or "").strip().lower()
            if not target:
                await self._emit_slash(emit, "Remove cancelled.")
                return
            if confirm not in {"yes", "y"}:
                await self._emit_slash(emit, f"Remove cancelled — `{target}` was not deleted.")
                return
            if not profiles.delete_profile(target):
                await self._emit_slash(emit, f"Failed to remove `{target}`.")
                return
            try:
                self.runtime.reload({})
            except Exception:
                pass
            await self._emit_slash(emit, f"Removed profile `{target}`.")
            await self._emit_init_done(emit)
            return

        # Unknown step — defensive reset.
        self._provider_flow = None

    async def _slash_skill(self, args: str, emit: EmitFn, *, run_now: bool = True) -> None:
        """Activate (and optionally invoke) a registered skill by name.

        Handles ``name``, ``name:sub``, and ``name:sub args`` forms. When
        ``run_now`` is true the skill's prompt section is injected into a new
        turn; otherwise the skill is just activated for future turns.
        """
        name = args.strip()
        if not name:
            await self._emit_slash(emit, "Usage: `/skill <name>` — use `/skills` to list available skills.")
            return
        skill_args = ""
        if ":" in name:
            name, skill_args = name.split(":", 1)
            name = name.strip()
            skill_args = skill_args.strip()

        self.runtime.discover_skills()
        skill = self.runtime.skill_registry.get(name)
        if skill is None:
            matches = self.runtime.skill_registry.search(name)
            if matches:
                await self._emit_slash(
                    emit, f"Skill '{name}' not found. Did you mean: {', '.join(s.name for s in matches[:5])}"
                )
            else:
                await self._emit_slash(emit, f"Skill '{name}' not found. Use /skills to list available skills.")
            return
        if not skill_matches_platform(skill):
            await self._emit_slash(emit, f"Skill '{name}' is not compatible with this platform ({sys.platform}).")
            return

        activate_skill(name)
        await emit("init_done", {"skills": self.runtime.discover_skills()})
        if not run_now:
            suffix = f"\nArguments: {skill_args}" if skill_args else ""
            await self._emit_slash(
                emit,
                f"Skill '{name}' enabled for future turns.{suffix}\n"
                "Its instructions will be included in the daemon session prompt.",
            )
            return

        await self._emit_slash(emit, f"Running skill '{name}'...")
        prompt_section = skill.to_prompt_section()
        config_block = inject_skill_config(skill)
        # Disambiguate sub-commands. The original bug: ``/autoresearch:learn``
        # arrived here as ``skill_args="learn"`` and got pasted into the prompt
        # as ``User request: learn`` — so the model saw an ambiguous one-word
        # "learn" instead of recognising the canonical sub-command. We now
        # reconstruct the full slash form when the first token of
        # skill_args matches one of the skill's declared sub-commands.
        declared_subs = skill.metadata.subcommands or []
        subcommand = ""
        free_form = skill_args
        if skill_args:
            first, _, rest = skill_args.partition(" ")
            if first in declared_subs:
                subcommand = first
                free_form = rest.strip()
        if subcommand and free_form:
            trigger = f"/{name}:{subcommand} {free_form}"
        elif subcommand:
            trigger = f"/{name}:{subcommand}"
        elif skill_args:
            trigger = skill_args
        else:
            trigger = f"Execute the '{name}' skill now."
        skill_message = f"[Skill '{name}' activated]{config_block}\n\n{prompt_section}\n\nUser request: {trigger}"
        await self._submit_turn(
            {
                "session_key": self._current_session_key,
                "text": skill_message,
                "mode": self._current_mode,
                "plan_mode": self._current_plan_mode,
            },
            emit,
        )

    async def _slash_skill_create(self, args: str, emit: EmitFn) -> None:
        """Start the multi-step ``/skill-create`` interview.

        If a slug is supplied inline (``/skill-create my-thing``) we skip the
        name prompt and go straight to the scope interview; otherwise we ask
        for the slug first. Each subsequent user message is intercepted in
        :meth:`_submit_turn` and routed to :meth:`_advance_skill_create` until
        every step in :data:`_SKILL_CREATE_STEPS` is filled.
        """
        raw = args.strip().split()[0] if args.strip() else ""
        safe_name = "".join(ch for ch in raw.lower() if ch.isalnum() or ch in {"-", "_"}).strip("-_")

        if not safe_name:
            # No slug yet — park for the name prompt; we'll start the scope
            # interview after the user answers that.
            self._pending_slash_arg = ("skill-create", self._current_session_key)
            await self._emit_slash(
                emit,
                "What should this skill be called? Type a short kebab-case slug "
                "(e.g. `commit-helper`). `/cancel` to abort.",
            )
            return

        # Slug present — pre-create the directory and start the scope interview.
        await self._start_skill_create_interview(safe_name, emit)

    async def _start_skill_create_interview(self, safe_name: str, emit: EmitFn) -> None:
        """Open the scope interview after a slug has been resolved."""
        target_dir = self.runtime.skills_dir / safe_name
        target_dir.mkdir(parents=True, exist_ok=True)
        self._pending_skill_create = {
            "session_key": self._current_session_key,
            "name": safe_name,
            "target_path": str(target_dir / "SKILL.md"),
            "answers": {},
        }
        await self._ask_next_skill_create_question(emit)

    async def _ask_next_skill_create_question(self, emit: EmitFn) -> None:
        """Emit the next unanswered question from ``_SKILL_CREATE_STEPS``."""
        state = self._pending_skill_create
        if state is None:
            return
        for key, question, _required in _SKILL_CREATE_STEPS:
            if key not in state["answers"]:
                await self._emit_slash(emit, question)
                return
        # All four answered — kick off the actual draft.
        await self._launch_skill_draft(emit)

    async def _advance_skill_create(self, text: str, emit: EmitFn) -> None:
        """Record one interview answer; either ask the next or launch the draft.

        Special inputs:

        * ``auto`` (case-insensitive) — fill *this* and *every remaining*
          field with the :data:`_SKILL_CREATE_AUTO` sentinel and immediately
          launch the draft. The synthesised prompt tells the model to fill
          those fields from session context.
        * empty input on a required step — re-prompt (the field is required).
        * empty input on an optional step (pitfalls) — accept as "skip".
        """
        state = self._pending_skill_create
        if state is None:
            return

        if text.strip().lower() in {"auto", "/auto"}:
            # Hand off every still-unanswered field to the model.
            for key, _question, _required in _SKILL_CREATE_STEPS:
                state["answers"].setdefault(key, _SKILL_CREATE_AUTO)
            await self._ask_next_skill_create_question(emit)
            return

        for key, _question, required in _SKILL_CREATE_STEPS:
            if key in state["answers"]:
                continue
            if required and not text.strip():
                await self._emit_slash(
                    emit,
                    "That field is required. Type an answer, `auto` to let me decide, or `/cancel` to abort.",
                )
                return
            state["answers"][key] = text.strip()
            break
        await self._ask_next_skill_create_question(emit)

    async def _launch_skill_draft(self, emit: EmitFn) -> None:
        """Synthesize the draft prompt from the collected answers and submit it.

        Each answer is either a literal user reply, an empty string (which
        only ``pitfalls`` allows), or :data:`_SKILL_CREATE_AUTO` to delegate
        the field to the model. The synthesized prompt renders each kind
        differently so the model knows which fields it must infer.
        """
        state = self._pending_skill_create
        if state is None:
            return
        self._pending_skill_create = None  # consume — even on failure the loop ends

        safe_name = state["name"]
        target_path = state["target_path"]
        answers = state["answers"]

        def render(label: str, key: str, infer_hint: str) -> str:
            value = answers.get(key, "").strip()
            if value == _SKILL_CREATE_AUTO:
                return f"**{label}:** _auto — {infer_hint}_\n\n"
            if not value:
                return ""
            return f"**{label}:** {value}\n\n"

        what_block = render(
            "What the skill should do",
            "what",
            "infer from what we worked on in this session.",
        )
        when_block = render(
            "Activation trigger",
            "when",
            "pick a sensible trigger (e.g. when the user says the skill name, "
            "or when the task description matches the work we just did).",
        )
        tools_block = render(
            "Tools / commands it uses",
            "tools",
            "list the tools we actually invoked this session.",
        )
        pitfalls_value = answers.get("pitfalls", "").strip()
        if pitfalls_value == _SKILL_CREATE_AUTO:
            pitfalls_block = (
                "**Pitfalls:** _auto — list any real issues we hit this session; "
                "omit the `# Pitfalls` section if none occurred._\n\n"
            )
        elif pitfalls_value:
            pitfalls_block = f"**User-reported pitfalls:** {pitfalls_value}\n\n"
        else:
            pitfalls_block = (
                "User reported no pitfalls — omit the `# Pitfalls` section unless "
                "something in this session genuinely went wrong.\n\n"
            )

        synthetic_prompt = (
            f"Write a reusable agent skill called **`{safe_name}`**. "
            f"Do not ask follow-up questions — write the SKILL.md directly. "
            "Any field marked _auto_ below is yours to fill in based on what we "
            "did in this session so far.\n\n"
            "## Inputs\n\n"
            f"{what_block}{when_block}{tools_block}{pitfalls_block}"
            "## Output\n\n"
            f"Write the file to **`{target_path}`** using the Write tool. The file must be "
            "valid Markdown with this exact structure:\n\n"
            "1. YAML frontmatter delimited by `---` lines, containing:\n"
            f"   - `name: {safe_name}` (use this exact slug)\n"
            '   - `description:` (one short line — derived from "what the skill should do")\n'
            "   - `version: 0.1.0`\n"
            "   - `tags: [...]` (short list of topics / domain hints)\n"
            "   - `required_tools: [...]` (tool names from the tools field)\n"
            "2. `# When to use` — based on the activation trigger.\n"
            "3. `# Procedure` — numbered steps grounded in the tool list.\n"
            "4. `# Pitfalls` — only if there were real pitfalls.\n"
            "5. `# Verification` — concrete signals the procedure succeeded.\n\n"
            "After writing, confirm the final path in one short sentence. Do not output the "
            "SKILL.md body in chat; the Write tool is the only delivery channel."
        )

        auto_keys = [k for k in answers if answers[k] == _SKILL_CREATE_AUTO]
        if auto_keys:
            announcement = (
                f"Drafting skill `{safe_name}` — inferring "
                f"{', '.join(auto_keys)} from session context, saving to `{target_path}`…"
            )
        else:
            announcement = f"Drafting skill `{safe_name}` from your answers — saving to `{target_path}`…"
        await self._emit_slash(emit, announcement)
        result = await self._submit_turn({"text": synthetic_prompt, "_internal_slash": True}, emit)

        # After the agent finishes writing the SKILL.md, re-discover skills
        # so the TUI's autocomplete cache picks up the new entry. Without
        # this, ``/<new-skill>`` still won't show in the completer until the
        # user restarts xerxes or runs ``/reload``.
        turn_task = result.get("turn_task") if isinstance(result, dict) else None
        if isinstance(turn_task, asyncio.Task):

            async def _refresh_skills_after_turn() -> None:
                try:
                    await turn_task
                except Exception:
                    # Even if the agent's turn errors mid-way it may still
                    # have written the file before failing — try to refresh
                    # regardless.
                    pass
                try:
                    skills = self.runtime.discover_skills()
                except Exception:
                    return
                try:
                    await emit("init_done", {"skills": skills})
                except Exception:
                    pass

            self._track_task(_refresh_skills_after_turn())

    async def _replay_session_history(self, session: Any, emit: EmitFn) -> None:
        """Stream prior user/assistant turns as ``history`` notifications.

        The TUI renders each emitted ``notification`` with ``category="history"``
        in its scrollback area. Tool replies and auto-injected skill prompts
        are skipped; a terminal ``type="resumed"`` marker tells the TUI the
        live transcript starts after this point.
        """

        count = 0
        for msg in session.state.messages:
            role = (msg.get("role") or "").lower()
            if role not in {"user", "assistant"}:
                continue
            content = msg.get("content")
            text = self._render_message_text(content)
            if not text:
                continue
            # Skill activations are auto-injected — they're not a user turn,
            # they're noise from the model's perspective.
            if role == "user" and self._looks_like_skill_activation(text):
                continue
            body = f"✨ {text}" if role == "user" else text
            await emit(
                "notification",
                {
                    "id": uuid.uuid4().hex[:12],
                    "category": "history",
                    "type": f"replay_{role}",
                    "severity": "info",
                    "title": "",
                    "body": body,
                    "payload": {},
                },
            )
            count += 1
        await emit(
            "notification",
            {
                "id": uuid.uuid4().hex[:12],
                "category": "history",
                "type": "resumed",
                "severity": "info",
                "title": "",
                "body": f"── resumed session {session.id} ({count} message{'s' if count != 1 else ''}) ──",
                "payload": {},
            },
        )

    @staticmethod
    def _render_message_text(content: Any) -> str:
        """Flatten an OpenAI/Anthropic content field (str | list | dict) into plain text."""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(str(p.get("text", "")))
                else:
                    parts.append(str(p))
            return "\n".join(p for p in parts if p).strip()
        if isinstance(content, dict):
            return str(content.get("text", "")).strip()
        return str(content or "").strip()

    @staticmethod
    def _looks_like_skill_activation(text: str) -> bool:
        """True when ``text`` is one of the ``[Skill 'X' activated]`` injections.

        Those headers are emitted by the skill dispatcher when the user types
        ``/skill_name`` — they're not real user input and shouldn't show in
        the replay.
        """
        head = text.lstrip()[:64]
        return head.startswith("[Skill") and "activated" in head

    async def _emit_init_done(self, emit: EmitFn) -> None:
        """Re-emit ``init_done`` so the TUI refreshes its banner / status bar.

        Used after any change that mutates the active model / provider — the
        TUI's welcome banner and footer cache the model name and only update
        when an ``init_done`` arrives. Without this, switching provider mid-
        session leaves the old model name visible in the banner and footer.

        Best-effort: missing attributes (e.g. in unit-test fixtures that bypass
        the normal constructor) degrade to empty strings rather than crashing.
        """
        sessions = getattr(self, "sessions", None)
        session_key = getattr(self, "_current_session_key", "")
        session = sessions.get(session_key) if sessions is not None and session_key else None
        workspaces = getattr(self, "workspaces", None)
        default_agent = getattr(workspaces, "default_agent_id", "default") if workspaces else "default"
        try:
            skills = self.runtime.discover_skills()
        except Exception:
            skills = []
        try:
            git_branch = self._git_branch()
        except Exception:
            git_branch = ""
        await emit(
            "init_done",
            {
                "model": getattr(self.runtime, "model", ""),
                "session_id": session.id if session else "",
                "cwd": str(Path.cwd()),
                "git_branch": git_branch,
                "context_limit": self._resolve_context_limit(),
                "agent_name": (session.agent_id if session else default_agent),
                "skills": skills,
            },
        )
        # Also push a status_update so the footer's ``context: x/y`` line
        # picks up the new max immediately — the banner refresh on its own
        # rewrites the welcome panel but leaves the bottom bar stale.
        try:
            await self._emit_status(emit)
        except Exception:
            pass

    def _resolve_context_limit(self) -> int:
        """Return the active model's context window in tokens.

        Resolution order:
          1. ``runtime_config["context_limit"]`` / ``["max_context"]`` if the
             operator explicitly overrode it.
          2. The pricing-table entry for the exact model id.
          3. The active provider's default context_limit from the registry.
          4. ``0`` as a last resort (renders as ``0/0`` in the TUI).
        """
        cfg = self.runtime.runtime_config
        override = cfg.get("context_limit") or cfg.get("max_context")
        if override:
            try:
                return int(override)
            except (TypeError, ValueError):
                pass
        try:
            from ..llms.registry import get_context_limit

            model = getattr(self.runtime, "model", "") or ""
            if model:
                return int(get_context_limit(model))
        except Exception:
            pass
        return 0

    @staticmethod
    async def _emit_slash(emit: EmitFn, body: str) -> None:
        """Emit a ``slash``-category ``notification`` carrying ``body``."""
        await emit(
            "notification",
            {
                "id": uuid.uuid4().hex[:12],
                "category": "slash",
                "type": "result",
                "severity": "info",
                "title": "",
                "body": body,
                "payload": {},
            },
        )

    # ============================================================
    # Bulk slash handlers — one per registered command. Each returns
    # nothing; they emit user-facing output via ``self._emit_slash``.
    # ============================================================

    async def _slash_new(self, args: str, emit: EmitFn) -> None:
        """Drop the cached session and start fresh — like a clean re-launch."""
        self.sessions.evict(self._current_session_key)
        session = self.sessions.open(self._current_session_key, self.workspaces.default_agent_id)
        await self._emit_slash(emit, f"New session `{session.id}` started. Scrollback cleared on TUI side.")

    async def _slash_stop(self, args: str, emit: EmitFn) -> None:
        """Cancel the currently in-flight tool / turn in this session."""
        cancelled = self.sessions.cancel(self._current_session_key)
        await self._emit_slash(emit, "Cancelled." if cancelled else "Nothing running to cancel.")

    async def _slash_cancel_all(self, args: str, emit: EmitFn) -> None:
        """Cancel every running turn across every session."""
        count = self.sessions.cancel_all()
        await self._emit_slash(emit, f"Cancelled {count} running turn{'s' if count != 1 else ''}.")

    async def _slash_compact(self, args: str, emit: EmitFn) -> None:
        """Compact the conversation by handing it to the agent with a brief.

        We don't run a local summariser — the model can do it cleanly inside a
        normal turn. The synthetic prompt asks the agent to summarise the
        thread and then continue with whatever comes next.
        """
        prompt = (
            "Please compact this conversation: summarise the relevant context so far "
            "in 3-6 bullet points (decisions made, open questions, key file paths) "
            "and then wait for the user's next instruction. Be terse."
        )
        await self._submit_turn({"text": prompt, "_internal_slash": True}, emit)
        await self._emit_slash(emit, "Compaction turn queued.")

    async def _slash_steer(self, args: str, emit: EmitFn) -> None:
        """Inject a steering hint into the active turn (``/btw`` and ``/steer``).

        Mid-turn the content is queued on the active session; the streaming
        loop drains it between tool iterations and prepends it as a synthetic
        user message before the next LLM request. With no active turn the
        steer lands directly in ``state.messages`` so the next turn sees it.
        ``steer_input`` is still emitted so the TUI can render the injection.
        """
        content = args.strip()
        if not content:
            await self._emit_slash(emit, "Usage: `/steer <hint>` (also `/btw`). Anything after the command is injected.")
            return
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session to steer.")
            return
        await emit("steer_input", {"content": content})
        if session.active_turn_id:
            session.pending_steers.put(content)
            await self._emit_slash(emit, "Steer queued — will land before the next tool round.")
        else:
            session.state.messages.append(
                {"role": "user", "content": f"[steer from user]\n{content}"}
            )
            await self._emit_slash(emit, "Steer injected — will land on the next turn.")

    async def _slash_model(self, args: str, emit: EmitFn) -> None:
        """Show the active model or switch to a new one.

        ``/model`` lists the active model + provider. ``/model <id>`` rebinds
        the runtime to that id (full path or short alias). Profiles are
        managed separately via ``/provider``.
        """
        target = args.strip()
        if not target:
            await self._emit_slash(
                emit,
                f"Active model: `{self.runtime.model or '(none)'}`\n"
                f"Base URL:      `{self.runtime.runtime_config.get('base_url', '(provider default)')}`\n"
                f"Switch with `/model <id>` or pick a profile with `/provider`.",
            )
            return
        try:
            self.runtime.reload({"model": target})
        except Exception as exc:
            await self._emit_slash(emit, f"Failed to switch model: `{exc}`")
            return
        await self._emit_slash(emit, f"Model set to `{self.runtime.model}`.")
        await self._emit_init_done(emit)

    async def _slash_sampling(self, args: str, emit: EmitFn) -> None:
        """Show or update sampling params (temperature, top_p, max_tokens)."""
        if not args.strip():
            cfg = self.runtime.runtime_config
            lines = ["Sampling:"]
            for key in ("temperature", "top_p", "top_k", "max_tokens", "presence_penalty", "frequency_penalty"):
                if key in cfg:
                    lines.append(f"  `{key}` = `{cfg[key]}`")
            if len(lines) == 1:
                lines.append("  (provider defaults — no overrides set)")
            lines.append("\nUpdate with `/sampling <key> <value>` (e.g. `/sampling temperature 0.7`).")
            await self._emit_slash(emit, "\n".join(lines))
            return
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            await self._emit_slash(emit, "Usage: `/sampling <key> <value>`")
            return
        key, raw = parts[0].strip(), parts[1].strip()
        # Coerce numerics; non-numeric values are stored as strings.
        value: Any = raw
        try:
            value = float(raw) if "." in raw else int(raw)
        except ValueError:
            pass
        self.runtime.runtime_config[key] = value
        await self._emit_slash(emit, f"Set sampling `{key}` = `{value}`.")

    async def _slash_config(self, args: str, emit: EmitFn) -> None:
        """Dump non-underscore runtime config keys, with secrets redacted."""
        cfg = self.runtime.runtime_config
        redact_keys = {"api_key", "auth_token", "token", "secret"}
        lines = ["Runtime config:"]
        for key in sorted(cfg):
            if key.startswith("_"):
                continue
            value = cfg[key]
            if any(s in key.lower() for s in redact_keys) and value:
                value = "***redacted***"
            lines.append(f"  `{key}` = `{value}`")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_title(self, args: str, emit: EmitFn) -> None:
        """Set or clear the current session's title."""
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session yet — start chatting first.")
            return
        title = args.strip()
        if not hasattr(session.state, "metadata") or session.state.metadata is None:
            session.state.metadata = {}
        if title:
            session.state.metadata["title"] = title
            await self._emit_slash(emit, f"Session title set to `{title}`.")
        else:
            current = session.state.metadata.get("title", "")
            await self._emit_slash(emit, f"Session title: `{current or '(unset)'}`.")

    async def _slash_workspace(self, args: str, emit: EmitFn) -> None:
        """Show the current project directory and the agent workspace path."""
        session = self.sessions.get(self._current_session_key)
        ws_path = str(session.workspace.path) if session is not None else "(no session)"
        lines = [
            f"Project dir:    `{self.config.project_dir or os.getcwd()}`",
            f"Agent workspace: `{ws_path}`",
            f"Agent id:        `{(session.agent_id if session else self.workspaces.default_agent_id)}`",
        ]
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_save(self, args: str, emit: EmitFn) -> None:
        """Force-persist the active session to disk right now."""
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session to save.")
            return
        self.sessions.save(session)
        path = self.sessions._session_path(session.id)
        await self._emit_slash(emit, f"Saved session `{session.id}` to `{path}`.")

    async def _slash_personality(self, args: str, emit: EmitFn) -> None:
        """Show the path to the workspace's persona file."""
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        path = session.workspace.path / "AGENTS.md"
        if not path.exists():
            await self._emit_slash(emit, f"`AGENTS.md` not found at `{path}`. Run any session prompt to seed it.")
            return
        await self._emit_slash(emit, f"Persona / instructions file: `{path}`\nEdit with your `$EDITOR`, then `/reload`.")

    async def _slash_soul(self, args: str, emit: EmitFn) -> None:
        """Show the path to the workspace's SOUL.md (or memory soul file)."""
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        path = session.workspace.path / "SOUL.md"
        if not path.exists():
            await self._emit_slash(emit, f"`SOUL.md` not found at `{path}`.")
            return
        await self._emit_slash(emit, f"Soul / values file: `{path}`\nEdit then `/reload` to pick up changes.")

    async def _slash_tools(self, args: str, emit: EmitFn) -> None:
        """List the tool names the runtime has registered."""
        names = sorted(t.get("function", {}).get("name", "") for t in self.runtime.tool_schemas)
        names = [n for n in names if n]
        if not names:
            await self._emit_slash(emit, "No tools loaded.")
            return
        lines = [f"Tools ({len(names)}):"]
        for name in names:
            lines.append(f"  `{name}`")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_toolsets(self, args: str, emit: EmitFn) -> None:
        """List subagent / toolset definitions discovered on disk."""
        try:
            from ..agents.definitions import list_agent_definitions
        except Exception:
            await self._emit_slash(emit, "Toolset listing unavailable in this build.")
            return
        defs = list_agent_definitions()
        if not defs:
            await self._emit_slash(emit, "No agent toolsets configured.")
            return
        lines = [f"Agent toolsets ({len(defs)}):"]
        for d in defs:
            name = getattr(d, "name", str(d))
            descr = getattr(d, "description", "") or ""
            lines.append(f"  `{name}` — {descr[:80]}")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_agents(self, args: str, emit: EmitFn) -> None:
        """List agent definitions and any running subagent tasks."""
        try:
            from ..agents.definitions import list_agent_definitions
        except Exception:
            list_agent_definitions = None  # type: ignore[assignment]
        defs = list_agent_definitions() if list_agent_definitions else []
        lines = [f"Agents ({len(defs)}):"]
        for d in defs:
            name = getattr(d, "name", str(d))
            descr = getattr(d, "description", "") or ""
            lines.append(f"  `{name}` — {descr[:80]}")
        # Running subagent tasks, if the runtime exposes them.
        try:
            mgr = self.runtime.subagent_manager  # may not exist
            tasks = [t for t in mgr.tasks.values() if t.status in {"pending", "running"}]
        except Exception:
            tasks = []
        if tasks:
            lines.append("")
            lines.append(f"Running subagent tasks ({len(tasks)}):")
            for t in tasks:
                lines.append(f"  `{t.id}` — `{t.name or t.agent_def_name}` ({t.status})")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_reload(self, args: str, emit: EmitFn) -> None:
        """Reload tools + skills from disk without restarting the daemon."""
        try:
            self.runtime.reload({})
        except Exception as exc:
            await self._emit_slash(emit, f"Reload failed: `{exc}`")
            return
        skills = self.runtime.discover_skills()
        await emit("init_done", {"skills": skills})
        await self._emit_slash(
            emit,
            f"Reloaded. Tools: {len(self.runtime.tool_schemas)} · Skills: {len(skills)}.",
        )

    async def _slash_reload_mcp(self, args: str, emit: EmitFn) -> None:
        """Reload MCP server connections."""
        try:
            from ..mcp.manager import reload_mcp_servers
        except Exception:
            await self._emit_slash(emit, "MCP support not available in this build.")
            return
        try:
            count = reload_mcp_servers()
        except Exception as exc:
            await self._emit_slash(emit, f"MCP reload failed: `{exc}`")
            return
        await self._emit_slash(emit, f"Reloaded {count} MCP server connection(s).")

    async def _slash_memory(self, args: str, emit: EmitFn) -> None:
        """Show where the agent's memory files live + their sizes."""
        from ..runtime.agent_memory import default_global_memory_dir, project_memory_dir_for

        global_dir = default_global_memory_dir()
        try:
            project_dir = project_memory_dir_for(self.config.project_dir or os.getcwd())
        except Exception:
            project_dir = None

        def _sz(path):
            try:
                return path.stat().st_size if path.exists() else 0
            except OSError:
                return 0

        lines = ["Memory:"]
        lines.append(f"  Global scope: `{global_dir}`")
        for name in ("SOUL.md", "IDENTITY.md", "USER.md", "MEMORY.md", "KNOWLEDGE.md", "INSIGHTS.md", "EXPERIENCES.md"):
            path = global_dir / name
            lines.append(f"    `{name}` — {_sz(path)} bytes" + ("" if path.exists() else "  (missing)"))
        if project_dir is not None:
            lines.append(f"  Project scope: `{project_dir}`")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_history(self, args: str, emit: EmitFn) -> None:
        """Show message and turn counts for the active session."""
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        st = session.state
        await self._emit_slash(
            emit,
            f"Messages: {len(st.messages)}\nTurns: {st.turn_count}\n"
            f"Input tokens: {st.total_input_tokens}\nOutput tokens: {st.total_output_tokens}",
        )

    async def _slash_usage(self, args: str, emit: EmitFn) -> None:
        """Show token usage for the active session (alias-ish for /context)."""
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        st = session.state
        total = st.total_input_tokens + st.total_output_tokens
        lines = [
            f"Input tokens:        {st.total_input_tokens}",
            f"Output tokens:       {st.total_output_tokens}",
            f"Total tokens:        {total}",
            f"Cache read tokens:   {st.total_cache_read_tokens}",
            f"Cache create tokens: {st.total_cache_creation_tokens}",
        ]
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_cost(self, args: str, emit: EmitFn) -> None:
        """Show the running USD cost for the active session."""
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        try:
            cost = session.state.cost
        except Exception:
            cost = 0.0
        await self._emit_slash(emit, f"Estimated cost: `${cost:.4f}` (model: `{self.runtime.model}`).")

    async def _slash_insights(self, args: str, emit: EmitFn) -> None:
        """Show top tools by call count from the session's execution log."""
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        counts: dict[str, int] = {}
        for ex in session.state.tool_executions:
            name = ex.get("name", "(unknown)")
            counts[name] = counts.get(name, 0) + 1
        if not counts:
            await self._emit_slash(emit, "No tools invoked in this session yet.")
            return
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        lines = ["Top tools this session:"]
        for name, n in top:
            lines.append(f"  `{name}` — {n} call{'s' if n != 1 else ''}")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_budget(self, args: str, emit: EmitFn) -> None:
        """Show context-window usage and remaining headroom."""
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        from ..llms.registry import get_context_limit

        model = self.runtime.model or ""
        limit = get_context_limit(model) if model else 0
        used = session.state.total_input_tokens + session.state.total_output_tokens
        remaining = max(0, limit - used)
        pct = (used / limit * 100) if limit else 0.0
        await self._emit_slash(
            emit,
            f"Context window: `{limit or '?'}` tokens for `{model or '(unknown)'}`\n"
            f"Used: {used} ({pct:.1f}%) · Remaining: {remaining}",
        )

    async def _slash_doctor(self, args: str, emit: EmitFn) -> None:
        """Run a quick self-check: provider configured, tools loaded, skills found."""
        lines = ["Diagnostics:"]
        ok = True
        if not self.runtime.model:
            lines.append("  ✗ No model configured — run `/provider` or set `XERXES_MODEL`.")
            ok = False
        else:
            lines.append(f"  ✓ Model: `{self.runtime.model}`")
        tools = len(self.runtime.tool_schemas)
        lines.append(f"  {'✓' if tools else '✗'} Tools loaded: {tools}")
        skills = len(self.runtime.discover_skills())
        lines.append(f"  ✓ Skills discovered: {skills}")
        sessions = len(self.sessions.list())
        lines.append(f"  ✓ Saved sessions on disk: {sessions}")
        lines.append("")
        lines.append("All good." if ok else "Something needs attention — see above.")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_update(self, args: str, emit: EmitFn) -> None:
        """Show the installed Xerxes version and an update hint."""
        try:
            from importlib.metadata import version

            ver = version("xerxes")
        except Exception:
            ver = "(unknown — running from source)"
        await self._emit_slash(
            emit,
            f"Xerxes `{ver}`\nUpdate with: `uv pip install -U xerxes` (or your package manager).",
        )

    async def _slash_nudge(self, args: str, emit: EmitFn) -> None:
        """Toggle Honcho-style background nudges."""
        cur = bool(self.runtime.runtime_config.get("nudge", True))
        action = args.strip().lower()
        if action == "on":
            new_value = True
        elif action == "off":
            new_value = False
        else:
            new_value = not cur
        self.runtime.runtime_config["nudge"] = new_value
        await self._emit_slash(emit, f"Nudge: {'ON' if new_value else 'OFF'}.")

    async def _slash_feedback(self, args: str, emit: EmitFn) -> None:
        """Show where to file feedback / report bugs."""
        await self._emit_slash(
            emit,
            "Feedback / issues:\n"
            "  • GitHub: https://github.com/erfanzar/Xerxes/issues\n"
            "  • Anything urgent? Mention it in the daemon log first (`~/.xerxes/daemon.log`).",
        )

    async def _slash_plugins(self, args: str, emit: EmitFn) -> None:
        """List loaded plugins and their slash registrations."""
        try:
            from ..extensions.plugins import list_loaded_plugins
        except Exception:
            list_loaded_plugins = None  # type: ignore[assignment]
        try:
            from ..extensions.slash_plugins import registered_slashes
        except Exception:
            registered_slashes = None  # type: ignore[assignment]

        lines = ["Plugins:"]
        plugins = list_loaded_plugins() if list_loaded_plugins else []
        if plugins:
            for p in plugins:
                name = getattr(p, "name", str(p))
                lines.append(f"  `{name}`")
        else:
            lines.append("  (no plugins loaded)")
        slashes = registered_slashes() if registered_slashes else []
        if slashes:
            lines.append("")
            lines.append("Plugin slash commands:")
            for s in slashes:
                lines.append(f"  `/{s.name}` — {getattr(s, 'description', '')}")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_platforms(self, args: str, emit: EmitFn) -> None:
        """List configured messaging channel platforms and their state."""
        lines = ["Channel platforms:"]
        for name, ch in self.channels.channels.items():
            enabled = "on " if ch.enabled else "off"
            ready = "ready" if ch.instance is not None else "not-started"
            lines.append(f"  [{enabled}] `{name}` — {ready}")
        if len(lines) == 1:
            lines.append("  (none configured)")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_browser(self, args: str, emit: EmitFn) -> None:
        """List browser sessions managed by the operator subsystem."""
        sessions = []
        for sess in self.sessions._sessions.values():
            ops = getattr(sess, "operator_state", None)
            if ops is None:
                continue
            mgr = getattr(ops, "browser_manager", None)
            if mgr is None:
                continue
            try:
                sessions.extend(mgr.list_sessions())
            except Exception:
                continue
        if not sessions:
            await self._emit_slash(emit, "No browser sessions open. Use the `start_browser` tool to create one.")
            return
        lines = [f"Browser sessions ({len(sessions)}):"]
        for s in sessions:
            lines.append(f"  `{getattr(s, 'id', '?')}` — `{getattr(s, 'url', '')}`")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_image(self, args: str, emit: EmitFn) -> None:
        """Generate an image from a prompt via the image-generation tool."""
        prompt = args.strip()
        if not prompt:
            await self._emit_slash(emit, "Usage: `/image <prompt>`")
            return
        # Defer to the agent — it has access to the image generation tool and
        # can pick a provider + model on its own.
        synthetic = (
            f"Generate an image matching this brief and report the saved path:\n\n{prompt}\n\n"
            "Use the `generate_image` tool. Do not write any code; one tool call is enough."
        )
        await self._submit_turn({"text": synthetic, "_internal_slash": True}, emit)
        await self._emit_slash(emit, "Image-generation turn queued.")

    async def _slash_cron(self, args: str, emit: EmitFn) -> None:
        """List cron jobs (sub-commands ``add``/``remove``/``run`` deferred to JSON-RPC)."""
        try:
            from ..cron import jobs as cron_jobs
        except Exception:
            await self._emit_slash(emit, "Cron support not available in this build.")
            return
        sub = args.strip().split(maxsplit=1)
        action = sub[0].lower() if sub else "list"
        if action in ("", "list"):
            try:
                items = cron_jobs.list_jobs()
            except Exception as exc:
                await self._emit_slash(emit, f"Cron list failed: `{exc}`")
                return
            if not items:
                await self._emit_slash(emit, "No cron jobs scheduled.")
                return
            lines = [f"Cron jobs ({len(items)}):"]
            for j in items:
                lines.append(
                    f"  `{getattr(j, 'id', '?')}` — `{getattr(j, 'schedule', '')}` "
                    f"({'paused' if getattr(j, 'paused', False) else 'active'})"
                )
            await self._emit_slash(emit, "\n".join(lines))
            return
        await self._emit_slash(
            emit,
            f"`/cron {action}` is not yet wired in the daemon. "
            "Use the corresponding JSON-RPC method, or `/cron list` to view jobs.",
        )

    async def _slash_fast(self, args: str, emit: EmitFn) -> None:
        """Toggle fast-mode (cheaper auxiliary model for summaries/titles/etc.)."""
        cur = bool(self.runtime.runtime_config.get("fast_mode", False))
        action = args.strip().lower()
        if action == "on":
            new_value = True
        elif action == "off":
            new_value = False
        else:
            new_value = not cur
        self.runtime.runtime_config["fast_mode"] = new_value
        await self._emit_slash(emit, f"Fast mode: {'ON' if new_value else 'OFF'}.")

    async def _slash_skin(self, args: str, emit: EmitFn) -> None:
        """Skin is a pure TUI concern; ack and tell the user to use the TUI command."""
        await self._emit_slash(emit, "Skin is a TUI-side setting. Use the TUI's `Ctrl+T` or the `xerxes-skin` CLI.")

    async def _slash_statusbar(self, args: str, emit: EmitFn) -> None:
        """Statusbar is TUI-side; ack with a hint."""
        await self._emit_slash(emit, "Statusbar visibility is a TUI-side setting.")

    async def _slash_paste(self, args: str, emit: EmitFn) -> None:
        """Paste is TUI-side; ack with a hint."""
        await self._emit_slash(emit, "Paste is handled in the TUI via `Ctrl+V` / `Alt+V`.")

    async def _slash_voice(self, args: str, emit: EmitFn) -> None:
        """Voice mode is TUI-side; ack with a hint."""
        await self._emit_slash(emit, "Voice mode is TUI-side. Use the `xerxes-voice` CLI or the TUI's voice key.")

    async def _slash_queue(self, args: str, emit: EmitFn) -> None:
        """The TUI manages its own input queue; ack and tell the user."""
        await self._emit_slash(
            emit, "Pending input queue is TUI-side. The footer shows the count when items are queued."
        )

    async def _slash_background(self, args: str, emit: EmitFn) -> None:
        """List in-flight daemon background tasks (turn runners + post-turn hooks)."""
        active = [t for t in self._background_tasks if not t.done()]
        if not active:
            await self._emit_slash(emit, "No background tasks running.")
            return
        lines = [f"Background tasks ({len(active)}):"]
        for t in active:
            lines.append(f"  `{t.get_name()}`")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_resume(self, args: str, emit: EmitFn) -> None:
        """List recent saved sessions; resuming requires re-launching with ``-r``."""
        records = self.sessions.list()
        if not records:
            await self._emit_slash(emit, "No saved sessions found.")
            return
        lines = ["Recent sessions:"]
        for r in records[:10]:
            sid = r.get("session_id", "?")
            updated = r.get("updated_at", "")
            turns = r.get("turn_count", 0)
            lines.append(f"  `{sid}` — {turns} turn(s), updated {updated}")
        lines.append("")
        lines.append("Resume from a fresh terminal: `xerxes -r <id>`.")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_restart(self, args: str, emit: EmitFn) -> None:
        """Schedule a daemon shutdown so the TUI respawns a fresh process on next launch."""
        await self._emit_slash(emit, "Restarting daemon — re-run `xerxes` after this shuts down.")
        self._track_task(self.shutdown())

    async def _slash_undo(self, args: str, emit: EmitFn) -> None:
        """Drop the last user/assistant turn pair from the active session."""
        session = self.sessions.get(self._current_session_key)
        if session is None or not session.state.messages:
            await self._emit_slash(emit, "Nothing to undo.")
            return
        msgs = session.state.messages
        # Pop assistant + matching user (and any intervening tool messages).
        dropped = 0
        while msgs and msgs[-1].get("role") != "user":
            msgs.pop()
            dropped += 1
        if msgs and msgs[-1].get("role") == "user":
            msgs.pop()
            dropped += 1
        session.state.turn_count = max(0, session.state.turn_count - 1)
        self.sessions.save(session)
        await self._emit_slash(emit, f"Undone — dropped {dropped} message(s) from the conversation.")

    async def _slash_retry(self, args: str, emit: EmitFn) -> None:
        """Resend the most recent user message after dropping the failed reply."""
        session = self.sessions.get(self._current_session_key)
        if session is None or not session.state.messages:
            await self._emit_slash(emit, "Nothing to retry.")
            return
        msgs = session.state.messages
        last_user_text = ""
        for msg in reversed(msgs):
            if msg.get("role") == "user":
                content = msg.get("content")
                last_user_text = content if isinstance(content, str) else str(content)
                break
        if not last_user_text:
            await self._emit_slash(emit, "No prior user message to retry.")
            return
        # Drop everything after the last user message so the model retries cleanly.
        while msgs and msgs[-1].get("role") != "user":
            msgs.pop()
        if msgs:
            msgs.pop()  # remove the user message too — _submit_turn re-appends it
        await self._emit_slash(emit, "Retrying the last prompt…")
        await self._submit_turn({"text": last_user_text, "_internal_slash": True}, emit)

    async def _slash_branch(self, args: str, emit: EmitFn) -> None:
        """Branch / fork the current session — saves a copy under a new id."""
        session = self.sessions.get(self._current_session_key)
        if session is None:
            await self._emit_slash(emit, "No active session to branch.")
            return
        import copy

        new_id = uuid.uuid4().hex[:12]
        clone = DaemonSession(
            id=new_id,
            key=new_id,
            agent_id=session.agent_id,
            workspace=session.workspace,
        )
        clone.state.messages = copy.deepcopy(session.state.messages)
        clone.state.turn_count = session.state.turn_count
        self.sessions._sessions[new_id] = clone
        self.sessions.save(clone)
        await self._emit_slash(
            emit,
            f"Branched to new session `{new_id}` ({len(clone.state.messages)} messages).\n"
            f"Open in a new terminal with `xerxes -r {new_id}`.",
        )

    async def _slash_branches(self, args: str, emit: EmitFn) -> None:
        """List every saved session id grouped by who their last update was."""
        records = self.sessions.list()
        if not records:
            await self._emit_slash(emit, "No branches / saved sessions.")
            return
        lines = [f"Branches / saved sessions ({len(records)}):"]
        for r in records[:20]:
            sid = r.get("session_id", "?")
            turns = r.get("turn_count", 0)
            updated = r.get("updated_at", "")
            lines.append(f"  `{sid}` — {turns} turn(s), updated {updated}")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_snapshot(self, args: str, emit: EmitFn) -> None:
        """Take a filesystem snapshot of the project working tree."""
        try:
            from ..session.snapshots import SnapshotManager
        except Exception:
            await self._emit_slash(emit, "Snapshot support not available in this build.")
            return
        try:
            mgr = SnapshotManager(self.config.project_dir or os.getcwd())
            record = mgr.snapshot(label=args.strip() or "manual")
        except Exception as exc:
            await self._emit_slash(emit, f"Snapshot failed: `{exc}`")
            return
        await self._emit_slash(emit, f"Snapshot `{getattr(record, 'id', '?')}` saved.")

    async def _slash_snapshots(self, args: str, emit: EmitFn) -> None:
        """List recent filesystem snapshots."""
        try:
            from ..session.snapshots import SnapshotManager
        except Exception:
            await self._emit_slash(emit, "Snapshot support not available in this build.")
            return
        try:
            mgr = SnapshotManager(self.config.project_dir or os.getcwd())
            items = mgr.list()
        except Exception as exc:
            await self._emit_slash(emit, f"Snapshot list failed: `{exc}`")
            return
        if not items:
            await self._emit_slash(emit, "No snapshots yet. Take one with `/snapshot [label]`.")
            return
        lines = [f"Snapshots ({len(items)}):"]
        for s in items[:20]:
            sid = getattr(s, "id", "?")
            label = getattr(s, "label", "") or ""
            when = getattr(s, "created_at", "") or ""
            lines.append(f"  `{sid}` — `{label}` @ {when}")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_rollback(self, args: str, emit: EmitFn) -> None:
        """Roll the working tree back to a snapshot (use ``/snapshots`` to find ids)."""
        target = args.strip()
        if not target:
            await self._emit_slash(emit, "Usage: `/rollback <snapshot-id>` — list with `/snapshots`.")
            return
        try:
            from ..session.snapshots import SnapshotManager
        except Exception:
            await self._emit_slash(emit, "Snapshot support not available in this build.")
            return
        try:
            mgr = SnapshotManager(self.config.project_dir or os.getcwd())
            mgr.rollback(target)
        except Exception as exc:
            await self._emit_slash(emit, f"Rollback failed: `{exc}`")
            return
        await self._emit_slash(emit, f"Rolled back to snapshot `{target}`.")


# ----- bulk dispatch table for the rest of COMMAND_REGISTRY ------------------
# Mapping cmd name → bound-method-style callable so a single
# ``handler(self, args, emit)`` line in ``_handle_slash`` covers everything.
# Aliases are duplicated here so resolving the canonical name still hits the
# right handler (``resolve_command`` already canonicalises before lookup).
_BULK_SLASH_HANDLERS: dict[str, Any] = {
    "new": DaemonServer._slash_new,
    "reset": DaemonServer._slash_new,
    "stop": DaemonServer._slash_stop,
    "cancel": DaemonServer._slash_stop,
    "cancel-all": DaemonServer._slash_cancel_all,
    "compact": DaemonServer._slash_compact,
    "compress": DaemonServer._slash_compact,
    "btw": DaemonServer._slash_steer,
    "steer": DaemonServer._slash_steer,
    "model": DaemonServer._slash_model,
    "sampling": DaemonServer._slash_sampling,
    "config": DaemonServer._slash_config,
    "title": DaemonServer._slash_title,
    "workspace": DaemonServer._slash_workspace,
    "save": DaemonServer._slash_save,
    "personality": DaemonServer._slash_personality,
    "soul": DaemonServer._slash_soul,
    "tools": DaemonServer._slash_tools,
    "toolsets": DaemonServer._slash_toolsets,
    "agents": DaemonServer._slash_agents,
    "reload": DaemonServer._slash_reload,
    "reload-mcp": DaemonServer._slash_reload_mcp,
    "memory": DaemonServer._slash_memory,
    "history": DaemonServer._slash_history,
    "usage": DaemonServer._slash_usage,
    "cost": DaemonServer._slash_cost,
    "insights": DaemonServer._slash_insights,
    "budget": DaemonServer._slash_budget,
    "doctor": DaemonServer._slash_doctor,
    "update": DaemonServer._slash_update,
    "nudge": DaemonServer._slash_nudge,
    "feedback": DaemonServer._slash_feedback,
    "plugins": DaemonServer._slash_plugins,
    "platforms": DaemonServer._slash_platforms,
    "browser": DaemonServer._slash_browser,
    "image": DaemonServer._slash_image,
    "cron": DaemonServer._slash_cron,
    "fast": DaemonServer._slash_fast,
    "skin": DaemonServer._slash_skin,
    "statusbar": DaemonServer._slash_statusbar,
    "paste": DaemonServer._slash_paste,
    "voice": DaemonServer._slash_voice,
    "queue": DaemonServer._slash_queue,
    "background": DaemonServer._slash_background,
    "resume": DaemonServer._slash_resume,
    "restart": DaemonServer._slash_restart,
    "undo": DaemonServer._slash_undo,
    "retry": DaemonServer._slash_retry,
    "branch": DaemonServer._slash_branch,
    "branches": DaemonServer._slash_branches,
    "snapshot": DaemonServer._slash_snapshot,
    "snapshots": DaemonServer._slash_snapshots,
    "rollback": DaemonServer._slash_rollback,
}


def main() -> None:
    """Parse CLI flags, build a :class:`DaemonServer`, and run it under ``asyncio.run``."""
    import argparse

    parser = argparse.ArgumentParser(description="Xerxes daemon")
    parser.add_argument("--project-dir", default="", help="Working directory")
    parser.add_argument("--host", default="", help="WebSocket host")
    parser.add_argument("--port", type=int, default=0, help="WebSocket port")
    parser.add_argument("--socket", default="", help="Unix socket path")
    args = parser.parse_args()

    config = load_config(project_dir=args.project_dir)
    if args.host:
        config.ws_host = args.host
    if args.port:
        config.ws_port = args.port
    if args.socket:
        config.socket_path = args.socket

    server = DaemonServer(config)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


__all__ = ["MIGRATED_ERROR", "DaemonServer", "main"]
