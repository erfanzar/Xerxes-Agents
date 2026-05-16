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
from .runtime import RuntimeManager, SessionManager, TurnRunner, WorkspaceManager
from .socket_channel import SocketChannel

MIGRATED_ERROR = (
    "Old daemon task API was removed; use session.open, turn.submit, turn.cancel, session.list, and runtime.status."
)
DAEMON_PROTOCOL_VERSION = (
    22  # bumped: bare-launch evicts the in-memory tui:default slot so fresh xerxes invocations are clean
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
            return {"ok": True, "profile": profile}
        if method == "provider_list":
            return {"ok": True, "profiles": profiles.list_profiles()}
        if method == "provider_select":
            ok = profiles.set_active(str(params.get("name", "")))
            if ok:
                self.runtime.reload()
            return {"ok": ok}
        if method == "provider_delete":
            return {"ok": profiles.delete_profile(str(params.get("name", "")))}
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
                "context_limit": int(
                    self.runtime.runtime_config.get("context_limit", self.runtime.runtime_config.get("max_context", 0))
                    or 0
                ),
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
                "max_context": int(
                    self.runtime.runtime_config.get("context_limit", self.runtime.runtime_config.get("max_context", 0))
                    or 0
                ),
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

        if cmd in {"exit", "quit", "q"}:
            await self._emit_slash(emit, "(use Ctrl+D or close the terminal to exit)")
            return

        if cmd == "clear":
            # Clear is a per-session concern; the TUI handles its own scrollback,
            # so we just acknowledge.
            await self._emit_slash(emit, "Cleared.")
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
