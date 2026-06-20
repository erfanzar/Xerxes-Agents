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

import argparse
import asyncio
import os
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Any

from ..bridge import profiles
from ..channels.types import ChannelMessage, MessageDirection
from ..context.window_usage import estimate_context_tokens
from ..runtime.config_context import get_event_callback, set_event_callback
from . import slash_commands as _slash_commands
from .channels import ChannelManager, ChannelWebhookServer
from .config import DaemonConfig, load_config
from .gateway import EmitFn, WebSocketGateway
from .log import DaemonLogger
from .provider_flow import ProviderFlowMixin
from .runtime import RuntimeManager, SessionManager, TurnRunner, WorkspaceManager
from .skill_create import SkillCreateMixin
from .socket_channel import SocketChannel

MIGRATED_ERROR = (
    "Old daemon task API was removed; use session.open, turn.submit, turn.cancel, session.list, and runtime.status."
)
DAEMON_PROTOCOL_VERSION = 35  # bumped: AskUserQuestionTool now drives the TUI panel via TurnRunner.ask_user_question

SlashCommandsMixin = _slash_commands.SlashCommandsMixin
_BULK_SLASH_HANDLERS = _slash_commands._BULK_SLASH_HANDLERS


class DaemonServer(SlashCommandsMixin, ProviderFlowMixin, SkillCreateMixin):
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
        # Per-connection session binding. The daemon multiplexes several
        # clients (Unix socket, websocket, channels) onto one loop, so a
        # single shared ``_current_session_key`` lets a second client's
        # ``prompt`` run against the first client's session. Each connection's
        # ``emit`` closure is a stable per-connection identity; we bind the
        # session key to it at ``initialize`` and resolve it per ``prompt`` so
        # concurrent turns never clobber one another. ``_current_session_key``
        # remains the fallback for legacy single-client paths (slash handlers,
        # un-initialized callers).
        self._connection_sessions: dict[EmitFn, str] = {}
        # Owning connection for each in-flight turn, keyed by session key.
        # ``TurnRunner.ask_user_question`` fires its ``question_request``
        # through the broadcast sink (it runs on a worker thread with no
        # per-connection ``emit``); we use this to route that question to the
        # connection that actually started the turn instead of fanning it out
        # to every client, and to reject answers from other connections.
        self._turn_owner_emits: dict[str, EmitFn] = {}
        # request_id -> owning emit, captured the first time a question is
        # routed so the matching ``question_response`` can be authorised.
        self._question_owner_emits: dict[str, EmitFn] = {}
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

        await self._socket.start(self._handle_rpc)
        gateway_started = True
        try:
            await self._gateway.start(self._handle_rpc)
        except OSError as exc:
            gateway_started = False
            self.logger.error("WebSocket gateway unavailable; continuing with Unix socket", error=str(exc))
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
            ws=f"ws://{self.config.ws_host}:{self.config.ws_port}" if gateway_started else "disabled",
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
            # Resolve the session bound to *this* connection rather than the
            # shared ``_current_session_key`` so a concurrent client's turn
            # can't redirect this prompt into its own session.
            return await self._submit_turn(
                {
                    "session_key": self._connection_session_key(emit),
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
            session = self.sessions.get(str(params.get("session_key") or self._connection_session_key(emit)))
            return {"ok": bool(session), "session": session.status() if session else None}
        if method == "turn.submit":
            return await self._submit_turn(params, emit)
        if method in {"turn.cancel", "cancel"}:
            return {"ok": self.sessions.cancel(str(params.get("session_key") or self._connection_session_key(emit)))}
        if method == "cancel_all":
            return {"ok": True, "cancelled": self.sessions.cancel_all()}
        if method == "turn.steer" or method == "steer":
            session_key = str(params.get("session_key") or self._connection_session_key(emit))
            await self._steer_session(session_key, str(params.get("content", "")), emit)
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
                # ``AskUserQuestionTool`` callback. Only the connection the
                # question was routed to may answer it — otherwise a second
                # client could resolve another client's turn with the wrong
                # user's input. If we never recorded an owner for this id
                # (e.g. a question that was broadcast because the owner was
                # ambiguous, or a test stub) we accept it for back-compat.
                question_owners = getattr(self, "_question_owner_emits", None)
                owner = question_owners.get(rid) if question_owners is not None else None
                if owner is not None and owner is not emit:
                    return {"ok": False, "error": "question owned by another connection"}
                # ``False`` means the request id was unknown (e.g. the turn was
                # already cancelled) — fine, drop silently.
                await self.turns.respond_question(rid, answers)
                if question_owners is not None:
                    question_owners.pop(rid, None)
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
        # Bind this session to the originating connection so its ``prompt``
        # calls resolve to the right session even while other clients are
        # connected to the same daemon. ``getattr`` guards test fixtures that
        # build the server via ``__new__`` and skip ``__init__``.
        conn_sessions = getattr(self, "_connection_sessions", None)
        if conn_sessions is not None:
            conn_sessions[emit] = self._current_session_key
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
            **self.runtime.status(),
            "ok": True,
            "session": session.status(),
            "daemon_protocol": DAEMON_PROTOCOL_VERSION,
        }

    def _connection_session_key(self, emit: EmitFn) -> str:
        """Return the session key bound to ``emit``'s connection.

        Falls back to the shared ``_current_session_key`` for connections
        that never sent ``initialize`` (e.g. a channel emit or a test stub),
        so behaviour for the single-client case is unchanged.
        """
        conn_sessions = getattr(self, "_connection_sessions", None)
        if conn_sessions is None:
            return self._current_session_key
        return conn_sessions.get(emit, self._current_session_key)

    async def _submit_turn(self, params: dict[str, Any], emit: EmitFn) -> dict[str, Any]:
        """Queue a new turn on the resolved session and stream events to ``emit``."""
        # Don't strip yet — empty/whitespace input is a *valid* answer for the
        # optional ``pitfalls`` step of the /skill-create interview ("Press
        # Enter to skip"). We re-strip per-branch below.
        raw_text = str(params.get("text") or params.get("prompt") or params.get("user_input") or "")
        text = raw_text.strip()
        session_key = str(params.get("session_key") or self._connection_session_key(emit))

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
        if not self.runtime.model:
            return {"ok": False, "error": "No model configured. Run /provider first or set XERXES_MODEL."}
        agent_id = str(params.get("agent_id") or self.workspaces.default_agent_id)
        mode = str(params.get("mode") or self._current_mode or "code")
        plan_mode = bool(params.get("plan_mode", self._current_plan_mode or mode == "plan"))
        # Keep this connection's binding current, but DON'T overwrite the
        # shared ``_current_session_key`` when the caller carried an explicit
        # session key — doing so is exactly the cross-client clobber bug. We
        # only fall back to mutating the shared key for legacy callers that
        # submitted without one. ``getattr`` guards ``__new__``-built test
        # fixtures that skip ``__init__``.
        conn_sessions = getattr(self, "_connection_sessions", None)
        if conn_sessions is not None and emit in conn_sessions:
            conn_sessions[emit] = session_key
        if params.get("session_key") is None:
            self._current_session_key = session_key
        self._current_mode = mode
        self._current_plan_mode = plan_mode
        session = self.sessions.open(session_key, agent_id)
        # Record the connection that owns this turn so an interactive
        # ``question_request`` (which TurnRunner emits through the broadcast
        # sink) gets routed back to it instead of every client.
        turn_owners = getattr(self, "_turn_owner_emits", None)
        if turn_owners is not None:
            turn_owners[session_key] = emit
        turn_task = self._track_task(self.turns.run_turn(session, text, emit, mode=mode, plan_mode=plan_mode))
        if turn_owners is not None and hasattr(turn_task, "add_done_callback"):
            turn_task.add_done_callback(lambda _t, key=session_key, e=emit: self._release_turn_owner(key, e))
        return {"ok": True, "session": session.status(), "turn_task": turn_task}

    def _release_turn_owner(self, session_key: str, emit: EmitFn) -> None:
        """Drop the turn-owner binding once a turn finishes.

        Only clears the slot if it still points at ``emit`` so a newer turn on
        the same session (queued before this one's done-callback fired) keeps
        its own owner.
        """
        if self._turn_owner_emits.get(session_key) is emit:
            self._turn_owner_emits.pop(session_key, None)

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
                "context_tokens": estimate_context_tokens(session.state.messages, model=self.runtime.model),
                "max_context": self._resolve_context_limit(),
                "mcp_status": {},
                "plan_mode": self._current_plan_mode,
                "mode": self._current_mode,
                "reasoning_effort": self.runtime.reasoning_state()["effort"],
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
        """Fan an event out to every connected socket and websocket client.

        ``question_request`` is the exception: an interactive question belongs
        to exactly one turn, so — like ``approval_request`` — it must reach
        only the connection that started that turn. If we can identify the
        owning connection we unicast to it and record the owner so the
        matching ``question_response`` can be authorised; otherwise we fall
        back to broadcasting (e.g. no turn owner known, or an unidentified
        single-client setup) to preserve the previous behaviour.
        """
        if event_type == "question_request":
            owner = self._resolve_question_owner(payload)
            if owner is not None:
                request_id = str(payload.get("id", ""))
                if request_id:
                    self._question_owner_emits[request_id] = owner
                self._track_task(owner(event_type, payload))
                return
        self._socket.broadcast(event_type, payload)
        self._gateway.broadcast(event_type, payload)

    def _resolve_question_owner(self, payload: dict[str, Any]) -> EmitFn | None:
        """Find the connection that owns an interactive ``question_request``.

        ``TurnRunner.ask_user_question`` runs on a turn's worker thread and has
        no per-connection ``emit``, so the request payload carries no session
        key. When exactly one turn is in flight we can attribute the question
        to it unambiguously; with several concurrent turns we can't tell which
        one asked, so we return ``None`` and let the caller broadcast rather
        than risk routing it to the wrong client.
        """
        owners = list(self._turn_owner_emits.values())
        if len(owners) == 1:
            return owners[0]
        return None

    def _on_background_task_done(self, task: asyncio.Task[Any]) -> None:
        """Drop completed tasks and log unexpected exceptions."""
        self._background_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.logger.error("Background task failed", error=str(exc))

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


def main() -> None:
    """Parse CLI flags, build a :class:`DaemonServer`, and run it under ``asyncio.run``."""
    parser = argparse.ArgumentParser(description="Xerxes daemon")
    parser.add_argument("--project-dir", default="", help="Working directory")
    parser.add_argument("--host", default="", help="WebSocket host")
    parser.add_argument("--port", type=int, default=None, help="WebSocket port")
    parser.add_argument("--socket", default="", help="Unix socket path")
    parser.add_argument("--pid-file", default="", help="Daemon pid file path")
    args = parser.parse_args()

    config = load_config(project_dir=args.project_dir)
    if args.host:
        config.ws_host = args.host
    if args.port is not None:
        config.ws_port = args.port
    if args.socket:
        config.socket_path = args.socket
    if args.pid_file:
        config.pid_file = args.pid_file

    server = DaemonServer(config)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


__all__ = ["MIGRATED_ERROR", "_BULK_SLASH_HANDLERS", "DaemonServer", "main"]
