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
"""Telegram webhook gateway for Xerxes.

Embeds a FastAPI app that accepts Telegram Bot API webhooks (or, in
polling mode, drives ``getUpdates`` itself), runs one Xerxes agent turn
per inbound message, and sends only the final response back to Telegram
— streaming the answer through edited preview messages when previews are
enabled.

Several defences live in this module because the Telegram chat surface
is uniquely exposed:

* fail-closed user / username allowlist (``_is_sender_allowed``);
* webhook authenticity via Telegram's secret-token header
  (``_const_time_eq``);
* hard cap on inbound POST size;
* per-chat in-flight queue cap to prevent lock-queue OOMs;
* prompt-injection scrub of inbound text and quote-fencing in the
  workspace journal (``_scan_inbound`` + ``_quote_user_block``);
* path / traceback redaction on outbound text (``_sanitize_outbound``);
* orphan tool-call / tool-result detection during session compaction
  (``_is_orphan_continuation``).
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import os
import re
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from ..bridge import profiles
from ..runtime.bootstrap import bootstrap
from ..runtime.bridge import build_tool_executor, populate_registry
from ..runtime.config_context import set_config as set_global_config
from ..security.prompt_scanner import scan_context_content
from ..streaming.events import AgentState, TextChunk, ToolEnd, ToolStart
from ..streaming.loop import run as run_agent_loop
from ._helpers import install_log_redaction
from .adapters.telegram import TelegramChannel
from .types import ChannelMessage, MessageDirection
from .workspace import MarkdownAgentWorkspace

install_log_redaction()

logger = logging.getLogger(__name__)

SubmitFn = Callable[[ChannelMessage], str]

# Maximum concurrent or queued turns per session_key. Past this we drop
# inbound silently with a log warning, otherwise a chat can grow the asyncio
# lock waiters list without bound.
_MAX_QUEUED_PER_CHAT = 4


def _parse_csv_set(value: str) -> set[str]:
    """Split a comma-separated config string into a deduplicated set.

    Stripped, empty-skipping. Used for ``allowed_user_ids`` and
    ``allowed_usernames`` config values.
    """
    return {part.strip() for part in (value or "").split(",") if part.strip()}


def _const_time_eq(a: str, b: str) -> bool:
    """Compare two strings in constant time to defeat timing oracles.

    Used to verify Telegram's ``X-Telegram-Bot-Api-Secret-Token`` header
    against the configured ``webhook_secret_token`` without leaking the
    expected length or first-differing byte via response timing.
    """
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _scan_inbound(text: str) -> str:
    """Run inbound channel text through the prompt-injection scanner.

    Replaces known injection patterns with neutralising placeholders so the
    surviving text can be journalled and re-loaded on later turns without
    smuggling instructions into the system prompt.

    Args:
        text: Raw user message body.

    Returns:
        The scanned text, with risky fragments replaced.
    """
    return scan_context_content(text, filename="telegram:inbound")


def _quote_user_block(text: str) -> str:
    """Wrap user text in a ``~~~user`` fenced block for safe journalling.

    The workspace loader treats the fenced content as quoted user input
    rather than agent guidance, so a malicious user cannot inject a
    Markdown heading like ``# System`` that the next turn's
    ``load_context`` would render as agent instruction.
    """
    fence = "~~~user"
    end = "~~~"
    return f"{fence}\n{text}\n{end}"


# Anything that looks like an absolute POSIX path under the user's home or
# common system roots, plus Python traceback noise, is redacted before being
# sent back to the channel. Errors should never leak directory layout.
_PATH_REDACT_RE = re.compile(
    r"(/Users/[^\s'\"]+|/home/[^\s'\"]+|/private/[^\s'\"]+|/var/[^\s'\"]+|/tmp/[^\s'\"]+|~/\.xerxes[^\s'\"]*)"
)
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):.*?(?=\n\n|\Z)", re.DOTALL)


def _is_orphan_continuation(msg: Any) -> bool:
    """Return whether ``msg`` would dangle if its pair message were dropped.

    Tool-result entries and assistant tool-call entries are only legal when
    paired with their counterpart; cutting a conversation history between
    such a pair leaves the kept side orphaned, and every major provider
    rejects ``tool_result`` without a matching ``tool_use``. Used by
    session compaction to advance the cut-point past dangling halves.

    Handles both the OpenAI shape (assistant message with non-empty
    ``tool_calls``) and the Anthropic shape (assistant message whose
    ``content`` list contains a ``tool_use`` or ``tool_result`` block).
    """
    if not isinstance(msg, dict):
        return False
    if msg.get("role") == "tool":
        return True
    if msg.get("role") == "assistant":
        # OpenAI-style: tool_calls is non-empty.
        if msg.get("tool_calls"):
            return True
        # Anthropic-style: content list contains a tool_use block.
        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(c, dict) and c.get("type") in ("tool_use", "tool_result") for c in content
        ):
            return True
    return False


def _sanitize_outbound(text: str) -> str:
    """Redact absolute paths and Python tracebacks before sending to Telegram.

    Defence-in-depth: even when the agent loop raises and something
    serialises an exception into the reply, the chat must not see the
    operator's home-directory layout or Python frames. Pattern-based: any
    ``/Users/...``, ``/home/...``, ``/private/...``, ``/var/...``, ``/tmp/...``,
    or ``~/.xerxes...`` token becomes ``[path redacted]`` and any
    ``Traceback ...`` block collapses to ``[traceback redacted]``.

    Args:
        text: Final reply text on its way to the channel.

    Returns:
        The cleaned text. Empty input is returned unchanged.
    """
    if not text:
        return text
    cleaned = _TRACEBACK_RE.sub("[traceback redacted]", text)
    cleaned = _PATH_REDACT_RE.sub("[path redacted]", cleaned)
    return cleaned


@dataclass
class TelegramGatewayConfig:
    """Runtime configuration for ``TelegramAgentGateway``.

    Attributes:
        token: Telegram bot token.
        host: Bind address for the FastAPI server in webhook mode.
        port: Bind port for the FastAPI server in webhook mode.
        webhook_path: HTTP path Telegram is told to POST to.
        webhook_url: Public webhook URL registered with ``setWebhook``.
            Empty triggers polling mode under ``transport='auto'``.
        workspace: Override for the Markdown workspace directory.
        project_dir: ``cwd`` for the agent. Defaults to the current dir.
        model: LLM model id; required when no active profile is configured.
        base_url: LLM provider base URL when no profile is selected.
        api_key: LLM provider API key when no profile is selected.
        permission_mode: Sandbox/permission mode forwarded to the runtime.
        transport: ``"polling"``, ``"webhook"``, or ``"auto"`` (uses webhook
            when ``webhook_url`` is set, otherwise polling).
        polling_timeout: Long-poll timeout in seconds.
        stream_previews: When ``True``, send a placeholder and progressively
            edit it as the agent streams.
        preview_interval: Minimum seconds between preview edits.
        session_keep_messages: Approximate ceiling on retained messages per
            session before compaction kicks in.
        allowed_user_ids: Comma-separated Telegram user ids permitted to
            drive the agent. Empty is fail-closed (every inbound refused).
        allowed_usernames: Comma-separated ``@usernames`` (no ``@``)
            allowed alongside the ids.
        webhook_secret_token: Secret echoed by Telegram in the
            ``X-Telegram-Bot-Api-Secret-Token`` header on every webhook.
            Empty disables the check (only safe inside a private network).
        max_payload_bytes: Hard cap on inbound webhook body size.
        bot_username: Bot's ``@username`` without the ``@``. Used for
            exact-mention matching in groups instead of substring search.
        ignore_edited_messages: When ``True`` (default), ``edited_message``
            updates are dropped to defeat replay-by-edit.
    """

    token: str
    host: str = "127.0.0.1"
    port: int = 11997
    webhook_path: str = "/telegram/webhook"
    webhook_url: str = ""
    workspace: str = ""
    project_dir: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    permission_mode: str = "accept-all"
    transport: str = "polling"
    polling_timeout: int = 30
    stream_previews: bool = True
    preview_interval: float = 1.0
    session_keep_messages: int = 40
    # Comma-separated Telegram user_ids permitted to drive the agent.
    # Empty => fail-closed: every inbound is refused. Polling mode bypasses
    # webhook auth, so the allowlist is the only line of defence then.
    allowed_user_ids: str = ""
    # Comma-separated Telegram usernames (no @) permitted alongside ids.
    allowed_usernames: str = ""
    # Telegram echoes this back in X-Telegram-Bot-Api-Secret-Token on every
    # webhook request when set on setWebhook. Empty disables the check (only
    # safe in polling mode).
    webhook_secret_token: str = ""
    # Hard cap on inbound webhook body. Real Telegram updates are <100KB.
    max_payload_bytes: int = 256 * 1024
    # Bot's @username (no leading @). Used for exact-mention matching in
    # groups instead of "xerxes in text" substring.
    bot_username: str = ""
    # If True, edited_message updates are ignored (default). Prevents replay-
    # by-edit triggering repeated agent runs.
    ignore_edited_messages: bool = True


class TelegramAgentGateway:
    """Drive a Xerxes agent loop from Telegram webhooks or long polling.

    Owns a ``TelegramChannel`` adapter, a FastAPI app for webhook mode, a
    background worker pool for synchronous agent calls, and a per-session
    asyncio lock map so each chat sees ordered, non-overlapping turns.
    Constructing the gateway does not start anything; call ``run`` for the
    blocking variant or ``start``/``stop``/``run_polling`` for embedded use.
    """

    def __init__(self, config: TelegramGatewayConfig, *, submit_fn: SubmitFn | None = None) -> None:
        """Build the gateway.

        Args:
            config: Gateway and runtime configuration.
            submit_fn: Optional synchronous hook returning the final reply
                text for a given ``ChannelMessage``. Bypasses the real agent
                loop and is used exclusively by tests.

        Raises:
            ValueError: ``config.token`` is empty.
        """
        if not config.token:
            raise ValueError("Telegram bot token is required")
        self.config = config
        self.workspace = MarkdownAgentWorkspace(config.workspace or None)
        self.channel = TelegramChannel(config.token, accept_edited_messages=not config.ignore_edited_messages)
        self.app = FastAPI(title="Xerxes Telegram Gateway")
        self._submit_fn = submit_fn
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._runtime_config: dict[str, Any] = {}
        self._base_system_prompt = ""
        self._tool_executor: Any = None
        self._tool_schemas: list[dict[str, Any]] = []
        self._sessions: dict[str, AgentState] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._polling_offset: int | None = None
        self._shutdown = False
        self._allowed_user_ids = _parse_csv_set(config.allowed_user_ids)
        self._allowed_usernames = {u.lstrip("@").lower() for u in _parse_csv_set(config.allowed_usernames)}
        self._bot_username = (config.bot_username or "").lstrip("@").lower()
        # Per-session in-flight counters. We refuse new turns when the queue
        # for a single chat goes past _MAX_QUEUED_PER_CHAT so a flooder cannot
        # grow the lock queue indefinitely and OOM the gateway.
        self._inflight: dict[str, int] = {}
        self._register_routes()

    def _register_routes(self) -> None:
        """Attach the ``/health`` and webhook endpoints to the FastAPI app."""

        @self.app.get("/health")
        async def health() -> dict[str, Any]:
            """Return a tiny liveness payload (workspace + active model)."""
            return {"ok": True, "workspace": str(self.workspace.path), "model": self._runtime_config.get("model", "")}

        @self.app.post(self.config.webhook_path)
        async def telegram_webhook(request: Request) -> PlainTextResponse:
            """Receive a Telegram webhook, authenticate it, and dispatch.

            Enforces (1) a content-length cap to defeat OOM POSTs, (2) the
            ``X-Telegram-Bot-Api-Secret-Token`` echo so only Telegram (or
            something holding the configured secret) can reach this endpoint,
            and (3) a post-read cap because attackers can lie about
            content-length.
            """
            # 1) Hard cap on inbound size — real Telegram updates are well under
            # 100KB. Without this an attacker can OOM the gateway with a giant
            # POST.
            cl = request.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > self.config.max_payload_bytes:
                return PlainTextResponse("payload too large", status_code=413)

            # 2) Webhook authenticity: Telegram echoes the secret set on
            # setWebhook in X-Telegram-Bot-Api-Secret-Token. Reject anything
            # without it. Operators who skip the secret (only safe inside a
            # private network) explicitly set webhook_secret_token="".
            expected = self.config.webhook_secret_token
            if expected:
                got = request.headers.get("x-telegram-bot-api-secret-token", "")
                if not _const_time_eq(got, expected):
                    return PlainTextResponse("unauthorized", status_code=401)

            body = await request.body()
            if len(body) > self.config.max_payload_bytes:
                return PlainTextResponse("payload too large", status_code=413)
            headers = dict(request.headers)
            response = await self.channel.handle_webhook(headers, body)
            return PlainTextResponse(response.body, status_code=response.status)

    async def start(self) -> None:
        """Bootstrap the runtime, start the channel, and register the webhook if needed.

        In webhook mode this calls Telegram's ``setWebhook`` so updates start
        flowing; in polling mode it is a no-op for the network side and the
        caller is expected to drive ``run_polling`` separately.
        """
        self._bootstrap()
        await self.channel.start(self._handle_inbound)
        if self._transport_mode() == "webhook" and self.config.webhook_url:
            await self._set_telegram_webhook(self.config.webhook_url)

    async def stop(self) -> None:
        """Stop the channel adapter and tear down the worker pool."""
        await self.channel.stop()
        self._pool.shutdown(wait=False)

    async def _handle_inbound(self, message: ChannelMessage) -> None:
        """Run one Xerxes turn for an inbound Telegram message.

        Performs all the per-turn safety work in order: empty-text drop,
        group addressing check, sender allowlist, prompt-injection scrub
        and fenced journalling of the user message, per-chat queue cap,
        snapshot-then-rollback around the agent run, outbound sanitisation,
        and journalling of the final reply.
        """
        if not message.text.strip():
            return
        if self._is_group_message_not_for_xerxes(message):
            return
        if not self._is_sender_allowed(message):
            # Fail-closed silently to avoid confirming bot identity to scanners.
            # Operator can grep daemon logs to see refusal counts.
            return

        # Quote-escape user text before journaling so a malicious message
        # cannot inject markdown headings / instructions that the workspace
        # loader later treats as agent guidance. We also redact known
        # prompt-injection patterns.
        safe_text = _quote_user_block(_scan_inbound(message.text))
        self.workspace.append_daily_note(
            f"[telegram:{message.room_id or message.channel_user_id}] user {message.channel_user_id}:\n{safe_text}"
        )

        session_key = self._session_key(message)
        # Shed-load: if the per-chat queue is already at capacity, drop the
        # new turn rather than letting it pile up behind the lock. A single
        # flooded chat must not be able to monopolise gateway memory.
        if self._inflight.get(session_key, 0) >= _MAX_QUEUED_PER_CHAT:
            logger.warning("dropping inbound for %s — queue full", session_key)
            return
        self._inflight[session_key] = self._inflight.get(session_key, 0) + 1
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        try:
            async with lock:
                # Snapshot AgentState.messages so we can roll back on a turn
                # that raises mid-stream. Without this, a failed run leaves
                # the conversation half-mutated and the next turn sees
                # nonsense like an orphan tool_call with no tool_result.
                state_snapshot = list(self._sessions.get(session_key, AgentState()).messages)
                try:
                    result = await self._run_message_with_preview(message, session_key)
                except Exception:
                    # Never echo internal paths / stack frames back to the chat.
                    # The operator inspects daemon logs for the real cause.
                    logger.warning("agent turn failed on session %s", session_key, exc_info=True)
                    state = self._sessions.get(session_key)
                    if state is not None:
                        state.messages = state_snapshot
                    result = "Sorry, that request failed. Check the daemon log for details."
                result = _sanitize_outbound(result).strip() or "(no response)"

                self.workspace.append_daily_note(
                    f"[telegram:{message.room_id or message.channel_user_id}] xerxes: {result[:500]}"
                )
        finally:
            # Always decrement so a stuck queue gradually drains.
            remaining = self._inflight.get(session_key, 1) - 1
            if remaining <= 0:
                self._inflight.pop(session_key, None)
            else:
                self._inflight[session_key] = remaining

    async def _run_message_with_preview(self, message: ChannelMessage, session_key: str) -> str:
        """Run a turn, streaming progress into an edited Telegram preview message.

        When ``stream_previews`` is disabled this falls through to a single
        final reply; otherwise a placeholder ``"..."`` is posted, the agent
        loop runs in the worker pool, ``on_progress`` chunks land in an
        asyncio queue, and ``_preview_editor`` rate-limits ``editMessageText``
        calls until the run finishes.
        """
        if not self.config.stream_previews:
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(self._pool, self._submit_message, message, session_key, None)
            result = _sanitize_outbound(raw)
            await self._send_final_reply(message, result)
            return result

        chat_id = message.room_id or message.channel_user_id or ""
        preview_response = await self.channel.send_text(
            chat_id=chat_id, text="...", reply_to=message.platform_message_id
        )
        preview_message_id = self._extract_message_id(preview_response)
        progress_queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_progress(text: str) -> None:
            """Push a streaming text chunk from the worker thread to the asyncio queue."""
            loop.call_soon_threadsafe(progress_queue.put_nowait, text)

        preview_task = asyncio.create_task(self._preview_editor(chat_id, preview_message_id, progress_queue))
        try:
            raw = await loop.run_in_executor(self._pool, self._submit_message, message, session_key, on_progress)
        finally:
            await progress_queue.put(None)
            await preview_task

        result = _sanitize_outbound(raw)
        if preview_message_id:
            await self.channel.edit_text(
                chat_id=chat_id, message_id=preview_message_id, text=result.strip() or "(no response)"
            )
        else:
            await self._send_final_reply(message, result)
        return result

    async def _preview_editor(
        self,
        chat_id: str,
        message_id: str,
        progress_queue: asyncio.Queue[str | None],
    ) -> None:
        """Edit the preview message at a bounded cadence as text streams in.

        Drains ``progress_queue`` until it receives the ``None`` sentinel.
        Only issues an ``editMessageText`` when the buffered text actually
        changed, capped at Telegram's 4096-character limit (we keep the tail
        so the most recent output is visible).
        """
        if not message_id:
            while await progress_queue.get() is not None:
                pass
            return

        buffer = ""
        last_sent = ""
        pending = True
        while pending:
            try:
                item = await asyncio.wait_for(progress_queue.get(), timeout=self.config.preview_interval)
            except TimeoutError:
                item = ""
            if item is None:
                pending = False
            elif item:
                buffer += item

            preview = buffer.strip()
            if preview and preview != last_sent:
                last_sent = preview
                await self.channel.edit_text(chat_id=chat_id, message_id=message_id, text=self._telegram_limit(preview))

    async def _send_final_reply(self, message: ChannelMessage, result: str) -> None:
        """Send the final agent reply as a fresh outbound Telegram message."""
        await self.channel.send(
            ChannelMessage(
                text=result.strip() or "(no response)",
                channel="telegram",
                channel_user_id=message.channel_user_id,
                room_id=message.room_id,
                reply_to=message.platform_message_id,
                direction=MessageDirection.OUTBOUND,
            )
        )

    def _submit_message(
        self,
        message: ChannelMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> str:
        """Run one agent turn synchronously and return the assembled text reply.

        Runs in the worker thread; never directly awaited. Builds the system
        prompt from the bootstrap prompt plus a fresh workspace snapshot,
        runs ``streaming.loop.run`` against the per-session ``AgentState``,
        forwards each ``TextChunk`` through ``on_progress``, and finally
        compacts the session before returning.
        """
        if self._submit_fn is not None:
            result = self._submit_fn(message)
            if on_progress:
                on_progress(result)
            return result

        workspace_context = self.workspace.load_context()
        system_prompt = "\n\n".join(
            part for part in (self._base_system_prompt.rstrip(), workspace_context.prompt) if part
        )
        prompt = self._format_channel_prompt(message)
        state = self._sessions.setdefault(session_key or self._session_key(message), AgentState())
        output_parts: list[str] = []

        for event in run_agent_loop(
            user_message=prompt,
            state=state,
            config=dict(self._runtime_config),
            system_prompt=system_prompt,
            tool_executor=self._tool_executor,
            tool_schemas=self._tool_schemas,
        ):
            if isinstance(event, TextChunk):
                output_parts.append(event.text)
                if on_progress:
                    on_progress(event.text)
            elif isinstance(event, (ToolStart, ToolEnd)):
                continue

        self._compact_session_if_needed(session_key or self._session_key(message), state)
        return "".join(output_parts)

    def _bootstrap(self) -> None:
        """One-time runtime setup: profile, model, tools, and base system prompt.

        Honours an active bridge profile if one is set, otherwise falls back
        to the per-config model/base_url/api_key. Refuses to continue when
        no model can be determined so the operator gets an explicit error
        instead of a silent loop with no LLM.

        Raises:
            RuntimeError: No model is configured anywhere.
        """
        if self.config.project_dir:
            os.chdir(Path(self.config.project_dir).expanduser())

        profile = profiles.get_active_profile()
        if profile:
            base_url = profile.get("base_url", "")
            api_key = profile.get("api_key", "")
            model = profile.get("model", "")
        else:
            base_url = self.config.base_url
            api_key = self.config.api_key
            model = self.config.model

        if not model:
            raise RuntimeError("No model configured. Run `xerxes` and configure a provider, or pass --model.")

        self._runtime_config = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "permission_mode": self.config.permission_mode,
        }
        if profile:
            self._runtime_config.update(profile.get("sampling", {}))
        set_global_config(self._runtime_config)

        boot = bootstrap(model=model)
        self._base_system_prompt = boot.system_prompt
        registry = populate_registry()
        self._tool_executor = build_tool_executor(registry=registry)
        self._tool_schemas = registry.tool_schemas()
        self.workspace.ensure()

    @staticmethod
    def _format_channel_prompt(message: ChannelMessage) -> str:
        """Render the user message into the structured prompt the agent sees."""
        username = message.metadata.get("username", "") if message.metadata else ""
        chat_type = message.metadata.get("chat_type", "") if message.metadata else ""
        return (
            "[Telegram message]\n"
            f"chat_id: {message.room_id or ''}\n"
            f"chat_type: {chat_type}\n"
            f"thread_id: {message.metadata.get('thread_id', '') if message.metadata else ''}\n"
            f"from_user_id: {message.channel_user_id or ''}\n"
            f"from_username: {username}\n\n"
            f"{message.text}"
        )

    @staticmethod
    def _session_key(message: ChannelMessage) -> str:
        """Compute the per-conversation lock / state key for a message.

        Groups and supergroups key on ``chat_id`` plus thread; DMs key on the
        sender's user id so each user has an isolated session.
        """
        chat_type = (message.metadata.get("chat_type", "") if message.metadata else "").lower()
        thread_id = message.metadata.get("thread_id", "") if message.metadata else ""
        if chat_type in {"group", "supergroup", "channel"}:
            return f"telegram:chat:{message.room_id or ''}:thread:{thread_id or 'main'}"
        return f"telegram:private:{message.channel_user_id or message.room_id or ''}"

    @staticmethod
    def _extract_message_id(response: dict[str, Any]) -> str:
        """Pull ``result.message_id`` out of a Telegram Bot API response."""
        result = response.get("result") if isinstance(response, dict) else None
        if isinstance(result, dict):
            raw = result.get("message_id", "")
            return str(raw) if raw else ""
        return ""

    @staticmethod
    def _telegram_limit(text: str) -> str:
        """Trim text to Telegram's 4096-character per-message limit (keeping the tail)."""
        return text if len(text) <= 4096 else text[-4096:]

    def _compact_session_if_needed(self, session_key: str, state: AgentState) -> None:
        """Drop oldest messages once the session exceeds ``session_keep_messages``.

        The cut-point is advanced past any orphan tool-call / tool-result
        pair so the resulting truncated history cannot violate the
        provider's tool-use invariants. A note is journalled to the daily
        memory so the agent has hints about durable content that may have
        scrolled out of the live window.
        """
        keep = max(4, self.config.session_keep_messages)
        if len(state.messages) <= keep:
            return
        # Snap the cut-point onto an "assistant" or "user" boundary so we
        # never keep a tool_result whose matching assistant tool_call was
        # just dropped — that combination crashes most providers ("tool
        # result without tool call"). We extend forward (drop a few more)
        # rather than backward, since older messages are the ones we wanted
        # gone anyway.
        cut = len(state.messages) - keep
        while cut < len(state.messages) and _is_orphan_continuation(state.messages[cut]):
            cut += 1
        removed = state.messages[:cut]
        if not removed:
            return
        self.workspace.append_daily_note(
            f"[session:{session_key}] compacted {len(removed)} old messages. "
            f"Recent context remains in live session; durable details should be kept in MEMORY.md."
        )
        state.messages = state.messages[cut:]

    def _is_group_message_not_for_xerxes(self, message: ChannelMessage) -> bool:
        """Return ``True`` for group messages that do not address the bot.

        DMs are always treated as addressed. In groups the message must
        either ``@``-mention the configured ``bot_username``, start with
        a ``/xerxes`` (or ``/<bot>``) command, or be the bare ``/xerxes``.
        Substring matches on ``"xerxes"`` are no longer accepted — they
        produced drive-by triggers from offhand chatter.
        """
        chat_type = (message.metadata.get("chat_type", "") if message.metadata else "").lower()
        if chat_type not in {"group", "supergroup"}:
            return False
        text = message.text.strip()
        text_lower = text.lower()
        # Exact mention with the configured bot username, or a /xerxes-prefixed
        # bot command at the start of the message. Substring "xerxes" anywhere
        # in chatter is no longer enough — that produced drive-by triggers
        # whenever someone said "xerxes is buggy" in the group.
        bot = self._bot_username
        if bot:
            if f"@{bot}" in text_lower:
                return False
            if text_lower.startswith(f"/xerxes@{bot}") or text_lower.startswith(f"/{bot}"):
                return False
        if text_lower.startswith("/xerxes ") or text_lower == "/xerxes":
            return False
        return True

    def _is_sender_allowed(self, message: ChannelMessage) -> bool:
        """Check the sender against the allowlists for both DMs and groups.

        Empty allowlists are deliberately fail-closed: the operator must
        opt people in by listing either Telegram user ids or @usernames.
        Returning ``False`` is silent — the gateway drops the message
        without acknowledgement so scanners cannot confirm the bot exists.
        """
        if not self._allowed_user_ids and not self._allowed_usernames:
            return False
        user_id = (message.channel_user_id or "").strip()
        if user_id and user_id in self._allowed_user_ids:
            return True
        username = (message.metadata or {}).get("username", "")
        return bool(username) and username.lstrip("@").lower() in self._allowed_usernames

    async def _set_telegram_webhook(self, webhook_url: str) -> None:
        """Register ``webhook_url`` with Telegram, attaching the secret token.

        Telegram echoes the secret back in ``X-Telegram-Bot-Api-Secret-Token``
        on every webhook delivery; that header is what the FastAPI route
        authenticates against in ``telegram_webhook``.

        Raises:
            RuntimeError: The ``setWebhook`` call failed (non-2xx response).
        """
        try:
            import httpx

            payload: dict[str, Any] = {"url": webhook_url}
            if self.config.webhook_secret_token:
                # Telegram echoes this back to us in
                # X-Telegram-Bot-Api-Secret-Token on every webhook delivery,
                # which is how we authenticate inbound requests in
                # telegram_webhook().
                payload["secret_token"] = self.config.webhook_secret_token
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{self.config.token}/setWebhook",
                    json=payload,
                )
                response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Failed to set Telegram webhook: {exc}") from exc

    async def _delete_telegram_webhook(self) -> None:
        """Best-effort ``deleteWebhook`` so polling mode can take over cleanly.

        Errors are swallowed because the only legitimate failure mode here
        is "no webhook was registered", which is benign.
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(f"https://api.telegram.org/bot{self.config.token}/deleteWebhook")
                response.raise_for_status()
        except Exception:
            pass

    async def run_polling(self) -> None:
        """Drive the gateway through Bot API long polling until shutdown.

        Clears any existing webhook registration so Telegram does not split
        delivery between webhook and polling, then loops ``_poll_once``
        until ``_shutdown`` is set (typically by ``KeyboardInterrupt`` in
        ``run``). Stops the channel cleanly in the ``finally`` clause.
        """
        await self.start()
        await self._delete_telegram_webhook()
        try:
            while not self._shutdown:
                await self._poll_once()
        finally:
            await self.stop()

    async def _poll_once(self) -> None:
        """Pull one batch of Telegram updates and dispatch each through the channel."""
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            self._pool,
            lambda: self.channel.get_updates(offset=self._polling_offset, timeout=self.config.polling_timeout),
        )
        for update in response.get("result", []) if isinstance(response, dict) else []:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._polling_offset = update_id + 1
            import json

            await self.channel.handle_webhook({}, json.dumps(update).encode("utf-8"))

    def run(self) -> None:
        """Block on the gateway in whichever transport mode is active.

        In polling mode runs ``asyncio.run(run_polling())`` and translates
        ``KeyboardInterrupt`` into clean shutdown. In webhook mode hands
        the FastAPI app to ``uvicorn.run`` with a lifespan context that
        calls ``start`` / ``stop`` around the server.
        """
        if self._transport_mode() == "polling":
            try:
                asyncio.run(self.run_polling())
            except KeyboardInterrupt:
                self._shutdown = True
            return

        @asynccontextmanager
        async def _lifespan(app: FastAPI):
            """FastAPI lifespan that brackets ``uvicorn.run`` with start/stop."""
            await self.start()
            try:
                yield
            finally:
                await self.stop()

        self.app.router.lifespan_context = _lifespan
        uvicorn.run(self.app, host=self.config.host, port=self.config.port)

    def _transport_mode(self) -> str:
        """Resolve ``"polling"`` vs ``"webhook"`` from the (possibly ``auto``) config."""
        mode = (self.config.transport or "auto").strip().lower()
        if mode == "auto":
            return "webhook" if self.config.webhook_url else "polling"
        return "webhook" if mode == "webhook" else "polling"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser backing ``python -m xerxes.channels.telegram_gateway``."""
    parser = argparse.ArgumentParser(description="Run Xerxes as a Telegram bot webhook gateway.")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""), help="Telegram bot token")
    parser.add_argument("--host", default=os.environ.get("XERXES_TELEGRAM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("XERXES_TELEGRAM_PORT", "11997")))
    parser.add_argument("--webhook-path", default=os.environ.get("XERXES_TELEGRAM_WEBHOOK_PATH", "/telegram/webhook"))
    parser.add_argument("--webhook-url", default=os.environ.get("XERXES_TELEGRAM_WEBHOOK_URL", ""))
    parser.add_argument(
        "--transport",
        choices=("auto", "polling", "webhook"),
        default=os.environ.get("XERXES_TELEGRAM_TRANSPORT", "auto"),
        help="Telegram transport. auto uses webhook when --webhook-url is set, otherwise polling.",
    )
    parser.add_argument(
        "--no-stream-previews",
        action="store_true",
        help="Disable Telegram edit-message streaming previews and send only final replies.",
    )
    parser.add_argument("--workspace", default=os.environ.get("XERXES_AGENT_WORKSPACE", ""))
    parser.add_argument("--project-dir", default=os.environ.get("XERXES_PROJECT_DIR", ""))
    parser.add_argument("--model", default=os.environ.get("XERXES_MODEL", ""))
    parser.add_argument("--base-url", default=os.environ.get("XERXES_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("XERXES_API_KEY", ""))
    parser.add_argument("--permission-mode", default=os.environ.get("XERXES_PERMISSION_MODE", "accept-all"))
    parser.add_argument(
        "--allowed-user-ids",
        default=os.environ.get("XERXES_TELEGRAM_ALLOWED_USER_IDS", ""),
        help="Comma-separated Telegram user_ids permitted to drive the agent. Empty = fail-closed.",
    )
    parser.add_argument(
        "--allowed-usernames",
        default=os.environ.get("XERXES_TELEGRAM_ALLOWED_USERNAMES", ""),
        help="Comma-separated Telegram @usernames (no @) permitted alongside --allowed-user-ids.",
    )
    parser.add_argument(
        "--webhook-secret-token",
        default=os.environ.get("XERXES_TELEGRAM_WEBHOOK_SECRET", ""),
        help="Secret echoed by Telegram in X-Telegram-Bot-Api-Secret-Token. Required for webhook mode.",
    )
    parser.add_argument(
        "--bot-username",
        default=os.environ.get("XERXES_TELEGRAM_BOT_USERNAME", ""),
        help="Bot @username (no @) used for exact-mention matching in group chats.",
    )
    parser.add_argument(
        "--max-payload-bytes",
        type=int,
        default=int(os.environ.get("XERXES_TELEGRAM_MAX_PAYLOAD", str(256 * 1024))),
        help="Hard cap on inbound webhook body size in bytes (default 256KB).",
    )
    parser.add_argument(
        "--allow-edited-messages",
        action="store_true",
        help="Process Telegram edited_message updates. Off by default to prevent replay-by-edit.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TelegramGatewayConfig:
    """Translate parsed CLI arguments into a ``TelegramGatewayConfig``."""
    return TelegramGatewayConfig(
        token=args.token,
        host=args.host,
        port=args.port,
        webhook_path=args.webhook_path,
        webhook_url=args.webhook_url,
        workspace=args.workspace,
        project_dir=args.project_dir,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        permission_mode=args.permission_mode,
        transport=args.transport,
        stream_previews=not args.no_stream_previews,
        allowed_user_ids=args.allowed_user_ids,
        allowed_usernames=args.allowed_usernames,
        webhook_secret_token=args.webhook_secret_token,
        bot_username=args.bot_username,
        max_payload_bytes=args.max_payload_bytes,
        ignore_edited_messages=not args.allow_edited_messages,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse args, build the gateway, and block on ``run``.

    ``KeyboardInterrupt`` is treated as a clean shutdown signal.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.token:
        parser.error("--token or TELEGRAM_BOT_TOKEN is required")
    gateway = TelegramAgentGateway(config_from_args(args))
    try:
        gateway.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])


__all__ = ["TelegramAgentGateway", "TelegramGatewayConfig", "build_arg_parser", "config_from_args", "main"]
