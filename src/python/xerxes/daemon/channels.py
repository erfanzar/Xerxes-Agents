# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Lifecycle for inbound messaging-channel adapters.

The daemon loads channel adapters declared under ``config.channels`` (Telegram,
Slack, Discord, Email/IMAP, Matrix, ...), constructs each one by reflecting
its ``__init__`` signature against the saved ``settings`` block, and runs
both poll-based ingestion (Telegram long-polling) and a FastAPI webhook
server for push-based providers. Inbound messages are handed to
:func:`DaemonServer._handle_channel_message` via the shared
:data:`xerxes.channels.base.InboundHandler` protocol.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from ..channels.adapters import (
    BlueBubblesChannel,
    DingTalkChannel,
    DiscordChannel,
    EmailChannel,
    FeishuChannel,
    HomeAssistantChannel,
    MatrixChannel,
    MattermostChannel,
    SignalChannel,
    SlackChannel,
    TelegramChannel,
    TwilioSMSChannel,
    WeComChannel,
    WhatsAppChannel,
)
from ..channels.base import Channel, InboundHandler
from .config import DaemonConfig

CHANNEL_CLASSES: dict[str, type[Channel]] = {
    "bluebubbles": BlueBubblesChannel,
    "dingtalk": DingTalkChannel,
    "discord": DiscordChannel,
    "email": EmailChannel,
    "email_imap": EmailChannel,
    "feishu": FeishuChannel,
    "home_assistant": HomeAssistantChannel,
    "matrix": MatrixChannel,
    "mattermost": MattermostChannel,
    "signal": SignalChannel,
    "slack": SlackChannel,
    "sms": TwilioSMSChannel,
    "telegram": TelegramChannel,
    "twilio_sms": TwilioSMSChannel,
    "wecom": WeComChannel,
    "whatsapp": WhatsAppChannel,
}


@dataclass
class LoadedChannel:
    """One declared channel adapter, post-construction.

    Attributes:
        name: Logical name (config key).
        type: Adapter type id used to look up the class.
        enabled: Whether the adapter should be started.
        settings: Resolved adapter-specific settings.
        instance: Live :class:`Channel` instance, or ``None`` if disabled/failed.
        error: Construction or runtime error message, or ``""``.
    """

    name: str
    type: str
    enabled: bool
    settings: dict[str, Any]
    instance: Channel | None = None
    error: str = ""

    def status(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot for ``channel.list``."""
        return {
            "name": self.name,
            "type": self.type,
            "enabled": self.enabled,
            "running": self.instance is not None and not self.error,
            "error": self.error,
        }


def _build_channel(channel_type: str, settings: dict[str, Any]) -> Channel:
    """Instantiate a channel by reflecting its ``__init__`` against ``settings``.

    Required positional parameters absent from ``settings`` raise
    ``ValueError`` so the daemon can surface a friendly error instead of
    failing at use time.
    """
    cls = CHANNEL_CLASSES.get(channel_type)
    if cls is None:
        raise ValueError(f"Unknown channel type: {channel_type}")

    signature = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}
    required: list[str] = []
    accepts_kwargs = False
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY and name in {"http_client", "session"}:
            continue
        if name in settings:
            kwargs[name] = settings[name]
        elif param.default is inspect.Parameter.empty:
            required.append(name)

    if accepts_kwargs:
        for key, value in settings.items():
            kwargs.setdefault(key, value)

    missing = [name for name in required if not kwargs.get(name)]
    if missing:
        raise ValueError(f"Missing settings for {channel_type}: {', '.join(missing)}")
    return cls(**kwargs)


class ChannelManager:
    """Construct, start, and stop the configured channel adapters."""

    def __init__(self, config: DaemonConfig) -> None:
        """Configure the manager; nothing is constructed until :meth:`load`."""
        self.config = config
        self.channels: dict[str, LoadedChannel] = {}
        self._polling_tasks: set[asyncio.Task[Any]] = set()

    def load(self) -> None:
        """Build :class:`LoadedChannel` records for every declared channel.

        Disabled channels are recorded with ``instance=None``; construction
        errors are captured into ``LoadedChannel.error`` so the daemon can
        keep running and surface the failure via ``channel.list``.
        """
        self.channels.clear()
        for name, raw in self.config.resolved_channels().items():
            channel_type = str(raw.get("type", name))
            enabled = bool(raw.get("enabled", False))
            settings = dict(raw.get("settings", {}))
            loaded = LoadedChannel(name=name, type=channel_type, enabled=enabled, settings=settings)
            if enabled:
                try:
                    loaded.instance = _build_channel(channel_type, settings)
                except Exception as exc:
                    loaded.error = str(exc)
            self.channels[name] = loaded

    async def start_all(self, handler: InboundHandler) -> None:
        """Start every healthy channel and spawn polling tasks where needed."""
        for channel in self.channels.values():
            if channel.instance is None or channel.error:
                continue
            try:
                await channel.instance.start(handler)
                if channel.type == "telegram" and self._telegram_uses_polling(channel):
                    task = asyncio.create_task(self._poll_telegram(channel))
                    self._polling_tasks.add(task)
                    task.add_done_callback(self._polling_tasks.discard)
            except Exception as exc:
                channel.error = str(exc)

    async def stop_all(self) -> None:
        """Cancel polling tasks and stop every adapter, swallowing teardown errors."""
        for task in list(self._polling_tasks):
            task.cancel()
        if self._polling_tasks:
            await asyncio.gather(*self._polling_tasks, return_exceptions=True)
        for channel in self.channels.values():
            if channel.instance is None:
                continue
            try:
                await channel.instance.stop()
            except Exception:
                pass

    def list(self) -> list[dict[str, Any]]:
        """Return :meth:`LoadedChannel.status` for every declared channel."""
        return [channel.status() for channel in self.channels.values()]

    def enable(self, name: str) -> bool:
        """Mark ``name`` enabled in both the live record and the config (no restart)."""
        if name not in self.channels:
            return False
        self.channels[name].enabled = True
        self.config.channels.setdefault(name, {})["enabled"] = True
        return True

    def disable(self, name: str) -> bool:
        """Mark ``name`` disabled in both the live record and the config."""
        if name not in self.channels:
            return False
        self.channels[name].enabled = False
        self.config.channels.setdefault(name, {})["enabled"] = False
        return True

    @staticmethod
    def _telegram_uses_polling(channel: LoadedChannel) -> bool:
        """True when the Telegram adapter should run long-polling instead of webhooks."""
        mode = str(channel.settings.get("transport", "auto")).lower()
        return mode == "polling" or (mode == "auto" and not channel.settings.get("webhook_url"))

    async def _poll_telegram(self, channel: LoadedChannel) -> None:
        """Long-poll a Telegram channel via ``get_updates`` and replay each update through ``handle_webhook``."""
        instance = channel.instance
        if instance is None or not hasattr(instance, "get_updates") or not hasattr(instance, "handle_webhook"):
            return
        offset: int | None = None
        timeout = int(channel.settings.get("polling_timeout", 30) or 30)
        while channel.enabled:
            try:
                response = await asyncio.to_thread(instance.get_updates, offset=offset, timeout=timeout)
                for update in response.get("result", []) if isinstance(response, dict) else []:
                    update_id = update.get("update_id")
                    if isinstance(update_id, int):
                        offset = update_id + 1
                    await instance.handle_webhook({}, json.dumps(update).encode("utf-8"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                channel.error = str(exc)
                await asyncio.sleep(2)


class ChannelWebhookServer:
    """FastAPI webhook receiver for push-based channel providers."""

    def __init__(self, manager: ChannelManager, *, host: str, port: int) -> None:
        """Build the FastAPI app and register routes; serving starts in :meth:`start`."""
        self.manager = manager
        self.host = host
        self.port = port
        self.app = FastAPI(title="Xerxes Daemon Channels")
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[Any] | None = None
        self._register_routes()

    def _register_routes(self) -> None:
        """Mount ``/channels`` (status) and ``/channels/{name}/webhook`` (delivery)."""

        @self.app.get("/channels")
        async def channel_list() -> dict[str, Any]:
            return {"ok": True, "channels": self.manager.list()}

        @self.app.post("/channels/{name}/webhook")
        async def channel_webhook(name: str, request: Request) -> PlainTextResponse:
            loaded = self.manager.channels.get(name)
            if loaded is None or loaded.instance is None:
                return PlainTextResponse(f"unknown channel {name!r}", status_code=404)
            if not hasattr(loaded.instance, "handle_webhook"):
                return PlainTextResponse("channel does not support webhooks", status_code=400)
            response = await loaded.instance.handle_webhook(dict(request.headers), await request.body())
            return PlainTextResponse(response.body, status_code=response.status, headers=response.headers or None)

    async def start(self) -> None:
        """Launch the uvicorn server on a background task (no-op if already running)."""
        if self._task is not None:
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning", loop="asyncio")
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        await asyncio.sleep(0)

    async def stop(self) -> None:
        """Ask uvicorn to exit and await the serving task."""
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        self._server = None
        self._task = None


__all__ = ["CHANNEL_CLASSES", "ChannelManager", "ChannelWebhookServer", "LoadedChannel"]
