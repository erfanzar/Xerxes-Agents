# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Configuration loader for the Xerxes daemon.

The daemon reads ``$XERXES_HOME/daemon/config.json`` (a nested document with
``runtime``, ``control``, ``workspace``, and ``channels`` blocks), then layers
``XERXES_DAEMON_*`` environment overrides on top. ``DaemonConfig`` exposes the
nested shape but keeps legacy flat attribute names (``ws_host``, ``socket_path``,
etc.) as properties so callers and tests untouched by the migration keep
working.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from xerxes.core.paths import xerxes_subdir

DAEMON_DIR = xerxes_subdir("daemon")
CONFIG_FILE = DAEMON_DIR / "config.json"


def _env_value(value: Any) -> Any:
    """Resolve an ``env:VAR`` indirection to the env var's current value."""
    if isinstance(value, str) and value.startswith("env:"):
        return os.environ.get(value[4:], "")
    return value


def _resolve_env_refs(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``*_env`` keys and ``env:VAR`` references into concrete values.

    A key ending in ``_env`` is replaced by the named environment variable's
    value, stored under the trimmed key. Other values run through
    :func:`_env_value` so plain ``env:VAR`` strings expand too.
    """
    resolved: dict[str, Any] = {}
    for key, value in settings.items():
        if key.endswith("_env") and isinstance(value, str):
            resolved[key[:-4]] = os.environ.get(value, "")
        else:
            resolved[key] = _env_value(value)
    return resolved


@dataclass
class DaemonConfig:
    """Nested daemon configuration with legacy flat-attribute access.

    The on-disk JSON is grouped into ``runtime`` (model/provider settings),
    ``control`` (sockets, ports, log directory, auth), ``workspace`` (agent
    workspaces root and defaults), and ``channels`` (messaging adapter
    settings). Properties expose the most-used keys at the top level so older
    service helpers and tests don't need to know about the nested shape.

    Attributes:
        runtime: LLM provider settings — ``model``, ``base_url``, ``api_key``,
            ``permission_mode``, sampling params.
        control: Listening surfaces — websocket host/port, Unix socket path,
            PID file, log directory, optional bearer token.
        workspace: Markdown workspace settings — ``root`` directory and
            ``default_agent_id``.
        channels: Messaging-channel adapters keyed by name. Each entry holds
            ``type``, ``enabled``, and provider-specific ``settings``.
        project_dir: Working directory the daemon ``cd``s into on launch.
        max_concurrent_turns: Worker count for the turn thread pool.
    """

    runtime: dict[str, Any] = field(default_factory=dict)
    control: dict[str, Any] = field(default_factory=dict)
    workspace: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    project_dir: str = ""
    max_concurrent_turns: int = 8

    @property
    def ws_host(self) -> str:
        return str(self.control.get("websocket_host", self.control.get("host", "127.0.0.1")))

    @ws_host.setter
    def ws_host(self, value: str) -> None:
        self.control["websocket_host"] = value

    @property
    def ws_port(self) -> int:
        return int(self.control.get("websocket_port", self.control.get("port", 11996)))

    @ws_port.setter
    def ws_port(self, value: int) -> None:
        self.control["websocket_port"] = int(value)

    @property
    def socket_path(self) -> str:
        return str(self.control.get("unix_socket", self.control.get("socket_path", DAEMON_DIR / "xerxes.sock")))

    @socket_path.setter
    def socket_path(self, value: str) -> None:
        self.control["unix_socket"] = value

    @property
    def pid_file(self) -> str:
        return str(self.control.get("pid_file", DAEMON_DIR / "daemon.pid"))

    @pid_file.setter
    def pid_file(self, value: str) -> None:
        self.control["pid_file"] = value

    @property
    def log_dir(self) -> str:
        return str(self.control.get("log_dir", DAEMON_DIR / "logs"))

    @log_dir.setter
    def log_dir(self, value: str) -> None:
        self.control["log_dir"] = value

    @property
    def auth_token(self) -> str:
        return str(self.control.get("auth_token", ""))

    @auth_token.setter
    def auth_token(self, value: str) -> None:
        self.control["auth_token"] = value

    @property
    def model(self) -> str:
        return str(self.runtime.get("model", ""))

    @model.setter
    def model(self, value: str) -> None:
        self.runtime["model"] = value

    @property
    def base_url(self) -> str:
        return str(self.runtime.get("base_url", ""))

    @base_url.setter
    def base_url(self, value: str) -> None:
        self.runtime["base_url"] = value

    @property
    def api_key(self) -> str:
        return str(self.runtime.get("api_key", ""))

    @api_key.setter
    def api_key(self, value: str) -> None:
        self.runtime["api_key"] = value

    @property
    def max_concurrent_tasks(self) -> int:
        return self.max_concurrent_turns

    @max_concurrent_tasks.setter
    def max_concurrent_tasks(self, value: int) -> None:
        self.max_concurrent_turns = int(value)

    def resolved_runtime(self) -> dict[str, Any]:
        """Return a copy of ``runtime`` with env-ref keys expanded."""
        return _resolve_env_refs(dict(self.runtime))

    def resolved_channels(self) -> dict[str, dict[str, Any]]:
        """Return a copy of ``channels`` with each adapter's ``settings`` expanded."""
        resolved: dict[str, dict[str, Any]] = {}
        for name, raw in self.channels.items():
            item = dict(raw or {})
            settings = dict(item.get("settings", {}))
            item["settings"] = _resolve_env_refs(settings)
            resolved[name] = item
        return resolved


def _merge_legacy_keys(cfg: DaemonConfig, data: dict[str, Any]) -> None:
    """Merge a parsed config dict onto ``cfg`` honouring both layouts.

    The new layout's top-level groups (``runtime``, ``control``, ``workspace``,
    ``channels``) are merged dict-wise; older flat keys (``ws_host``,
    ``model``, etc.) are routed through the legacy property setters so they
    land in the right group.
    """
    for key in ("runtime", "control", "workspace", "channels"):
        if isinstance(data.get(key), dict):
            getattr(cfg, key).update(data[key])

    legacy = {
        "ws_host": "ws_host",
        "ws_port": "ws_port",
        "socket_path": "socket_path",
        "pid_file": "pid_file",
        "log_dir": "log_dir",
        "auth_token": "auth_token",
        "model": "model",
        "base_url": "base_url",
        "api_key": "api_key",
        "max_concurrent_tasks": "max_concurrent_tasks",
        "max_concurrent_turns": "max_concurrent_tasks",
        "project_dir": "project_dir",
    }
    for source, target in legacy.items():
        if source in data:
            setattr(cfg, target, data[source])


def load_config(project_dir: str = "") -> DaemonConfig:
    """Build a :class:`DaemonConfig` from defaults, ``config.json``, and env vars.

    Defaults seed the control/workspace blocks, then ``$XERXES_HOME/daemon/config.json``
    (if present) is merged on top via :func:`_merge_legacy_keys`. Finally
    ``XERXES_DAEMON_*`` and ``XERXES_*`` environment variables override
    matching fields. ``XERXES_DAEMON_ENABLE_TELEGRAM`` injects a minimal
    Telegram channel entry pulling the bot token from ``TELEGRAM_BOT_TOKEN``.
    """

    cfg = DaemonConfig(project_dir=project_dir or os.getcwd())
    cfg.ws_host = "127.0.0.1"
    cfg.ws_port = 11996
    cfg.socket_path = str(DAEMON_DIR / "xerxes.sock")
    cfg.pid_file = str(DAEMON_DIR / "daemon.pid")
    cfg.log_dir = str(DAEMON_DIR / "logs")
    cfg.workspace.setdefault("root", str(xerxes_subdir("agents")))
    cfg.workspace.setdefault("default_agent_id", "default")

    if CONFIG_FILE.exists():
        try:
            _merge_legacy_keys(cfg, json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass

    if value := os.environ.get("XERXES_DAEMON_HOST"):
        cfg.ws_host = value
    if value := os.environ.get("XERXES_DAEMON_PORT"):
        cfg.ws_port = int(value)
    if value := os.environ.get("XERXES_DAEMON_SOCKET"):
        cfg.socket_path = value
    if value := os.environ.get("XERXES_DAEMON_TOKEN"):
        cfg.auth_token = value
    if value := os.environ.get("XERXES_MAX_TASKS"):
        cfg.max_concurrent_tasks = int(value)
    if value := os.environ.get("XERXES_MAX_TURNS"):
        cfg.max_concurrent_turns = int(value)

    for env, key in (
        ("XERXES_MODEL", "model"),
        ("XERXES_BASE_URL", "base_url"),
        ("XERXES_API_KEY", "api_key"),
        ("XERXES_PERMISSION_MODE", "permission_mode"),
    ):
        if value := os.environ.get(env):
            cfg.runtime[key] = value

    if os.environ.get("XERXES_DAEMON_ENABLE_TELEGRAM"):
        settings = cfg.channels.setdefault("telegram", {"type": "telegram", "enabled": True, "settings": {}})
        settings["enabled"] = True
        settings.setdefault("type", "telegram")
        settings.setdefault("settings", {}).setdefault("token_env", "TELEGRAM_BOT_TOKEN")

    return cfg


__all__ = ["CONFIG_FILE", "DAEMON_DIR", "DaemonConfig", "load_config"]
