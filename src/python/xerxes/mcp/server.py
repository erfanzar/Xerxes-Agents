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
"""Expose Xerxes as an MCP server for external clients.

Ten tools let MCP clients — Claude Desktop, Claude Code, Cursor, Cline —
drive Xerxes sessions: list and read
conversations, send messages, poll/wait for events, fetch attachments, and
list/answer pending permission requests. The Xerxes daemon remains the
source of truth; :class:`XerxesMcpServer` is a thin facade over its public
surface that depends on a :class:`SessionReader` for in-process session
inspection and a :class:`DaemonBridge` of callables for daemon-mediated
actions. Bindings are pluggable so tests can drive the server in-process,
while :func:`main` wires the real daemon to whichever MCP transport the
upstream ``mcp`` package provides (optional ``[mcp]`` extra).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SessionReader(Protocol):
    """Read-only interface the MCP server needs from a session store."""

    def list_sessions(self, workspace_id: str | None = None) -> list[str]: ...
    def get_session(self, session_id: str) -> Any: ...


@dataclass
class DaemonBridge:
    """Bundle of callables the MCP server uses to act on the daemon.

    Each slot is independently optional so tests can build a bridge with
    only the surface they exercise; missing slots cause the corresponding
    MCP tool to return ``{"error": "..."}``.

    Attributes:
        list_channels: ``() -> list[dict]`` — list connected channels.
        send_message: ``(session_id, text, files=None) -> dict``.
        events_poll: ``(session_id, since_ts) -> list[dict]``.
        events_wait: ``(session_id, since_ts, timeout_sec) -> list[dict]`` (may be async).
        list_pending_permissions: ``() -> list[dict]``.
        respond_permission: ``(permission_id, allow) -> dict``.
        fetch_attachment: ``(attachment_id) -> dict``.
    """

    list_channels: Callable[[], list[dict[str, Any]]] | None = None
    send_message: Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]] | None = None
    events_poll: Callable[..., list[dict[str, Any]]] | None = None
    events_wait: Callable[..., Awaitable[list[dict[str, Any]]] | list[dict[str, Any]]] | None = None
    list_pending_permissions: Callable[[], list[dict[str, Any]]] | None = None
    respond_permission: Callable[..., dict[str, Any]] | None = None
    fetch_attachment: Callable[..., dict[str, Any]] | None = None


def _redact(obj: Any) -> Any:
    """Best-effort scrub of obvious credential fields on outbound payloads.

    Replaces ``api_key`` / ``apikey`` / ``token`` / ``password`` dict values
    with ``"[redacted]"``; recurses into nested dicts and lists.
    """
    if isinstance(obj, dict):
        return {
            k: ("[redacted]" if k.lower() in {"api_key", "apikey", "token", "password"} else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


# ----- the 10 tools as plain callables --------------------------------------


@dataclass
class XerxesMcpServer:
    """In-process implementation of the 10 MCP tools.

    Each public method maps to one entry in :data:`MCP_TOOLS` and returns
    JSON-serialisable Python (transport encoding lives in :func:`main`).
    Construct with a :class:`SessionReader` (read side) and a
    :class:`DaemonBridge` (write side) so tests and the production daemon
    can supply their own bindings.

    Attributes:
        sessions: Session-store reader.
        bridge: Adapter for daemon-mediated actions.
    """

    sessions: SessionReader
    bridge: DaemonBridge

    # 1. conversations_list
    def conversations_list(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        """Summarise every known session (optionally filtered to ``workspace_id``)."""
        ids = self.sessions.list_sessions(workspace_id)
        out: list[dict[str, Any]] = []
        for sid in ids:
            sess = self.sessions.get_session(sid)
            if sess is None:
                continue
            out.append(
                {
                    "session_id": getattr(sess, "session_id", sid),
                    "workspace_id": getattr(sess, "workspace_id", None),
                    "agent_id": getattr(sess, "agent_id", None),
                    "created_at": getattr(sess, "created_at", ""),
                    "updated_at": getattr(sess, "updated_at", ""),
                    "turn_count": len(getattr(sess, "turns", []) or []),
                }
            )
        return out

    # 2. conversation_get
    def conversation_get(self, session_id: str) -> dict[str, Any] | None:
        """Return the redacted full session dict for ``session_id`` (or ``None`` if absent)."""
        sess = self.sessions.get_session(session_id)
        if sess is None:
            return None
        return _redact(sess.to_dict() if hasattr(sess, "to_dict") else dict(sess.__dict__))

    # 3. messages_read
    def messages_read(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return the (optionally tail-limited) message history for ``session_id``."""
        sess = self.sessions.get_session(session_id)
        if sess is None:
            return []
        turns = list(getattr(sess, "turns", []) or [])
        if limit is not None:
            turns = turns[-limit:]
        return [_redact(t.to_dict() if hasattr(t, "to_dict") else dict(t.__dict__)) for t in turns]

    # 4. attachments_fetch
    def attachments_fetch(self, attachment_id: str) -> dict[str, Any]:
        """Fetch one attachment by id via the daemon bridge."""
        if self.bridge.fetch_attachment is None:
            return {"error": "attachments_fetch not supported by this daemon"}
        return _redact(self.bridge.fetch_attachment(attachment_id))

    # 5. events_poll
    def events_poll(self, session_id: str, since_ts: str = "") -> list[dict[str, Any]]:
        """Non-blocking peek at wire events newer than ``since_ts``."""
        if self.bridge.events_poll is None:
            return []
        out = self.bridge.events_poll(session_id, since_ts)
        return [_redact(e) for e in out]

    # 6. events_wait (sync wrapper — async transport calls it async-aware)
    def events_wait(self, session_id: str, since_ts: str = "", timeout_sec: float = 30.0) -> list[dict[str, Any]]:
        """Long-poll for new events; falls back to :meth:`events_poll` if the bridge omits it.

        Returns the awaitable verbatim when the bridge supplies an async
        callable, so the transport layer can ``await`` it.
        """
        if self.bridge.events_wait is None:
            return self.events_poll(session_id, since_ts)
        out = self.bridge.events_wait(session_id, since_ts, timeout_sec)
        # If the bridge returned a coroutine, surface that to the caller — the
        # transport layer is responsible for awaiting it.
        if hasattr(out, "__await__"):
            return out  # type: ignore[return-value]
        return [_redact(e) for e in out]  # type: ignore[union-attr]

    # 7. messages_send
    def messages_send(
        self,
        session_id: str,
        text: str,
        *,
        files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Submit a new user message (optionally with file attachments) to ``session_id``."""
        if self.bridge.send_message is None:
            return {"error": "messages_send not supported by this daemon"}
        out = self.bridge.send_message(session_id, text, files=files)
        if hasattr(out, "__await__"):
            return out  # type: ignore[return-value]
        return _redact(out)  # type: ignore[arg-type]

    # 8. permissions_list_open
    def permissions_list_open(self) -> list[dict[str, Any]]:
        """List every pending permission prompt across all sessions."""
        if self.bridge.list_pending_permissions is None:
            return []
        return [_redact(p) for p in self.bridge.list_pending_permissions()]

    # 9. permissions_respond
    def permissions_respond(self, permission_id: str, allow: bool) -> dict[str, Any]:
        """Approve or deny a pending permission request by id."""
        if self.bridge.respond_permission is None:
            return {"error": "permissions_respond not supported by this daemon"}
        return _redact(self.bridge.respond_permission(permission_id, allow))

    # 10. channels_list
    def channels_list(self) -> list[dict[str, Any]]:
        """List currently-connected messaging channels."""
        if self.bridge.list_channels is None:
            return []
        return [_redact(c) for c in self.bridge.list_channels()]


# ----- public tool table ----------------------------------------------------

MCP_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "conversations_list",
        "description": "List all known Xerxes sessions (optionally scoped to a workspace).",
        "input_schema": {
            "type": "object",
            "properties": {"workspace_id": {"type": "string"}},
        },
    },
    {
        "name": "conversation_get",
        "description": "Fetch metadata + transitions for one session by id.",
        "input_schema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {"session_id": {"type": "string"}},
        },
    },
    {
        "name": "messages_read",
        "description": "Read the message history of a session (most recent N turns).",
        "input_schema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {"session_id": {"type": "string"}, "limit": {"type": "integer"}},
        },
    },
    {
        "name": "attachments_fetch",
        "description": "Download an attachment by id.",
        "input_schema": {
            "type": "object",
            "required": ["attachment_id"],
            "properties": {"attachment_id": {"type": "string"}},
        },
    },
    {
        "name": "events_poll",
        "description": "Non-blocking poll for new wire events on a session.",
        "input_schema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {"session_id": {"type": "string"}, "since_ts": {"type": "string"}},
        },
    },
    {
        "name": "events_wait",
        "description": "Long-poll for events on a session, with a timeout.",
        "input_schema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string"},
                "since_ts": {"type": "string"},
                "timeout_sec": {"type": "number"},
            },
        },
    },
    {
        "name": "messages_send",
        "description": "Send a message into a Xerxes session (resumes a turn).",
        "input_schema": {
            "type": "object",
            "required": ["session_id", "text"],
            "properties": {
                "session_id": {"type": "string"},
                "text": {"type": "string"},
                "files": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "permissions_list_open",
        "description": "List pending permission requests across all sessions.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "permissions_respond",
        "description": "Approve or deny a pending permission request by id.",
        "input_schema": {
            "type": "object",
            "required": ["permission_id", "allow"],
            "properties": {"permission_id": {"type": "string"}, "allow": {"type": "boolean"}},
        },
    },
    {
        "name": "channels_list",
        "description": "List currently connected messaging channels.",
        "input_schema": {"type": "object", "properties": {}},
    },
)


def main() -> None:
    """Wire :class:`XerxesMcpServer` to the upstream ``mcp`` stdio transport.

    Falls back to a stub stdio loop when the optional ``[mcp]`` extra isn't
    installed (the caller gets a helpful error on stderr).
    """
    try:
        import mcp.server.stdio  # type: ignore
        import mcp.types as mcp_types  # type: ignore  # noqa: F401
        from mcp.server import Server  # type: ignore
    except ImportError:
        _fallback_stdio_loop()
        return

    from xerxes.runtime.session import open_default_session_manager  # type: ignore[attr-defined]

    server: Server = Server("xerxes")  # type: ignore[no-untyped-call]
    bridge = DaemonBridge()
    sessions = open_default_session_manager()  # may raise; the user will get a clear error
    impl = XerxesMcpServer(sessions=sessions, bridge=bridge)

    @server.list_tools()  # type: ignore[misc]
    async def list_tools():  # type: ignore[no-untyped-def]
        return [
            {"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]} for t in MCP_TOOLS
        ]

    @server.call_tool()  # type: ignore[misc]
    async def call_tool(name: str, arguments: dict[str, Any]):  # type: ignore[no-untyped-def]
        method = getattr(impl, name, None)
        if not callable(method):
            return [{"type": "text", "text": json.dumps({"error": f"unknown tool: {name}"})}]
        result = method(**(arguments or {}))
        if hasattr(result, "__await__"):
            result = await result
        return [{"type": "text", "text": json.dumps(result, default=str)}]

    import asyncio

    asyncio.run(mcp.server.stdio.run_stdio(server))


def _fallback_stdio_loop() -> None:
    """Stand-in for the ``[mcp]`` extra; prints an install hint and exits."""

    import sys

    print(json.dumps({"error": "Install xerxes-agent[mcp] for full MCP transport"}), file=sys.stderr)


__all__ = ["MCP_TOOLS", "DaemonBridge", "SessionReader", "XerxesMcpServer", "main"]
