# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Minimal WebSocket transport for the daemon JSON-RPC surface.

Implements just enough of RFC 6455 to upgrade an incoming HTTP/1.1 request,
parse a single frame at a time, and round-trip JSON-RPC 2.0 messages.
``broadcast`` fans an ``event`` notification to every live client without
waiting for acknowledgements; per-connection requests are dispatched through
a caller-supplied :data:`RPCHandler`. Optional Bearer-token auth (from
``Authorization`` header or ``?token=`` query) gates the upgrade handshake.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
from collections.abc import Awaitable, Callable
from typing import Any

EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]
RPCHandler = Callable[[str, dict[str, Any], EmitFn], Awaitable[dict[str, Any]]]


class WebSocketGateway:
    """Tiny RFC 6455 server speaking JSON-RPC 2.0 to remote daemon clients."""

    def __init__(self, host: str, port: int, auth_token: str | None = None) -> None:
        """Configure the gateway; the listener is opened by :meth:`start`."""
        self._host = host
        self._port = port
        self._auth_token = auth_token or None
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._handler: RPCHandler | None = None

    async def start(self, handler: RPCHandler) -> None:
        """Bind the listener and store ``handler`` for per-message dispatch."""
        self._handler = handler
        self._server = await asyncio.start_server(self._handle_connection, self._host, self._port)

    async def stop(self) -> None:
        """Close every live connection and tear down the listener."""
        for writer in list(self._clients):
            try:
                writer.close()
            except Exception:
                pass
        self._clients.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Encode one ``method=event`` frame and write it to every live client.

        Dead writers are dropped silently — there's no retry or backpressure;
        broadcast is fire-and-forget by design.
        """
        msg = json.dumps(
            {"jsonrpc": "2.0", "method": "event", "params": {"type": event_type, "payload": data}}, default=str
        )
        frame = self._encode_ws_frame(msg)
        dead: list[asyncio.StreamWriter] = []
        for writer in self._clients:
            try:
                writer.write(frame)
            except Exception:
                dead.append(writer)
        for writer in dead:
            self._clients.discard(writer)

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Complete the WebSocket upgrade and pump frames through :meth:`_handle_message`."""
        try:
            request = b""
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                request += line
                if line == b"\r\n":
                    break

            decoded = request.decode(errors="replace")
            headers = self._parse_headers(decoded)
            ws_key = headers.get("sec-websocket-key", "")
            if not ws_key:
                writer.close()
                return

            if self._auth_token and not self._is_authorized(decoded, headers):
                writer.write(
                    b'HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer realm="xerxes"\r\nContent-Length: 0\r\n\r\n'
                )
                await writer.drain()
                writer.close()
                return

            writer.write(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {self._ws_accept_key(ws_key)}\r\n"
                    "\r\n"
                ).encode()
            )
            await writer.drain()
            self._clients.add(writer)

            async def emit(event_type: str, payload: dict[str, Any]) -> None:
                self._send_ws(
                    writer,
                    {"jsonrpc": "2.0", "method": "event", "params": {"type": event_type, "payload": payload}},
                )

            while True:
                raw = await self._read_ws_frame(reader)
                if raw is None:
                    break
                await self._handle_message(raw, writer, emit)
        except (TimeoutError, ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            self._clients.discard(writer)
            try:
                writer.close()
            except Exception:
                pass

    async def _handle_message(self, raw: str, writer: asyncio.StreamWriter, emit: EmitFn) -> None:
        """Parse one JSON-RPC frame, dispatch via the handler, and send back the response."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self._send_ws(writer, {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Invalid JSON"}})
            return

        msg_id = msg.get("id")
        method = str(msg.get("method") or msg.get("type") or "")
        params = msg.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if not self._handler:
            result = {"ok": False, "error": "Daemon not ready"}
        else:
            try:
                result = await self._handler(method, params, emit)
            except Exception as exc:
                self._send_ws(writer, {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32000, "message": str(exc)}})
                return
        self._send_ws(writer, {"jsonrpc": "2.0", "id": msg_id, "result": result})

    def _send_ws(self, writer: asyncio.StreamWriter, data: dict[str, Any]) -> None:
        """Best-effort encode + write of a JSON object as one WebSocket text frame."""
        try:
            writer.write(self._encode_ws_frame(json.dumps(data, ensure_ascii=False, default=str)))
        except Exception:
            pass

    @staticmethod
    def _encode_ws_frame(text: str) -> bytes:
        """Build an unmasked single-frame server-to-client text message."""
        payload = text.encode("utf-8")
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", 0x81, length)
        elif length < 65536:
            header = struct.pack("!BBH", 0x81, 126, length)
        else:
            header = struct.pack("!BBQ", 0x81, 127, length)
        return header + payload

    @staticmethod
    async def _read_ws_frame(reader: asyncio.StreamReader) -> str | None:
        """Read one client-to-server frame.

        Returns the decoded payload, an empty string for a ping frame the
        caller should ignore, or ``None`` to signal close/EOF.
        """
        try:
            header = await reader.readexactly(2)
        except (asyncio.IncompleteReadError, ConnectionError):
            return None

        opcode = header[0] & 0x0F
        if opcode == 0x8:
            return None
        if opcode == 0x9:
            return ""

        masked = bool(header[1] & 0x80)
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", await reader.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", await reader.readexactly(8))[0]

        mask_key = await reader.readexactly(4) if masked else b""
        payload = await reader.readexactly(length)
        if masked:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return payload.decode("utf-8", errors="replace")

    def _is_authorized(self, request: str, headers: dict[str, str]) -> bool:
        """Constant-time compare a Bearer token from ``Authorization`` or ``?token=``."""
        import hmac

        expected = self._auth_token or ""
        if not expected:
            return True

        auth_header = headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            if hmac.compare_digest(auth_header[7:].strip(), expected):
                return True

        try:
            request_line = request.split("\r\n", 1)[0]
            parts = request_line.split(" ")
            if len(parts) >= 2 and "?" in parts[1]:
                query = parts[1].split("?", 1)[1]
                for pair in query.split("&"):
                    if "=" not in pair:
                        continue
                    key, value = pair.split("=", 1)
                    if key == "token" and hmac.compare_digest(value, expected):
                        return True
        except (IndexError, ValueError):
            pass
        return False

    @staticmethod
    def _parse_headers(request: str) -> dict[str, str]:
        """Split a raw HTTP request into a lower-cased header dict."""
        headers: dict[str, str] = {}
        for line in request.split("\r\n")[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key.lower()] = value
        return headers

    @staticmethod
    def _ws_accept_key(key: str) -> str:
        """Compute the ``Sec-WebSocket-Accept`` header value for the handshake."""
        import base64

        magic = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        return base64.b64encode(hashlib.sha1(magic.encode()).digest()).decode()


__all__ = ["EmitFn", "RPCHandler", "WebSocketGateway"]
