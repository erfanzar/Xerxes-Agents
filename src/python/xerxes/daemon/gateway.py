# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""WebSocket gateway — central control plane for external clients.

Implements a minimal RFC 6455 WebSocket server using raw ``asyncio``
streams (no external dependencies). Clients connect, submit tasks via
JSON messages, and receive streamed progress notifications.

Protocol::

    Client -> Server:
        {"type": "task.submit", "id": "req-1", "prompt": "..."}
        {"type": "task.cancel", "id": "req-2", "task_id": "..."}
        {"type": "task.list",   "id": "req-3"}
        {"type": "status",      "id": "req-4"}

    Server -> Client (broadcast):
        {"type": "task.started",   "task_id": "...", "prompt": "..."}
        {"type": "task.progress",  "task_id": "...", "text": "..."}
        {"type": "task.tool",      "task_id": "...", "name": "..."}
        {"type": "task.completed", "task_id": "...", "result": "..."}
        {"type": "task.failed",    "task_id": "...", "error": "..."}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
from collections.abc import Awaitable, Callable
from typing import Any

SubmitFn = Callable[[str, str, Callable[[str, dict[str, Any]], None]], Awaitable[str]]


class WebSocketGateway:
    """Minimal WebSocket server using raw asyncio (no external deps).

    Implements RFC 6455 basics: handshake, text frames, ping/pong, close.

    Optional bearer-token auth: if ``auth_token`` is non-empty, every
    incoming connection must present the token via either an
    ``Authorization: Bearer <token>`` header or a ``?token=<token>``
    query parameter on the WebSocket upgrade request. Connections that
    fail auth receive a ``401 Unauthorized`` response and are closed
    before the WebSocket handshake completes.
    """

    def __init__(self, host: str, port: int, auth_token: str | None = None) -> None:
        self._host = host
        self._port = port
        self._auth_token = auth_token or None
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._submit_fn: SubmitFn | None = None
        self._list_fn: Callable[[], list[dict[str, Any]]] | None = None
        self._status_fn: Callable[[], dict[str, Any]] | None = None
        self._cancel_fn: Callable[[str], bool] | None = None

    async def start(
        self,
        submit_fn: SubmitFn,
        list_fn: Callable[[], list[dict[str, Any]]],
        status_fn: Callable[[], dict[str, Any]],
        cancel_fn: Callable[[str], bool],
    ) -> None:
        self._submit_fn = submit_fn
        self._list_fn = list_fn
        self._status_fn = status_fn
        self._cancel_fn = cancel_fn

        self._server = await asyncio.start_server(
            self._handle_connection,
            self._host,
            self._port,
        )

    async def stop(self) -> None:
        for writer in list(self._clients):
            try:
                writer.close()
            except Exception:
                pass
        self._clients.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Send an event to all connected WebSocket clients."""
        msg = json.dumps({"type": event_type, **data}, default=str)
        frame = self._encode_ws_frame(msg)
        dead: list[asyncio.StreamWriter] = []
        for writer in self._clients:
            try:
                writer.write(frame)
            except Exception:
                dead.append(writer)
        for w in dead:
            self._clients.discard(w)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a raw TCP connection — do WS handshake, then message loop."""
        try:
            request = b""
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                request += line
                if line == b"\r\n":
                    break

            decoded = request.decode()
            headers = self._parse_headers(decoded)
            ws_key = headers.get("sec-websocket-key", "")
            if not ws_key:
                writer.close()
                return

            if self._auth_token and not self._is_authorized(decoded, headers):
                response = (
                    'HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer realm="xerxes"\r\nContent-Length: 0\r\n\r\n'
                )
                writer.write(response.encode())
                await writer.drain()
                writer.close()
                return

            accept = self._ws_accept_key(ws_key)
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            )
            writer.write(response.encode())
            await writer.drain()

            self._clients.add(writer)

            while True:
                msg = await self._read_ws_frame(reader)
                if msg is None:
                    break
                await self._handle_message(msg, writer)

        except (TimeoutError, ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            self._clients.discard(writer)
            try:
                writer.close()
            except Exception:
                pass

    async def _handle_message(self, raw: str, writer: asyncio.StreamWriter) -> None:
        """Dispatch a parsed WebSocket text message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self._send_ws(writer, {"type": "error", "error": "Invalid JSON"})
            return

        msg_type = msg.get("type", "")
        msg_id = msg.get("id", "")

        if msg_type == "task.submit":
            prompt = msg.get("prompt", "").strip()
            if not prompt:
                self._send_ws(writer, {"type": "error", "id": msg_id, "error": "Empty prompt"})
                return

            asyncio.create_task(self._handle_submit(prompt, msg_id, writer))  # noqa: RUF006

        elif msg_type == "task.cancel":
            task_id = msg.get("task_id", "")
            if self._cancel_fn and task_id:
                ok = self._cancel_fn(task_id)
                self._send_ws(writer, {"type": "task.cancel.ack", "id": msg_id, "ok": ok})
            else:
                self._send_ws(writer, {"type": "error", "id": msg_id, "error": "Invalid task_id"})

        elif msg_type == "task.list":
            tasks = self._list_fn() if self._list_fn else []
            self._send_ws(writer, {"type": "task.list.result", "id": msg_id, "tasks": tasks})

        elif msg_type == "task.status":
            task_id = msg.get("task_id", "")
            tasks = self._list_fn() if self._list_fn else []
            task = next((t for t in tasks if t.get("id") == task_id), None)
            self._send_ws(writer, {"type": "task.status.result", "id": msg_id, "task": task})

        elif msg_type == "status":
            status = self._status_fn() if self._status_fn else {}
            self._send_ws(writer, {"type": "status.result", "id": msg_id, **status})

        else:
            self._send_ws(writer, {"type": "error", "id": msg_id, "error": f"Unknown type: {msg_type}"})

    async def _handle_submit(self, prompt: str, msg_id: str, writer: asyncio.StreamWriter) -> None:
        """Submit a task and stream progress to the requesting client."""
        if not self._submit_fn:
            self._send_ws(writer, {"type": "error", "id": msg_id, "error": "Not ready"})
            return

        def on_event(event_type: str, data: dict[str, Any]) -> None:
            self.broadcast(event_type, data)

        result = await self._submit_fn(prompt, f"ws:{msg_id}", on_event)

        self._send_ws(writer, {"type": "task.submit.ack", "id": msg_id, "result": result[:500]})

    def _send_ws(self, writer: asyncio.StreamWriter, data: dict[str, Any]) -> None:
        try:
            frame = self._encode_ws_frame(json.dumps(data, default=str))
            writer.write(frame)
        except Exception:
            pass

    @staticmethod
    def _encode_ws_frame(text: str) -> bytes:
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
        """Read a single WebSocket text frame. Returns None on close/error."""
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
        """Verify the bearer token via Authorization header or `?token=` query param.

        Uses constant-time comparison to avoid timing oracles. Returns
        ``True`` when no auth token is configured (auth disabled).
        """
        import hmac

        expected = self._auth_token or ""
        if not expected:
            return True

        auth_header = headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            presented = auth_header[7:].strip()
            if hmac.compare_digest(presented, expected):
                return True

        try:
            request_line = request.split("\r\n", 1)[0]
            parts = request_line.split(" ")
            if len(parts) >= 2 and "?" in parts[1]:
                query = parts[1].split("?", 1)[1]
                for pair in query.split("&"):
                    if "=" not in pair:
                        continue
                    k, v = pair.split("=", 1)
                    if k == "token" and hmac.compare_digest(v, expected):
                        return True
        except (IndexError, ValueError):
            pass

        return False

    @staticmethod
    def _parse_headers(request: str) -> dict[str, str]:
        headers: dict[str, str] = {}
        for line in request.split("\r\n")[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key.lower()] = value
        return headers

    @staticmethod
    def _ws_accept_key(key: str) -> str:
        import base64

        magic = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        digest = hashlib.sha1(magic.encode()).digest()
        return base64.b64encode(digest).decode()
