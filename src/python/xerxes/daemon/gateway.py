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
"""WebSocket gateway for the Xerxes daemon.

Implements a lightweight WebSocket server using ``asyncio`` streams. Clients
can submit tasks, list running tasks, query status, and cancel tasks over a
plain-text WebSocket protocol.
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
    """WebSocket server that exposes daemon operations to remote clients.

    Args:
        host (str): IN: Bind address for the TCP server. OUT: Passed to
            ``asyncio.start_server``.
        port (int): IN: TCP port to listen on. OUT: Passed to
            ``asyncio.start_server``.
        auth_token (str | None): IN: Optional Bearer token for client auth.
            OUT: Checked in ``_is_authorized`` during handshake.
    """

    def __init__(self, host: str, port: int, auth_token: str | None = None) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            host (str): IN: host. OUT: Consumed during execution.
            port (int): IN: port. OUT: Consumed during execution.
            auth_token (str | None, optional): IN: auth token. Defaults to None. OUT: Consumed during execution."""
        self._host = host
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            host (str): IN: host. OUT: Consumed during execution.
            port (int): IN: port. OUT: Consumed during execution.
            auth_token (str | None, optional): IN: auth token. Defaults to None. OUT: Consumed during execution."""
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            host (str): IN: host. OUT: Consumed during execution.
            port (int): IN: port. OUT: Consumed during execution.
            auth_token (str | None, optional): IN: auth token. Defaults to None. OUT: Consumed during execution."""
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
        """Start the WebSocket server and bind handler callbacks.

        Args:
            submit_fn (SubmitFn): IN: Async callback that accepts a prompt,
                source label, and event callback. OUT: Invoked for
                ``task.submit`` messages.
            list_fn (Callable): IN: Callback returning a list of task dicts.
                OUT: Invoked for ``task.list`` messages.
            status_fn (Callable): IN: Callback returning daemon status dict.
                OUT: Invoked for ``status`` messages.
            cancel_fn (Callable[[str], bool]): IN: Callback accepting a task
                ID and returning success. OUT: Invoked for ``task.cancel``.

        Returns:
            None: OUT: Server is running when the coroutine completes.
        """
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
        """Gracefully close all client connections and stop the server.

        Returns:
            None: OUT: All sockets are closed and the server is shut down.
        """
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
        """Send a JSON event to every connected WebSocket client.

        Dead clients are automatically removed from the internal set.

        Args:
            event_type (str): IN: Event name inserted as ``"type"``. OUT:
                Serialized into the JSON frame.
            data (dict[str, Any]): IN: Additional payload fields. OUT:
                Merged with ``event_type`` and encoded as a WebSocket text
                frame.

        Returns:
            None: OUT: Frame is written (best-effort) to all live clients.
        """

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
        """Handle a single TCP connection and upgrade it to WebSocket.

        Args:
            reader (asyncio.StreamReader): IN: Async read interface for the
                client socket. OUT: Consumed during HTTP handshake and frame
                reads.
            writer (asyncio.StreamWriter): IN: Async write interface for the
                client socket. OUT: Used to send handshake response and
                WebSocket frames.

        Returns:
            None: OUT: Connection is closed on error or client disconnect.
        """

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
        """Dispatch an incoming WebSocket text message to the correct handler.

        Args:
            raw (str): IN: UTF-8 decoded text frame payload. OUT: Parsed as
                JSON and routed by ``msg["type"]``.
            writer (asyncio.StreamWriter): IN: Client socket writer. OUT:
                Used to send JSON response frames.

        Returns:
            None: OUT: Response frame is sent (or connection closed on error).
        """

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

            self._submit_task = asyncio.create_task(self._handle_submit(prompt, msg_id, writer))

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
        """Forward a task submission to the daemon and reply with the result.

        Args:
            prompt (str): IN: User prompt text. OUT: Passed to ``_submit_fn``.
            msg_id (str): IN: Client message identifier. OUT: Echoed in the
                acknowledgment frame.
            writer (asyncio.StreamWriter): IN: Client socket writer. OUT: Used
                to send the ``task.submit.ack`` frame.

        Returns:
            None: OUT: Acknowledgment frame is sent after task completion.
        """

        if not self._submit_fn:
            self._send_ws(writer, {"type": "error", "id": msg_id, "error": "Not ready"})
            return

        def on_event(event_type: str, data: dict[str, Any]) -> None:
            """On event.

            Args:
                event_type (str): IN: event type. OUT: Consumed during execution.
                data (dict[str, Any]): IN: data. OUT: Consumed during execution."""
            self.broadcast(event_type, data)
            """On event.

            Args:
                event_type (str): IN: event type. OUT: Consumed during execution.
                data (dict[str, Any]): IN: data. OUT: Consumed during execution."""
            """On event.

            Args:
                event_type (str): IN: event type. OUT: Consumed during execution.
                data (dict[str, Any]): IN: data. OUT: Consumed during execution."""

        result = await self._submit_fn(prompt, f"ws:{msg_id}", on_event)

        self._send_ws(writer, {"type": "task.submit.ack", "id": msg_id, "result": result[:500]})

    def _send_ws(self, writer: asyncio.StreamWriter, data: dict[str, Any]) -> None:
        """Encode ``data`` as JSON and write a WebSocket text frame.

        Args:
            writer (asyncio.StreamWriter): IN: Target client writer. OUT:
                Receives the encoded frame bytes.
            data (dict[str, Any]): IN: Payload dict. OUT: JSON-encoded and
                wrapped in a WebSocket frame.

        Returns:
            None: OUT: Frame is written best-effort; errors are swallowed.
        """
        try:
            frame = self._encode_ws_frame(json.dumps(data, default=str))
            writer.write(frame)
        except Exception:
            pass

    @staticmethod
    def _encode_ws_frame(text: str) -> bytes:
        """Encode a UTF-8 string as a WebSocket text frame.

        Args:
            text (str): IN: Payload text. OUT: UTF-8 encoded and prefixed with
                a WebSocket header.

        Returns:
            bytes: OUT: Complete WebSocket frame ready for socket write.
        """
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
        """Read and decode a single WebSocket text frame.

        Args:
            reader (asyncio.StreamReader): IN: Async reader for the connection.
                OUT: Consumed to read the frame header, mask, and payload.

        Returns:
            str | None: OUT: Decoded UTF-8 payload, empty string for ping, or
            ``None`` on connection close / error.
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
        """Validate the Bearer token from headers or query string.

        Args:
            request (str): IN: Full HTTP request text. OUT: Parsed for a
                ``?token=…`` query parameter.
            headers (dict[str, str]): IN: Lower-cased header mapping. OUT:
                Checked for ``authorization`` header.

        Returns:
            bool: OUT: ``True`` if the presented token matches ``_auth_token``.
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
        """Parse an HTTP request string into a lower-cased header dict.

        Args:
            request (str): IN: Raw HTTP request text. OUT: Split on ``\r\n``
                and parsed for ``Key: Value`` lines.

        Returns:
            dict[str, str]: OUT: Mapping of lower-cased header names to values.
        """
        headers: dict[str, str] = {}
        for line in request.split("\r\n")[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key.lower()] = value
        return headers

    @staticmethod
    def _ws_accept_key(key: str) -> str:
        """Compute the ``Sec-WebSocket-Accept`` response value per RFC 6455.

        Args:
            key (str): IN: Client ``sec-websocket-key`` header value. OUT:
                Concatenated with the WebSocket magic string and hashed.

        Returns:
            str: OUT: Base64-encoded SHA-1 digest.
        """
        import base64

        magic = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        digest = hashlib.sha1(magic.encode()).digest()
        return base64.b64encode(digest).decode()
