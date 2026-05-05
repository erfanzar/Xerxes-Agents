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
"""Unix domain socket channel for local daemon communication.

``SocketChannel`` exposes a line-delimited JSON-RPC-like interface over a
Unix domain socket so local clients can submit tasks without a network hop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

SubmitFn = Callable[[str, str], Awaitable[str]]


class SocketChannel:
    """Unix socket server that accepts JSON-line requests.

    Args:
        socket_path (str): IN: Filesystem path for the Unix socket. OUT:
            Expanded, cleaned up, and bound by ``asyncio.start_unix_server``.
    """

    def __init__(self, socket_path: str) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            socket_path (str): IN: socket path. OUT: Consumed during execution."""
        self._path = Path(socket_path).expanduser()
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            socket_path (str): IN: socket path. OUT: Consumed during execution."""
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            socket_path (str): IN: socket path. OUT: Consumed during execution."""
        self._server: asyncio.AbstractServer | None = None
        self._submit_fn: SubmitFn | None = None
        self._list_fn: Callable[[], list[dict[str, Any]]] | None = None
        self._status_fn: Callable[[], dict[str, Any]] | None = None

    async def start(
        self,
        submit_fn: SubmitFn,
        list_fn: Callable[[], list[dict[str, Any]]],
        status_fn: Callable[[], dict[str, Any]],
    ) -> None:
        """Bind callbacks and start the Unix socket server.

        Args:
            submit_fn (SubmitFn): IN: Async callback for task submission.
                OUT: Invoked for ``submit`` method requests.
            list_fn (Callable): IN: Callback returning task list. OUT: Invoked
                for ``list`` method requests.
            status_fn (Callable): IN: Callback returning daemon status. OUT:
                Invoked for ``status`` method requests.

        Returns:
            None: OUT: Server is listening when the coroutine completes.
        """
        self._submit_fn = submit_fn
        self._list_fn = list_fn
        self._status_fn = status_fn

        if self._path.exists():
            self._path.unlink()
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._path),
        )

    async def stop(self) -> None:
        """Stop the server and remove the socket file.

        Returns:
            None: OUT: Socket is closed and filesystem entry is removed.
        """
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._path.exists():
            self._path.unlink(missing_ok=True)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Process JSON-line requests from a single client.

        Args:
            reader (asyncio.StreamReader): IN: Async reader for the socket.
                OUT: Consumed line-by-line.
            writer (asyncio.StreamWriter): IN: Async writer for the socket.
                OUT: Used to send JSON response lines.

        Returns:
            None: OUT: Connection is closed after the first ``submit`` or on
            error.
        """
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                line = data.decode().strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    resp = {"ok": False, "error": "Invalid JSON"}
                    writer.write((json.dumps(resp) + "\n").encode())
                    await writer.drain()
                    continue

                method = msg.get("method", "")
                resp = await self._dispatch(method, msg.get("params", {}))
                writer.write((json.dumps(resp, default=str) + "\n").encode())
                await writer.drain()

                if method == "submit":
                    break
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    async def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Route a parsed request to the appropriate handler.

        Args:
            method (str): IN: Request method name. OUT: Used for routing.
            params (dict[str, Any]): IN: Method parameters. OUT: Passed to the
                handler (e.g. ``prompt`` for ``submit``).

        Returns:
            dict[str, Any]: OUT: Response dict with ``ok`` and result fields.
        """
        if method == "submit":
            prompt = params.get("prompt", "").strip()
            if not prompt:
                return {"ok": False, "error": "Empty prompt"}
            if self._submit_fn:
                result = await self._submit_fn(prompt, "socket")
                return {"ok": True, "result": result}
            return {"ok": False, "error": "Daemon not ready"}

        if method == "list":
            if self._list_fn:
                return {"ok": True, "tasks": self._list_fn()}
            return {"ok": True, "tasks": []}

        if method == "status":
            if self._status_fn:
                return {"ok": True, **self._status_fn()}
            return {"ok": True, "status": "running"}

        return {"ok": False, "error": f"Unknown method: {method}"}
