# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.

"""Unix domain socket channel for local ``xerxes send`` commands.

Provides a newline-delimited JSON protocol over a Unix socket at
``~/.xerxes/daemon/xerxes.sock`` for fast, auth-free local task
submission and daemon status queries.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

SubmitFn = Callable[[str, str], Awaitable[str]]


class SocketChannel:
    """Unix domain socket server for local task submission.

    Protocol: newline-delimited JSON.
    Requests:
        {"method": "submit", "params": {"prompt": "..."}}
        {"method": "list"}
        {"method": "status"}
    Responses:
        {"ok": true, "task_id": "...", "result": "..."} or
        {"ok": false, "error": "..."}
    """

    def __init__(self, socket_path: str) -> None:
        self._path = Path(socket_path).expanduser()
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
