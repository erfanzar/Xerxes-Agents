# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Newline-delimited JSON-RPC over a Unix domain socket.

This is the local-only sibling of :mod:`xerxes.daemon.gateway`: TUIs and other
co-tenanted clients connect to ``$XERXES_HOME/daemon/xerxes.sock`` and speak
plain JSON-RPC 2.0, one message per line. Outbound ``event`` notifications
are broadcast to every live writer; inbound requests are dispatched via the
caller-supplied :data:`RPCHandler`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]
RPCHandler = Callable[[str, dict[str, Any], EmitFn], Awaitable[dict[str, Any]]]


class SocketChannel:
    """JSON-RPC 2.0 server over a Unix domain socket using newline framing."""

    def __init__(self, socket_path: str) -> None:
        """Configure the path; the listener is opened by :meth:`start`."""
        self._path = Path(socket_path).expanduser()
        self._server: asyncio.AbstractServer | None = None
        self._handler: RPCHandler | None = None
        self._clients: dict[asyncio.StreamWriter, asyncio.Lock] = {}
        self._broadcast_tasks: set[asyncio.Task[None]] = set()

    async def start(self, handler: RPCHandler) -> None:
        """Bind the socket, remove any stale file, and accept connections."""
        self._handler = handler
        if self._path.exists():
            self._path.unlink()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self._path))

    async def stop(self) -> None:
        """Close every client, cancel in-flight broadcasts, and remove the socket file."""
        for writer in list(self._clients):
            try:
                writer.close()
            except Exception:
                pass
        self._clients.clear()
        for task in list(self._broadcast_tasks):
            task.cancel()
        self._broadcast_tasks.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._path.unlink(missing_ok=True)

    def broadcast(self, event_type: str, payload: dict[str, Any]) -> None:
        """Push an ``event`` notification line to every connected client."""
        data = {"jsonrpc": "2.0", "method": "event", "params": {"type": event_type, "payload": payload}}
        for writer, lock in list(self._clients.items()):
            task = asyncio.create_task(self._send(writer, lock, data))
            self._broadcast_tasks.add(task)
            task.add_done_callback(self._broadcast_tasks.discard)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Pump one connected client's request loop until disconnect."""
        write_lock = asyncio.Lock()
        self._clients[writer] = write_lock

        async def send(data: dict[str, Any]) -> None:
            await self._send(writer, write_lock, data)

        async def emit(event_type: str, payload: dict[str, Any]) -> None:
            await send({"jsonrpc": "2.0", "method": "event", "params": {"type": event_type, "payload": payload}})

        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                try:
                    msg = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    await send({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Invalid JSON"}})
                    continue

                req_id = msg.get("id")
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
                        await send(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32000, "message": str(exc)},
                            }
                        )
                        continue
                await send({"jsonrpc": "2.0", "id": req_id, "result": result})
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            self._clients.pop(writer, None)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _send(self, writer: asyncio.StreamWriter, lock: asyncio.Lock, data: dict[str, Any]) -> None:
        """Serialise ``data`` and write it as a single newline-terminated line."""
        try:
            async with lock:
                writer.write((json.dumps(data, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError, OSError):
            self._clients.pop(writer, None)


__all__ = ["EmitFn", "RPCHandler", "SocketChannel"]
