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
"""Bridge ACP-style approvals to Xerxes ``PermissionRequest`` semantics.

ACP exposes approvals as request/response correlated by id. The
adapter holds pending requests in a dict and surfaces them via
``snapshot_pending`` so the daemon can list them; ``resolve``
records a decision."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AcpPermissionRequest:
    """Mirror of a pending approval surfaced to an ACP client.

    Attributes:
        id: stable request id (uuid).
        session_id: the session asking.
        tool_name: tool that needs approval.
        description: human-facing description.
        inputs: the tool inputs to be approved.
        decided: ``True`` once ``resolve`` is called.
        allowed: outcome (only meaningful when ``decided``).
        metadata: extra fields a client may attach."""

    id: str
    session_id: str
    tool_name: str
    description: str
    inputs: dict[str, Any]
    decided: bool = False
    allowed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def route_permission(
    *,
    session_id: str,
    tool_name: str,
    description: str,
    inputs: dict[str, Any],
) -> AcpPermissionRequest:
    """Construct a new pending request with a fresh id."""
    return AcpPermissionRequest(
        id=uuid.uuid4().hex,
        session_id=session_id,
        tool_name=tool_name,
        description=description,
        inputs=dict(inputs),
    )


class AcpPermissionBoard:
    """Hold pending ACP approval requests in memory.

    Tools call ``await_decision(request)`` to block until the client
    responds. The ACP server thread calls ``snapshot_pending()`` and
    ``resolve(id, allow)`` based on incoming RPC."""

    def __init__(self) -> None:
        """Initialise empty request + event maps under a single mutex."""
        self._requests: dict[str, AcpPermissionRequest] = {}
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def submit(self, req: AcpPermissionRequest) -> threading.Event:
        """Register ``req`` and return an Event the tool thread can wait on."""
        with self._lock:
            self._requests[req.id] = req
            ev = threading.Event()
            self._events[req.id] = ev
            return ev

    def resolve(self, req_id: str, allow: bool) -> bool:
        """Record the client's decision and wake any waiter.

        Returns False if the request is unknown or already decided
        (so duplicate responses are idempotent)."""
        with self._lock:
            req = self._requests.get(req_id)
            if req is None or req.decided:
                return False
            req.decided = True
            req.allowed = allow
            ev = self._events.get(req_id)
        if ev is not None:
            ev.set()
        return True

    def snapshot_pending(self) -> list[AcpPermissionRequest]:
        """Return requests still awaiting a decision."""
        with self._lock:
            return [r for r in self._requests.values() if not r.decided]

    def get(self, req_id: str) -> AcpPermissionRequest | None:
        """Look up a single request by id."""
        with self._lock:
            return self._requests.get(req_id)

    def drop(self, req_id: str) -> None:
        """Forget ``req_id`` entirely (request + waiter)."""
        with self._lock:
            self._requests.pop(req_id, None)
            self._events.pop(req_id, None)


__all__ = ["AcpPermissionBoard", "AcpPermissionRequest", "route_permission"]
