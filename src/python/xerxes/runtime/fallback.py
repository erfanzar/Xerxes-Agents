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
"""Capability fallback chains and tool health probes.

:class:`FallbackChain` records a preferred provider/tool plus ordered
alternatives for one capability; :class:`FallbackRegistry` is the
thread-safe map keyed by capability name. :class:`ToolHealthProber` runs
caller-supplied probe functions on a per-tool interval and surfaces the
latest :class:`HealthSnapshot` for routing decisions.
"""

from __future__ import annotations

import logging
import threading
import time
import typing as tp
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FallbackChain:
    """Ordered chain of providers/tools for one capability.

    Attributes:
        capability: Capability name (e.g. ``"llm.summary"``).
        preferred: Primary choice; tried first by :meth:`order`.
        alternatives: Ordered backup options tried after ``preferred``.
    """

    capability: str
    preferred: str
    alternatives: list[str] = field(default_factory=list)

    def order(self) -> list[str]:
        """Return the deduplicated try-order starting with ``preferred``."""

        seen: set[str] = set()
        out: list[str] = []
        for x in [self.preferred, *self.alternatives]:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out


class FallbackRegistry:
    """Thread-safe map of capability name → :class:`FallbackChain`."""

    def __init__(self) -> None:
        """Create an empty registry guarded by a reentrant lock."""
        self._chains: dict[str, FallbackChain] = {}
        self._lock = threading.RLock()

    def set(self, capability: str, preferred: str, *, alternatives: tp.Iterable[str] = ()) -> None:
        """Register (or replace) the fallback chain for ``capability``."""
        with self._lock:
            self._chains[capability] = FallbackChain(
                capability=capability,
                preferred=preferred,
                alternatives=list(alternatives),
            )

    def get(self, capability: str) -> FallbackChain | None:
        """Return the registered chain for ``capability`` or ``None``."""
        with self._lock:
            return self._chains.get(capability)

    def order_for(self, capability: str) -> list[str]:
        """Return the ordered fallback list, or an empty list if unregistered."""
        with self._lock:
            chain = self._chains.get(capability)
            return chain.order() if chain else []

    def next_after(self, capability: str, current: str) -> str | None:
        """Return the next backup after ``current`` in ``capability``'s chain, or ``None``."""
        order = self.order_for(capability)
        try:
            idx = order.index(current)
        except ValueError:
            return None
        return order[idx + 1] if idx + 1 < len(order) else None

    def remove(self, capability: str) -> bool:
        """Drop ``capability``; return ``True`` when something was removed."""
        with self._lock:
            return self._chains.pop(capability, None) is not None

    def all(self) -> dict[str, FallbackChain]:
        """Return a shallow copy of every registered chain."""
        with self._lock:
            return dict(self._chains)


@dataclass
class HealthSnapshot:
    """Last observation of a probed tool's health.

    Attributes:
        name: Tool name the snapshot belongs to.
        status: ``"ok"``, ``"down"``, or ``"unknown"``.
        latency_ms: Last probe latency in milliseconds.
        last_checked: Wall-clock seconds at which the probe ran.
        message: Optional short human-readable diagnostic message.
    """

    name: str
    status: str = "unknown"
    latency_ms: float = 0.0
    last_checked: float = 0.0
    message: str = ""


ProbeFn = tp.Callable[[], "HealthSnapshot | bool | None"]
"""Probe signature: returns a :class:`HealthSnapshot`, ``bool``, or ``None``."""


class ToolHealthProber:
    """Run periodic probe functions and remember each tool's latest health."""

    def __init__(self) -> None:
        """Create an empty prober."""
        self._probes: dict[str, tuple[ProbeFn, float]] = {}
        self._snapshots: dict[str, HealthSnapshot] = {}
        self._next_due: dict[str, float] = {}
        self._lock = threading.Lock()

    def register(self, name: str, probe: ProbeFn, *, interval_seconds: float = 60.0) -> None:
        """Register ``probe`` for tool ``name`` to run every ``interval_seconds``."""

        with self._lock:
            self._probes[name] = (probe, float(interval_seconds))
            self._next_due[name] = 0.0
            self._snapshots.setdefault(name, HealthSnapshot(name=name))

    def unregister(self, name: str) -> None:
        """Drop the probe and accumulated snapshot for ``name``."""

        with self._lock:
            self._probes.pop(name, None)
            self._next_due.pop(name, None)
            self._snapshots.pop(name, None)

    def run_one(self, name: str, *, now: float | None = None) -> HealthSnapshot:
        """Invoke ``name``'s probe immediately, ignoring its scheduled interval.

        Records the resulting snapshot, schedules the next due time, and
        returns the snapshot. ``True``/``None``/``False`` probe return values
        are mapped to ``"ok"``/``"down"`` statuses; raised exceptions become
        ``"down"`` snapshots with the exception text.
        """

        now = time.time() if now is None else now
        with self._lock:
            entry = self._probes.get(name)
        if entry is None:
            snapshot = HealthSnapshot(name=name, status="unknown", last_checked=now, message="not registered")
            self._snapshots[name] = snapshot
            return snapshot
        probe, interval = entry
        start = time.perf_counter()
        try:
            raw = probe()
            duration = (time.perf_counter() - start) * 1000.0
            if isinstance(raw, HealthSnapshot):
                snapshot = HealthSnapshot(
                    name=raw.name or name,
                    status=raw.status or "ok",
                    latency_ms=raw.latency_ms or duration,
                    last_checked=now,
                    message=raw.message,
                )
            elif raw is True:
                snapshot = HealthSnapshot(name=name, status="ok", latency_ms=duration, last_checked=now)
            elif raw is False or raw is None:
                snapshot = HealthSnapshot(name=name, status="down", latency_ms=duration, last_checked=now, message="")
            else:
                snapshot = HealthSnapshot(
                    name=name, status="ok", latency_ms=duration, last_checked=now, message=str(raw)[:120]
                )
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000.0
            snapshot = HealthSnapshot(
                name=name, status="down", latency_ms=duration, last_checked=now, message=f"{type(exc).__name__}: {exc}"
            )
        with self._lock:
            self._snapshots[name] = snapshot
            self._next_due[name] = now + interval
        return snapshot

    def run_due(self, *, now: float | None = None) -> list[HealthSnapshot]:
        """Run every probe whose next-due time has elapsed and return new snapshots."""

        now = time.time() if now is None else now
        with self._lock:
            due_names = [n for n, due in self._next_due.items() if due <= now]
        return [self.run_one(name, now=now) for name in due_names]

    def snapshot(self, name: str) -> HealthSnapshot | None:
        """Return the latest :class:`HealthSnapshot` for ``name``, or ``None``."""
        with self._lock:
            return self._snapshots.get(name)

    def snapshots(self) -> dict[str, HealthSnapshot]:
        """Return a shallow copy of every tool snapshot held."""
        with self._lock:
            return dict(self._snapshots)

    def healthy(self, name: str) -> bool:
        """Return ``True`` when ``name``'s most recent snapshot has status ``"ok"``."""
        snap = self.snapshot(name)
        return bool(snap and snap.status == "ok")


__all__ = [
    "FallbackChain",
    "FallbackRegistry",
    "HealthSnapshot",
    "ProbeFn",
    "ToolHealthProber",
]
