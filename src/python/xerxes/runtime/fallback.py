# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Fallback agent registry + periodic tool health probes.

Two cooperating helpers:

- :class:`FallbackRegistry` lets the orchestrator declare ordered
  fallback chains per capability (e.g. "if `summarise` fails on agent
  ``opus``, try ``sonnet``; if that also fails, try ``haiku``"). Used
  by ``AgentOrchestrator`` for graceful degradation.
- :class:`ToolHealthProber` runs supplied probe callables on a periodic
  schedule and exposes the latest health snapshot. Probes have a status
  (``"ok"`` / ``"degraded"`` / ``"down"``) and an error message; the
  daemon renders these via the gateway.
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
    """Ordered preference for a single capability.

    Attributes:
        capability: Logical capability name (e.g. ``"summarise"``).
        preferred: First-choice agent identifier.
        alternatives: Ordered list of alternates.
    """

    capability: str
    preferred: str
    alternatives: list[str] = field(default_factory=list)

    def order(self) -> list[str]:
        """All identifiers in fallback order, dedup'd, preserving order."""
        seen: set[str] = set()
        out: list[str] = []
        for x in [self.preferred, *self.alternatives]:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out


class FallbackRegistry:
    """Per-capability fallback chains.

    Threadsafe; chains may be mutated at runtime (e.g. by tests or by
    the daemon when a probe reports an agent is down).

    Example:
        >>> r = FallbackRegistry()
        >>> r.set("summarise", "opus", alternatives=["sonnet", "haiku"])
        >>> r.order_for("summarise")
        ['opus', 'sonnet', 'haiku']
        >>> r.next_after("summarise", "opus")
        'sonnet'
    """

    def __init__(self) -> None:
        """Initialise with an empty chain map guarded by a reentrant lock."""
        self._chains: dict[str, FallbackChain] = {}
        self._lock = threading.RLock()

    def set(self, capability: str, preferred: str, *, alternatives: tp.Iterable[str] = ()) -> None:
        """Define (or replace) the chain for *capability*."""
        with self._lock:
            self._chains[capability] = FallbackChain(
                capability=capability,
                preferred=preferred,
                alternatives=list(alternatives),
            )

    def get(self, capability: str) -> FallbackChain | None:
        """Return the :class:`FallbackChain` for *capability*, or ``None`` if unset."""
        with self._lock:
            return self._chains.get(capability)

    def order_for(self, capability: str) -> list[str]:
        """Return the ordered ``[preferred, *alternatives]`` list for *capability*."""
        with self._lock:
            chain = self._chains.get(capability)
            return chain.order() if chain else []

    def next_after(self, capability: str, current: str) -> str | None:
        """Return the agent that should be tried after *current* fails."""
        order = self.order_for(capability)
        try:
            idx = order.index(current)
        except ValueError:
            return None
        return order[idx + 1] if idx + 1 < len(order) else None

    def remove(self, capability: str) -> bool:
        """Drop the chain for *capability*; returns ``True`` when one existed."""
        with self._lock:
            return self._chains.pop(capability, None) is not None

    def all(self) -> dict[str, FallbackChain]:
        """Return a shallow copy of every registered chain."""
        with self._lock:
            return dict(self._chains)


@dataclass
class HealthSnapshot:
    """Outcome of one probe execution.

    Attributes:
        name: Probe identifier.
        status: ``"ok"`` / ``"degraded"`` / ``"down"`` / ``"unknown"``.
        latency_ms: Probe wall-clock time.
        last_checked: ``time.time()`` at the moment of the check.
        message: Optional human-readable detail (e.g. error text).
    """

    name: str
    status: str = "unknown"
    latency_ms: float = 0.0
    last_checked: float = 0.0
    message: str = ""


ProbeFn = tp.Callable[[], "HealthSnapshot | bool | None"]


class ToolHealthProber:
    """Runs registered probes, tracks the last result.

    The actual *scheduling* is intentionally left to the caller — the
    daemon, a cron loop, or test code — by invoking
    :meth:`run_due` periodically. This keeps the prober dependency-free.

    Probe semantics:

    - Returning a :class:`HealthSnapshot` adopts it directly.
    - Returning ``True`` is treated as ``status="ok"``.
    - Returning ``False`` / ``None`` is treated as ``status="down"``.
    - Raising any exception is treated as ``status="down"`` with the
      exception text as the message.
    """

    def __init__(self) -> None:
        """Initialise empty probe/snapshot/schedule maps."""
        self._probes: dict[str, tuple[ProbeFn, float]] = {}
        self._snapshots: dict[str, HealthSnapshot] = {}
        self._next_due: dict[str, float] = {}
        self._lock = threading.Lock()

    def register(self, name: str, probe: ProbeFn, *, interval_seconds: float = 60.0) -> None:
        """Register *probe* to be run at most every *interval_seconds*."""
        with self._lock:
            self._probes[name] = (probe, float(interval_seconds))
            self._next_due[name] = 0.0
            self._snapshots.setdefault(name, HealthSnapshot(name=name))

    def unregister(self, name: str) -> None:
        """Remove probe *name* along with its schedule and cached snapshot."""
        with self._lock:
            self._probes.pop(name, None)
            self._next_due.pop(name, None)
            self._snapshots.pop(name, None)

    def run_one(self, name: str, *, now: float | None = None) -> HealthSnapshot:
        """Force-run probe *name*; returns the resulting snapshot."""
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
        """Run every probe whose interval has elapsed; returns fresh snapshots."""
        now = time.time() if now is None else now
        with self._lock:
            due_names = [n for n, due in self._next_due.items() if due <= now]
        return [self.run_one(name, now=now) for name in due_names]

    def snapshot(self, name: str) -> HealthSnapshot | None:
        """Return the most recent cached snapshot for probe *name*, if any."""
        with self._lock:
            return self._snapshots.get(name)

    def snapshots(self) -> dict[str, HealthSnapshot]:
        """Return a shallow copy of every cached probe snapshot."""
        with self._lock:
            return dict(self._snapshots)

    def healthy(self, name: str) -> bool:
        """Return ``True`` iff the last snapshot for *name* reports ``status == "ok"``."""
        snap = self.snapshot(name)
        return bool(snap and snap.status == "ok")


__all__ = [
    "FallbackChain",
    "FallbackRegistry",
    "HealthSnapshot",
    "ProbeFn",
    "ToolHealthProber",
]
