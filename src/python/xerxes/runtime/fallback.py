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
"""Fallback module for Xerxes.

Exports:
    - logger
    - FallbackChain
    - FallbackRegistry
    - HealthSnapshot
    - ProbeFn
    - ToolHealthProber"""

from __future__ import annotations

import logging
import threading
import time
import typing as tp
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FallbackChain:
    """Fallback chain.

    Attributes:
        capability (str): capability.
        preferred (str): preferred.
        alternatives (list[str]): alternatives."""

    capability: str
    preferred: str
    alternatives: list[str] = field(default_factory=list)

    def order(self) -> list[str]:
        """Order.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[str]: OUT: Result of the operation."""

        seen: set[str] = set()
        out: list[str] = []
        for x in [self.preferred, *self.alternatives]:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out


class FallbackRegistry:
    """Fallback registry."""

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self._chains: dict[str, FallbackChain] = {}
        self._lock = threading.RLock()

    def set(self, capability: str, preferred: str, *, alternatives: tp.Iterable[str] = ()) -> None:
        """Set.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            capability (str): IN: capability. OUT: Consumed during execution.
            preferred (str): IN: preferred. OUT: Consumed during execution.
            alternatives (tp.Iterable[str], optional): IN: alternatives. Defaults to (). OUT: Consumed during execution."""

        with self._lock:
            self._chains[capability] = FallbackChain(
                capability=capability,
                preferred=preferred,
                alternatives=list(alternatives),
            )

    def get(self, capability: str) -> FallbackChain | None:
        """Get.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            capability (str): IN: capability. OUT: Consumed during execution.
        Returns:
            FallbackChain | None: OUT: Result of the operation."""

        with self._lock:
            return self._chains.get(capability)

    def order_for(self, capability: str) -> list[str]:
        """Order for.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            capability (str): IN: capability. OUT: Consumed during execution.
        Returns:
            list[str]: OUT: Result of the operation."""

        with self._lock:
            chain = self._chains.get(capability)
            return chain.order() if chain else []

    def next_after(self, capability: str, current: str) -> str | None:
        """Next after.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            capability (str): IN: capability. OUT: Consumed during execution.
            current (str): IN: current. OUT: Consumed during execution.
        Returns:
            str | None: OUT: Result of the operation."""

        order = self.order_for(capability)
        try:
            idx = order.index(current)
        except ValueError:
            return None
        return order[idx + 1] if idx + 1 < len(order) else None

    def remove(self, capability: str) -> bool:
        """Remove.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            capability (str): IN: capability. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        with self._lock:
            return self._chains.pop(capability, None) is not None

    def all(self) -> dict[str, FallbackChain]:
        """All.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, FallbackChain]: OUT: Result of the operation."""

        with self._lock:
            return dict(self._chains)


@dataclass
class HealthSnapshot:
    """Health snapshot.

    Attributes:
        name (str): name.
        status (str): status.
        latency_ms (float): latency ms.
        last_checked (float): last checked.
        message (str): message."""

    name: str
    status: str = "unknown"
    latency_ms: float = 0.0
    last_checked: float = 0.0
    message: str = ""


ProbeFn = tp.Callable[[], "HealthSnapshot | bool | None"]


class ToolHealthProber:
    """Tool health prober."""

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

        self._probes: dict[str, tuple[ProbeFn, float]] = {}
        self._snapshots: dict[str, HealthSnapshot] = {}
        self._next_due: dict[str, float] = {}
        self._lock = threading.Lock()

    def register(self, name: str, probe: ProbeFn, *, interval_seconds: float = 60.0) -> None:
        """Register.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            probe (ProbeFn): IN: probe. OUT: Consumed during execution.
            interval_seconds (float, optional): IN: interval seconds. Defaults to 60.0. OUT: Consumed during execution."""

        with self._lock:
            self._probes[name] = (probe, float(interval_seconds))
            self._next_due[name] = 0.0
            self._snapshots.setdefault(name, HealthSnapshot(name=name))

    def unregister(self, name: str) -> None:
        """Unregister.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution."""

        with self._lock:
            self._probes.pop(name, None)
            self._next_due.pop(name, None)
            self._snapshots.pop(name, None)

    def run_one(self, name: str, *, now: float | None = None) -> HealthSnapshot:
        """Run one.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
            now (float | None, optional): IN: now. Defaults to None. OUT: Consumed during execution.
        Returns:
            HealthSnapshot: OUT: Result of the operation."""

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
        """Run due.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            now (float | None, optional): IN: now. Defaults to None. OUT: Consumed during execution.
        Returns:
            list[HealthSnapshot]: OUT: Result of the operation."""

        now = time.time() if now is None else now
        with self._lock:
            due_names = [n for n, due in self._next_due.items() if due <= now]
        return [self.run_one(name, now=now) for name in due_names]

    def snapshot(self, name: str) -> HealthSnapshot | None:
        """Snapshot.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
        Returns:
            HealthSnapshot | None: OUT: Result of the operation."""

        with self._lock:
            return self._snapshots.get(name)

    def snapshots(self) -> dict[str, HealthSnapshot]:
        """Snapshots.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            dict[str, HealthSnapshot]: OUT: Result of the operation."""

        with self._lock:
            return dict(self._snapshots)

    def healthy(self, name: str) -> bool:
        """Healthy.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            name (str): IN: name. OUT: Consumed during execution.
        Returns:
            bool: OUT: Result of the operation."""

        snap = self.snapshot(name)
        return bool(snap and snap.status == "ok")


__all__ = [
    "FallbackChain",
    "FallbackRegistry",
    "HealthSnapshot",
    "ProbeFn",
    "ToolHealthProber",
]
