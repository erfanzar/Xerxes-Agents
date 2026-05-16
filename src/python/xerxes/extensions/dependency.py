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
"""Dependency resolution and version constraint checking for plugins.

Provides ``VersionConstraint`` (PEP-440-ish), ``DependencySpec``,
``DependencyResolver``, and ``CircularDependencyError`` for validating and
ordering plugin dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _parse_version_tuple(version: str, *, pad: bool = True) -> tuple[int, ...]:
    """Split a dotted version string into a tuple of integer segments.

    Args:
        version: Version string such as ``"1.2.3"``.
        pad: When True, zero-extend the result to at least three segments.
    """

    parts = []
    for part in version.strip().split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    if pad:
        while len(parts) < 3:
            parts.append(0)
    return tuple(parts)


class VersionConstraint:
    """Parse and evaluate version constraints like ``>=1.0`` or ``~=1.2.3``."""

    _OP_PATTERN = re.compile(r"^\s*(~=|==|!=|>=|<=|>|<)\s*(.+)\s*$")

    def __init__(self, constraint_str: str) -> None:
        """Tokenise ``constraint_str`` into ``(op, version, raw_parts)`` triples.

        Multiple constraints may be combined with commas.
        """

        self.raw = constraint_str.strip()
        self._constraints: list[tuple[str, tuple[int, ...], int]] = []
        if not self.raw:
            return
        for part in self.raw.split(","):
            part = part.strip()
            if not part:
                continue
            m = self._OP_PATTERN.match(part)
            if m:
                op = m.group(1)
                raw_ver = _parse_version_tuple(m.group(2), pad=False)
                ver = _parse_version_tuple(m.group(2))
                self._constraints.append((op, ver, len(raw_ver)))
            else:
                raw_ver = _parse_version_tuple(part, pad=False)
                self._constraints.append(("==", _parse_version_tuple(part), len(raw_ver)))

    def satisfies(self, version: str) -> bool:
        """Return whether ``version`` satisfies every stored constraint."""

        if not self._constraints:
            return True
        ver = _parse_version_tuple(version)
        for op, target, raw_parts in self._constraints:
            if not self._check(op, ver, target, raw_parts):
                return False
        return True

    @staticmethod
    def _check(op: str, ver: tuple[int, ...], target: tuple[int, ...], raw_parts: int = 3) -> bool:
        """Evaluate a single constraint of the form ``ver op target``.

        ``raw_parts`` determines the upper bound for the ``~=`` compatible release
        operator and is otherwise unused.
        """

        if op == "==":
            return ver == target
        if op == "!=":
            return ver != target
        if op == ">=":
            return ver >= target
        if op == "<=":
            return ver <= target
        if op == ">":
            return ver > target
        if op == "<":
            return ver < target
        if op == "~=":
            if ver < target:
                return False
            raw_target = list(target[:raw_parts])
            prefix = raw_target[:-1]
            if prefix:
                prefix[-1] += 1
                upper = tuple(prefix) + (0,) * (len(target) - len(prefix))
                return ver < upper
            return True
        return False

    def __repr__(self) -> str:
        """Return ``VersionConstraint(<raw>)`` for debugging."""

        return f"VersionConstraint({self.raw!r})"


@dataclass
class DependencySpec:
    """Named dependency with an optional version constraint.

    Attributes:
        name: Dependency identifier.
        version_constraint: Raw constraint text (e.g. ``">=1.0"``) or ``None``.
    """

    name: str
    version_constraint: str | None = None

    def to_version_constraint(self) -> VersionConstraint:
        """Return a ``VersionConstraint`` built from ``version_constraint``."""

        return VersionConstraint(self.version_constraint or "")


def parse_dependency(dep_str: str) -> DependencySpec:
    """Parse ``name`` or ``name<op>version`` into a ``DependencySpec``."""

    dep_str = dep_str.strip()
    m = re.match(r"^([A-Za-z0-9_\-]+)(.*)", dep_str)
    if m:
        name = m.group(1)
        constraint = m.group(2).strip()
        return DependencySpec(name=name, version_constraint=constraint if constraint else None)
    return DependencySpec(name=dep_str, version_constraint=None)


@dataclass
class ResolveResult:
    """Outcome of a dependency resolution pass.

    Attributes:
        satisfied: ``True`` when no requirements are missing or conflicting.
        missing: Names of dependencies that could not be resolved.
        conflicts: Human-readable version-conflict messages.
        resolution_order: Intended load order, populated by topological sort.
    """

    satisfied: bool
    missing: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    resolution_order: list[str] = field(default_factory=list)


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected during topological sort."""

    def __init__(self, cycle: list[str]) -> None:
        """Store ``cycle`` on the exception and embed it in the message."""

        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")


class DependencyResolver:
    """Validate plugin dependencies and compute topological load order."""

    def resolve(
        self,
        available: dict[str, str],
        requirements: list[DependencySpec],
    ) -> ResolveResult:
        """Check whether ``requirements`` are satisfied by ``available``.

        Args:
            available: Mapping from plugin name to installed version.
            requirements: Desired dependency specifications.
        """

        missing: list[str] = []
        conflicts: list[str] = []

        for req in requirements:
            if req.name not in available:
                missing.append(req.name)
            elif req.version_constraint:
                vc = req.to_version_constraint()
                actual_version = available[req.name]
                if not vc.satisfies(actual_version):
                    conflicts.append(f"{req.name}: requires {req.version_constraint}, found {actual_version}")

        satisfied = not missing and not conflicts
        return ResolveResult(
            satisfied=satisfied,
            missing=missing,
            conflicts=conflicts,
            resolution_order=[],
        )

    def topological_sort(
        self,
        graph: dict[str, list[str]],
    ) -> list[str]:
        """Return nodes of ``graph`` in dependency-respecting order.

        Args:
            graph: Mapping from node name to its dependency list.

        Raises:
            CircularDependencyError: A cycle was encountered during traversal.
        """

        state: dict[str, int] = {node: 0 for node in graph}
        order: list[str] = []
        path: list[str] = []

        def visit(node: str) -> None:
            """DFS visitor that records traversal state and detects cycles."""

            if state.get(node) == 2:
                return
            if state.get(node) == 1:
                cycle_start = path.index(node)
                cycle = [*path[cycle_start:], node]
                raise CircularDependencyError(cycle)

            state[node] = 1
            path.append(node)
            for dep in graph.get(node, []):
                if dep in graph:
                    visit(dep)
            path.pop()
            state[node] = 2
            order.append(node)

        for node in sorted(graph.keys()):
            if state.get(node) == 0:
                visit(node)

        return order
