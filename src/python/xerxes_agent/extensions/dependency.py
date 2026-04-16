# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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


"""Dependency resolution for Xerxes plugins and skills.

Provides version constraint parsing, dependency specification, and
topological sorting with circular dependency detection for local
plugin/skill dependency graphs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _parse_version_tuple(version: str, *, pad: bool = True) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers.

    Splits a dotted version string (e.g., ``'1.2.3'``) on ``'.'`` and converts
    each segment to an integer. Non-numeric trailing segments are silently
    dropped.  When *pad* is ``True`` the tuple is zero-padded to at least
    three elements so that ``'1.2'`` becomes ``(1, 2, 0)``.

    Args:
        version: A dotted version string such as ``'1.2.3'`` or ``'2.0'``.
        pad: If ``True`` (the default), pad the result with zeros so it
            contains at least three elements.

    Returns:
        A tuple of integers representing the parsed version components.

    Example:
        >>> _parse_version_tuple("1.2.3")
        (1, 2, 3)
        >>> _parse_version_tuple("1.2", pad=False)
        (1, 2)
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
    """Parse and evaluate PEP 440-style version constraint strings.

    A ``VersionConstraint`` encapsulates one or more comparison operators
    applied to version strings.  Supported operators are ``==``, ``!=``,
    ``>=``, ``<=``, ``>``, ``<``, and ``~=`` (compatible release).  Multiple
    constraints can be combined with commas (e.g., ``">=1.0,<2.0"``), in
    which case *all* constraints must be satisfied.

    Attributes:
        raw: The original constraint string passed to the constructor.

    Example:
        >>> vc = VersionConstraint(">=1.0.0")
        >>> vc.satisfies("1.2.3")
        True
        >>> vc = VersionConstraint(">=1.0,<2.0")
        >>> vc.satisfies("2.0.0")
        False
        >>> vc = VersionConstraint("~=1.2")
        >>> vc.satisfies("1.3.0")
        True
    """

    _OP_PATTERN = re.compile(r"^\s*(~=|==|!=|>=|<=|>|<)\s*(.+)\s*$")

    def __init__(self, constraint_str: str) -> None:
        """Initialize a VersionConstraint by parsing the constraint string.

        The constraint string is split on commas and each segment is parsed
        into an ``(operator, version_tuple, raw_parts)`` triple stored in
        ``self._constraints``.  If a segment contains no recognized operator
        the ``==`` operator is assumed.

        Args:
            constraint_str: A version constraint expression such as
                ``">=1.0.0"``, ``">=1.0,<2.0"``, or ``"~=1.2"``.  An empty
                string produces a constraint that matches any version.
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
        """Check whether *version* satisfies every constraint.

        If no constraints were parsed (e.g., the constraint string was empty),
        any version is considered satisfying.

        Args:
            version: A dotted version string to test (e.g., ``"1.2.3"``).

        Returns:
            ``True`` if *version* satisfies all individual constraints,
            ``False`` otherwise.
        """
        if not self._constraints:
            return True
        ver = _parse_version_tuple(version)
        for op, target, raw_parts in self._constraints:
            if not self._check(op, ver, target, raw_parts):
                return False
        return True

    @staticmethod
    def _check(op: str, ver: tuple[int, ...], target: tuple[int, ...], raw_parts: int = 3) -> bool:
        """Evaluate a single version comparison operation.

        For the ``~=`` (compatible release) operator the check ensures that
        *ver* is at least *target* and shares the same prefix up to the
        second-to-last specified segment (e.g., ``~=1.2`` allows ``>=1.2,<2.0``).

        Args:
            op: The comparison operator string (one of ``==``, ``!=``,
                ``>=``, ``<=``, ``>``, ``<``, ``~=``).
            ver: The version being tested, as a tuple of ints.
            target: The constraint version to compare against.
            raw_parts: The number of segments originally specified in the
                constraint (before zero-padding).  Used by ``~=`` to determine
                the prefix length.

        Returns:
            ``True`` if the comparison holds, ``False`` otherwise.  Returns
            ``None`` implicitly if *op* is unrecognized (should not happen
            in normal use).
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

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            A string of the form ``VersionConstraint('<raw>')``.
        """
        return f"VersionConstraint({self.raw!r})"


@dataclass
class DependencySpec:
    """A single dependency requirement consisting of a name and optional version constraint.

    Attributes:
        name: The identifier of the required dependency (e.g., ``"my_plugin"``).
        version_constraint: An optional version constraint string such as
            ``">=1.0.0"`` or ``"~=2.1"``.  ``None`` means any version is
            acceptable.

    Example:
        >>> spec = DependencySpec(name="my_plugin", version_constraint=">=1.0")
        >>> spec.to_version_constraint().satisfies("1.5.0")
        True
    """

    name: str
    version_constraint: str | None = None

    def to_version_constraint(self) -> VersionConstraint:
        """Convert the version constraint string into a ``VersionConstraint`` object.

        Returns:
            A ``VersionConstraint`` parsed from :attr:`version_constraint`.
            If :attr:`version_constraint` is ``None``, an unconstrained
            (always-satisfied) ``VersionConstraint`` is returned.
        """
        return VersionConstraint(self.version_constraint or "")


def parse_dependency(dep_str: str) -> DependencySpec:
    """Parse a dependency string into a ``DependencySpec``.

    The string is expected to start with an alphanumeric-plus-hyphens/underscores
    name, optionally followed by a version constraint expression.  If no
    constraint is present, ``version_constraint`` on the returned spec will be
    ``None``.

    Args:
        dep_str: A dependency declaration such as ``"my_plugin>=1.0.0"`` or
            plain ``"my_plugin"``.

    Returns:
        A ``DependencySpec`` with the parsed ``name`` and optional
        ``version_constraint``.

    Example:
        >>> parse_dependency("my_plugin>=1.0.0")
        DependencySpec(name='my_plugin', version_constraint='>=1.0.0')
        >>> parse_dependency("my_plugin")
        DependencySpec(name='my_plugin', version_constraint=None)
    """
    dep_str = dep_str.strip()
    m = re.match(r"^([A-Za-z0-9_\-]+)(.*)", dep_str)
    if m:
        name = m.group(1)
        constraint = m.group(2).strip()
        return DependencySpec(name=name, version_constraint=constraint if constraint else None)
    return DependencySpec(name=dep_str, version_constraint=None)


@dataclass
class ResolveResult:
    """Result of a dependency resolution pass.

    Attributes:
        satisfied: ``True`` when all requirements are met with no missing
            dependencies and no version conflicts.
        missing: Names of required packages that were not found in the
            available set.
        conflicts: Human-readable descriptions of version mismatches
            (e.g., ``"foo: requires >=2.0, found 1.3.0"``).
        resolution_order: An ordered list of package names representing a
            safe loading sequence (populated when topological sorting is
            performed separately).
    """

    satisfied: bool
    missing: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    resolution_order: list[str] = field(default_factory=list)


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected during topological sorting.

    Attributes:
        cycle: An ordered list of node names that form the cycle.  The first
            and last elements are identical, illustrating the loop
            (e.g., ``["A", "B", "C", "A"]``).
    """

    def __init__(self, cycle: list[str]) -> None:
        """Initialize with the detected dependency cycle path.

        Args:
            cycle: The list of node names forming the cycle.  The last
                element should repeat the first to close the loop.
        """
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")


class DependencyResolver:
    """Resolve a dependency graph, detecting missing deps, version conflicts,
    and circular dependencies, and produce a topological load order.

    The resolver provides two main operations:

    * :meth:`resolve` -- checks that a flat list of ``DependencySpec``
      requirements are met by a set of available packages with known versions.
    * :meth:`topological_sort` -- orders nodes in a directed acyclic graph so
      that every dependency appears before the nodes that depend on it.

    Example:
        >>> resolver = DependencyResolver()
        >>> available = {"A": "1.0.0", "B": "2.0.0"}
        >>> reqs = [DependencySpec("A", ">=1.0")]
        >>> result = resolver.resolve(available, reqs)
        >>> result.satisfied
        True
    """

    def resolve(
        self,
        available: dict[str, str],
        requirements: list[DependencySpec],
    ) -> ResolveResult:
        """Check that all *requirements* are met by *available* packages.

        Args:
            available: Mapping of package name to version string.
            requirements: List of dependency specifications to check.

        Returns:
            A ResolveResult with satisfaction status and details.
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
        """Topological sort of a dependency graph.

        Args:
            graph: Mapping of node -> list of nodes it depends on.

        Returns:
            List of node names in dependency-safe order (dependencies first).

        Raises:
            CircularDependencyError: If a cycle is detected.
        """
        state: dict[str, int] = {node: 0 for node in graph}
        order: list[str] = []
        path: list[str] = []

        def visit(node: str) -> None:
            """Recursively visit a node using depth-first search.

            Uses a tri-colour marking scheme (0 = unvisited, 1 = in-progress,
            2 = finished) to detect back-edges that indicate cycles.

            Args:
                node: The graph node to visit.

            Raises:
                CircularDependencyError: If a back-edge (cycle) is detected.
            """
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
