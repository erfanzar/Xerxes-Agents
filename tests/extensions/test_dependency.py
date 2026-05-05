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
"""Tests for xerxes.dependency — version constraints and dependency resolution."""

import pytest
from xerxes.extensions.dependency import (
    CircularDependencyError,
    DependencyResolver,
    DependencySpec,
    VersionConstraint,
    parse_dependency,
)


class TestVersionConstraint:
    def test_exact_match(self):
        vc = VersionConstraint("==1.0.0")
        assert vc.satisfies("1.0.0")
        assert not vc.satisfies("1.0.1")

    def test_not_equal(self):
        vc = VersionConstraint("!=1.0.0")
        assert not vc.satisfies("1.0.0")
        assert vc.satisfies("1.0.1")

    def test_greater_equal(self):
        vc = VersionConstraint(">=1.0.0")
        assert vc.satisfies("1.0.0")
        assert vc.satisfies("1.2.3")
        assert not vc.satisfies("0.9.9")

    def test_less_than(self):
        vc = VersionConstraint("<2.0.0")
        assert vc.satisfies("1.9.9")
        assert not vc.satisfies("2.0.0")
        assert not vc.satisfies("2.0.1")

    def test_greater_than(self):
        vc = VersionConstraint(">1.0.0")
        assert not vc.satisfies("1.0.0")
        assert vc.satisfies("1.0.1")

    def test_less_equal(self):
        vc = VersionConstraint("<=2.0.0")
        assert vc.satisfies("2.0.0")
        assert vc.satisfies("1.9.9")
        assert not vc.satisfies("2.0.1")

    def test_combined_constraints(self):
        vc = VersionConstraint(">=1.0,<2.0")
        assert vc.satisfies("1.0.0")
        assert vc.satisfies("1.9.9")
        assert not vc.satisfies("0.9.0")
        assert not vc.satisfies("2.0.0")

    def test_compatible_release(self):
        vc = VersionConstraint("~=1.2")
        assert vc.satisfies("1.2.0")
        assert vc.satisfies("1.9.0")
        assert not vc.satisfies("2.0.0")
        assert not vc.satisfies("1.1.0")

    def test_compatible_release_three_part(self):
        vc = VersionConstraint("~=1.2.3")
        assert vc.satisfies("1.2.3")
        assert vc.satisfies("1.2.9")
        assert not vc.satisfies("1.3.0")
        assert not vc.satisfies("1.2.2")

    def test_empty_constraint_matches_all(self):
        vc = VersionConstraint("")
        assert vc.satisfies("0.0.1")
        assert vc.satisfies("99.0.0")

    def test_bare_version_treated_as_exact(self):
        vc = VersionConstraint("1.0.0")
        assert vc.satisfies("1.0.0")
        assert not vc.satisfies("1.0.1")

    def test_two_part_version(self):
        vc = VersionConstraint(">=1.2")
        assert vc.satisfies("1.2.0")
        assert vc.satisfies("1.3")
        assert not vc.satisfies("1.1")

    def test_repr(self):
        vc = VersionConstraint(">=1.0")
        assert ">=1.0" in repr(vc)


class TestParseDependency:
    def test_name_only(self):
        spec = parse_dependency("my_plugin")
        assert spec.name == "my_plugin"
        assert spec.version_constraint is None

    def test_name_with_constraint(self):
        spec = parse_dependency("my_plugin>=1.0.0")
        assert spec.name == "my_plugin"
        assert spec.version_constraint == ">=1.0.0"

    def test_name_with_combined_constraint(self):
        spec = parse_dependency("core>=1.0,<2.0")
        assert spec.name == "core"
        assert spec.version_constraint == ">=1.0,<2.0"

    def test_name_with_exact_constraint(self):
        spec = parse_dependency("lib==2.0.0")
        assert spec.name == "lib"
        assert spec.version_constraint == "==2.0.0"

    def test_whitespace_handling(self):
        spec = parse_dependency("  my_plugin >= 1.0  ")
        assert spec.name == "my_plugin"
        assert spec.version_constraint is not None

    def test_hyphenated_name(self):
        spec = parse_dependency("my-plugin>=1.0")
        assert spec.name == "my-plugin"
        assert spec.version_constraint == ">=1.0"


class TestDependencyResolver:
    def test_satisfied_no_constraints(self):
        resolver = DependencyResolver()
        available = {"A": "1.0.0", "B": "2.0.0"}
        reqs = [DependencySpec("A"), DependencySpec("B")]
        result = resolver.resolve(available, reqs)
        assert result.satisfied
        assert result.missing == []
        assert result.conflicts == []

    def test_satisfied_with_constraints(self):
        resolver = DependencyResolver()
        available = {"A": "1.5.0", "B": "2.0.0"}
        reqs = [DependencySpec("A", ">=1.0"), DependencySpec("B", ">=2.0,<3.0")]
        result = resolver.resolve(available, reqs)
        assert result.satisfied

    def test_missing_dependency(self):
        resolver = DependencyResolver()
        available = {"A": "1.0.0"}
        reqs = [DependencySpec("A"), DependencySpec("C")]
        result = resolver.resolve(available, reqs)
        assert not result.satisfied
        assert "C" in result.missing

    def test_version_conflict(self):
        resolver = DependencyResolver()
        available = {"A": "1.0.0"}
        reqs = [DependencySpec("A", ">=2.0")]
        result = resolver.resolve(available, reqs)
        assert not result.satisfied
        assert len(result.conflicts) == 1
        assert "A" in result.conflicts[0]

    def test_empty_requirements(self):
        resolver = DependencyResolver()
        result = resolver.resolve({"A": "1.0"}, [])
        assert result.satisfied


class TestTopologicalSort:
    def test_simple_chain(self):
        resolver = DependencyResolver()
        graph = {"C": ["B"], "B": ["A"], "A": []}
        order = resolver.topological_sort(graph)
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_no_dependencies(self):
        resolver = DependencyResolver()
        graph = {"A": [], "B": [], "C": []}
        order = resolver.topological_sort(graph)
        assert set(order) == {"A", "B", "C"}

    def test_diamond_dependency(self):
        resolver = DependencyResolver()

        graph = {"D": ["B", "C"], "B": ["A"], "C": ["A"], "A": []}
        order = resolver.topological_sort(graph)
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_circular_dependency_raises(self):
        resolver = DependencyResolver()
        graph = {"A": ["B"], "B": ["C"], "C": ["A"]}
        with pytest.raises(CircularDependencyError) as exc_info:
            resolver.topological_sort(graph)
        assert len(exc_info.value.cycle) >= 2

    def test_self_referencing_circular(self):
        resolver = DependencyResolver()
        graph = {"A": ["A"]}
        with pytest.raises(CircularDependencyError):
            resolver.topological_sort(graph)

    def test_deterministic_order(self):
        resolver = DependencyResolver()
        graph = {"Z": [], "A": [], "M": []}
        order1 = resolver.topological_sort(graph)
        order2 = resolver.topological_sort(graph)
        assert order1 == order2
