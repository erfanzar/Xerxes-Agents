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
"""Parity audit module for Xerxes.

Exports:
    - ModuleStatus
    - ParityAuditResult
    - run_parity_audit"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModuleStatus:
    """Module status.

    Attributes:
        name (str): name.
        category (str): category.
        expected (bool): expected.
        present (bool): present.
        file_count (int): file count.
        notes (str): notes."""

    name: str
    category: str = ""
    expected: bool = True
    present: bool = False
    file_count: int = 0
    notes: str = ""


@dataclass
class ParityAuditResult:
    """Parity audit result.

    Attributes:
        modules (list[ModuleStatus]): modules.
        total_files (int): total files."""

    modules: list[ModuleStatus] = field(default_factory=list)
    total_files: int = 0

    @property
    def missing(self) -> list[ModuleStatus]:
        """Return Missing.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[ModuleStatus]: OUT: Result of the operation."""
        return [m for m in self.modules if m.expected and not m.present]
        """Return Missing.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[ModuleStatus]: OUT: Result of the operation."""
        """Return Missing.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[ModuleStatus]: OUT: Result of the operation."""

    @property
    def extra(self) -> list[ModuleStatus]:
        """Return Extra.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[ModuleStatus]: OUT: Result of the operation."""
        return [m for m in self.modules if not m.expected and m.present]
        """Return Extra.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[ModuleStatus]: OUT: Result of the operation."""
        """Return Extra.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[ModuleStatus]: OUT: Result of the operation."""

    @property
    def present(self) -> list[ModuleStatus]:
        """Return Present.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[ModuleStatus]: OUT: Result of the operation."""
        return [m for m in self.modules if m.expected and m.present]
        """Return Present.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[ModuleStatus]: OUT: Result of the operation."""
        """Return Present.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            list[ModuleStatus]: OUT: Result of the operation."""

    @property
    def coverage_pct(self) -> float:
        """Return Coverage pct.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            float: OUT: Result of the operation."""
        expected = [m for m in self.modules if m.expected]
        """Return Coverage pct.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            float: OUT: Result of the operation."""
        """Return Coverage pct.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            float: OUT: Result of the operation."""
        if not expected:
            return 100.0
        return len(self.present) / len(expected) * 100

    def as_markdown(self) -> str:
        """As markdown.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""

        lines = [
            """As markdown.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""
            """As markdown.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
        Returns:
            str: OUT: Result of the operation."""
            "# Parity Audit",
            "",
            f"Coverage: {self.coverage_pct:.1f}%",
            f"Total files: {self.total_files}",
            f"Present: {len(self.present)} / {len([m for m in self.modules if m.expected])}",
            f"Missing: {len(self.missing)}",
            f"Extra: {len(self.extra)}",
            "",
        ]

        if self.missing:
            lines.append("## Missing Modules")
            for m in self.missing:
                lines.append(f"- **{m.name}** ({m.category}) — {m.notes or 'not implemented'}")
            lines.append("")

        if self.extra:
            lines.append("## Extra Modules (not in reference)")
            for m in self.extra:
                lines.append(f"- **{m.name}** ({m.category}, {m.file_count} files)")
            lines.append("")

        if self.present:
            lines.append("## Present Modules")
            for m in self.present:
                lines.append(f"- {m.name} ({m.category}, {m.file_count} files)")

        return "\n".join(lines)


REFERENCE_SURFACE: list[dict[str, str]] = [
    {"name": "llms", "category": "core", "notes": "LLM provider integrations"},
    {"name": "llms.registry", "category": "core", "notes": "Provider registry with auto-detection"},
    {"name": "llms.compat", "category": "core", "notes": "OpenAI-compatible provider wrapper"},
    {"name": "tools", "category": "core", "notes": "Tool definitions and registry"},
    {"name": "memory", "category": "core", "notes": "Memory system (short/long/entity/user)"},
    {"name": "types", "category": "core", "notes": "Type definitions"},
    {"name": "cortex", "category": "orchestration", "notes": "Multi-agent orchestration"},
    {"name": "mcp", "category": "integration", "notes": "Model Context Protocol"},
    {"name": "runtime", "category": "runtime", "notes": "Runtime execution context"},
    {"name": "runtime.execution_registry", "category": "runtime", "notes": "Command/tool routing"},
    {"name": "runtime.query_engine", "category": "runtime", "notes": "Multi-turn conversation engine"},
    {"name": "runtime.transcript", "category": "runtime", "notes": "Transcript with compaction"},
    {"name": "runtime.history", "category": "runtime", "notes": "Session event history"},
    {"name": "runtime.cost_tracker", "category": "runtime", "notes": "Cost tracking"},
    {"name": "runtime.session", "category": "runtime", "notes": "Full session state"},
    {"name": "runtime.tool_pool", "category": "runtime", "notes": "Permission-filtered tool pool"},
    {"name": "runtime.bootstrap", "category": "runtime", "notes": "Bootstrap and system init"},
    {"name": "streaming", "category": "streaming", "notes": "Streaming event protocol"},
    {"name": "streaming.events", "category": "streaming", "notes": "Typed streaming events"},
    {"name": "streaming.messages", "category": "streaming", "notes": "Neutral message format"},
    {"name": "streaming.permissions", "category": "streaming", "notes": "Permission modes"},
    {"name": "streaming.loop", "category": "streaming", "notes": "Generator-based agent loop"},
    {"name": "security", "category": "security", "notes": "Tool policies and sandboxing"},
    {"name": "session", "category": "persistence", "notes": "Session persistence"},
    {"name": "audit", "category": "observability", "notes": "Audit event system"},
    {"name": "context", "category": "context", "notes": "Context management and compaction"},
    {"name": "extensions", "category": "extensions", "notes": "Plugins, skills, hooks"},
    {"name": "tui", "category": "ui", "notes": "Terminal UI"},
    {"name": "api_server", "category": "ui", "notes": "OpenAI-compatible API server"},
    {"name": "operators", "category": "runtime", "notes": "Operator state and config"},
]


def run_parity_audit(
    package_root: Path | None = None,
    reference: list[dict[str, str]] | None = None,
) -> ParityAuditResult:
    """Run parity audit.

    Args:
        package_root (Path | None, optional): IN: package root. Defaults to None. OUT: Consumed during execution.
        reference (list[dict[str, str]] | None, optional): IN: reference. Defaults to None. OUT: Consumed during execution.
    Returns:
        ParityAuditResult: OUT: Result of the operation."""

    root = package_root or Path(__file__).resolve().parent.parent
    """Run parity audit.

    Args:
        package_root (Path | None, optional): IN: package root. Defaults to None. OUT: Consumed during execution.
        reference (list[dict[str, str]] | None, optional): IN: reference. Defaults to None. OUT: Consumed during execution.
    Returns:
        ParityAuditResult: OUT: Result of the operation."""
    """Run parity audit.

    Args:
        package_root (Path | None, optional): IN: package root. Defaults to None. OUT: Consumed during execution.
        reference (list[dict[str, str]] | None, optional): IN: reference. Defaults to None. OUT: Consumed during execution.
    Returns:
        ParityAuditResult: OUT: Result of the operation."""
    ref = reference or REFERENCE_SURFACE
    result = ParityAuditResult()

    result.total_files = sum(1 for _ in root.rglob("*.py"))

    expected_names = set()
    for entry in ref:
        name = entry["name"]
        expected_names.add(name)
        module_name = f"xerxes.{name}"

        present = False
        file_count = 0
        try:
            mod = importlib.import_module(module_name)
            present = True
            mod_path = getattr(mod, "__file__", None)
            if mod_path:
                mod_dir = Path(mod_path).parent
                if mod_dir.is_dir():
                    file_count = sum(1 for _ in mod_dir.rglob("*.py"))
                else:
                    file_count = 1
        except ImportError:
            pass

        result.modules.append(
            ModuleStatus(
                name=name,
                category=entry.get("category", ""),
                expected=True,
                present=present,
                file_count=file_count,
                notes=entry.get("notes", ""),
            )
        )

    for item in sorted(root.iterdir()):
        if item.is_dir() and not item.name.startswith("_"):
            mod_name = item.name
            if mod_name not in expected_names:
                file_count = sum(1 for _ in item.rglob("*.py"))
                result.modules.append(
                    ModuleStatus(
                        name=mod_name,
                        category="unknown",
                        expected=False,
                        present=True,
                        file_count=file_count,
                    )
                )

    return result


__all__ = [
    "ModuleStatus",
    "ParityAuditResult",
    "run_parity_audit",
]
