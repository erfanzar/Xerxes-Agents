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
"""Codebase-aware repo map for system-prompt injection.

Extracts function/class signatures from source files, ranks them by
cross-file reference frequency and modification recency, and renders a
compact ranked tree within a token budget. This gives the agent
structural awareness of the project without needing to Read/Grep every
file on each task — the single highest-impact context feature for coding
agents.

Python files are parsed with :mod:`ast` for precise signature
extraction. Other languages fall back to a lightweight regex sweep
that recognises common ``def``/``func``/``function``/``class``/``fn``
/``struct``/``interface`` patterns.  The output format is a flat ranked
list grouped by file path, similar to ``ctags --format=2`` or Aider's
repo-map output.

Example output (truncated)::

    src/python/xerxes/llms/registry.py
      # detect_provider (19 refs)
      # get_api_key (12 refs)
      class ProviderConfig
      COSTS: dict[str, tuple[float, float]]

    src/python/xerxes/streaming/loop.py
      def run (31 refs)
      def arun (8 refs)
      class _ThinkingParser
"""

from __future__ import annotations

import ast
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_DEFAULT_TOKEN_BUDGET = 2048
_DEFAULT_MAX_FILES = 200
_DEFAULT_MAX_SYMBOLS_PER_FILE = 15

_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".tox",
        ".eggs",
        "dist",
        "build",
        ".idea",
        ".vscode",
        "*.egg-info",
    }
)

_IGNORE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".o",
        ".a",
        ".wasm",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".bmp",
        ".tiff",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".lock",
        ".min.js",
        ".min.css",
    }
)

_SOURCE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".rb",
        ".php",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".hpp",
        ".swift",
        ".scala",
        ".lua",
        ".sh",
        ".bash",
        ".zsh",
    }
)

# Lightweight regex for non-Python source files. Matches common
# function/class/method declaration patterns across languages.
_REGEX_SYMBOL: list[tuple[str, re.Pattern[str]]] = [
    ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.MULTILINE)),
    (
        "const_arrow",
        re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\(", re.MULTILINE),
    ),
    ("class", re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.MULTILINE)),
    ("def", re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)),
    ("fn", re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)),
    ("func", re.compile(r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)),
    ("interface", re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.MULTILINE)),
    ("struct", re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)),
]


@dataclass(frozen=True)
class Symbol:
    """One extracted code symbol (function, class, or method).

    Attributes:
        name: The symbol identifier as it appears in source.
        kind: ``"function"``, ``"class"``, ``"method"``, ``"constant"``, etc.
        file: Relative path from the repo root.
        line: 1-indexed source line where the symbol is declared.
        signature: Compact one-line signature (Python only; empty for regex-extracted symbols).
    """

    name: str
    kind: str
    file: str
    line: int
    signature: str = ""


@dataclass
class RepoMapConfig:
    """Tunable parameters for :class:`RepoMapper`.

    Attributes:
        token_budget: Maximum tokens (estimated at 4 chars/token) the rendered map may occupy.
        max_files: Hard cap on files scanned.
        max_symbols_per_file: Maximum symbols included per file after ranking.
        reference_weight: Multiplier applied to reference-count rank score.
        recency_weight: Multiplier applied to modification-recency rank score.
    """

    token_budget: int = _DEFAULT_TOKEN_BUDGET
    max_files: int = _DEFAULT_MAX_FILES
    max_symbols_per_file: int = _DEFAULT_MAX_SYMBOLS_PER_FILE
    reference_weight: float = 3.0
    recency_weight: float = 1.0


@dataclass
class RepoMapResult:
    """Rendered repo map and metadata.

    Attributes:
        text: The compact ranked tree string ready for prompt injection.
        file_count: Number of files scanned.
        symbol_count: Total symbols extracted before budget truncation.
        included_count: Symbols actually included in the rendered output.
        estimated_tokens: Token estimate of the rendered text.
    """

    text: str
    file_count: int
    symbol_count: int
    included_count: int
    estimated_tokens: int


def _is_ignored(path: Path, name: str) -> bool:
    """Return True if a directory or file should be skipped during the walk."""

    if name in _IGNORE_DIRS:
        return True
    if path.suffix in _IGNORE_SUFFIXES:
        return True
    return False


def _gitignore_patterns(root: Path) -> list[str]:
    """Read top-level .gitignore patterns (simple prefix/glob matching only).

    Returns a list of raw pattern strings. Matching is intentionally
    conservative — only directory-name and suffix patterns are honoured to
    avoid false positives from complex negation rules.
    """

    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return []
    try:
        lines = gitignore.read_text(errors="replace").splitlines()
    except OSError:
        return []
    patterns: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("!"):
            continue
        patterns.append(stripped.rstrip("/"))
    return patterns


def _matches_gitignore(rel_path: str, patterns: list[str]) -> bool:
    """Conservative .gitignore prefix/suffix match."""

    parts = Path(rel_path).parts
    for pat in patterns:
        if "/" not in pat and pat in parts:
            return True
        if rel_path.endswith(pat):
            return True
        if pat.startswith("*.") and rel_path.endswith(pat[1:]):
            return True
    return False


def _extract_python_symbols(file_path: Path, rel_path: str) -> list[Symbol]:
    """Parse a Python file with :mod:`ast` and extract top-level symbols.

    Returns function defs, async function defs, class defs, and top-level
    assignments to UPPER_CASE names (constants). Methods inside classes are
    included so the map shows class structure. Nested functions are skipped
    to keep the map compact.
    """

    try:
        source = file_path.read_text(errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []

    symbols: list[Symbol] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            sig = _python_signature(node)
            kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            symbols.append(
                Symbol(
                    name=node.name,
                    kind=kind,
                    file=rel_path,
                    line=node.lineno,
                    signature=sig,
                )
            )
        elif isinstance(node, ast.ClassDef):
            methods = _extract_class_methods(node, rel_path)
            symbols.append(
                Symbol(
                    name=node.name,
                    kind="class",
                    file=rel_path,
                    line=node.lineno,
                    signature=f"class {node.name}",
                )
            )
            symbols.extend(methods)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    symbols.append(
                        Symbol(
                            name=target.id,
                            kind="constant",
                            file=rel_path,
                            line=node.lineno,
                        )
                    )

    return symbols


def _python_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a compact ``name(arg1, arg2) -> return_type`` signature string."""

    args: list[str] = []
    for a in node.args.args:
        args.append(a.arg)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    sig = f"{prefix} {node.name}({', '.join(args)})"
    if node.returns and isinstance(node.returns, ast.Name):
        sig += f" -> {node.returns.id}"
    return sig


def _extract_class_methods(cls_node: ast.ClassDef, rel_path: str) -> list[Symbol]:
    """Extract public method names from a class (skip dunder and private)."""

    methods: list[Symbol] = []
    for node in ast.iter_child_nodes(cls_node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            methods.append(
                Symbol(
                    name=f"{cls_node.name}.{node.name}",
                    kind="method",
                    file=rel_path,
                    line=node.lineno,
                )
            )
    return methods


def _extract_regex_symbols(source: str, rel_path: str) -> list[Symbol]:
    """Extract symbols from non-Python source via regex patterns."""

    symbols: list[Symbol] = []
    seen: set[str] = set()
    for kind, pattern in _REGEX_SYMBOL:
        for match in pattern.finditer(source):
            name = match.group(1)
            key = f"{name}:{kind}"
            if key in seen:
                continue
            seen.add(key)
            line = source.count("\n", 0, match.start()) + 1
            symbols.append(Symbol(name=name, kind=kind, file=rel_path, line=line))
    return symbols


def _count_references(symbol_names: list[str], source_files: list[tuple[str, str]]) -> Counter[str]:
    """Count how many files reference each symbol name.

    A symbol is "referenced" by a file if its bare name appears as a whole
    word in that file's source. The file that *defines* the symbol also
    counts, so the minimum reference count is 1.
    """

    name_set = set(symbol_names)
    if not name_set:
        return Counter()
    refs: Counter[str] = Counter()
    for _rel, source in source_files:
        tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", source))
        for name in name_set & tokens:
            refs[name] += 1
    return refs


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ceiling(len / 4)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


class RepoMapper:
    """Builds a ranked, token-budgeted repo map from source files.

    Results are cached per-file keyed on (path, mtime, size). Subsequent
    :meth:`build` calls only re-scan files whose mtime changed since the
    last scan, making incremental updates cheap enough to run every turn.
    Call :meth:`invalidate` to force a full rescan, or pass ``force=True``
    to :meth:`build`.
    """

    def __init__(self, config: RepoMapConfig | None = None) -> None:
        """Initialise with optional config overrides."""
        self.config = config or RepoMapConfig()
        self._file_cache: dict[str, tuple[float, int, list[Symbol]]] = {}
        self._gitignore_cache: dict[str, list[str]] = {}

    def build(self, root: str | Path, *, force: bool = False) -> RepoMapResult:
        """Scan ``root`` and return a rendered :class:`RepoMapResult`.

        Walks the tree (respecting .gitignore and common ignore dirs),
        extracts symbols, ranks them, and renders the top entries within
        the configured token budget. Uses an incremental file cache —
        only files whose mtime/size changed since the last scan are
        re-parsed. Pass ``force=True`` to bypass the cache.

        Args:
            root: Absolute or relative path to the repository root.
            force: Bypass the incremental cache and do a full rescan.

        Returns:
            A :class:`RepoMapResult` with the rendered map text and
            metadata. Returns an empty result if ``root`` does not exist
            or contains no source files.
        """

        root_path = Path(root).resolve()
        if not root_path.is_dir():
            return RepoMapResult(text="", file_count=0, symbol_count=0, included_count=0, estimated_tokens=0)

        str(root_path)
        if force:
            self._file_cache.clear()

        gitignore = _gitignore_patterns(root_path)

        source_files: list[tuple[Path, str]] = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if not _is_ignored(Path(dirpath) / d, d)]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if _is_ignored(fpath, fname):
                    continue
                rel = str(fpath.relative_to(root_path))
                if _matches_gitignore(rel, gitignore):
                    continue
                if fpath.suffix not in _SOURCE_SUFFIXES:
                    continue
                source_files.append((fpath, rel))

        source_files.sort(key=lambda x: x[1])
        if len(source_files) > self.config.max_files:
            source_files = source_files[: self.config.max_files]

        all_symbols: list[Symbol] = []
        raw_sources: list[tuple[str, str]] = []

        for fpath, rel in source_files:
            try:
                st = fpath.stat()
            except OSError:
                continue
            mtime, size = st.st_mtime, st.st_size
            cached = self._file_cache.get(rel)
            if cached and cached[0] == mtime and cached[1] == size:
                syms = cached[2]
                # Still need the source text for reference counting
                try:
                    source = fpath.read_text(errors="replace")
                except OSError:
                    continue
                raw_sources.append((rel, source))
                all_symbols.extend(syms)
                continue

            try:
                source = fpath.read_text(errors="replace")
            except OSError:
                continue
            raw_sources.append((rel, source))
            if fpath.suffix == ".py":
                syms = _extract_python_symbols(fpath, rel)
            else:
                syms = _extract_regex_symbols(source, rel)
            self._file_cache[rel] = (mtime, size, syms)
            all_symbols.extend(syms)

        # Remove cache entries for files that no longer exist
        current_rels = {rel for _, rel in source_files}
        stale = [k for k in self._file_cache if k not in current_rels]
        for k in stale:
            self._file_cache.pop(k, None)

        if not all_symbols:
            return RepoMapResult(
                text="", file_count=len(source_files), symbol_count=0, included_count=0, estimated_tokens=0
            )

        name_list = [s.name.split(".")[-1] for s in all_symbols]
        ref_counts = _count_references(name_list, raw_sources)

        mtimes = {}
        for fpath, rel in source_files:
            try:
                mtimes[rel] = fpath.stat().st_mtime
            except OSError:
                mtimes[rel] = 0.0
        if mtimes:
            max_mtime = max(mtimes.values()) or 1.0
        else:
            max_mtime = 1.0

        scored: list[tuple[float, Symbol]] = []
        for sym in all_symbols:
            base_name = sym.name.split(".")[-1]
            refs = ref_counts.get(base_name, 1)
            recency = mtimes.get(sym.file, 0.0) / max_mtime if max_mtime > 0 else 0.0
            score = refs * self.config.reference_weight + recency * self.config.recency_weight
            scored.append((score, sym))

        scored.sort(key=lambda x: (-x[0], x[1].file, x[1].line))

        return self._render(scored, ref_counts)

    def _render(self, scored: list[tuple[float, Symbol]], ref_counts: Counter[str]) -> RepoMapResult:
        """Render scored symbols into a grouped, budgeted string."""

        budget = self.config.token_budget
        per_file: dict[str, list[tuple[float, Symbol]]] = {}
        for score, sym in scored:
            per_file.setdefault(sym.file, []).append((score, sym))

        ranked_files = sorted(per_file.keys(), key=lambda f: -max(s for s, _ in per_file[f]))

        lines: list[str] = []
        total_tokens = 0
        included = 0

        for file_path in ranked_files:
            file_header = f"\n{file_path}"
            header_tokens = _estimate_tokens(file_header)
            if total_tokens + header_tokens > budget:
                break

            lines.append(file_header)
            total_tokens += header_tokens

            syms = per_file[file_path][: self.config.max_symbols_per_file]
            for _score, sym in syms:
                base_name = sym.name.split(".")[-1]
                refs = ref_counts.get(base_name, 1)
                ref_tag = f"({refs} refs)" if refs > 1 else ""
                sig = sym.signature or sym.name
                entry = f"  {sym.kind}: {sig} {ref_tag}".rstrip()
                entry_tokens = _estimate_tokens(entry)
                if total_tokens + entry_tokens > budget:
                    lines.append("  ...")
                    break
                lines.append(entry)
                total_tokens += entry_tokens
                included += 1

        text = "\n".join(lines).strip()
        return RepoMapResult(
            text=text,
            file_count=len(per_file),
            symbol_count=len(scored),
            included_count=included,
            estimated_tokens=total_tokens,
        )

    def invalidate(self, file_path: str | None = None) -> None:
        """Drop cached symbols for one file (or all files when ``None``)."""

        if file_path is None:
            self._file_cache.clear()
        else:
            self._file_cache.pop(file_path, None)


def build_repo_map(root: str | Path, config: RepoMapConfig | None = None) -> str:
    """Convenience wrapper: return just the rendered map text for ``root``.

    Returns an empty string when the path is invalid or contains no
    extractable symbols. Use :class:`RepoMapper` directly when you need
    the metadata (:class:`RepoMapResult`).
    """

    return RepoMapper(config).build(root).text


__all__ = [
    "RepoMapConfig",
    "RepoMapResult",
    "RepoMapper",
    "Symbol",
    "build_repo_map",
]
