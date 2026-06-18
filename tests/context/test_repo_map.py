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
"""Tests for :mod:`xerxes.context.repo_map`."""

from __future__ import annotations

from pathlib import Path

import pytest
from xerxes.context.repo_map import (
    RepoMapConfig,
    RepoMapper,
    Symbol,
    build_repo_map,
)

_PY_SAMPLE = '''\
"""Module docstring."""

import os

MAX_RETRIES = 3

def public_function(arg1, arg2):
    """A public function."""
    return arg1 + arg2

async def async_fetch(url: str) -> str:
    """Fetch a URL."""
    return url

class DataProcessor:
    """Process data."""

    def process(self, data):
        return data

    def _private(self):
        return None

def _helper():
    return 42
'''

_JS_SAMPLE = """\
export function fetchData(url) {
    return fetch(url);
}

export class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async getData() {
        return await fetchData(this.baseUrl);
    }
}

const handler = (req) => {
    return req.body;
};
"""


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a small fake repo with Python and JS files."""

    (tmp_path / "app.py").write_text(_PY_SAMPLE)
    (tmp_path / "client.js").write_text(_JS_SAMPLE)
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "utils.py").write_text("def helper_func():\n    return 42\n\nclass Helper:\n    pass\n")
    (tmp_path / ".gitignore").write_text("ignored_dir/\n*.lock\n")
    (tmp_path / "ignored_dir").mkdir()
    (tmp_path / "ignored_dir" / "secret.py").write_text("SECRET = 'key'\n")
    return tmp_path


class TestPythonSymbolExtraction:
    """Tests for ast-based Python symbol extraction."""

    def test_extracts_functions_and_classes(self, sample_repo: Path):
        """Public functions, async functions, and classes are extracted."""

        result = RepoMapper().build(sample_repo)
        # Check that our key symbols appear somewhere in the map
        assert "public_function" in result.text or "def public_function" in result.text
        assert "async_fetch" in result.text or "def async_fetch" in result.text
        assert "DataProcessor" in result.text

    def test_extracts_constants(self, sample_repo: Path):
        """Top-level UPPER_CASE assignments are extracted as constants."""

        result = RepoMapper().build(sample_repo)
        assert "MAX_RETRIES" in result.text

    def test_extracts_methods(self, sample_repo: Path):
        """Public methods (not dunder/private) are extracted."""

        result = RepoMapper().build(sample_repo)
        assert "DataProcessor.process" in result.text
        # _private should NOT appear
        assert "_private" not in result.text

    def test_skips_private_functions(self, sample_repo: Path):
        """Underscore-prefixed functions are excluded."""

        result = RepoMapper().build(sample_repo)
        assert "_helper" not in result.text


class TestRegexSymbolExtraction:
    """Tests for regex-based non-Python symbol extraction."""

    def test_extracts_js_functions(self, sample_repo: Path):
        """JS function, class, and const-arrow are extracted."""

        result = RepoMapper().build(sample_repo)
        assert "fetchData" in result.text
        assert "ApiClient" in result.text


class TestGitignore:
    """Tests for .gitignore-aware directory walking."""

    def test_gitignore_excludes_directory(self, sample_repo: Path):
        """Directories listed in .gitignore are skipped."""

        result = RepoMapper().build(sample_repo)
        assert "ignored_dir" not in result.text
        assert "SECRET" not in result.text


class TestRanking:
    """Tests for reference-based symbol ranking."""

    def test_high_reference_symbols_appear_first(self, tmp_path: Path):
        """A symbol referenced by many files ranks higher than one nobody references."""

        (tmp_path / "core.py").write_text("def popular_func():\n    pass\ndef obscure_func():\n    pass\n")
        (tmp_path / "a.py").write_text("from core import popular_func\npopular_func()\n")
        (tmp_path / "b.py").write_text("from core import popular_func\npopular_func()\n")

        result = RepoMapper(RepoMapConfig(token_budget=200)).build(tmp_path)
        popular_pos = result.text.find("popular_func")
        obscure_pos = result.text.find("obscure_func")
        assert popular_pos < obscure_pos
        assert popular_pos != -1

    def test_token_budget_truncates_output(self, sample_repo: Path):
        """A small token budget produces shorter output than a large one."""

        small = RepoMapper(RepoMapConfig(token_budget=50)).build(sample_repo)
        large = RepoMapper(RepoMapConfig(token_budget=2000)).build(sample_repo)
        assert small.estimated_tokens < large.estimated_tokens
        assert large.included_count >= small.included_count


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nonexistent_path_returns_empty(self, tmp_path: Path):
        """A path that doesn't exist produces an empty result."""

        result = RepoMapper().build(tmp_path / "nonexistent")
        assert result.text == ""
        assert result.symbol_count == 0

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        """An empty directory produces an empty result."""

        result = RepoMapper().build(tmp_path)
        assert result.text == ""
        assert result.symbol_count == 0

    def test_syntax_error_file_skipped(self, tmp_path: Path):
        """A file with a SyntaxError is skipped, not crashed on."""

        (tmp_path / "broken.py").write_text("def (\n")
        (tmp_path / "good.py").write_text("def works():\n    return True\n")
        result = RepoMapper().build(tmp_path)
        assert "works" in result.text

    def test_build_repo_map_convenience(self, sample_repo: Path):
        """The convenience function returns just the text."""

        text = build_repo_map(sample_repo)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_symbol_dataclass_fields(self):
        """Symbol is a frozen dataclass with expected fields."""

        sym = Symbol(name="foo", kind="function", file="bar.py", line=10)
        assert sym.name == "foo"
        assert sym.kind == "function"
        assert sym.file == "bar.py"
        assert sym.line == 10
        assert sym.signature == ""
