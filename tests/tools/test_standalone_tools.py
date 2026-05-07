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
"""Tests for xerxes.tools.standalone module."""

import pytest
from xerxes.tools.standalone import (
    AppendFile,
    ExecutePythonCode,
    ExecuteShell,
    ListDir,
    ReadFile,
    WriteFile,
)


class TestReadFile:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = ReadFile.static_call(str(f))
        assert result == "hello world"

    def test_read_truncated(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 10000)
        result = ReadFile.static_call(str(f), max_chars=100)
        assert len(result) < 200
        assert "truncated" in result

    def test_read_large_file_not_truncated_by_default(self, tmp_path):
        f = tmp_path / "big.txt"
        content = "x" * 10000
        f.write_text(content)
        result = ReadFile.static_call(str(f))
        assert result == content

    def test_read_no_truncation(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("hello")
        result = ReadFile.static_call(str(f), max_chars=None)
        assert result == "hello"

    def test_read_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            ReadFile.static_call("/nonexistent/file.txt")


class TestWriteFile:
    def test_write_new_file(self, tmp_path):
        f = tmp_path / "output.txt"
        result = WriteFile.static_call(str(f), "hello")
        assert "Wrote" in result
        assert f.read_text() == "hello"

    def test_write_creates_dirs(self, tmp_path):
        f = tmp_path / "sub" / "dir" / "file.txt"
        WriteFile.static_call(str(f), "content")
        assert f.exists()

    def test_write_no_overwrite(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old")
        with pytest.raises(FileExistsError):
            WriteFile.static_call(str(f), "new", overwrite=False)

    def test_write_overwrite(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old")
        WriteFile.static_call(str(f), "new", overwrite=True)
        assert f.read_text() == "new"


class TestListDir:
    def test_list_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = ListDir.static_call(str(tmp_path))
        assert "a.py" in result
        assert "b.txt" in result

    def test_list_with_filter(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = ListDir.static_call(str(tmp_path), extension_filter=".py")
        assert "a.py" in result
        assert "b.txt" not in result

    def test_list_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            ListDir.static_call("/nonexistent/dir")

    def test_list_excludes_dirs(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("")
        result = ListDir.static_call(str(tmp_path))
        assert "subdir" not in result
        assert "file.txt" in result


class TestExecutePythonCode:
    def test_simple_code(self):
        result = ExecutePythonCode.static_call("print('hello')")
        assert result["stdout"].strip() == "hello"
        assert result["stderr"] == ""

    def test_error_code(self):
        result = ExecutePythonCode.static_call("raise ValueError('oops')")
        assert "ValueError" in result["stderr"]

    def test_multiline_code(self):
        code = "x = 1 + 2\nprint(x)"
        result = ExecutePythonCode.static_call(code)
        assert "3" in result["stdout"]


class TestExecuteShell:
    def test_simple_command(self):
        result = ExecuteShell.static_call("echo hello")
        assert "hello" in result["stdout"]

    def test_with_cwd(self, tmp_path):
        result = ExecuteShell.static_call("pwd", cwd=str(tmp_path))
        assert str(tmp_path) in result["stdout"] or tmp_path.name in result["stdout"]


class TestAppendFile:
    def test_append_new_file(self, tmp_path):
        f = tmp_path / "log.txt"
        AppendFile.static_call(str(f), "line1")
        assert "line1" in f.read_text()

    def test_append_existing(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("line1\n")
        AppendFile.static_call(str(f), "line2")
        content = f.read_text()
        assert "line1" in content
        assert "line2" in content

    def test_append_creates_dirs(self, tmp_path):
        f = tmp_path / "sub" / "log.txt"
        AppendFile.static_call(str(f), "data")
        assert f.exists()
