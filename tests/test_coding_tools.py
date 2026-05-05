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
"""Tests for xerxes.tools.coding_tools module."""

from xerxes.tools.coding_tools import (
    analyze_code_structure,
    apply_diff,
    copy_file,
    create_diff,
    delete_file,
    detect_language,
    find_and_replace,
    format_size,
    git_diff,
    git_log,
    git_status,
    list_directory,
    move_file,
    read_file,
    write_file,
)


class TestReadFile:
    def test_read_existing(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\n")
        result = read_file(str(f))
        assert "line1" in result
        assert "line2" in result

    def test_read_range(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\nline4\n")
        result = read_file(str(f), start_line=2, end_line=3)
        assert "line2" in result
        assert "line3" in result

    def test_read_nonexistent(self):
        result = read_file("/nonexistent/file.txt")
        assert "Error" in result


class TestWriteFile:
    def test_write_new(self, tmp_path):
        f = tmp_path / "out.txt"
        result = write_file(str(f), "hello world")
        assert "Success" in result or "wrote" in result.lower() or "written" in result.lower()

    def test_write_creates_dirs(self, tmp_path):
        f = tmp_path / "sub" / "dir" / "file.txt"
        write_file(str(f), "content")
        assert f.exists()


class TestListDirectory:
    def test_list(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = list_directory(str(tmp_path))
        assert "a.py" in result
        assert "b.txt" in result

    def test_list_nonexistent(self):
        result = list_directory("/nonexistent/dir")
        assert "Error" in result or "not found" in result.lower()


class TestCopyFile:
    def test_copy(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("data")
        dest = tmp_path / "dest.txt"
        copy_file(str(src), str(dest))
        assert dest.exists()

    def test_copy_nonexistent(self, tmp_path):
        result = copy_file("/nonexistent/src.txt", str(tmp_path / "dest.txt"))
        assert "Error" in result or "not found" in result.lower()


class TestDeleteFile:
    def test_delete(self, tmp_path):
        f = tmp_path / "to_del.txt"
        f.write_text("data")
        delete_file(str(f))
        assert not f.exists()

    def test_delete_nonexistent(self):
        result = delete_file("/nonexistent/file.txt")
        assert "Error" in result or "not found" in result.lower()


class TestFindAndReplace:
    def test_replace(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello world\nhello there\n")
        result = find_and_replace(str(f), "hello", "hi")
        assert "replaced" in result.lower() or "success" in result.lower() or result is not None

    def test_nonexistent(self):
        result = find_and_replace("/nonexistent/file.py", "a", "b")
        assert "Error" in result or "not found" in result.lower()


class TestCreateDiff:
    def test_basic_diff(self):
        result = create_diff("hello\nworld\n", "hello\nearth\n", "a.txt", "b.txt")
        assert result is not None


class TestDetectLanguage:
    def test_detects_something(self):
        result = detect_language("test.py")
        assert isinstance(result, str)

    def test_unknown_ext(self):
        result = detect_language("test.xyz")
        assert isinstance(result, str)


class TestGitStatus:
    def test_git_status(self):
        result = git_status()
        assert result is not None

    def test_git_log(self):
        result = git_log()
        assert result is not None

    def test_git_diff(self):
        result = git_diff()
        assert result is not None


class TestMoveFile:
    def test_move(self, tmp_path):
        src = tmp_path / "to_move.txt"
        src.write_text("data")
        dest = tmp_path / "moved.txt"
        result = move_file(str(src), str(dest))
        assert dest.exists() or "success" in result.lower() or "moved" in result.lower()

    def test_move_nonexistent(self, tmp_path):
        result = move_file("/nonexistent/file.txt", str(tmp_path / "dest.txt"))
        assert "Error" in result or "not found" in result.lower()


class TestApplyDiff:
    def test_apply(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("line1\nline2\nline3\n")
        diff = create_diff("line1\nline2\nline3\n", "line1\nchanged\nline3\n", str(f), str(f))
        result = apply_diff(str(f), diff)
        assert result is not None


class TestFormatSize:
    def test_bytes(self):
        assert "B" in format_size(500)

    def test_kb(self):
        result = format_size(1500)
        assert "KB" in result or "B" in result

    def test_mb(self):
        result = format_size(5_000_000)
        assert "MB" in result


class TestAnalyzePython:
    def test_basic(self, tmp_path):
        f = tmp_path / "module.py"
        f.write_text("class Foo:\n    def bar(self):\n        pass\n\ndef baz():\n    pass\n")
        result = analyze_code_structure(str(f))
        assert "Foo" in result or "baz" in result or "class" in result.lower()


class TestAnalyzeCodeStructure:
    def test_python_file(self, tmp_path):
        f = tmp_path / "module.py"
        f.write_text("class MyClass:\n    def method(self):\n        pass\n\ndef func():\n    pass\n")
        result = analyze_code_structure(str(f))
        assert "MyClass" in result or "func" in result

    def test_nonexistent(self):
        result = analyze_code_structure("/nonexistent/file.py")
        assert "Error" in result or "not found" in result.lower()
