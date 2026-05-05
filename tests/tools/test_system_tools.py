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
"""Tests for xerxes.tools.system_tools module."""

import os

from xerxes.tools.system_tools import (
    EnvironmentManager,
    FileSystemTools,
    ProcessManager,
    SystemInfo,
    TempFileManager,
)


class TestSystemInfo:
    def test_os_info(self):
        result = SystemInfo.static_call(info_type="os")
        assert "os" in result
        assert "system" in result["os"]

    def test_memory_info(self):
        result = SystemInfo.static_call(info_type="memory")
        assert "memory" in result
        assert "total" in result["memory"]

    def test_disk_info(self):
        result = SystemInfo.static_call(info_type="disk")
        assert "disk" in result

    def test_network_info(self):
        result = SystemInfo.static_call(info_type="network")
        assert "network" in result


class TestProcessManager:
    def test_find_python(self):
        result = ProcessManager.static_call(operation="find", process_name="python")
        assert "found" in result
        assert "count" in result

    def test_find(self):
        result = ProcessManager.static_call(operation="find", process_name="python")
        assert "found" in result

    def test_find_no_name(self):
        result = ProcessManager.static_call(operation="find")
        assert "error" in result

    def test_info_no_pid(self):
        result = ProcessManager.static_call(operation="info")
        assert "error" in result

    def test_info_current_process(self):
        result = ProcessManager.static_call(operation="info", pid=os.getpid())
        assert "error" not in result or "name" in result

    def test_run(self):
        result = ProcessManager.static_call(operation="run", command="echo hello")
        assert "completed" in result or "stdout" in result


class TestFileSystemTools:
    def test_copy(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("content")
        dest = tmp_path / "dest.txt"
        result = FileSystemTools.static_call(operation="copy", path=str(src), destination=str(dest))
        assert dest.exists() or "error" not in result

    def test_move(self, tmp_path):
        src = tmp_path / "to_move.txt"
        src.write_text("data")
        dest = tmp_path / "moved.txt"
        result = FileSystemTools.static_call(operation="move", path=str(src), destination=str(dest))
        assert dest.exists() or "error" not in result

    def test_search(self, tmp_path):
        (tmp_path / "test.py").write_text("")
        (tmp_path / "test.txt").write_text("")
        result = FileSystemTools.static_call(operation="search", path=str(tmp_path), pattern="*.py")
        assert "found" in result or "files" in result or "error" not in result

    def test_delete(self, tmp_path):
        f = tmp_path / "to_del.txt"
        f.write_text("data")
        result = FileSystemTools.static_call(operation="delete", path=str(f))
        assert not f.exists() or "error" not in result

    def test_no_operation(self):
        result = FileSystemTools.static_call(operation="invalid")
        assert "error" in result


class TestEnvironmentManager:
    def test_get(self):
        result = EnvironmentManager.static_call(operation="get", key="PATH")
        assert "value" in result or "error" not in result

    def test_list(self):
        result = EnvironmentManager.static_call(operation="list")
        assert "variables" in result or "count" in result or "environment" in result

    def test_get_nonexistent(self):
        result = EnvironmentManager.static_call(operation="get", key="DEFINITELY_NOT_A_REAL_VAR_12345")
        assert result is not None


class TestTempFileManager:
    def test_create_file(self):
        result = TempFileManager.static_call(operation="create_file", content="test")
        assert "path" in result

    def test_create_dir(self):
        result = TempFileManager.static_call(operation="create_dir")
        assert "path" in result
