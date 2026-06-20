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
"""Regression tests for tool-call JSON nulls on read-file chunk arguments."""

from xerxes.tools.coding_tools import read_file
from xerxes.tools.standalone import ReadFile


def test_standalone_read_file_null_chunk_args_use_defaults(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("".join(f"payload-{idx}\n" for idx in range(1000)))

    result = ReadFile.static_call(str(path), offset=None, limit=None)

    assert "payload-0" in result
    assert "payload-399" in result
    assert "payload-400" not in result
    assert "Continue with offset=400, limit=400" in result


def test_coding_read_file_null_line_args_use_defaults(tmp_path):
    path = tmp_path / "big.py"
    path.write_text("".join(f"payload-{idx}\n" for idx in range(1000)))

    result = read_file(str(path), start_line=None, end_line=None)

    assert "payload-0" in result
    assert "payload-399" in result
    assert "payload-400" not in result
    assert "Continue with start_line=401" in result
