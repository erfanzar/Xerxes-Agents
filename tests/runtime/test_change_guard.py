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

from xerxes.runtime.change_guard import (
    analyze_status_lines,
    format_change_guard_notification,
    parse_porcelain_status,
)


def test_deleted_tests_are_error_even_with_verification() -> None:
    report = analyze_status_lines(
        [
            " D tests/tools/test_standalone_tools.py",
            " M src/python/xerxes/tools/standalone.py",
        ],
        [
            {
                "name": "exec_command",
                "inputs": {"cmd": "uv run pytest tests/tools/test_read_file_null_defaults.py -q"},
            }
        ],
    )

    assert report.should_notify
    assert report.severity == "error"
    assert [finding.code for finding in report.findings] == ["deleted-tests", "runtime-critical-changed"]
    assert report.verification_commands == ("uv run pytest tests/tools/test_read_file_null_defaults.py -q",)


def test_runtime_change_without_verification_notifies() -> None:
    report = analyze_status_lines([" M src/python/xerxes/daemon/runtime.py"], [])

    assert report.should_notify
    assert report.severity == "warning"
    assert report.findings[0].code == "runtime-critical-changed"
    assert "No recent pytest" in format_change_guard_notification(report)


def test_runtime_change_with_verification_suppresses_warning() -> None:
    report = analyze_status_lines(
        [" M src/python/xerxes/daemon/runtime.py"],
        [{"name": "exec_command", "inputs": {"cmd": "uv run ruff check src/python/xerxes/daemon/runtime.py"}}],
    )

    assert not report.should_notify
    assert report.verification_commands == ("uv run ruff check src/python/xerxes/daemon/runtime.py",)


def test_parse_porcelain_rename_uses_new_path() -> None:
    changes = parse_porcelain_status(["R  tests/test_old.py -> tests/test_new.py"])

    assert changes[0].old_path == "tests/test_old.py"
    assert changes[0].path == "tests/test_new.py"
