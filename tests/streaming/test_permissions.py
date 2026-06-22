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

from __future__ import annotations

from xerxes.streaming.permissions import PermissionMode, check_permission, is_safe_bash


def test_cd_is_safe_bash_without_shell_metacharacters() -> None:
    assert is_safe_bash("cd")
    assert is_safe_bash("cd /tmp")
    assert is_safe_bash("cd ../project && pwd")
    assert is_safe_bash("cd /tmp && git diff")
    assert is_safe_bash("cd /tmp && rg TODO")


def test_cd_with_shell_metacharacters_still_requires_approval() -> None:
    assert not is_safe_bash("cd /tmp; rm -rf build")
    assert not is_safe_bash("cd $(mktemp -d)")
    assert not is_safe_bash("cd /tmp && npm install")


def test_execute_shell_cd_auto_permission() -> None:
    assert check_permission(
        {"name": "ExecuteShell", "input": {"command": "cd /tmp && git diff"}},
        PermissionMode.AUTO,
    )


def test_ask_user_question_is_safe_in_auto_mode() -> None:
    assert check_permission(
        {"name": "AskUserQuestionTool", "input": {"question": "Continue?"}},
        PermissionMode.AUTO,
    )
