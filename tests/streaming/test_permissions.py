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
