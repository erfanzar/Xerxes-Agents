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

from pathlib import Path

from xerxes.runtime.change_guard import ChangeGuardFinding, ChangeGuardReport
from xerxes.streaming.events import AgentState
from xerxes.streaming.loop import _inject_workspace_guard_message


def test_workspace_guard_injects_model_message_once(monkeypatch) -> None:
    report = ChangeGuardReport(
        findings=(
            ChangeGuardFinding(
                severity="error",
                code="deleted-tests",
                path="tests/test_example.py",
                message="1 tracked test file(s) were deleted",
            ),
        ),
        verification_commands=(),
    )

    def _fake_analyze(cwd: Path, tool_executions: list[dict]) -> ChangeGuardReport:
        return report

    monkeypatch.setattr("xerxes.streaming.loop.analyze_workspace_changes", _fake_analyze)
    state = AgentState()

    _inject_workspace_guard_message(state, config={"project_dir": "/tmp/project"})
    _inject_workspace_guard_message(state, config={"project_dir": "/tmp/project"})

    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "user"
    assert "[Workspace guard]" in state.messages[0]["content"]
    assert "Do not claim completion" in state.messages[0]["content"]
    assert state.metadata["last_change_guard_model_fingerprint"] == report.fingerprint
