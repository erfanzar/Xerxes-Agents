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
"""Bootstrap prompt regressions for tool-selection guidance."""

from xerxes.runtime.bootstrap import _build_system_prompt


def test_bootstrap_prompt_prefers_terminal_sessions_for_commands() -> None:
    prompt = _build_system_prompt({})

    assert "exec_command(cmd=..., yield_time_ms=1000)" in prompt
    assert "poll with write_stdin(session_id=..., chars='')" in prompt
    assert "close_terminal_session(session_id=...)" in prompt
    assert "Use ExecuteShell(command=...) only for short blocking one-shot commands." in prompt
