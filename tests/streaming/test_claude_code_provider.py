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
"""Claude Code CLI provider bridge tests."""

from __future__ import annotations

import io
import json

from xerxes.streaming import loop as loop_module
from xerxes.streaming.events import TextChunk
from xerxes.streaming.tool_markers import extract_assistant_tool_call_markers, strip_assistant_tool_call_markers


class _Stdin:
    def __init__(self) -> None:
        self.text = ""
        self.closed = False

    def write(self, text: str) -> int:
        self.text += text
        return len(text)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, lines: list[str]) -> None:
        self.stdin = _Stdin()
        self.stdout = iter(lines)
        self.stderr = io.StringIO("")

    def wait(self) -> int:
        return 0


def test_claude_code_command_maps_model_and_effort() -> None:
    argv = loop_module._claude_code_command("sonnet", {"thinking": True, "reasoning_effort": "high"})
    removed_shell_tool = "Execute" + "Shell"

    assert argv[:6] == ["claude", "-p", "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
    assert "--disable-slash-commands" in argv
    assert argv[argv.index("--tools") + 1] == ""
    disallowed_tools = argv[argv.index("--disallowedTools") + 1]
    assert "Bash" in disallowed_tools
    assert removed_shell_tool in disallowed_tools
    assert ["--model", "sonnet"] == argv[-4:-2]
    assert ["--effort", "high"] == argv[-2:]
    assert "--bare" not in argv


def test_claude_code_default_model_omits_model_flag() -> None:
    argv = loop_module._claude_code_command("default", {})
    removed_shell_tool = "Execute" + "Shell"

    assert "--model" not in argv
    assert argv[:6] == ["claude", "-p", "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
    assert "--disable-slash-commands" in argv
    assert argv[argv.index("--tools") + 1] == ""
    disallowed_tools = argv[argv.index("--disallowedTools") + 1]
    assert "Bash" in disallowed_tools
    assert removed_shell_tool in disallowed_tools


def test_claude_code_env_strips_anthropic_api_credentials(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bad-token")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "bad-token")

    env = loop_module._claude_code_env({})

    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_TOKEN" not in env


def test_claude_code_auth_hint_names_api_env(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad-key")

    hinted = loop_module._claude_code_auth_hint(
        "Failed to authenticate. API Error: 401 Invalid authentication credentials"
    )

    assert "ANTHROPIC_API_KEY" in hinted
    assert "uv run xerxes --refresh" in hinted


def test_claude_code_auth_hint_names_rejected_model(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)

    hinted = loop_module._claude_code_auth_hint(
        "Failed to authenticate. API Error: 401 Invalid authentication credentials",
        "opus",
    )

    assert "explicit model override `opus`" in hinted
    assert "/model claude-code/default" in hinted


def test_claude_code_stream_uses_local_cli_and_parses_function_blocks(monkeypatch, tmp_path) -> None:
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "text",
                    "text": 'Let me read.\n<function=ReadFile>{"file_path":"x.py"}</function>',
                }
            ]
        },
    }
    fake = _FakeProcess([json.dumps(event) + "\n"])
    captured: dict[str, object] = {}

    def _fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return fake

    monkeypatch.setattr(loop_module.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(loop_module.subprocess, "Popen", _fake_popen)

    events = list(
        loop_module._stream_claude_code_cli(
            "sonnet",
            "System instructions",
            [{"role": "user", "content": "inspect x"}],
            [{"name": "ReadFile", "description": "read", "input_schema": {"type": "object"}}],
            {"project_dir": str(tmp_path), "thinking": True, "reasoning_effort": "high"},
        )
    )

    text = "".join(event.text for event in events if isinstance(event, TextChunk))
    final = next(event for event in events if isinstance(event, dict))
    assert text == "Let me read."
    assert final["tool_calls"] == [{"id": "call_cc_0", "name": "ReadFile", "input": {"file_path": "x.py"}}]
    argv = captured["argv"]
    assert argv[:6] == [
        "/usr/bin/claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
    ]
    assert "--disable-slash-commands" in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert "Bash" in argv[argv.index("--disallowedTools") + 1]
    assert ("Execute" + "Shell") in argv[argv.index("--disallowedTools") + 1]
    assert ["--model", "sonnet"] == argv[-4:-2]
    assert ["--effort", "high"] == argv[-2:]
    assert captured["kwargs"]["cwd"] == str(tmp_path)
    assert "env" in captured["kwargs"]
    assert "Xerxes Tool Calling" in fake.stdin.text
    assert "For shell work use Xerxes `exec_command`" in fake.stdin.text
    assert fake.stdin.closed is True


def test_claude_code_stream_strips_and_parses_assistant_tool_call_marker(monkeypatch, tmp_path) -> None:
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Switching to researcher mode.\n"
                        'ASSISTANT_TOOL_CALLS: [{"id":"call_1","name":"SetInteractionModeTool",'
                        '"input":{"mode":"researcher","reason":"read-only audit"}}]'
                    ),
                }
            ]
        },
    }
    fake = _FakeProcess([json.dumps(event) + "\n"])

    def _fake_popen(argv, **kwargs):
        return fake

    monkeypatch.setattr(loop_module.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(loop_module.subprocess, "Popen", _fake_popen)

    events = list(
        loop_module._stream_claude_code_cli(
            "default",
            "System instructions",
            [{"role": "user", "content": "audit only"}],
            [{"name": "SetInteractionModeTool", "description": "mode", "input_schema": {"type": "object"}}],
            {"project_dir": str(tmp_path)},
        )
    )

    text = "".join(event.text for event in events if isinstance(event, TextChunk))
    final = next(event for event in events if isinstance(event, dict))
    assert text == "Switching to researcher mode."
    assert "ASSISTANT_TOOL_CALLS" not in text
    assert final["tool_calls"] == [
        {
            "id": "call_1",
            "name": "SetInteractionModeTool",
            "input": {"mode": "researcher", "reason": "read-only audit"},
        }
    ]


def test_claude_code_stream_drops_native_tool_call_markers(monkeypatch, tmp_path) -> None:
    removed_shell_tool = "Execute" + "Shell"
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "text",
                    "text": "Running tools.\nASSISTANT_TOOL_CALLS: "
                    + json.dumps(
                        [
                            {"id": "call_1", "name": removed_shell_tool, "input": {"command": "pwd"}},
                            {"id": "call_2", "name": "exec_command", "input": {"cmd": "pwd"}},
                        ]
                    ),
                }
            ]
        },
    }
    fake = _FakeProcess([json.dumps(event) + "\n"])

    def _fake_popen(argv, **kwargs):
        return fake

    monkeypatch.setattr(loop_module.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(loop_module.subprocess, "Popen", _fake_popen)

    events = list(
        loop_module._stream_claude_code_cli(
            "default",
            "System instructions",
            [{"role": "user", "content": "count files"}],
            [{"name": "exec_command", "description": "run", "input_schema": {"type": "object"}}],
            {"project_dir": str(tmp_path)},
        )
    )

    text = "".join(event.text for event in events if isinstance(event, TextChunk))
    final = next(event for event in events if isinstance(event, dict))
    assert text == "Running tools."
    assert final["tool_calls"] == [{"id": "call_2", "name": "exec_command", "input": {"cmd": "pwd"}}]


def test_claude_code_stream_drops_native_function_blocks(monkeypatch, tmp_path) -> None:
    removed_shell_tool = "Execute" + "Shell"
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Need shell.\n"
                        f'<function={removed_shell_tool}>{{"command":"pwd"}}</function>'
                        '<function=exec_command>{"cmd":"pwd"}</function>'
                    ),
                }
            ]
        },
    }
    fake = _FakeProcess([json.dumps(event) + "\n"])

    def _fake_popen(argv, **kwargs):
        return fake

    monkeypatch.setattr(loop_module.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(loop_module.subprocess, "Popen", _fake_popen)

    events = list(
        loop_module._stream_claude_code_cli(
            "default",
            "System instructions",
            [{"role": "user", "content": "count files"}],
            [{"name": "exec_command", "description": "run", "input_schema": {"type": "object"}}],
            {"project_dir": str(tmp_path)},
        )
    )

    text = "".join(event.text for event in events if isinstance(event, TextChunk))
    final = next(event for event in events if isinstance(event, dict))
    assert text == "Need shell."
    assert final["tool_calls"] == [{"id": "call_cc_1", "name": "exec_command", "input": {"cmd": "pwd"}}]


def test_claude_code_stream_missing_cli_points_to_xerxes_installer(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loop_module.shutil, "which", lambda _name: None)

    events = list(
        loop_module._stream_claude_code_cli(
            "default",
            "System instructions",
            [{"role": "user", "content": "hi"}],
            [],
            {"project_dir": str(tmp_path)},
        )
    )

    text = "".join(event.text for event in events if isinstance(event, TextChunk))
    final = next(event for event in events if isinstance(event, dict))
    assert "xerxes install --cloud-code" in text
    assert "claude auth login" in text
    assert final["tool_calls"] == []


def test_claude_code_text_extractor_ignores_thinking_blocks() -> None:
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "hidden chain"},
                {"type": "text", "text": "visible answer"},
            ]
        },
    }

    assert loop_module._claude_code_text_from_event(event) == "visible answer"
    assert loop_module._claude_code_text_from_event({"type": "thinking_delta", "delta": {"text": "hidden"}}) == ""
    assert loop_module._claude_code_text_from_event({"type": "tool_result", "content": "hidden"}) == ""


def test_tool_marker_strip_removes_provider_tool_context() -> None:
    cleaned = strip_assistant_tool_call_markers(
        'Confirmed.\nTOOL: {"content":"raw file payload","next_offset":200}\nTOOL_CALL_ID: call_cc_0\nNext point.'
    )

    assert cleaned == "Confirmed.\nNext point."


def test_tool_marker_strip_handles_python_repr_and_glued_markers() -> None:
    cleaned, calls = extract_assistant_tool_call_markers(
        "TOOL: {'ok': True, 'path': '/tmp/result.md'}\n"
        'TOOL_CALL_ID: call_cc_0ASSISTANT_TOOL_CALLS: [{"id":"call_1","name":"SetInteractionModeTool",'
        '"input":{"mode":"researcher"}}]\n'
        "Round 2 complete."
    )

    assert cleaned == "Round 2 complete."
    assert calls == [{"id": "call_1", "name": "SetInteractionModeTool", "input": {"mode": "researcher"}}]


def test_tool_marker_extracts_claude_code_invoke_blocks() -> None:
    cleaned, calls = extract_assistant_tool_call_markers(
        "Updating todos.\n"
        '<invoke name="TodoWriteTool">'
        '<parameter name="todos">[{"id":"1","content":"verify","status":"completed"}]</parameter>'
        "</invoke>\n"
        "<system-reminder>Tool result pending. Write any text first.</system-reminder>\n"
        "Done.",
        id_prefix="call_cc",
    )

    assert cleaned == "Updating todos.\nDone."
    assert calls == [
        {
            "id": "call_cc_0",
            "name": "TodoWriteTool",
            "input": {"todos": [{"id": "1", "content": "verify", "status": "completed"}]},
        }
    ]


def test_claude_code_stream_strips_native_invoke_text(monkeypatch, tmp_path) -> None:
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I'll verify.\n"
                        '<invoke name="exec_command">'
                        '<parameter name="cmd">uv run pytest tests/mcp/</parameter>'
                        '<parameter name="yield_time_ms">120000</parameter>'
                        "</invoke>"
                        "<system-reminder>Tool result pending.</system-reminder>"
                    ),
                }
            ]
        },
    }
    fake = _FakeProcess([json.dumps(event) + "\n"])

    def _fake_popen(argv, **kwargs):
        return fake

    monkeypatch.setattr(loop_module.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(loop_module.subprocess, "Popen", _fake_popen)

    events = list(
        loop_module._stream_claude_code_cli(
            "default",
            "System instructions",
            [{"role": "user", "content": "verify"}],
            [{"name": "exec_command", "description": "run", "input_schema": {"type": "object"}}],
            {"project_dir": str(tmp_path)},
        )
    )

    text = "".join(event.text for event in events if isinstance(event, TextChunk))
    final = next(event for event in events if isinstance(event, dict))
    assert text == "I'll verify."
    assert "<invoke" not in text
    assert "<system-reminder" not in text
    assert final["tool_calls"] == [
        {
            "id": "call_cc_0",
            "name": "exec_command",
            "input": {"cmd": "uv run pytest tests/mcp/", "yield_time_ms": 120000},
        }
    ]
