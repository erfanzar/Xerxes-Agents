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

import json
from pathlib import Path

from xerxes import __main__


def _write_session(home: Path, *, session_id: str, project: Path, title: str, updated_at: str) -> Path:
    sessions = home / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / f"{session_id}.json"
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "key": session_id,
                "agent_id": "default",
                "cwd": str(project),
                "workspace": str(home / "agents" / "default"),
                "updated_at": updated_at,
                "messages": [
                    {"role": "user", "content": f"read {project.name}"},
                    {"role": "assistant", "content": f"report for {project.name}"},
                ],
                "turn_count": 1,
                "interaction_mode": "code",
                "plan_mode": False,
                "total_input_tokens": 10,
                "total_output_tokens": 20,
                "metadata": {"title": title},
                "thinking_content": ["thought"],
                "tool_executions": [{"name": "ReadFile", "ok": True}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_export_session_writes_full_json_trace(monkeypatch, tmp_path, capsys) -> None:
    home = tmp_path / "home"
    project = tmp_path / "lovely-pirate"
    project.mkdir()
    monkeypatch.setenv("XERXES_HOME", str(home))
    record_path = _write_session(
        home,
        session_id="abc12345",
        project=project,
        title="clone lovely pirate",
        updated_at="2026-06-27T10:00:00+00:00",
    )
    record_path.with_name("abc12345.archive.jsonl").write_text(
        json.dumps({"role": "system", "content": "archived setup"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "trace.json"

    __main__.main(["export", "--project", str(project), "--session", "abc12345", "--output", str(output)])

    assert "Exported session abc12345" in capsys.readouterr().out
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema"] == "xerxes.session.export.v1"
    assert data["session"]["project_dir"] == str(project.resolve())
    assert data["archive_included"] is True
    assert [message["content"] for message in data["messages"]] == [
        "archived setup",
        "read lovely-pirate",
        "report for lovely-pirate",
    ]
    assert data["live_messages"][0]["content"] == "read lovely-pirate"
    assert data["archive_messages"][0]["content"] == "archived setup"
    assert data["tool_executions"] == [{"name": "ReadFile", "ok": True}]


def test_export_list_filters_by_project(monkeypatch, tmp_path, capsys) -> None:
    home = tmp_path / "home"
    project = tmp_path / "lovely-pirate"
    other = tmp_path / "xerxes-agents"
    project.mkdir()
    other.mkdir()
    monkeypatch.setenv("XERXES_HOME", str(home))
    _write_session(
        home,
        session_id="abc12345",
        project=project,
        title="target session",
        updated_at="2026-06-27T10:00:00+00:00",
    )
    _write_session(
        home,
        session_id="def67890",
        project=other,
        title="other session",
        updated_at="2026-06-27T11:00:00+00:00",
    )

    __main__.main(["export", "--list", "--project", str(project)])

    output = capsys.readouterr().out
    assert "abc12345" in output
    assert "target session" in output
    assert "def67890" not in output


def test_export_defaults_to_latest_session_for_project(monkeypatch, tmp_path, capsys) -> None:
    home = tmp_path / "home"
    project = tmp_path / "lovely-pirate"
    project.mkdir()
    monkeypatch.setenv("XERXES_HOME", str(home))
    _write_session(
        home,
        session_id="old11111",
        project=project,
        title="old",
        updated_at="2026-06-27T09:00:00+00:00",
    )
    _write_session(
        home,
        session_id="new22222",
        project=project,
        title="new",
        updated_at="2026-06-27T12:00:00+00:00",
    )

    __main__.main(["export", "--project", str(project), "--format", "jsonl"])

    lines = capsys.readouterr().out.splitlines()
    header = json.loads(lines[0])
    assert header["type"] == "session"
    assert header["session"]["id"] == "new22222"


def test_export_lovely_pirate_jsonl_uses_external_agent_events(monkeypatch, tmp_path, capsys) -> None:
    home = tmp_path / "home"
    project = tmp_path / "lovely-pirate"
    project.mkdir()
    monkeypatch.setenv("XERXES_HOME", str(home))
    record_path = _write_session(
        home,
        session_id="abc12345",
        project=project,
        title="target session",
        updated_at="2026-06-27T10:00:00+00:00",
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["metadata"] = {"title": "target session", "provider": "claude-code", "model": "claude-code/opus"}
    record["messages"] = [
        {"role": "user", "content": [{"type": "text", "text": "read lovely pirate"}]},
        {
            "role": "assistant",
            "content": "I will inspect the repo.",
            "reasoning_content": "brief plan",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ReadFile", "arguments": '{"file_path":"README.md"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "ReadFile", "content": {"text": "README body"}},
        {"role": "assistant", "content": "Done."},
    ]
    record_path.write_text(json.dumps(record), encoding="utf-8")

    __main__.main(["export", "--project", str(project), "--session", "abc12345", "--format", "lovely-pirate"])

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["type"] for row in rows] == [
        "external_session_meta",
        "external_message",
        "external_message",
        "external_message",
        "external_message",
    ]
    assert rows[0]["payload"]["source"] == "xerxes"
    assert rows[0]["payload"]["cwd"] == str(project.resolve())
    assert rows[0]["payload"]["model"] == "claude-code/opus"
    assert rows[1]["role"] == "user"
    assert rows[1]["content"] == "read lovely pirate"
    assert rows[2]["reasoning_content"] == "brief plan"
    assert rows[2]["tool_calls"][0]["function"]["name"] == "ReadFile"
    assert rows[3]["role"] == "tool"
    assert rows[3]["tool_call_id"] == "call_1"
    assert rows[3]["name"] == "ReadFile"
