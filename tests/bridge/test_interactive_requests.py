from __future__ import annotations

import io
import json

from xerxes.bridge.server import BridgeServer
from xerxes.streaming.wire_events import ApprovalRequest, QuestionRequest, event_from_dict


def _wire_payloads(output: str) -> list[dict]:
    return [json.loads(line)["params"]["payload"] for line in output.splitlines()]


def test_permission_request_payload_parses_in_tui_wire_model() -> None:
    server = BridgeServer(wire_mode=True)
    output = io.StringIO()
    server._stdout = output

    server._emit_wire_permission_request("tool_1", "ExecuteShell", "Run: git diff")

    payload = _wire_payloads(output.getvalue())[0]
    payload["type"] = "ApprovalRequest"
    event = event_from_dict(payload)

    assert isinstance(event, ApprovalRequest)
    assert event.id
    assert event.tool_call_id == "tool_1"
    assert event.action == "ExecuteShell"


def test_question_request_payload_parses_in_tui_wire_model() -> None:
    server = BridgeServer(wire_mode=True)
    output = io.StringIO()
    server._stdout = output

    server._emit_wire_question_request(
        [
            {
                "id": "q1",
                "question": "Continue?",
                "options": ["yes", "no"],
                "allow_free_form": False,
            }
        ]
    )

    payload = _wire_payloads(output.getvalue())[0]
    payload["type"] = "QuestionRequest"
    event = event_from_dict(payload)

    assert isinstance(event, QuestionRequest)
    assert event.id
    assert event.questions[0]["question"] == "Continue?"


def test_subagent_done_only_clears_transient_preview() -> None:
    server = BridgeServer(wire_mode=True)
    output = io.StringIO()
    server._stdout = output

    server._emit_subagent_summary(
        "agent_done",
        {
            "task_id": "task_123",
            "agent_name": "worker",
            "status": "completed",
            "result": "final output should not be appended",
        },
    )

    payloads = _wire_payloads(output.getvalue())

    assert len(payloads) == 1
    assert payloads[0]["category"] == "subagent_stream"
    assert payloads[0]["body"] == ""
