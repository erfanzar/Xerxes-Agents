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
"""Freeze the TUI ⇄ daemon wire contract consumed by ``src/ui-tui/``.

The TypeScript/Ink frontend in ``src/ui-tui/`` is a JSON-RPC peer that cannot
import Python; it depends on the snake_case ⇄ PascalCase event-name map and the
field names of each wire event staying stable. This test is the canary: if it
fails, ``src/ui-tui/PROTOCOL.md`` and ``src/ui-tui/src/gatewayTypes.ts`` must change in
lockstep with the Python side.
"""

from __future__ import annotations

import dataclasses

from xerxes.streaming import wire_events as we

# The exact wire names the TS client keys on. Adding an event is allowed (extend
# this set), but renaming or removing one is a breaking change for src/ui-tui.
EXPECTED_EVENT_NAME_MAP: dict[str, str] = {
    "init_done": "InitDone",
    "turn_begin": "TurnBegin",
    "turn_end": "TurnEnd",
    "step_begin": "StepBegin",
    "step_end": "StepEnd",
    "step_interrupted": "StepInterrupted",
    "steer_input": "SteerInput",
    "compaction_begin": "CompactionBegin",
    "compaction_end": "CompactionEnd",
    "hook_triggered": "HookTriggered",
    "hook_resolved": "HookResolved",
    "mcp_loading_begin": "MCPLoadingBegin",
    "mcp_loading_end": "MCPLoadingEnd",
    "btw_begin": "BtwBegin",
    "btw_end": "BtwEnd",
    "text_part": "TextPart",
    "think_part": "ThinkPart",
    "image_url_part": "ImageURLPart",
    "audio_url_part": "AudioURLPart",
    "video_url_part": "VideoURLPart",
    "tool_call": "ToolCall",
    "tool_call_part": "ToolCallPart",
    "tool_result": "ToolResult",
    "tool_call_request": "ToolCallRequest",
    "approval_request": "ApprovalRequest",
    "approval_response": "ApprovalResponse",
    "question_request": "QuestionRequest",
    "question_response": "QuestionResponse",
    "status_update": "StatusUpdate",
    "notification": "Notification",
    "plan_display": "PlanDisplay",
    "subagent_event": "SubagentEvent",
}

# Per-event payload fields the TS gatewayTypes.ts relies on. Each entry is a
# subset that MUST exist; extra fields are fine.
EXPECTED_EVENT_FIELDS: dict[str, set[str]] = {
    "init_done": {
        "model",
        "session_id",
        "cwd",
        "git_branch",
        "head_hash",
        "context_limit",
        "agent_name",
        "skills",
        "skill_descriptions",
        "mode",
        "version",
    },
    "text_part": {"text"},
    "think_part": {"think"},
    "tool_call": {"id", "name", "arguments"},
    "tool_call_part": {"arguments_part"},
    "tool_result": {"tool_call_id", "return_value", "duration_ms", "display_blocks"},
    "tool_call_request": {"id", "tool_call_id", "name", "arguments"},
    "approval_request": {"id", "tool_call_id", "action", "description"},
    "question_request": {"id", "tool_call_id", "questions"},
    "status_update": {"context_tokens", "max_context", "mcp_status", "plan_mode", "mode", "reasoning_effort"},
    "notification": {"id", "category", "type", "severity", "title", "body", "payload"},
    "plan_display": {"content", "file_path"},
    "subagent_event": {"parent_tool_call_id", "agent_id", "subagent_type", "event"},
    "step_begin": {"n"},
    "step_end": {"n"},
    "steer_input": {"content"},
}


def test_event_name_map_is_frozen() -> None:
    """Every documented event keeps its snake_case ⇄ PascalCase pairing."""
    for snake, pascal in EXPECTED_EVENT_NAME_MAP.items():
        assert we.to_kimi_event_name(snake) == pascal, f"{snake} must serialise as {pascal}"
        assert we.to_internal_event_name(pascal) == snake, f"{pascal} must decode to {snake}"


def test_registry_accepts_both_names() -> None:
    """``event_from_dict`` decodes both wire spellings to the same class."""
    for snake, pascal in EXPECTED_EVENT_NAME_MAP.items():
        from_snake = we.event_from_dict({"type": snake})
        from_pascal = we.event_from_dict({"type": pascal})
        assert type(from_snake) is type(from_pascal), f"{snake}/{pascal} must map to one class"
        # Unknown discriminators fall back to GenericWireEvent; the documented
        # set must all be concrete.
        assert not isinstance(from_snake, we.GenericWireEvent), f"{snake} should be a concrete WireEvent"


def test_event_payload_fields_present() -> None:
    """Each event still carries the fields gatewayTypes.ts reads."""
    for snake, required in EXPECTED_EVENT_FIELDS.items():
        cls = we.event_from_dict({"type": snake}).__class__
        assert dataclasses.is_dataclass(cls), f"{snake} -> {cls!r} must be a dataclass"
        field_names = {f.name for f in dataclasses.fields(cls)}
        missing = required - field_names
        assert not missing, f"{snake} payload lost fields: {sorted(missing)}"


def test_event_to_dict_round_trips_to_pascalcase() -> None:
    """Serialisation emits the PascalCase ``{type, payload}`` the client parses.

    Uses a top-level :class:`WireEvent` subclass (carries ``event_type``).
    Content *parts* such as ``TextPart`` use a ``type`` field instead and are
    broadcast directly as ``{type, payload}`` rather than via ``event_to_dict``.
    """
    wire = we.event_to_dict(we.InitDone(model="m", agent_name="Xerxes"))
    assert wire["type"] == "InitDone"
    assert wire["payload"]["model"] == "m"
    assert wire["payload"]["agent_name"] == "Xerxes"
    assert "event_type" not in wire["payload"]
