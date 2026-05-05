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
"""Wire events module for Xerxes.

Exports:
    - BriefDisplayBlock
    - DiffDisplayBlock
    - TodoDisplayBlock
    - BackgroundTaskDisplayBlock
    - GenericDisplayBlock
    - DisplayBlock
    - TextPart
    - ThinkPart
    - ImageURLPart
    - AudioURLPart
    - ... and 44 more."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class BriefDisplayBlock:
    """A short plain-text display block.

    Args:
        type (Literal["brief"]): IN: fixed tag "brief". OUT: identifies the block kind.
        body (str): IN: text to display. OUT: rendered as brief content.
    """

    type: Literal["brief"] = "brief"
    body: str = ""


@dataclass(frozen=True)
class DiffDisplayBlock:
    """A code-diff display block.

    Args:
        type (Literal["diff"]): IN: fixed tag "diff". OUT: identifies the block kind.
        diff (str): IN: diff text content. OUT: rendered as a diff.
        language (str): IN: programming language for syntax highlighting. OUT: used by the renderer.
    """

    type: Literal["diff"] = "diff"
    diff: str = ""
    language: str = ""


@dataclass(frozen=True)
class TodoDisplayBlock:
    """A todo-list display block.

    Args:
        type (Literal["todo"]): IN: fixed tag "todo". OUT: identifies the block kind.
        items (list[dict[str, Any]]): IN: list of todo item dicts. OUT: rendered as a checklist.
    """

    type: Literal["todo"] = "todo"
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class BackgroundTaskDisplayBlock:
    """A background-task status display block.

    Args:
        type (Literal["background_task"]): IN: fixed tag "background_task". OUT: identifies the block kind.
        title (str): IN: human-readable task title. OUT: shown in the UI.
        status (str): IN: current status string. OUT: shown in the UI.
    """

    type: Literal["background_task"] = "background_task"
    title: str = ""
    status: str = ""


@dataclass(frozen=True)
class GenericDisplayBlock:
    """A generic catch-all display block.

    Args:
        type (Literal["generic"]): IN: fixed tag "generic". OUT: identifies the block kind.
        content (str): IN: raw content string. OUT: rendered as plain text.
    """

    type: Literal["generic"] = "generic"
    content: str = ""


DisplayBlock = (
    BriefDisplayBlock
    | DiffDisplayBlock
    | TodoDisplayBlock
    | BackgroundTaskDisplayBlock
    | GenericDisplayBlock
)


@dataclass(frozen=True)
class TextPart:
    """A plain-text content part.

    Args:
        type (Literal["text"]): IN: fixed tag "text". OUT: identifies the part kind.
        text (str): IN: text payload. OUT: rendered as message text.
    """

    type: Literal["text"] = "text"
    text: str = ""


@dataclass(frozen=True)
class ThinkPart:
    """A "thinking" content part (model reasoning).

    Args:
        type (Literal["think"]): IN: fixed tag "think". OUT: identifies the part kind.
        think (str): IN: reasoning text. OUT: shown in a collapsible think block.
    """

    type: Literal["think"] = "think"
    think: str = ""


@dataclass(frozen=True)
class ImageURLPart:
    """An image referenced by URL.

    Args:
        type (Literal["image_url"]): IN: fixed tag "image_url". OUT: identifies the part kind.
        url (str): IN: image URL. OUT: loaded and displayed by the client.
        alt (str | None): IN: optional alt text. OUT: used for accessibility.
    """

    type: Literal["image_url"] = "image_url"
    url: str = ""
    alt: str | None = None


@dataclass(frozen=True)
class AudioURLPart:
    """An audio clip referenced by URL.

    Args:
        type (Literal["audio_url"]): IN: fixed tag "audio_url". OUT: identifies the part kind.
        url (str): IN: audio URL. OUT: played by the client.
    """

    type: Literal["audio_url"] = "audio_url"
    url: str = ""


@dataclass(frozen=True)
class VideoURLPart:
    """A video referenced by URL.

    Args:
        type (Literal["video_url"]): IN: fixed tag "video_url". OUT: identifies the part kind.
        url (str): IN: video URL. OUT: played by the client.
        alt (str | None): IN: optional alt text. OUT: used for accessibility.
    """

    type: Literal["video_url"] = "video_url"
    url: str = ""
    alt: str | None = None


ContentPart = TextPart | ThinkPart | ImageURLPart | AudioURLPart | VideoURLPart


@dataclass(frozen=True)
class QuestionItem:
    """A single question in an interactive question request.

    Args:
        id (str): IN: unique question identifier. OUT: used to match answers.
        question (str): IN: question text. OUT: displayed to the user.
        options (list[str]): IN: selectable answer options. OUT: shown as choices.
        allow_free_form (bool): IN: whether free-text input is allowed. OUT: controls UI behaviour.
    """

    id: str = ""
    question: str = ""
    options: list[str] = field(default_factory=list)
    allow_free_form: bool = False


@dataclass(frozen=True)
class WireEvent:
    """Base class for all wire events.

    Args:
        event_type (str): IN: event discriminator. OUT: used to look up the correct subclass.
    """

    event_type: str = ""


@dataclass(frozen=True)
class InitDone(WireEvent):
    """Sent when the agent has finished initialising.

    Args:
        event_type (str): IN: fixed "init_done". OUT: identifies the event kind.
        model (str): IN: model name in use. OUT: displayed in the UI.
        session_id (str): IN: current session identifier. OUT: used for session management.
        cwd (str): IN: current working directory. OUT: shown in the UI.
        git_branch (str): IN: active Git branch. OUT: shown in the UI.
        context_limit (int): IN: maximum context token limit. OUT: shown in the UI.
        agent_name (str): IN: name of the running agent. OUT: shown in the UI.
        skills (list[str]): IN: list of loaded skill names. OUT: shown in the UI.
    """

    event_type: str = "init_done"
    model: str = ""
    session_id: str = ""
    cwd: str = ""
    git_branch: str = ""
    context_limit: int = 0
    agent_name: str = ""
    skills: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TurnBegin(WireEvent):
    """Signals the start of a new user turn.

    Args:
        event_type (str): IN: fixed "turn_begin". OUT: identifies the event kind.
        user_input (str | list[dict[str, Any]]): IN: raw user input or multimodal parts. OUT: processed by the agent.
    """

    event_type: str = "turn_begin"
    user_input: str | list[dict[str, Any]] = ""


@dataclass(frozen=True)
class TurnEnd(WireEvent):
    """Signals the end of a user turn.

    Args:
        event_type (str): IN: fixed "turn_end". OUT: identifies the event kind.
    """

    event_type: str = "turn_end"


@dataclass(frozen=True)
class StepBegin(WireEvent):
    """Signals the start of an agent processing step.

    Args:
        event_type (str): IN: fixed "step_begin". OUT: identifies the event kind.
        n (int): IN: step number. OUT: shown in the UI.
    """

    event_type: str = "step_begin"
    n: int = 0


@dataclass(frozen=True)
class StepEnd(WireEvent):
    """Signals the end of an agent processing step.

    Args:
        event_type (str): IN: fixed "step_end". OUT: identifies the event kind.
        n (int): IN: step number. OUT: shown in the UI.
    """

    event_type: str = "step_end"
    n: int = 0


@dataclass(frozen=True)
class StepInterrupted(WireEvent):
    """Signals that the current step was interrupted.

    Args:
        event_type (str): IN: fixed "step_interrupted". OUT: identifies the event kind.
    """

    event_type: str = "step_interrupted"


@dataclass(frozen=True)
class SteerInput(WireEvent):
    """User steering input injected mid-generation.

    Args:
        event_type (str): IN: fixed "steer_input". OUT: identifies the event kind.
        content (str): IN: steering text from the user. OUT: forwarded to the agent loop.
    """

    event_type: str = "steer_input"
    content: str = ""


@dataclass(frozen=True)
class CompactionBegin(WireEvent):
    """Signals the start of a context compaction operation.

    Args:
        event_type (str): IN: fixed "compaction_begin". OUT: identifies the event kind.
    """

    event_type: str = "compaction_begin"


@dataclass(frozen=True)
class CompactionEnd(WireEvent):
    """Signals the end of a context compaction operation.

    Args:
        event_type (str): IN: fixed "compaction_end". OUT: identifies the event kind.
    """

    event_type: str = "compaction_end"


@dataclass(frozen=True)
class HookTriggered(WireEvent):
    """Sent when a lifecycle hook has been triggered.

    Args:
        event_type (str): IN: fixed "hook_triggered". OUT: identifies the event kind.
        hook_name (str): IN: name of the triggered hook. OUT: shown in the UI.
        trigger_type (str): IN: type of trigger (e.g. "pre", "post"). OUT: shown in the UI.
    """

    event_type: str = "hook_triggered"
    hook_name: str = ""
    trigger_type: str = ""


@dataclass(frozen=True)
class HookResolved(WireEvent):
    """Sent when a triggered lifecycle hook has finished.

    Args:
        event_type (str): IN: fixed "hook_resolved". OUT: identifies the event kind.
        hook_name (str): IN: name of the resolved hook. OUT: shown in the UI.
    """

    event_type: str = "hook_resolved"
    hook_name: str = ""


@dataclass(frozen=True)
class MCPLoadingBegin(WireEvent):
    """Signals the start of MCP server loading.

    Args:
        event_type (str): IN: fixed "mcp_loading_begin". OUT: identifies the event kind.
        server_name (str): IN: name of the MCP server being loaded. OUT: shown in the UI.
    """

    event_type: str = "mcp_loading_begin"
    server_name: str = ""


@dataclass(frozen=True)
class MCPLoadingEnd(WireEvent):
    """Signals the end of MCP server loading.

    Args:
        event_type (str): IN: fixed "mcp_loading_end". OUT: identifies the event kind.
        server_name (str): IN: name of the MCP server. OUT: shown in the UI.
        success (bool): IN: whether loading succeeded. OUT: controls UI status indicator.
    """

    event_type: str = "mcp_loading_end"
    server_name: str = ""
    success: bool = False


@dataclass(frozen=True)
class BtwBegin(WireEvent):
    """Signals the start of a "by the way" side note.

    Args:
        event_type (str): IN: fixed "btw_begin". OUT: identifies the event kind.
    """

    event_type: str = "btw_begin"


@dataclass(frozen=True)
class BtwEnd(WireEvent):
    """Signals the end of a "by the way" side note.

    Args:
        event_type (str): IN: fixed "btw_end". OUT: identifies the event kind.
    """

    event_type: str = "btw_end"


@dataclass(frozen=True)
class ToolCall(WireEvent):
    """Represents a tool call emitted by the model.

    Args:
        event_type (str): IN: fixed "tool_call". OUT: identifies the event kind.
        id (str): IN: unique tool-call identifier. OUT: used to correlate with results.
        name (str): IN: name of the tool being called. OUT: shown in the UI.
        arguments (str | None): IN: JSON-encoded arguments. OUT: parsed before execution.
    """

    event_type: str = "tool_call"
    id: str = ""
    name: str = ""
    arguments: str | None = None


@dataclass(frozen=True)
class ToolCallPart(WireEvent):
    """A partial/ streaming chunk of tool-call arguments.

    Args:
        event_type (str): IN: fixed "tool_call_part". OUT: identifies the event kind.
        arguments_part (str): IN: incremental argument JSON. OUT: appended to the buffer.
    """

    event_type: str = "tool_call_part"
    arguments_part: str = ""


@dataclass(frozen=True)
class ToolResult(WireEvent):
    """The result of executing a tool call.

    Args:
        event_type (str): IN: fixed "tool_result". OUT: identifies the event kind.
        tool_call_id (str): IN: identifier of the matching tool call. OUT: used for correlation.
        return_value (str): IN: serialised return value. OUT: displayed or fed back to the model.
        display_blocks (list[DisplayBlock]): IN: rich display blocks. OUT: rendered in the UI.
    """

    event_type: str = "tool_result"
    tool_call_id: str = ""
    return_value: str = ""
    display_blocks: list[DisplayBlock] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCallRequest(WireEvent):
    """A request from the UI to execute a tool call.

    Args:
        event_type (str): IN: fixed "tool_call_request". OUT: identifies the event kind.
        id (str): IN: request identifier. OUT: echoed in the response.
        tool_call_id (str): IN: tool-call identifier. OUT: used for correlation.
        name (str): IN: tool name. OUT: used to dispatch the correct tool.
        arguments (dict[str, Any]): IN: parsed arguments dict. OUT: passed to the tool handler.
    """

    event_type: str = "tool_call_request"
    id: str = ""
    tool_call_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovalRequest(WireEvent):
    """A request for user approval before executing an action.

    Args:
        event_type (str): IN: fixed "approval_request". OUT: identifies the event kind.
        id (str): IN: request identifier. OUT: echoed in the response.
        tool_call_id (str): IN: related tool-call identifier. OUT: used for correlation.
        action (str): IN: description of the action. OUT: shown to the user.
        description (str): IN: detailed explanation. OUT: shown to the user.
    """

    event_type: str = "approval_request"
    id: str = ""
    tool_call_id: str = ""
    action: str = ""
    description: str = ""


@dataclass(frozen=True)
class ApprovalResponse(WireEvent):
    """The user's response to an approval request.

    Args:
        event_type (str): IN: fixed "approval_response". OUT: identifies the event kind.
        request_id (str): IN: identifier of the original request. OUT: used for correlation.
        response (Literal["approve", "approve_for_session", "reject"]): IN: user decision. OUT: controls execution flow.
        feedback (str | None): IN: optional free-form feedback. OUT: logged or shown.
    """

    event_type: str = "approval_response"
    request_id: str = ""
    response: Literal["approve", "approve_for_session", "reject"] = "reject"
    feedback: str | None = None


@dataclass(frozen=True)
class QuestionRequest(WireEvent):
    """A request for the user to answer one or more questions.

    Args:
        event_type (str): IN: fixed "question_request". OUT: identifies the event kind.
        id (str): IN: request identifier. OUT: echoed in the response.
        tool_call_id (str): IN: related tool-call identifier. OUT: used for correlation.
        questions (list[QuestionItem]): IN: questions to present. OUT: rendered in the UI.
    """

    event_type: str = "question_request"
    id: str = ""
    tool_call_id: str = ""
    questions: list[QuestionItem] = field(default_factory=list)


@dataclass(frozen=True)
class QuestionResponse(WireEvent):
    """The user's answers to a question request.

    Args:
        event_type (str): IN: fixed "question_response". OUT: identifies the event kind.
        id (str): IN: request identifier. OUT: used for correlation.
        answers (dict[str, str]): IN: mapping of question id -> answer text. OUT: processed by the agent.
    """

    event_type: str = "question_response"
    id: str = ""
    answers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StatusUpdate(WireEvent):
    """Periodic status update broadcast to the UI.

    Args:
        event_type (str): IN: fixed "status_update". OUT: identifies the event kind.
        context_tokens (int): IN: current token count. OUT: shown in the UI.
        max_context (int): IN: maximum allowed tokens. OUT: shown in the UI.
        mcp_status (dict[str, Any]): IN: per-server status dict. OUT: shown in the UI.
        plan_mode (bool): IN: whether plan mode is active. OUT: controls UI state.
    """

    event_type: str = "status_update"
    context_tokens: int = 0
    max_context: int = 0
    mcp_status: dict[str, Any] = field(default_factory=dict)
    plan_mode: bool = False


@dataclass(frozen=True)
class Notification(WireEvent):
    """A general notification event.

    Args:
        event_type (str): IN: fixed "notification". OUT: identifies the event kind.
        id (str): IN: notification identifier. OUT: used for deduplication.
        category (str): IN: notification category. OUT: used for filtering / styling.
        type (str): IN: notification sub-type. OUT: used for filtering / styling.
        severity (Literal["info", "success", "warning", "error"]): IN: severity level. OUT: controls UI colour / icon.
        title (str): IN: short title. OUT: shown in the UI.
        body (str): IN: detailed message. OUT: shown in the UI.
        payload (dict[str, Any]): IN: arbitrary extra data. OUT: available to consumers.
    """

    event_type: str = "notification"
    id: str = ""
    category: str = ""
    type: str = ""
    severity: Literal["info", "success", "warning", "error"] = "info"
    title: str = ""
    body: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanDisplay(WireEvent):
    """Displays a plan or plan update.

    Args:
        event_type (str): IN: fixed "plan_display". OUT: identifies the event kind.
        content (str): IN: plan markdown text. OUT: rendered in the UI.
        file_path (str | None): IN: optional associated file path. OUT: shown as a link.
    """

    event_type: str = "plan_display"
    content: str = ""
    file_path: str | None = None


@dataclass(frozen=True)
class SubagentEvent(WireEvent):
    """Wraps an event produced by a sub-agent.

    Args:
        event_type (str): IN: fixed "subagent_event". OUT: identifies the event kind.
        parent_tool_call_id (str | None): IN: parent tool-call id. OUT: used for correlation.
        agent_id (str | None): IN: sub-agent identifier. OUT: shown in the UI.
        subagent_type (str | None): IN: type of sub-agent. OUT: shown in the UI.
        event (WireEvent | None): IN: nested wire event. OUT: unwrapped and displayed.
    """

    event_type: str = "subagent_event"
    parent_tool_call_id: str | None = None
    agent_id: str | None = None
    subagent_type: str | None = None
    event: WireEvent | None = None


WireEventType = (
    TurnBegin
    | TurnEnd
    | StepBegin
    | StepEnd
    | StepInterrupted
    | SteerInput
    | CompactionBegin
    | CompactionEnd
    | HookTriggered
    | HookResolved
    | MCPLoadingBegin
    | MCPLoadingEnd
    | BtwBegin
    | BtwEnd
    | TextPart
    | ThinkPart
    | ImageURLPart
    | AudioURLPart
    | VideoURLPart
    | ToolCall
    | ToolCallPart
    | ToolResult
    | ToolCallRequest
    | ApprovalRequest
    | ApprovalResponse
    | QuestionRequest
    | QuestionResponse
    | StatusUpdate
    | Notification
    | PlanDisplay
    | SubagentEvent
)


_WIRE_EVENT_REGISTRY: dict[str, type[WireEvent]] = {
    "init_done": InitDone,
    "turn_begin": TurnBegin,
    "turn_end": TurnEnd,
    "step_begin": StepBegin,
    "step_end": StepEnd,
    "step_interrupted": StepInterrupted,
    "steer_input": SteerInput,
    "compaction_begin": CompactionBegin,
    "compaction_end": CompactionEnd,
    "hook_triggered": HookTriggered,
    "hook_resolved": HookResolved,
    "mcp_loading_begin": MCPLoadingBegin,
    "mcp_loading_end": MCPLoadingEnd,
    "btw_begin": BtwBegin,
    "btw_end": BtwEnd,
    "text_part": TextPart,
    "think_part": ThinkPart,
    "image_url_part": ImageURLPart,
    "audio_url_part": AudioURLPart,
    "video_url_part": VideoURLPart,
    "tool_call": ToolCall,
    "tool_call_part": ToolCallPart,
    "tool_result": ToolResult,
    "tool_call_request": ToolCallRequest,
    "approval_request": ApprovalRequest,
    "approval_response": ApprovalResponse,
    "question_request": QuestionRequest,
    "question_response": QuestionResponse,
    "status_update": StatusUpdate,
    "notification": Notification,
    "plan_display": PlanDisplay,
    "subagent_event": SubagentEvent,
}

_KIMI_EVENT_NAME_BY_INTERNAL: dict[str, str] = {
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
_INTERNAL_EVENT_NAME_BY_KIMI = {v: k for k, v in _KIMI_EVENT_NAME_BY_INTERNAL.items()}

for kimi_name, internal_name in _INTERNAL_EVENT_NAME_BY_KIMI.items():
    _WIRE_EVENT_REGISTRY[kimi_name] = _WIRE_EVENT_REGISTRY[internal_name]


def to_kimi_event_name(event_type: str) -> str:
    """Convert an internal event type string to the Kimi/PascalCase name.

    Args:
        event_type (str): IN: internal snake_case event type. OUT: looked up in the mapping.

    Returns:
        str: OUT: PascalCase name if known, otherwise the original string.
    """

    return _KIMI_EVENT_NAME_BY_INTERNAL.get(event_type, event_type)


def to_internal_event_name(event_type: str) -> str:
    """Convert a Kimi/PascalCase event name to the internal snake_case type.

    Args:
        event_type (str): IN: PascalCase event name. OUT: looked up in the mapping.

    Returns:
        str: OUT: snake_case type if known, otherwise the original string.
    """

    return _INTERNAL_EVENT_NAME_BY_KIMI.get(event_type, event_type)


def event_from_dict(data: dict[str, Any]) -> WireEvent:
    """Deserialise a dictionary into the correct WireEvent subclass.

    The dictionary must contain a "type" key.  Unknown types fall back to
    GenericWireEvent.

    Args:
        data (dict[str, Any]): IN: serialised event dict. OUT: used to look up the class and instantiate it.

    Returns:
        WireEvent: OUT: concrete event instance (or GenericWireEvent for unknown types).
    """

    event_type = data.get("type", "")
    cls = _WIRE_EVENT_REGISTRY.get(event_type)
    if cls is None:
        return GenericWireEvent(raw=data)
    payload = {k: v for k, v in data.items() if k != "type"}
    if cls is SubagentEvent and isinstance(payload.get("event"), dict):
        nested = dict(payload["event"])
        nested_type = nested.get("type")
        nested_payload = nested.get("payload")
        if nested_type is not None and isinstance(nested_payload, dict):
            nested_payload = dict(nested_payload)
            nested_payload["type"] = nested_type
            payload["event"] = event_from_dict(nested_payload)
        elif nested_type is not None:
            payload["event"] = event_from_dict(nested)
    return cls(**payload)


def event_to_dict(event: WireEvent) -> dict[str, Any]:
    """Serialise a WireEvent into a dictionary suitable for the wire format.

    Args:
        event (WireEvent): IN: event instance to serialise. OUT: converted to a dict.

    Returns:
        dict[str, Any]: OUT: dict with "type" (PascalCase) and "payload" keys.
    """

    if is_dataclass(event):
        data = asdict(event)
    else:
        data = dict(event)
    event_type = data.pop("event_type", getattr(event, "event_type", ""))
    return {"type": to_kimi_event_name(event_type), "payload": data}


@dataclass(frozen=True)
class GenericWireEvent(WireEvent):
    """Fallback event for unknown event types.

    Args:
        event_type (str): IN: fixed "generic". OUT: identifies the event kind.
        raw (dict[str, Any]): IN: original raw dict. OUT: preserved for inspection.
    """

    event_type: str = "generic"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JSONRPCEventMessage:
    """A JSON-RPC notification wrapping an event.

    Args:
        jsonrpc (Literal["2.0"]): IN: JSON-RPC version. OUT: always "2.0".
        method (Literal["event"]): IN: JSON-RPC method. OUT: always "event".
        params (dict[str, Any]): IN: event payload. OUT: forwarded to consumers.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["event"] = "event"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JSONRPCRequestMessage:
    """A JSON-RPC request message.

    Args:
        jsonrpc (Literal["2.0"]): IN: JSON-RPC version. OUT: always "2.0".
        method (Literal["request"]): IN: JSON-RPC method. OUT: always "request".
        id (str): IN: request identifier. OUT: echoed in the response.
        params (dict[str, Any]): IN: request parameters. OUT: used by the handler.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["request"] = "request"
    id: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JSONRPCSuccessResponse:
    """A JSON-RPC success response.

    Args:
        jsonrpc (Literal["2.0"]): IN: JSON-RPC version. OUT: always "2.0".
        id (str | int): IN: identifier matching the request. OUT: used for correlation.
        result (dict[str, Any]): IN: result payload. OUT: returned to the caller.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int = ""
    result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JSONRPCErrorResponse:
    """A JSON-RPC error response.

    Args:
        jsonrpc (Literal["2.0"]): IN: JSON-RPC version. OUT: always "2.0".
        id (str | int): IN: identifier matching the request. OUT: used for correlation.
        error (dict[str, Any]): IN: error details. OUT: returned to the caller.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int = ""
    error: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptParams:
    """Parameters for a prompt/request message.

    Args:
        user_input (str): IN: user text input. OUT: processed by the agent.
        plan_mode (bool): IN: whether plan mode is requested. OUT: controls agent behaviour.
    """

    user_input: str = ""
    plan_mode: bool = False


@dataclass(frozen=True)
class InitializeParams:
    """Parameters for the initialisation handshake.

    Args:
        model (str): IN: model identifier. OUT: selects the backend model.
        base_url (str): IN: API base URL. OUT: used for the model provider endpoint.
        api_key (str): IN: API authentication key. OUT: sent with requests.
        permission_mode (str): IN: default permission mode. OUT: controls approval behaviour.
    """

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    permission_mode: str = "auto"


@dataclass(frozen=True)
class PermissionResponseParams:
    """Parameters for a permission/approval response.

    Args:
        request_id (str): IN: identifier of the approval request. OUT: used for correlation.
        response (Literal["approve", "approve_for_session", "reject"]): IN: user decision. OUT: controls execution flow.
        feedback (str | None): IN: optional free-form feedback. OUT: logged or shown.
    """

    request_id: str = ""
    response: Literal["approve", "approve_for_session", "reject"] = "reject"
    feedback: str | None = None


__all__ = [
    "ApprovalRequest",
    "ApprovalResponse",
    "AudioURLPart",
    "BackgroundTaskDisplayBlock",
    "BriefDisplayBlock",
    "BtwBegin",
    "BtwEnd",
    "CompactionBegin",
    "CompactionEnd",
    "ContentPart",
    "DiffDisplayBlock",
    "DisplayBlock",
    "GenericDisplayBlock",
    "GenericWireEvent",
    "HookResolved",
    "HookTriggered",
    "ImageURLPart",
    "InitDone",
    "InitializeParams",
    "JSONRPCErrorResponse",
    "JSONRPCEventMessage",
    "JSONRPCRequestMessage",
    "JSONRPCSuccessResponse",
    "MCPLoadingBegin",
    "MCPLoadingEnd",
    "Notification",
    "PermissionResponseParams",
    "PlanDisplay",
    "PromptParams",
    "QuestionItem",
    "QuestionRequest",
    "QuestionResponse",
    "StatusUpdate",
    "SteerInput",
    "StepBegin",
    "StepEnd",
    "StepInterrupted",
    "SubagentEvent",
    "TextPart",
    "ThinkPart",
    "TodoDisplayBlock",
    "ToolCall",
    "ToolCallPart",
    "ToolCallRequest",
    "ToolResult",
    "TurnBegin",
    "TurnEnd",
    "VideoURLPart",
    "WireEvent",
    "WireEventType",
    "event_from_dict",
    "event_to_dict",
    "to_internal_event_name",
    "to_kimi_event_name",
]
