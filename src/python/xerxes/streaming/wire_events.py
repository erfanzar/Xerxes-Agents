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
"""Wire-format event vocabulary for the daemon/bridge JSON-RPC protocol.

Wire events are the *external* contract between the agent process and any
client (TUI, web UI, headless bridge consumer). Internal streaming events
from :mod:`xerxes.streaming.events` are translated into these frozen
dataclasses, serialised via :func:`event_to_dict`, and shipped as JSON-RPC
notifications. ``event_from_dict`` performs the reverse translation, including
unwrapping ``SubagentEvent``\\ s recursively.

The module also defines display blocks (rich UI payloads attached to tool
results), content parts (text/think/image/audio/video deltas), and the
JSON-RPC envelope dataclasses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class BriefDisplayBlock:
    """Plain-text display block attached to a tool result.

    Attributes:
        type: Block discriminator, always ``"brief"``.
        body: Text to render.
    """

    type: Literal["brief"] = "brief"
    body: str = ""


@dataclass(frozen=True)
class DiffDisplayBlock:
    """Code-diff display block.

    Attributes:
        type: Block discriminator, always ``"diff"``.
        diff: Unified-diff text.
        language: Language hint for syntax highlighting.
    """

    type: Literal["diff"] = "diff"
    diff: str = ""
    language: str = ""


@dataclass(frozen=True)
class TodoDisplayBlock:
    """Todo-list display block.

    Attributes:
        type: Block discriminator, always ``"todo"``.
        items: Todo entries; each dict carries at least a ``content`` and
            ``status`` field as defined by the TodoTool schema.
    """

    type: Literal["todo"] = "todo"
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class BackgroundTaskDisplayBlock:
    """Background-task status display block.

    Attributes:
        type: Block discriminator, always ``"background_task"``.
        title: Human-readable task title.
        status: Current status string (e.g. ``"running"``, ``"failed"``).
    """

    type: Literal["background_task"] = "background_task"
    title: str = ""
    status: str = ""


@dataclass(frozen=True)
class GenericDisplayBlock:
    """Catch-all display block for tool results without a specialised renderer.

    Attributes:
        type: Block discriminator, always ``"generic"``.
        content: Raw text to render verbatim.
    """

    type: Literal["generic"] = "generic"
    content: str = ""


DisplayBlock = BriefDisplayBlock | DiffDisplayBlock | TodoDisplayBlock | BackgroundTaskDisplayBlock | GenericDisplayBlock


@dataclass(frozen=True)
class TextPart:
    """Visible assistant text delta.

    Attributes:
        type: Part discriminator, always ``"text"``.
        text: Incremental text fragment.
    """

    type: Literal["text"] = "text"
    text: str = ""


@dataclass(frozen=True)
class ThinkPart:
    """Hidden reasoning delta (model thinking channel).

    Attributes:
        type: Part discriminator, always ``"think"``.
        think: Incremental reasoning fragment.
    """

    type: Literal["think"] = "think"
    think: str = ""


@dataclass(frozen=True)
class ImageURLPart:
    """Image content referenced by URL.

    Attributes:
        type: Part discriminator, always ``"image_url"``.
        url: Image URL (may be a ``data:`` URI).
        alt: Optional alt text for accessibility.
    """

    type: Literal["image_url"] = "image_url"
    url: str = ""
    alt: str | None = None


@dataclass(frozen=True)
class AudioURLPart:
    """Audio clip referenced by URL.

    Attributes:
        type: Part discriminator, always ``"audio_url"``.
        url: Audio URL.
    """

    type: Literal["audio_url"] = "audio_url"
    url: str = ""


@dataclass(frozen=True)
class VideoURLPart:
    """Video referenced by URL.

    Attributes:
        type: Part discriminator, always ``"video_url"``.
        url: Video URL.
        alt: Optional alt text.
    """

    type: Literal["video_url"] = "video_url"
    url: str = ""
    alt: str | None = None


ContentPart = TextPart | ThinkPart | ImageURLPart | AudioURLPart | VideoURLPart


@dataclass(frozen=True)
class QuestionItem:
    """One question inside an interactive ``QuestionRequest``.

    Attributes:
        id: Stable identifier used to key the eventual answer.
        question: Prompt text shown to the user.
        options: Selectable answer choices; empty for free-form.
        allow_free_form: Whether a free-text answer is accepted in addition
            to (or instead of) the listed options.
    """

    id: str = ""
    question: str = ""
    options: list[str] = field(default_factory=list)
    allow_free_form: bool = False


@dataclass(frozen=True)
class WireEvent:
    """Base class for every wire-protocol event.

    Subclasses set ``event_type`` to the snake_case discriminator used in the
    registry and JSON envelope.

    Attributes:
        event_type: Snake_case event discriminator.
    """

    event_type: str = ""


@dataclass(frozen=True)
class InitDone(WireEvent):
    """Emitted once the agent process is ready for prompts.

    Attributes:
        event_type: Always ``"init_done"``.
        model: Model identifier the agent will use.
        session_id: Session identifier (also used for replay).
        cwd: Working directory.
        git_branch: Active git branch in ``cwd``, or empty string.
        context_limit: Maximum context window for the active model.
        agent_name: Name of the loaded agent definition.
        skills: List of loaded skill identifiers.
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
    """Marks the start of a new user-driven turn.

    Attributes:
        event_type: Always ``"turn_begin"``.
        user_input: User input — plain string for text-only turns, or a list
            of multimodal content dicts.
    """

    event_type: str = "turn_begin"
    user_input: str | list[dict[str, Any]] = ""


@dataclass(frozen=True)
class TurnEnd(WireEvent):
    """Marks the end of a user-driven turn.

    Attributes:
        event_type: Always ``"turn_end"``.
    """

    event_type: str = "turn_end"


@dataclass(frozen=True)
class StepBegin(WireEvent):
    """Marks the start of a single assistant step within a turn.

    Attributes:
        event_type: Always ``"step_begin"``.
        n: 1-based step index within the current turn.
    """

    event_type: str = "step_begin"
    n: int = 0


@dataclass(frozen=True)
class StepEnd(WireEvent):
    """Marks the end of a single assistant step within a turn.

    Attributes:
        event_type: Always ``"step_end"``.
        n: Step index this event pairs with.
    """

    event_type: str = "step_end"
    n: int = 0


@dataclass(frozen=True)
class StepInterrupted(WireEvent):
    """Signals that the in-flight step was cancelled (e.g. by user Ctrl-C).

    Attributes:
        event_type: Always ``"step_interrupted"``.
    """

    event_type: str = "step_interrupted"


@dataclass(frozen=True)
class SteerInput(WireEvent):
    """User text injected mid-generation to redirect the model.

    Attributes:
        event_type: Always ``"steer_input"``.
        content: Steering text appended to the in-flight context.
    """

    event_type: str = "steer_input"
    content: str = ""


@dataclass(frozen=True)
class CompactionBegin(WireEvent):
    """Marks the start of an automatic context compaction.

    Attributes:
        event_type: Always ``"compaction_begin"``.
    """

    event_type: str = "compaction_begin"


@dataclass(frozen=True)
class CompactionEnd(WireEvent):
    """Marks the end of an automatic context compaction.

    Attributes:
        event_type: Always ``"compaction_end"``.
    """

    event_type: str = "compaction_end"


@dataclass(frozen=True)
class HookTriggered(WireEvent):
    """Emitted when a lifecycle hook fires.

    Attributes:
        event_type: Always ``"hook_triggered"``.
        hook_name: Hook identifier (e.g. ``"PreToolUse"``).
        trigger_type: Trigger phase, typically ``"pre"`` or ``"post"``.
    """

    event_type: str = "hook_triggered"
    hook_name: str = ""
    trigger_type: str = ""


@dataclass(frozen=True)
class HookResolved(WireEvent):
    """Emitted after a hook finishes (success or controlled failure).

    Attributes:
        event_type: Always ``"hook_resolved"``.
        hook_name: Hook identifier paired with the prior trigger.
    """

    event_type: str = "hook_resolved"
    hook_name: str = ""


@dataclass(frozen=True)
class MCPLoadingBegin(WireEvent):
    """Emitted while an MCP server is being negotiated.

    Attributes:
        event_type: Always ``"mcp_loading_begin"``.
        server_name: MCP server identifier.
    """

    event_type: str = "mcp_loading_begin"
    server_name: str = ""


@dataclass(frozen=True)
class MCPLoadingEnd(WireEvent):
    """Emitted when MCP server loading completes.

    Attributes:
        event_type: Always ``"mcp_loading_end"``.
        server_name: MCP server identifier.
        success: ``True`` when tools were registered successfully.
    """

    event_type: str = "mcp_loading_end"
    server_name: str = ""
    success: bool = False


@dataclass(frozen=True)
class BtwBegin(WireEvent):
    """Marks the start of a "by the way" side-channel note.

    Attributes:
        event_type: Always ``"btw_begin"``.
    """

    event_type: str = "btw_begin"


@dataclass(frozen=True)
class BtwEnd(WireEvent):
    """Marks the end of a "by the way" side-channel note.

    Attributes:
        event_type: Always ``"btw_end"``.
    """

    event_type: str = "btw_end"


@dataclass(frozen=True)
class ToolCall(WireEvent):
    """A completed tool call request emitted by the model.

    Attributes:
        event_type: Always ``"tool_call"``.
        id: Provider-issued call id used to correlate with the result.
        name: Tool identifier.
        arguments: Fully-assembled argument JSON, or ``None`` if streamed in
            pieces via :class:`ToolCallPart`.
    """

    event_type: str = "tool_call"
    id: str = ""
    name: str = ""
    arguments: str | None = None


@dataclass(frozen=True)
class ToolCallPart(WireEvent):
    """A streaming fragment of tool-call argument JSON.

    Clients accumulate consecutive ``ToolCallPart`` events until a matching
    :class:`ToolCall` arrives with the full arguments.

    Attributes:
        event_type: Always ``"tool_call_part"``.
        arguments_part: Incremental argument-JSON fragment.
    """

    event_type: str = "tool_call_part"
    arguments_part: str = ""


@dataclass(frozen=True)
class ToolResult(WireEvent):
    """The result of executing a tool call.

    Attributes:
        event_type: Always ``"tool_result"``.
        tool_call_id: Matching tool call id.
        return_value: Serialised return value (rendered in the UI).
        duration_ms: Wall-clock execution time.
        display_blocks: Optional rich-render blocks (diff, todo, etc.).
    """

    event_type: str = "tool_result"
    tool_call_id: str = ""
    return_value: str = ""
    duration_ms: float = 0.0
    display_blocks: list[DisplayBlock] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCallRequest(WireEvent):
    """A client-issued request to invoke a tool directly.

    Used by harnesses that drive tool execution out-of-band rather than via
    the LLM loop.

    Attributes:
        event_type: Always ``"tool_call_request"``.
        id: Request id echoed in the response.
        tool_call_id: Tool-call correlation id.
        name: Tool identifier.
        arguments: Parsed argument dict.
    """

    event_type: str = "tool_call_request"
    id: str = ""
    tool_call_id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApprovalRequest(WireEvent):
    """Request for user approval before executing a sensitive action.

    Attributes:
        event_type: Always ``"approval_request"``.
        id: Request id echoed in the response.
        tool_call_id: Tool call being gated.
        action: Short action label (e.g. ``"Run shell command"``).
        description: Longer explanation (the command, target file, etc.).
    """

    event_type: str = "approval_request"
    id: str = ""
    tool_call_id: str = ""
    action: str = ""
    description: str = ""


@dataclass(frozen=True)
class ApprovalResponse(WireEvent):
    """User reply to an :class:`ApprovalRequest`.

    Attributes:
        event_type: Always ``"approval_response"``.
        request_id: Id of the originating request.
        response: ``"approve"`` (this call only), ``"approve_for_session"``
            (remember for the rest of the session), or ``"reject"``.
        feedback: Optional free-form note attached to the decision.
    """

    event_type: str = "approval_response"
    request_id: str = ""
    response: Literal["approve", "approve_for_session", "reject"] = "reject"
    feedback: str | None = None


@dataclass(frozen=True)
class QuestionRequest(WireEvent):
    """Request the user to answer one or more interactive questions.

    Attributes:
        event_type: Always ``"question_request"``.
        id: Request id echoed in the response.
        tool_call_id: Tool call that issued the question batch.
        questions: Ordered list of questions to present.
    """

    event_type: str = "question_request"
    id: str = ""
    tool_call_id: str = ""
    questions: list[QuestionItem] = field(default_factory=list)


@dataclass(frozen=True)
class QuestionResponse(WireEvent):
    """User answers to a :class:`QuestionRequest`.

    Attributes:
        event_type: Always ``"question_response"``.
        id: Id of the originating request.
        answers: Mapping of question id -> answer text.
    """

    event_type: str = "question_response"
    id: str = ""
    answers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StatusUpdate(WireEvent):
    """Periodic status snapshot for the UI status line.

    Attributes:
        event_type: Always ``"status_update"``.
        context_tokens: Current context window usage.
        max_context: Model context limit.
        mcp_status: Per-server MCP status dict (name -> state).
        plan_mode: Whether plan mode is engaged.
        mode: Active interaction mode (``"code"``, ``"plan"``, ``"chat"`` …).
    """

    event_type: str = "status_update"
    context_tokens: int = 0
    max_context: int = 0
    mcp_status: dict[str, Any] = field(default_factory=dict)
    plan_mode: bool = False
    mode: str = "code"


@dataclass(frozen=True)
class Notification(WireEvent):
    """General-purpose notification surfaced to the UI.

    Attributes:
        event_type: Always ``"notification"``.
        id: Notification id (used for dedup / dismissal).
        category: High-level category (e.g. ``"system"``, ``"tool"``).
        type: Finer-grained sub-type within the category.
        severity: ``"info"``, ``"success"``, ``"warning"``, or ``"error"``.
        title: Short headline.
        body: Detailed message body.
        payload: Arbitrary extra data forwarded to consumers.
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
    """Renders or updates a plan document in the UI.

    Attributes:
        event_type: Always ``"plan_display"``.
        content: Plan markdown.
        file_path: Optional on-disk path of the underlying plan file.
    """

    event_type: str = "plan_display"
    content: str = ""
    file_path: str | None = None


@dataclass(frozen=True)
class SubagentEvent(WireEvent):
    """Wraps an event produced by a spawned subagent.

    The nested ``event`` retains its original wire type and is recursively
    decoded by :func:`event_from_dict`.

    Attributes:
        event_type: Always ``"subagent_event"``.
        parent_tool_call_id: Tool call id of the spawning ``Agent`` call.
        agent_id: Subagent identifier.
        subagent_type: Subagent template name (``"researcher"``, etc.).
        event: The inner wire event.
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
    """Translate an internal snake_case event type to its Kimi PascalCase name.

    Args:
        event_type: Internal snake_case discriminator.

    Returns:
        Matching PascalCase name, or ``event_type`` unchanged if unknown.
    """

    return _KIMI_EVENT_NAME_BY_INTERNAL.get(event_type, event_type)


def to_internal_event_name(event_type: str) -> str:
    """Translate a Kimi PascalCase event name to its internal snake_case form.

    Args:
        event_type: PascalCase name as emitted by Kimi-style clients.

    Returns:
        Matching snake_case discriminator, or ``event_type`` unchanged if
        unknown.
    """

    return _INTERNAL_EVENT_NAME_BY_KIMI.get(event_type, event_type)


def event_from_dict(data: dict[str, Any]) -> WireEvent:
    """Deserialise a wire dict into the matching :class:`WireEvent` subclass.

    Looks up the ``"type"`` field in the registry, which accepts both
    snake_case and PascalCase keys. Nested ``SubagentEvent`` payloads are
    decoded recursively. Unknown types fall back to :class:`GenericWireEvent`
    so callers can still inspect the raw payload.

    Args:
        data: Wire dict with a ``"type"`` discriminator and per-event fields.

    Returns:
        Concrete ``WireEvent`` (or ``GenericWireEvent`` for unknown types).
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
    """Serialise a :class:`WireEvent` to a ``{"type", "payload"}`` wire dict.

    The discriminator is converted to PascalCase via :func:`to_kimi_event_name`
    so the output is directly consumable by Kimi-style clients.

    Args:
        event: Event to serialise.

    Returns:
        Dict with a PascalCase ``"type"`` and a ``"payload"`` body holding the
        remaining fields.
    """

    if is_dataclass(event):
        data = asdict(event)
    else:
        data = dict(event)
    event_type = data.pop("event_type", getattr(event, "event_type", ""))
    return {"type": to_kimi_event_name(event_type), "payload": data}


@dataclass(frozen=True)
class GenericWireEvent(WireEvent):
    """Fallback wrapper for wire dicts with an unknown ``type``.

    Attributes:
        event_type: Always ``"generic"``.
        raw: Verbatim payload preserved for inspection or forwarding.
    """

    event_type: str = "generic"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JSONRPCEventMessage:
    """JSON-RPC notification envelope wrapping a wire event.

    Attributes:
        jsonrpc: Protocol version, always ``"2.0"``.
        method: Method name, always ``"event"``.
        params: Wire-event payload produced by :func:`event_to_dict`.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["event"] = "event"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JSONRPCRequestMessage:
    """JSON-RPC request envelope.

    Attributes:
        jsonrpc: Protocol version, always ``"2.0"``.
        method: Method name, always ``"request"``.
        id: Request id echoed in the response.
        params: Request parameters consumed by the handler.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["request"] = "request"
    id: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JSONRPCSuccessResponse:
    """JSON-RPC success response envelope.

    Attributes:
        jsonrpc: Protocol version, always ``"2.0"``.
        id: Request id correlated with the originating call.
        result: Response payload.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int = ""
    result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JSONRPCErrorResponse:
    """JSON-RPC error response envelope.

    Attributes:
        jsonrpc: Protocol version, always ``"2.0"``.
        id: Request id correlated with the originating call.
        error: Error dict with at minimum ``code`` and ``message`` fields.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int = ""
    error: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptParams:
    """Parameters for a ``prompt`` JSON-RPC request.

    Attributes:
        user_input: User text input for the turn.
        plan_mode: Whether the turn should run in plan mode.
    """

    user_input: str = ""
    plan_mode: bool = False


@dataclass(frozen=True)
class InitializeParams:
    """Parameters for the ``initialize`` handshake.

    Attributes:
        model: Model identifier.
        base_url: Override base URL for the LLM provider.
        api_key: Explicit API key (overrides env vars).
        permission_mode: Default permission mode (``"auto"``, ``"manual"``, …).
    """

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    permission_mode: str = "auto"


@dataclass(frozen=True)
class PermissionResponseParams:
    """Parameters for a ``permission_response`` JSON-RPC request.

    Attributes:
        request_id: Id of the original approval request.
        response: Decision — ``"approve"``, ``"approve_for_session"``, or
            ``"reject"``.
        feedback: Optional free-form note logged with the decision.
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
