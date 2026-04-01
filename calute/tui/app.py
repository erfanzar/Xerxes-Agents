# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Textual-based terminal UI for Calute.

Implements the interactive TUI application built on the Textual framework.
The main class, ``CaluteTUI``, renders a multi-pane interface with a
transcript view, an operations/draft panel, a sidebar, and a slash-command
hint system.  Helper dataclasses (``ChatEntry``, ``ToolActivity``,
``SlashCommandHint``) model the display state, while ``TextualLauncher``
and ``launch_tui`` provide convenient entry points.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import time
import typing as tp
from dataclasses import dataclass
from urllib.parse import urlparse

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.suggester import Suggester
from textual.widgets import Static, TextArea

from ..context.token_counter import ProviderTokenCounter
from ..llms import create_llm
from ..runtime.profiles import PromptProfile
from ..types import (
    Agent,
    AgentSwitch,
    AssistantMessage,
    Completion,
    FunctionCallsExtracted,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    MessagesHistory,
    ReinvokeSignal,
    StreamChunk,
    ToolCallStreamChunk,
    UserMessage,
)
from .model_discovery import discover_available_models
from .terminal_config import SAMPLING_PARAM_KEYS, TerminalConfigStore, TerminalProfile

if tp.TYPE_CHECKING:
    from ..calute import Calute


_UNSET = object()
SUPPORTED_TUI_PROVIDERS = [
    "openai",
    "anthropic",
    "gemini",
    "ollama",
]
PROVIDER_ALIASES = {
    "oai": "openai",
    "claude": "anthropic",
    "google": "gemini",
    "local": "ollama",
}
SAMPLING_PARAM_ALIASES = {
    "temp": "temperature",
    "temperature": "temperature",
    "top-p": "top_p",
    "top_p": "top_p",
    "max-tokens": "max_tokens",
    "max_tokens": "max_tokens",
    "top-k": "top_k",
    "top_k": "top_k",
    "min-p": "min_p",
    "min_p": "min_p",
    "presence-penalty": "presence_penalty",
    "presence_penalty": "presence_penalty",
    "frequency-penalty": "frequency_penalty",
    "frequency_penalty": "frequency_penalty",
    "repetition-penalty": "repetition_penalty",
    "repetition_penalty": "repetition_penalty",
}
SAMPLING_PARAM_RANGES: dict[str, tuple[float | int | None, float | int | None, type]] = {
    "temperature": (0.0, 2.0, float),
    "top_p": (0.0, 1.0, float),
    "max_tokens": (1, None, int),
    "top_k": (0, None, int),
    "min_p": (0.0, 1.0, float),
    "presence_penalty": (-2.0, 2.0, float),
    "frequency_penalty": (-2.0, 2.0, float),
    "repetition_penalty": (0.1, 2.0, float),
}
DEFAULT_SAMPLING_PARAMS = {key: Agent.model_fields[key].default for key in SAMPLING_PARAM_KEYS}


@dataclass
class ChatEntry:
    """Single chat entry shown in the transcript pane.

    Each entry is rendered as a ``rich.Panel`` in the scrollable
    transcript area.  The ``role`` field controls visual styling and
    labelling.

    Attributes:
        role: Entry role identifier.  Common values: ``"user"``,
            ``"assistant"``, ``"tool"``, ``"note"``, ``"question"``,
            ``"error"``.
        content: Main textual body of the entry.
        meta: Optional secondary text rendered above the content (e.g.
            reasoning output for assistant entries, or instructional
            text for question entries).
        streaming: When ``True``, indicates that the entry is still
            receiving incremental content from the model stream.
        title: Optional panel title override; when ``None``, a sensible
            default is derived from *role*.
        key: Application-defined key for de-duplicating entries (used
            primarily for pending question cards).
    """

    role: str
    content: str
    meta: str | None = None
    streaming: bool = False
    title: str | None = None
    key: str | None = None


@dataclass
class ToolActivity:
    """Compact tracked state for a tool call shown in the operations panel.

    Entries are created when a ``FunctionExecutionStart`` event arrives and
    updated on ``FunctionExecutionComplete``.  They persist briefly after
    completion to give the user visual feedback.

    Attributes:
        function_id: Unique identifier for the tool invocation.
        function_name: Human-readable function/tool name.
        status: Current lifecycle state: ``"running"``, ``"success"``, or
            ``"failed"``.
        progress: Optional short progress string (e.g. ``"2/5"``).
        agent_id: Identifier of the agent that triggered the call.
        preview: Truncated preview of the successful result.
        error: Error message when *status* is ``"failed"``.
        started_at: Monotonic timestamp (``time.perf_counter``) when the
            tool call started.
        finished_at: Monotonic timestamp when the call completed, or
            ``None`` while still running.
    """

    function_id: str
    function_name: str
    status: str = "running"
    progress: str | None = None
    agent_id: str | None = None
    arguments_preview: str | None = None
    preview: str | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None


@dataclass(frozen=True)
class SlashCommandHint:
    """Single slash-command hint row shown in the hint panel.

    Displayed as a two-column row (usage example, description) inside
    the ``Slash Hints`` panel at the bottom of the TUI.

    Attributes:
        usage: Command syntax string, e.g. ``"/provider <name>"``.
        description: One-line description of what the command does.
    """

    usage: str
    description: str


class SlashCommandSuggester(Suggester):
    """Inline suggester for slash commands and their arguments.

    Integrates with Textual's ``Input`` widget to provide real-time
    tab-completable suggestions as the user types a ``/`` command.
    Uses both prefix matching and fuzzy matching (via ``difflib``) to
    tolerate minor typos.

    Attributes:
        _supplier: Callable that returns the current list of valid
            suggestion strings on each invocation.
    """

    def __init__(self, supplier: tp.Callable[[], list[str]]):
        """Initialise the suggester.

        Args:
            supplier: A zero-argument callable that returns the current
                list of complete slash-command strings (e.g.
                ``["/help", "/model gpt-4", ...]``).
        """
        super().__init__(use_cache=False, case_sensitive=False)
        self._supplier = supplier

    async def get_suggestion(self, value: str) -> str | None:
        """Return the best inline suggestion for the current input value.

        Tries direct prefix matching first, then falls back to fuzzy
        matching on the command token.

        Args:
            value: Current text in the input widget.

        Returns:
            A full suggestion string if a match is found, or ``None``
            when the input does not start with ``/`` or no candidate
            matches.
        """
        stripped = value.lstrip()
        if not stripped.startswith("/"):
            return None

        candidates = self._supplier()
        if not candidates:
            return None

        lowered = stripped.casefold()
        direct = [candidate for candidate in candidates if candidate.casefold().startswith(lowered)]
        if direct:
            return direct[0]

        token = stripped[1:].split(" ", 1)[0].strip().casefold()
        if not token:
            return candidates[0]

        command_map: dict[str, str] = {}
        for candidate in candidates:
            key = candidate[1:].split(" ", 1)[0].strip().casefold()
            command_map.setdefault(key, candidate)

        closest = difflib.get_close_matches(token, list(command_map.keys()), n=1, cutoff=0.45)
        if closest:
            return command_map[closest[0]]
        return None


def preview_value(value: tp.Any, max_length: int = 240) -> str:
    """Render a compact preview string for tool results and errors.

    Strings are stripped; other types are JSON-serialised (falling back to
    ``repr`` on failure).  The result is truncated with an ellipsis when it
    exceeds *max_length*.

    Args:
        value: The value to preview.  Strings are used directly; other
            types are serialised via ``json.dumps``.
        max_length: Maximum character length of the returned preview.

    Returns:
        A human-readable, truncated preview string.
    """
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            text = repr(value)
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def parse_command(command: str) -> tuple[str, str]:
    """Parse a slash command into its verb and argument payload.

    The leading ``/`` is stripped, and the remaining text is split on the
    first whitespace into a lowercased verb and an argument tail.

    Args:
        command: Raw input text starting with ``/``.

    Returns:
        A ``(verb, argument)`` tuple.  Both are empty strings when the
        command body is blank.

    Example:
        >>> parse_command("/model gpt-4")
        ('model', 'gpt-4')
        >>> parse_command("/clear")
        ('clear', '')
    """
    body = command.strip()[1:].strip()
    if not body:
        return "", ""
    if " " not in body:
        return body.lower(), ""
    verb, arg = body.split(" ", 1)
    return verb.lower(), arg.strip()


class TextualLauncher:
    """Thin launcher that wraps ``CaluteTUI`` construction and execution.

    Created by ``launch_tui`` so that callers can hold a deferred reference
    and call ``launch()`` when ready.

    Attributes:
        executor: The ``Calute`` runtime that provides LLM and agent
            orchestration.
        agent: Initial agent (or agent ID) to activate in the TUI.
        profile: Optional saved terminal profile to restore.
        config_store: Optional persistent config store for saving profile
            changes made inside the TUI.
    """

    def __init__(
        self,
        executor: Calute,
        agent: Agent | str | None = None,
        profile: TerminalProfile | None = None,
        config_store: TerminalConfigStore | None = None,
    ):
        """Initialise the launcher.

        Args:
            executor: Calute runtime instance.
            agent: Agent instance, agent ID string, or ``None`` to use
                the orchestrator's current agent.
            profile: Saved terminal profile, or ``None``.
            config_store: Config store for persistence, or ``None``.
        """
        self.executor = executor
        self.agent = agent
        self.profile = profile
        self.config_store = config_store

    def launch(self) -> None:
        """Construct and run the Textual application.

        Blocks until the TUI is closed by the user.
        """
        CaluteTUI(
            executor=self.executor,
            agent=self.agent,
            profile=self.profile,
            config_store=self.config_store,
        ).run()


class CaluteTUI(App[None]):
    """Interactive Textual-based terminal UI for Calute sessions.

    Renders a multi-pane layout comprising:

    * A **left rail** with a sidebar (session info, agents, keyboard
      shortcuts) and an operations/draft panel.
    * A **central chat column** with a scrollable transcript and a
      slash-command hint bar.
    * A **bottom input** bar with inline slash-command suggestions.

    Supports live model streaming, tool-call visualisation, agent
    switching, provider/model hot-swapping, sampling parameter tuning,
    prompt-profile cycling, operator session introspection, and
    structured plan summaries -- all driven by slash commands and
    keyboard shortcuts.

    Attributes:
        SPINNER_FRAMES: Animation frames for the tool-running indicator.
        DEFAULT_INPUT_PLACEHOLDER: Default composer hint shown when idle.
        CSS: Textual CSS stylesheet for the application layout.
        BINDINGS: Keyboard shortcut bindings.
        PROFILE_ORDER: Ordered list of prompt profiles for cycling.
        executor: The ``Calute`` runtime instance.
        current_agent_id: ID of the currently active agent.
        profile: The active ``TerminalProfile``, or ``None``.
        config_store: Persistent configuration store, or ``None``.
        available_models: Cached list of discovered model names.
        command_suggester: Inline slash-command suggestion engine.
        chat_history: Ordered list of ``ChatEntry`` objects displayed in
            the transcript.
        activity_history: Plain-text log of event messages.
        tool_activity: Mapping of function IDs to ``ToolActivity`` state.
        tool_activity_order: Ordered list of function IDs for display.
        current_draft: Partial assistant text while streaming.
        current_reasoning: Partial reasoning text while streaming.
        is_busy: ``True`` while a model response is being streamed.
        prompt_tokens_used: Estimated prompt token count for the current
            turn.
        output_tokens_used: Estimated output token count for the current
            turn.
        tokens_per_second: Estimated output tokens per second.
    """

    SPINNER_FRAMES: tp.ClassVar = ["|", "/", "-", "\\"]
    DEFAULT_INPUT_PLACEHOLDER = "Paste/type multiple lines. Cmd+Enter to send. Type / for palette."

    CSS = """
    Screen {
        background: #0d1117;
        color: #c9d1d9;
        layout: vertical;
    }

    #header-bar {
        height: 3;
        width: 100%;
        background: #161b22;
        border-bottom: heavy #7B8CFF;
        padding: 1 2 0 2;
        text-style: bold;
    }

    #main-body {
        height: 1fr;
        width: 100%;
    }

    #transcript-scroll {
        height: 1fr;
        background: #0d1117;
        padding: 1 3;
        scrollbar-color: #30363d;
        scrollbar-color-hover: #7B8CFF;
        scrollbar-color-active: #B07FFF;
        scrollbar-background: #0d1117;
        scrollbar-background-hover: #161b22;
        scrollbar-background-active: #161b22;
    }

    #transcript {
        padding: 0 0 1 0;
    }

    #status-bar {
        height: 1;
        width: 100%;
        background: #161b22;
        border-top: solid #21262d;
        border-bottom: solid #21262d;
        padding: 0 2;
    }

    #footer-bar {
        height: 1;
        width: 100%;
        background: #0d1117;
        padding: 0 2;
    }

    #input {
        height: auto;
        min-height: 1;
        max-height: 6;
        margin: 0 2 1 2;
        padding: 0 1;
        border: heavy #21262d;
        background: #161b22;
        color: #c9d1d9;
    }

    #input:focus {
        border: heavy #7B8CFF;
    }
    """

    BINDINGS: tp.ClassVar = [
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+enter,meta+enter", "submit_input", "Send"),
        Binding("ctrl+j", "focus_input", "Focus Input"),
        Binding("ctrl+n", "next_agent", "Next Agent"),
        Binding("ctrl+p", "cycle_profile", "Cycle Profile"),
        Binding("ctrl+m", "cycle_model", "Cycle Model"),
    ]

    PROFILE_ORDER: tp.ClassVar = [
        PromptProfile.FULL,
        PromptProfile.COMPACT,
        PromptProfile.MINIMAL,
        PromptProfile.NONE,
    ]

    def __init__(
        self,
        executor: Calute,
        agent: Agent | str | None = None,
        profile: TerminalProfile | None = None,
        config_store: TerminalConfigStore | None = None,
    ):
        """Initialise the TUI application.

        Args:
            executor: The ``Calute`` runtime providing LLM interaction and
                agent orchestration.
            agent: An ``Agent`` instance, an agent ID string, or ``None``
                to default to the orchestrator's current agent.
            profile: Optional saved ``TerminalProfile`` to restore
                provider/model/key settings from.
            config_store: Optional ``TerminalConfigStore`` for persisting
                profile changes during the session.
        """
        super().__init__()
        self.executor = executor
        self.current_agent_id = self._resolve_agent_id(agent)
        self.profile = profile
        self.config_store = config_store
        self.available_models = list(profile.available_models) if profile else []
        self.command_suggester = SlashCommandSuggester(self._command_suggestion_values)
        self.chat_history: list[ChatEntry] = []
        self.activity_history: list[str] = []
        self.tool_activity: dict[str, ToolActivity] = {}
        self.tool_activity_order: list[str] = []
        self.current_draft: str = ""
        self.current_reasoning: str = ""
        self.is_busy = False
        self.prompt_tokens_used = 0
        self.output_tokens_used = 0
        self.tokens_per_second = 0.0
        self._stream_started_at: float | None = None
        self._last_pending_question_id: str | None = None
        self._last_escape_shortcut_at: float | None = None
        self._vscode_meta_enter_fallback = bool(
            os.environ.get("TERM_PROGRAM", "").strip().lower() == "vscode" or os.environ.get("VSCODE_PID")
        )

    def compose(self) -> ComposeResult:
        """Build and yield the Textual widget tree for the application layout."""
        yield Static(id="header-bar")
        with Vertical(id="main-body"):
            with VerticalScroll(id="transcript-scroll"):
                yield Static(id="transcript")
        yield Static(id="status-bar")
        yield Static(id="footer-bar")
        yield TextArea(
            "",
            id="input",
            soft_wrap=True,
            show_line_numbers=False,
        )

    def on_mount(self) -> None:
        """Handle the Textual mount lifecycle event.

        Sets the application title, performs initial UI refreshes, focuses
        the input widget, starts the operations-panel tick timer, and
        optionally kicks off background model discovery.
        """
        self.title = "Calute TUI"
        self.sub_title = "Streaming terminal runtime"
        self._refresh_header_bar()
        self._refresh_transcript()
        self._refresh_footer_bar()
        self._set_input_hint(self.DEFAULT_INPUT_PLACEHOLDER)
        self._set_status("Ready")
        self.query_one("#input", TextArea).focus()
        self.set_interval(0.12, self._tick_operations)
        if not self.available_models and self._provider_name() in {"openai", "ollama", "local"}:
            self.run_worker(
                self._refresh_available_models(),
                name="model-discovery",
                group="metadata",
                description="Fetch available models",
            )

    def action_focus_input(self) -> None:
        """Move keyboard focus to the input bar (bound to Ctrl+J)."""
        self.query_one("#input", TextArea).focus()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """React to input text changes by refreshing the status bar hints."""
        stripped = event.text_area.text.lstrip()
        if stripped.startswith("/"):
            hints = self._match_slash_hints(stripped)
            if hints:
                hint_text = "  ".join(f"{h.usage}" for h in hints[:4])
                self._set_status(hint_text)
            else:
                self._set_status("No matching commands. Try /help.")
        elif not self.is_busy:
            self._set_status("")

    def on_key(self, event: events.Key) -> None:
        """Handle VS Code's ``Cmd+Enter`` terminal fallback as ``Esc`` + ``Enter``.

        The integrated VS Code terminal can be configured to forward
        ``Cmd+Enter`` to terminal apps by sending an escape prefix followed
        by carriage return. Textual doesn't reliably collapse that into a
        single ``meta+enter`` binding in every terminal, so Calute treats a
        tight ``Esc`` -> ``Enter`` sequence in the focused composer as a
        submit shortcut when running inside VS Code.
        """
        if not self._vscode_meta_enter_fallback:
            return
        try:
            input_focused = self.focused is self._input_widget()
        except NoMatches:
            input_focused = False

        if not input_focused:
            self._last_escape_shortcut_at = None
            return

        if event.key == "escape":
            self._last_escape_shortcut_at = time.perf_counter()
            return

        if event.key != "enter":
            self._last_escape_shortcut_at = None
            return

        if self._last_escape_shortcut_at is None:
            return

        elapsed = time.perf_counter() - self._last_escape_shortcut_at
        self._last_escape_shortcut_at = None
        if elapsed > 0.35:
            return

        event.prevent_default()
        event.stop()
        self.run_worker(
            self.action_submit_input(),
            name="submit-input-shortcut",
            group="input-shortcuts",
            exclusive=True,
            description="Submit input via VS Code Cmd+Enter fallback",
        )

    def _input_widget(self) -> TextArea:
        """Return the multiline composer widget."""
        return self.query_one("#input", TextArea)

    def _set_input_hint(self, hint: str, *, title: str = "composer") -> None:
        """Render the current composer hint in the widget chrome."""
        input_widget = self._input_widget()
        input_widget.border_title = f" {title} "
        input_widget.border_subtitle = f" {hint} "

    async def action_submit_input(self) -> None:
        """Submit the current composer contents using the multiline shortcuts."""
        input_widget = self._input_widget()
        raw_text = input_widget.text
        text = raw_text if raw_text.strip() else ""
        input_widget.text = ""

        if not text:
            return

        if self._pending_user_prompt() is not None:
            self._submit_pending_user_answer(text)
            return

        if self.is_busy:
            self._set_status("A response is already streaming.")
            return

        stripped = text.strip()
        if stripped.startswith("/") and "\n" not in stripped and self._handle_command(stripped):
            return

        self.run_worker(
            self._stream_prompt(text),
            name="conversation",
            group="conversation",
            exclusive=True,
            description=f"Prompt: {self._truncate(text, 40)}",
        )

    def action_clear_chat(self) -> None:
        """Clear the transcript pane and reset draft/reasoning buffers (bound to Ctrl+L)."""
        self.chat_history.clear()
        self.current_draft = ""
        self.current_reasoning = ""
        self._refresh_transcript()
        self._append_event("Transcript cleared.", style="bold yellow")
        self._set_status("Transcript cleared.")

    def action_next_agent(self) -> None:
        """Cycle to the next registered agent (bound to Ctrl+N)."""
        agent_ids = self._available_agent_ids()
        if not agent_ids:
            self._set_status("No agents registered.")
            return
        current = self.current_agent_id or agent_ids[0]
        try:
            next_index = (agent_ids.index(current) + 1) % len(agent_ids)
        except ValueError:
            next_index = 0
        self._switch_agent(agent_ids[next_index])

    def action_cycle_profile(self) -> None:
        """Cycle to the next prompt profile in PROFILE_ORDER (bound to Ctrl+P)."""
        profile = self._current_prompt_profile()
        try:
            current_index = self.PROFILE_ORDER.index(profile)
        except ValueError:
            current_index = 0
        next_profile = self.PROFILE_ORDER[(current_index + 1) % len(self.PROFILE_ORDER)]
        self._set_prompt_profile(next_profile)
        self._append_event(f"Prompt profile set to [bold]{next_profile.value}[/].", style="bold cyan")
        self._set_status(f"Prompt profile: {next_profile.value}")

    def action_cycle_model(self) -> None:
        """Cycle to the next discovered model (bound to Ctrl+M)."""
        if not self.available_models:
            self._set_status("No discovered models to cycle.")
            return
        current = self._current_model_name()
        try:
            current_index = self.available_models.index(current) if current else -1
        except ValueError:
            current_index = -1
        next_model = self.available_models[(current_index + 1) % len(self.available_models)]
        self._set_active_model(next_model)
        self._append_event(f"Model set to [bold]{next_model}[/].", style="bold cyan")
        self._set_status(f"Model: {next_model}")

    async def _stream_prompt(self, prompt: str) -> None:
        """Stream a user prompt through the Calute executor and update the UI.

        Creates user and placeholder assistant entries in the transcript,
        then iterates over the streamed response events to update
        content, handle tool calls, agent switches, and completion.

        Args:
            prompt: The user's message text.
        """
        self.is_busy = True
        input_widget = self._input_widget()
        input_widget.disabled = True
        conversation_messages = self._conversation_messages()
        self._start_run_stats(prompt, messages=conversation_messages)
        self.chat_history.append(ChatEntry(role="user", content=prompt))
        self._begin_streaming_assistant_phase()
        self._refresh_footer_bar()
        self._set_status("Thinking...")
        self._append_event(f"[bold green]user[/] {prompt}")

        assistant_buffer = ""
        reasoning_buffer = ""

        try:
            self._ensure_live_client_ready()
            stream = await self.executor.create_response(
                prompt=prompt,
                agent_id=self.current_agent_id,
                messages=conversation_messages,
                stream=True,
            )

            async for item in stream:
                if isinstance(item, StreamChunk):
                    tool_preview_updated = False
                    if item.streaming_tool_calls:
                        for tool_call in item.streaming_tool_calls:
                            self._update_streaming_tool_call(
                                tool_call,
                                agent_id=item.agent_id or self.current_agent_id,
                            )
                            tool_preview_updated = True
                    if item.content:
                        assistant_buffer += item.content
                        self.current_draft = self._sanitize_display_text(assistant_buffer)
                    if item.buffered_reasoning_content:
                        reasoning_buffer = item.buffered_reasoning_content
                        self.current_reasoning = self._sanitize_display_text(reasoning_buffer)
                    elif item.reasoning_content and not reasoning_buffer:
                        reasoning_buffer = item.reasoning_content
                        self.current_reasoning = self._sanitize_display_text(reasoning_buffer)
                    self._update_run_stats(assistant_buffer, reasoning_buffer)
                    self._update_streaming_assistant(
                        self._sanitize_display_text(assistant_buffer),
                        self._sanitize_display_text(reasoning_buffer),
                    )
                    if tool_preview_updated:
                        self._refresh_transcript()
                    self._refresh_footer_bar()
                elif isinstance(item, FunctionDetection):
                    self._append_event("Detected tool call(s).", style="bold cyan")
                    self._set_status("Detected tool calls.")
                elif isinstance(item, FunctionCallsExtracted):
                    self._close_streaming_assistant_phase(assistant_buffer, reasoning_buffer)
                    assistant_buffer = ""
                    reasoning_buffer = ""
                    for function_call in item.function_calls:
                        self._queue_tool_call_entry(
                            function_id=function_call.id,
                            function_name=function_call.name,
                            agent_id=item.agent_id or self.current_agent_id,
                        )
                    self._refresh_transcript()
                    names = ", ".join(f"{fc.name}#{fc.id}" for fc in item.function_calls)
                    self._append_event(f"Extracted: {names}", style="cyan")
                elif isinstance(item, FunctionExecutionStart):
                    progress = f" {item.progress}" if item.progress else ""
                    self._start_tool_entry(
                        function_id=item.function_id,
                        function_name=item.function_name,
                        progress=item.progress,
                        agent_id=item.agent_id or self.current_agent_id,
                    )
                    self._append_event(
                        f"[bold magenta]tool[/] {item.function_name}{progress}",
                        style="bold magenta",
                    )
                    self._set_status(f"Running {item.function_name}{progress}")
                elif isinstance(item, FunctionExecutionComplete):
                    self._complete_tool_entry(
                        function_id=item.function_id,
                        function_name=item.function_name,
                        status=item.status,
                        result=item.result,
                        error=item.error,
                        agent_id=item.agent_id or self.current_agent_id,
                    )
                    if item.status == "success":
                        preview = preview_value(item.result)
                        self._append_event(
                            f"[bold magenta]{item.function_name}[/] -> {preview}",
                            style="magenta",
                        )
                        self._set_status(f"{item.function_name} completed")
                    else:
                        self._append_event(
                            f"[bold red]{item.function_name}[/] failed: {item.error or 'unknown error'}",
                            style="bold red",
                        )
                        self._set_status(f"{item.function_name} failed")
                elif isinstance(item, ReinvokeSignal):
                    self._begin_streaming_assistant_phase()
                    self._append_event(item.message, style="bold yellow")
                    self._set_status("Reinvoking with tool results.")
                elif isinstance(item, AgentSwitch):
                    self.current_agent_id = item.to_agent
                    self._refresh_header_bar()
                    self._append_event(
                        f"Switched agent [bold]{item.from_agent}[/] -> [bold]{item.to_agent}[/]",
                        style="bold blue",
                    )
                    self._set_status(f"Active agent: {item.to_agent}")
                elif isinstance(item, Completion):
                    final_content = (
                        self._sanitize_display_text(item.final_content or assistant_buffer) or "(empty response)"
                    )
                    final_reasoning_raw = item.reasoning_content or reasoning_buffer or None
                    final_reasoning = self._sanitize_display_text(final_reasoning_raw or "") or None
                    self._finalize_run_stats(
                        item.final_content or assistant_buffer, final_reasoning_raw or reasoning_buffer
                    )
                    self._finalize_streaming_assistant(final_content, final_reasoning)
                    self.current_draft = ""
                    self.current_reasoning = ""
                    self._refresh_transcript()
                    self._refresh_footer_bar()
                    self._append_event(
                        f"[bold green]done[/] {len(final_content)} chars, {item.function_calls_executed} tool call(s)",
                        style="bold green",
                    )
                    self._set_status("Done.")
        except Exception as exc:  # pragma: no cover - exercised through the app
            message = self._format_runtime_error(exc)
            self._update_run_stats(assistant_buffer, reasoning_buffer)
            self._discard_streaming_assistant()
            self.chat_history.append(ChatEntry(role="error", content=message))
            self.current_draft = ""
            self.current_reasoning = ""
            self._refresh_transcript()
            self._refresh_footer_bar()
            self._append_event(f"[bold red]error[/] {message}", style="bold red")
            self._set_status(f"Error: {message}")
        finally:
            self.is_busy = False
            input_widget.disabled = False
            input_widget.focus()

    def _handle_command(self, text: str) -> bool:
        command, argument = parse_command(text)
        if command == "":
            self._set_status("Empty command.")
            return True
        if command == "clear":
            self.action_clear_chat()
            return True
        if command == "help":
            self._add_note(
                "Commands: /providers, /provider <name>, /endpoint <url>, /apikey <key>, /clear, /agent <id>,"
                " /profile <full|compact|minimal|none>, /models, /model <name>, /tools,"
                " /power on|off, /sessions, /agents, /plans,"
                " /sampling, /set <param> <value>, /reset-sampling, /help"
            )
            self._set_status("Help added to the transcript.")
            return True
        if command == "providers":
            self._add_note(
                "Providers: openai (alias: oai), anthropic (alias: claude), gemini (alias: google), ollama (alias: local)",
            )
            self._set_status("Listed available providers.")
            return True
        if command == "provider":
            if not argument:
                self._add_note(
                    "Current provider: "
                    f"{self._provider_name()}. Options: openai/oai, anthropic/claude, gemini/google, ollama/local",
                )
                self._set_status("Listed provider options.")
                return True
            self._set_provider(argument)
            return True
        if command == "endpoint":
            if not argument:
                endpoint = self._base_url() or "(not set)"
                self._add_note(f"Current endpoint: {endpoint}")
                self._set_status("Displayed current endpoint.")
                return True
            self._set_endpoint(argument)
            return True
        if command in {"apikey", "api-key"}:
            if not argument:
                state = self._masked_secret(self._api_key()) or "(not set)"
                self._add_note(f"Current API key: {state}")
                self._set_status("Displayed API key status.")
                return True
            self._set_api_key(argument)
            return True
        if command == "agent":
            if not argument:
                self._add_note(f"Agents: {', '.join(self._available_agent_ids()) or 'none'}")
                self._set_status("Listed available agents.")
                return True
            self._switch_agent(argument)
            return True
        if command == "profile":
            if not argument:
                self._set_status(f"Prompt profile: {self._current_prompt_profile().value}")
                return True
            try:
                profile = PromptProfile(argument.strip().lower())
            except ValueError:
                self._set_status("Unknown profile. Use full, compact, minimal, or none.")
                return True
            self._set_prompt_profile(profile)
            self._append_event(f"Prompt profile set to [bold]{profile.value}[/].", style="bold cyan")
            self._set_status(f"Prompt profile: {profile.value}")
            return True
        if command == "models":
            if self.available_models:
                self._add_note(f"Models: {', '.join(self.available_models)}")
                self._set_status(f"{len(self.available_models)} model(s) available.")
            else:
                self.run_worker(
                    self._refresh_available_models(),
                    name="model-discovery",
                    group="metadata",
                    description="Fetch available models",
                )
            self._set_status("Fetching available models...")
            return True
        if command == "tools":
            tools = self._current_agent().get_available_functions() if self._current_agent() else []
            self._add_note(f"Tools: {', '.join(tools) if tools else 'none'}")
            self._set_status(f"{len(tools)} tool(s) listed.")
            return True
        if command == "model":
            if not argument:
                current = self._current_model_name() or "none"
                self._add_note(f"Current model: {current}")
                return True
            self._set_active_model(argument)
            self._append_event(f"Model set to [bold]{argument}[/].", style="bold cyan")
            self._set_status(f"Model: {argument}")
            return True
        if command == "power":
            operator_state = self._operator_state()
            if operator_state is None:
                self._set_status("Operator tooling is not enabled for this runtime.")
                return True
            if not argument:
                state = "on" if self._power_tools_enabled() else "off"
                self._add_note(f"Power tools: {state}")
                self._set_status(f"Power tools: {state}")
                return True
            normalized = argument.strip().lower()
            if normalized not in {"on", "off", "true", "false", "1", "0"}:
                self._set_status("Usage: /power on|off")
                return True
            enabled = normalized in {"on", "true", "1"}
            operator_state.set_power_tools_enabled(enabled)
            self._persist_profile(power_tools_enabled=enabled)
            self._refresh_header_bar()
            self._refresh_footer_bar()
            state = "on" if enabled else "off"
            self._add_note(f"Power tools turned {state}.")
            self._set_status(f"Power tools: {state}")
            return True
        if command == "sessions":
            operator_state = self._operator_state()
            if operator_state is None:
                self._set_status("Operator tooling is not enabled for this runtime.")
                return True
            state = operator_state.list_operator_state()
            pty_sessions = state["pty_sessions"]
            browser_pages = state["browser_pages"]
            note = (
                f"PTY sessions ({len(pty_sessions)}): "
                f"{', '.join(session['session_id'] for session in pty_sessions) or 'none'}\n"
                f"Browser pages ({len(browser_pages)}): "
                f"{', '.join(page['ref_id'] for page in browser_pages) or 'none'}"
            )
            self._add_note(note)
            self._set_status("Operator sessions listed.")
            return True
        if command == "agents":
            operator_state = self._operator_state()
            if operator_state is None:
                self._set_status("Operator tooling is not enabled for this runtime.")
                return True
            handles = operator_state.list_operator_state()["spawned_agents"]
            if not handles:
                self._add_note("Spawned agents: none")
            else:
                rows = []
                for handle in handles:
                    row = f"{handle['id']} ({handle['status']})"
                    if handle.get("last_input"):
                        row += f" task={self._truncate(handle['last_input'], 48)}"
                    if handle.get("queue_size"):
                        row += f" queued={handle['queue_size']}"
                    rows.append(row)
                self._add_note("Spawned agents:\n- " + "\n- ".join(rows))
            self._set_status("Spawned agents listed.")
            return True
        if command == "plans":
            operator_state = self._operator_state()
            if operator_state is None:
                self._set_status("Operator tooling is not enabled for this runtime.")
                return True
            plan = operator_state.list_operator_state()["plan"]
            steps = plan.get("steps", [])
            if not steps:
                self._add_note(f"Plan revision {plan.get('revision', 0)}: no steps")
            else:
                lines = [f"Plan revision {plan.get('revision', 0)}:"]
                lines.extend(f"- {step['status']}: {step['step']}" for step in steps)
                self._add_note("\n".join(lines))
            self._set_status("Plan summary added to the transcript.")
            return True
        if command == "sampling":
            self._add_note(self._sampling_summary())
            self._set_status("Sampling settings added to the transcript.")
            return True
        if command == "set":
            if not argument:
                self._set_status("Usage: /set <param> <value>")
                return True
            self._set_sampling_from_command(argument)
            return True
        if command == "reset-sampling":
            self._reset_sampling_params()
            return True
        self._set_status(f"Unknown command: {command}")
        return True

    def _command_suggestion_values(self) -> list[str]:
        """Return slash-command suggestions used for inline completion."""
        suggestions = [
            "/help",
            "/clear",
            "/providers",
            "/provider",
            "/endpoint",
            "/apikey",
            "/models",
            "/tools",
            "/agent",
            "/profile",
            "/model",
            "/power",
            "/sessions",
            "/agents",
            "/plans",
            "/sampling",
            "/set",
            "/reset-sampling",
        ]
        suggestions.extend(f"/provider {provider}" for provider in self._provider_command_names())
        suggestions.extend(f"/profile {profile.value}" for profile in self.PROFILE_ORDER)
        suggestions.extend(f"/agent {agent_id}" for agent_id in self._available_agent_ids())
        suggestions.extend(f"/model {model_name}" for model_name in self.available_models)
        suggestions.extend(["/power on", "/power off"])
        suggestions.extend(f"/set {name}" for name in SAMPLING_PARAM_KEYS)

        deduped: list[str] = []
        seen: set[str] = set()
        for suggestion in suggestions:
            if suggestion not in seen:
                deduped.append(suggestion)
                seen.add(suggestion)
        return deduped

    def _base_command_hints(self) -> list[SlashCommandHint]:
        """Return base slash-command help rows."""
        return [
            SlashCommandHint("/help", "Show command help."),
            SlashCommandHint("/clear", "Clear the transcript pane."),
            SlashCommandHint("/providers", "List supported providers and aliases."),
            SlashCommandHint(
                "/provider <name>", "Switch provider: openai/oai, anthropic/claude, gemini/google, ollama/local."
            ),
            SlashCommandHint("/endpoint <url>", "Set the provider endpoint or OpenAI-compatible base URL."),
            SlashCommandHint("/apikey <key>", "Set or clear the API key for the current provider."),
            SlashCommandHint("/agent <id>", "Switch the active agent."),
            SlashCommandHint("/profile <mode>", "Set prompt mode: full, compact, minimal, none."),
            SlashCommandHint("/models", "Show discovered models for the current endpoint."),
            SlashCommandHint("/tools", "List tools available on the active agent."),
            SlashCommandHint("/model <name>", "Switch the active model and save it."),
            SlashCommandHint("/power on|off", "Enable or disable high-power operator tools."),
            SlashCommandHint("/sessions", "Inspect live PTY and browser session state."),
            SlashCommandHint("/agents", "Inspect spawned background agent handles."),
            SlashCommandHint("/plans", "Show the current structured plan state."),
            SlashCommandHint("/sampling", "Show current agent sampling parameters."),
            SlashCommandHint("/set <param> <value>", "Update a supported sampling parameter."),
            SlashCommandHint("/reset-sampling", "Reset sampling parameters to Agent defaults."),
        ]

    def _match_slash_hints(self, value: str) -> list[SlashCommandHint]:
        """Compute the most relevant slash-command hints for the current input."""
        stripped = value.lstrip()
        if not stripped.startswith("/"):
            return []

        body = stripped[1:]
        command_token, argument = ([*body.split(" ", 1), ""])[:2] if body else ["", ""]
        command_token = command_token.strip().lower()
        argument = argument.strip()
        base_hints = self._base_command_hints()

        if not command_token:
            return base_hints

        if not argument:
            matches = [hint for hint in base_hints if hint.usage.split(" ", 1)[0][1:].lower().startswith(command_token)]
            if matches:
                return matches

            names = [hint.usage.split(" ", 1)[0][1:].lower() for hint in base_hints]
            fuzzy = difflib.get_close_matches(command_token, names, n=4, cutoff=0.4)
            if fuzzy:
                return [hint for hint in base_hints if hint.usage.split(" ", 1)[0][1:].lower() in fuzzy]
            return []

        if command_token == "profile":
            return self._match_dynamic_options(
                prefix="/profile",
                values=[profile.value for profile in self.PROFILE_ORDER],
                query=argument,
                description_factory=lambda profile: f"Use prompt profile '{profile}'.",
            )

        if command_token == "provider":
            return self._match_dynamic_options(
                prefix="/provider",
                values=self._provider_command_names(),
                query=argument,
                description_factory=lambda provider: f"Switch provider to '{provider}'.",
            )

        if command_token == "agent":
            return self._match_dynamic_options(
                prefix="/agent",
                values=self._available_agent_ids(),
                query=argument,
                description_factory=lambda agent_id: f"Switch active agent to '{agent_id}'.",
            )

        if command_token == "model":
            return self._match_dynamic_options(
                prefix="/model",
                values=self.available_models,
                query=argument,
                description_factory=lambda model_name: f"Use model '{model_name}'.",
            )

        if command_token == "power":
            return self._match_dynamic_options(
                prefix="/power",
                values=["on", "off"],
                query=argument,
                description_factory=lambda value: f"Turn power tools {value}.",
            )

        if command_token == "set":
            if " " not in argument:
                return self._match_dynamic_options(
                    prefix="/set",
                    values=list(SAMPLING_PARAM_KEYS),
                    query=argument,
                    description_factory=lambda name: f"Change sampling parameter '{name}'.",
                )

        return []

    def _match_dynamic_options(
        self,
        *,
        prefix: str,
        values: list[str],
        query: str,
        description_factory: tp.Callable[[str], str],
    ) -> list[SlashCommandHint]:
        """Match dynamic command arguments using prefix and fuzzy search."""
        if not values:
            return []

        lowered_query = query.lower()
        matches = [value for value in values if value.lower().startswith(lowered_query)]
        if not matches and lowered_query:
            matches = difflib.get_close_matches(lowered_query, values, n=6, cutoff=0.4)

        return [SlashCommandHint(f"{prefix} {value}", description_factory(value)) for value in matches[:6]]

    def _update_status_from_input(self, value: str) -> None:
        """Update status bar based on current input context."""
        stripped = value.lstrip()
        pending = self._pending_user_prompt()
        if pending is not None and not stripped.startswith("/"):
            self._set_status(f"? {self._truncate(pending['question'], 60)}")
            return
        if stripped.startswith("/"):
            hints = self._match_slash_hints(stripped)
            if hints:
                self._set_status("  ".join(f"{h.usage}" for h in hints[:4]))
            else:
                self._set_status("No matching commands. Try /help.")
            return
        if not self.is_busy:
            self._set_status("")

    def _switch_agent(self, agent_id: str) -> None:
        if agent_id not in self.executor.orchestrator.agents:
            self._set_status(f"Unknown agent: {agent_id}")
            return
        self.executor.orchestrator.switch_agent(agent_id, "TUI user selection")
        self.current_agent_id = agent_id
        self._refresh_header_bar()
        self._append_event(f"Active agent -> [bold]{agent_id}[/]", style="bold blue")
        self._set_status(f"Switched to agent {agent_id}")

    def _current_prompt_profile(self) -> PromptProfile:
        runtime_state = self.executor._runtime_features_state
        if runtime_state is None:
            return PromptProfile.FULL
        profile = runtime_state.get_prompt_profile(self.current_agent_id)
        if isinstance(profile, PromptProfile):
            return profile
        if isinstance(profile, str):
            return PromptProfile(profile.strip().lower())
        if profile is not None:
            return profile.profile
        return PromptProfile.FULL

    def _set_prompt_profile(self, profile: PromptProfile) -> None:
        runtime_state = self.executor._runtime_features_state
        if runtime_state is None:
            self._set_status("Prompt profiles require runtime_features.enabled=True.")
            return

        if self.current_agent_id:
            overrides = runtime_state.config.agent_overrides.setdefault(
                self.current_agent_id, runtime_state.get_agent_overrides(self.current_agent_id)
            )
            overrides.prompt_profile = profile
        else:
            runtime_state.config.default_prompt_profile = profile
        self._persist_profile(prompt_profile=profile.value)
        self._refresh_header_bar()

    def _resolve_agent_id(self, agent: Agent | str | None) -> str | None:
        if isinstance(agent, Agent):
            return agent.id
        if isinstance(agent, str):
            return agent
        return self.executor.orchestrator.current_agent_id

    def _available_agent_ids(self) -> list[str]:
        return list(self.executor.orchestrator.agents.keys())

    def _append_event(self, message: str, style: str | None = None) -> None:
        line = Text.from_markup(message) if "[" in message and "]" in message else Text(message)
        if style:
            line.stylize(style)
        self.activity_history.append(line.plain)

    def _tick_operations(self) -> None:
        """Refresh status while tools or sub-agents are active."""
        self._sync_pending_user_prompt()
        if self._has_running_tools() or self._has_running_subagents():
            self._refresh_footer_bar()

    def _add_note(self, message: str) -> None:
        self.chat_history.append(ChatEntry(role="note", content=message))
        self._refresh_transcript()

    def _set_status(self, message: str) -> None:
        try:
            status = Text()
            if self.is_busy:
                frame = self.SPINNER_FRAMES[int(time.perf_counter() * 10) % len(self.SPINNER_FRAMES)]
                status.append(f" {frame} ", style="bold #7B8CFF")
            else:
                status.append(" \u2502 ", style="#30363d")
            status.append(message, style="#484f58")
            self.query_one("#status-bar", Static).update(status)
        except NoMatches:
            return

    def _refresh_header_bar(self) -> None:
        """Render the header bar with connection and agent info."""
        agent_id = self.current_agent_id or "none"
        provider = self._provider_name()
        model = self._current_model_name() or "-"
        profile = self._current_prompt_profile().value
        tools = len(self._current_agent().functions) if self._current_agent() is not None else 0

        line1 = Text()
        line1.append(" \u2588\u2588 ", style="bold #B07FFF")
        line1.append("CALUTE", style="bold #e0e4ed")
        line1.append("  ", style="")
        line1.append(f" {agent_id} ", style="bold #c9d1d9 on #1f2937")
        line1.append("  ", style="")
        line1.append(f"{provider}", style="#7B8CFF")
        line1.append(" / ", style="#30363d")
        line1.append(f"{model}", style="#B07FFF")
        line1.append("  ", style="")
        line1.append(f"[{profile}]", style="#484f58")
        line1.append(f"  tools:{tools}", style="#484f58")

        try:
            self.query_one("#header-bar", Static).update(line1)
        except NoMatches:
            return

    def _refresh_transcript(self) -> None:
        if not self.chat_history:
            welcome = Text()
            welcome.append("\n\n\n")
            welcome.append("   \u2588\u2588  ", style="bold #B07FFF")
            welcome.append("C A L U T E\n\n", style="bold #c9d1d9")
            welcome.append("   Streaming terminal runtime\n\n", style="#484f58")
            welcome.append("   Type a message below or ", style="#30363d")
            welcome.append("/help", style="#7B8CFF")
            welcome.append(" for commands\n", style="#30363d")
            self.query_one("#transcript", Static).update(welcome)
            self._scroll_transcript_to_end()
            return

        renderables: list[tp.Any] = []
        for entry in self.chat_history:
            renderables.append(self._build_message_renderable(entry))
        self.query_one("#transcript", Static).update(Group(*renderables))
        self._scroll_transcript_to_end()

    def _refresh_footer_bar(self) -> None:
        """Render the footer bar with token stats and shortcuts."""
        power = "on" if self._power_tools_enabled() else "off"
        footer = Text()
        footer.append(" ^L", style="bold #7B8CFF")
        footer.append(" clear ", style="#484f58")
        footer.append(" ^S", style="bold #7B8CFF")
        footer.append(" send ", style="#484f58")
        footer.append(" ^J", style="bold #7B8CFF")
        footer.append(" input ", style="#484f58")
        footer.append(" ^N", style="bold #7B8CFF")
        footer.append(" agent ", style="#484f58")
        footer.append(" ^P", style="bold #7B8CFF")
        footer.append(" profile ", style="#484f58")
        footer.append(" ^M", style="bold #7B8CFF")
        footer.append(" model ", style="#484f58")
        footer.append("  \u2502  ", style="#30363d")
        footer.append(f"power:{power}", style="#484f58")
        footer.append("  ", style="")
        if self.prompt_tokens_used or self.output_tokens_used:
            footer.append(f"in:{self.prompt_tokens_used}", style="#484f58")
            footer.append("  ", style="")
            footer.append(f"out:{self.output_tokens_used}", style="#484f58")
            footer.append("  ", style="")
            footer.append(f"tps:{self.tokens_per_second:.1f}", style="#484f58")
        try:
            self.query_one("#footer-bar", Static).update(footer)
        except NoMatches:
            return

    def _visible_tool_activity(self) -> list[ToolActivity]:
        """Return running tools first, then the most recent completed ones."""
        ordered = [self.tool_activity[key] for key in self.tool_activity_order if key in self.tool_activity]
        running = [activity for activity in ordered if activity.status == "running"]
        recent = [activity for activity in ordered if activity.status != "running"][:4]
        return running + recent

    def _has_running_tools(self) -> bool:
        """Report whether any tracked tool call is still active."""
        return any(activity.status == "running" for activity in self.tool_activity.values())

    def _has_running_subagents(self) -> bool:
        """Report whether any spawned agent is currently active."""
        operator_state = self._operator_state()
        if operator_state is None:
            return False
        snapshot = operator_state.list_operator_state()
        return any(handle.get("status") == "running" for handle in snapshot["spawned_agents"])

    @staticmethod
    def _truncate(value: str, max_length: int) -> str:
        """Truncate long text for display."""
        text = value.strip().replace("\n", " ")
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    def _refresh_bars(self) -> None:
        """Convenience method to refresh both header and footer bars."""
        self._refresh_header_bar()
        self._refresh_footer_bar()

    def _count_tokens(self, text_or_messages: str | list[dict[str, str]]) -> int:
        """Estimate token usage for the current provider/model."""
        return ProviderTokenCounter.count_tokens_for_provider(
            text_or_messages,
            provider=self._canonical_provider_name(self._provider_name()),
            model=self._current_model_name(),
            llm_client=self.executor.llm_client,
        )

    def _inline_value_preview(self, value: tp.Any, max_length: int = 56) -> str:
        """Render a compact one-line preview for tool args and results."""
        if isinstance(value, str):
            return json.dumps(self._truncate(value, max_length), ensure_ascii=False)
        return self._truncate(preview_value(value, max_length=max_length * 4), max_length)

    def _format_tool_arguments_preview(self, arguments: dict[str, tp.Any] | str | None) -> str | None:
        """Format tool arguments into a compact Claude-style input summary."""
        if arguments is None:
            return None

        payload: tp.Any = arguments
        partial = False
        if isinstance(arguments, str):
            stripped = arguments.strip()
            if not stripped:
                return None
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = stripped
                partial = True

        if isinstance(payload, dict):
            items: list[str] = []
            for index, (key, value) in enumerate(payload.items()):
                if index >= 3:
                    items.append(f"+{len(payload) - 3} more")
                    break
                items.append(f"{key}={self._inline_value_preview(value, max_length=48)}")
            return ", ".join(items) if items else "{}"

        prefix = "partial args: " if partial else ""
        return prefix + self._truncate(str(payload), 120)

    def _summarize_tool_result(self, result: tp.Any) -> str:
        """Condense tool results into a short readable summary."""
        if result is None:
            return "completed"

        if isinstance(result, dict):
            if isinstance(result.get("results"), list):
                results = result["results"]
                parts: list[str] = []
                query = result.get("query")
                if query:
                    parts.append(f"query={self._inline_value_preview(query, max_length=40)}")
                parts.append(f"{len(results)} result(s)")
                if results and isinstance(results[0], dict):
                    top_hit = results[0].get("title") or results[0].get("name") or results[0].get("url")
                    if top_hit:
                        parts.append(f"top={self._inline_value_preview(str(top_hit), max_length=60)}")
                return "; ".join(parts)

            for collection_key in ("items", "matches", "data"):
                collection = result.get(collection_key)
                if isinstance(collection, list):
                    parts = [f"{len(collection)} {collection_key}"]
                    if collection:
                        parts.append(f"first={self._inline_value_preview(collection[0], max_length=60)}")
                    return "; ".join(parts)

            parts = []
            for key in ("query", "url", "title", "name", "status", "message", "answer", "text", "ref_id"):
                value = result.get(key)
                if value in (None, "", [], {}):
                    continue
                parts.append(f"{key}={self._inline_value_preview(value, max_length=52)}")
                if len(parts) >= 3:
                    break
            if parts:
                return ", ".join(parts)

        if isinstance(result, list):
            summary = f"{len(result)} item(s)"
            if result:
                summary += f"; first={self._inline_value_preview(result[0], max_length=60)}"
            return summary

        if isinstance(result, str):
            lines = [line.strip() for line in result.splitlines() if line.strip()]
            if not lines:
                return "completed"
            summary = self._truncate(lines[0], 160)
            if len(lines) > 1:
                summary += f" (+{len(lines) - 1} lines)"
            return summary

        return self._inline_value_preview(result, max_length=160)

    def _summarize_tool_error(self, error: str | None) -> str:
        """Condense long tool errors into one readable line."""
        if not error:
            return "unknown error"
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        if not lines:
            return self._truncate(str(error), 160)

        summary = self._truncate(lines[0], 160)
        if len(lines) > 1:
            summary = f"{summary} / {self._truncate(lines[1], 120)}"
        if len(lines) > 2:
            summary += f" (+{len(lines) - 2} lines)"
        return summary

    def _find_tool_chat_entry(self, function_id: str) -> ChatEntry | None:
        """Return the existing transcript card for a tool call if present."""
        for entry in reversed(self.chat_history):
            if entry.role == "tool" and entry.key == function_id:
                return entry
        return None

    def _queue_tool_call_entry(
        self,
        function_id: str,
        function_name: str,
        arguments_preview: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Create or update a tool card while the model is still emitting it."""
        activity = self.tool_activity.get(function_id)
        if activity is None:
            activity = ToolActivity(
                function_id=function_id,
                function_name=function_name,
                agent_id=agent_id,
                arguments_preview=arguments_preview,
                started_at=time.perf_counter(),
            )
            self.tool_activity[function_id] = activity
            self.tool_activity_order = [function_id, *[key for key in self.tool_activity_order if key != function_id]][
                :10
            ]
        else:
            activity.function_name = function_name
            activity.agent_id = agent_id
            if arguments_preview:
                activity.arguments_preview = arguments_preview

        entry = self._find_tool_chat_entry(function_id)
        if entry is None:
            self.chat_history.append(
                ChatEntry(
                    role="tool",
                    title=function_name,
                    meta=arguments_preview,
                    content="drafting call...",
                    streaming=True,
                    key=function_id,
                )
            )
            return

        entry.title = function_name
        if arguments_preview:
            entry.meta = arguments_preview
        if not entry.content or entry.content == "queued...":
            entry.content = "drafting call..."
        entry.streaming = True

    def _update_streaming_tool_call(self, tool_call: ToolCallStreamChunk, agent_id: str | None = None) -> None:
        """Reflect partial streamed tool-call data in the transcript."""
        self._queue_tool_call_entry(
            function_id=tool_call.id,
            function_name=tool_call.function_name or "tool",
            arguments_preview=self._format_tool_arguments_preview(tool_call.arguments),
            agent_id=agent_id,
        )

    def _conversation_messages(self) -> MessagesHistory:
        """Convert visible transcript turns into model message history.

        Keeps only user/assistant turns so follow-up prompts preserve the
        conversational state without replaying TUI-only panels such as tool
        cards, notes, or hidden reasoning.
        """
        messages: list[UserMessage | AssistantMessage] = []
        for entry in self.chat_history:
            if entry.streaming:
                continue
            if entry.role == "user":
                messages.append(UserMessage(content=entry.content))
            elif entry.role == "assistant":
                messages.append(AssistantMessage(content=entry.content or None))
        return MessagesHistory(messages=messages)

    def _start_run_stats(self, prompt: str, messages: MessagesHistory | None = None) -> None:
        """Reset and initialize token/TPS counters for a new streamed turn."""
        self.prompt_tokens_used = 0
        self.output_tokens_used = 0
        self.tokens_per_second = 0.0
        self._stream_started_at = time.perf_counter()

        agent = self._current_agent()
        if agent is None:
            self.prompt_tokens_used = self._count_tokens(prompt)
            self._refresh_bars()
            return

        try:
            prompt_messages = self.executor.manage_messages(agent=agent, prompt=prompt, messages=messages).to_openai()[
                "messages"
            ]
            self.prompt_tokens_used = self._count_tokens(prompt_messages)
        except Exception:
            self.prompt_tokens_used = self._count_tokens(prompt)
        self._refresh_bars()

    def _update_run_stats(self, assistant_content: str, reasoning_content: str) -> None:
        """Refresh output token and TPS estimates while streaming."""
        output_text = "".join(part for part in [assistant_content, reasoning_content] if part)
        self.output_tokens_used = self._count_tokens(output_text) if output_text else 0
        if self._stream_started_at is not None:
            elapsed = max(time.perf_counter() - self._stream_started_at, 1e-6)
            self.tokens_per_second = (
                self.output_tokens_used / elapsed if self.output_tokens_used and elapsed >= 0.25 else 0.0
            )
        self._refresh_bars()

    def _finalize_run_stats(self, assistant_content: str, reasoning_content: str) -> None:
        """Lock in final output token and TPS estimates for the completed turn."""
        self._update_run_stats(assistant_content, reasoning_content)

    def _build_message_renderable(self, entry: ChatEntry) -> tp.Any:
        """Build a clean renderable for a chat entry (OpenClaw style)."""
        renderables: list[tp.Any] = []

        if entry.role == "user":
            renderables.append(Text(""))
            label = Text()
            label.append(" \u25cf ", style="#7B8CFF")
            label.append(entry.title or "you", style="bold #c9d1d9")
            renderables.append(label)
            if entry.content:
                content = Text()
                for line in entry.content.split("\n"):
                    content.append(f"   {line}\n", style="#c9d1d9")
                renderables.append(content)

        elif entry.role == "assistant":
            renderables.append(Text(""))
            agent_name = entry.title or self.current_agent_id or "assistant"
            label = Text()
            label.append(" \u25cf ", style="#B07FFF")
            label.append(agent_name, style="bold #c9d1d9")
            if entry.streaming:
                label.append("  \u2026", style="#484f58")
            renderables.append(label)
            if entry.meta:
                think_panel = Panel(
                    Text(entry.meta, style="italic #484f58"),
                    border_style="#30363d",
                    title=Text("thinking", style="#484f58"),
                    title_align="left",
                    padding=(0, 1),
                )
                renderables.append(think_panel)
            if entry.content:
                content = Text()
                for line in entry.content.split("\n"):
                    content.append(f"   {line}\n", style="#c9d1d9")
                renderables.append(content)
            elif entry.streaming:
                frame = self.SPINNER_FRAMES[int(time.perf_counter() * 10) % len(self.SPINNER_FRAMES)]
                renderables.append(Text(f"   {frame}", style="#7B8CFF"))

        elif entry.role == "tool":
            border_style = "#30363d"
            title_style = "#484f58"
            title_icon = "\u2500"
            detail_style = "#c9d1d9"
            detail_label_style = "bold #8b949e"
            if entry.streaming:
                border_style = "#7B8CFF"
                title_style = "#7B8CFF"
                title_icon = "\u25cb"
            elif entry.title and "\u2713" in entry.title:
                border_style = "#238636"
                title_style = "#3fb950"
                title_icon = "\u2713"
            elif entry.title and "\u2717" in entry.title:
                border_style = "#da3633"
                title_style = "#f85149"
                title_icon = "\u2717"

            def _tool_line(label: str, value: str, value_style: str = detail_style) -> Text:
                line = Text()
                line.append(f" {label:<6}", style=detail_label_style)
                line.append(value, style=value_style)
                return line

            tool_parts: list[tp.Any] = []
            if entry.meta:
                tool_parts.append(_tool_line("input", entry.meta))
            if entry.content:
                content_label = "status"
                content_style = detail_style
                if entry.streaming:
                    content_style = "#7B8CFF"
                elif entry.title and "\u2713" in entry.title:
                    content_label = "result"
                elif entry.title and "\u2717" in entry.title:
                    content_label = "error"
                    content_style = "#f85149"
                tool_parts.append(_tool_line(content_label, entry.content, value_style=content_style))
            tool_name = (entry.title or "tool").replace(" \u2713", "").replace(" \u2717", "")
            title_text = Text(f" {title_icon} {tool_name} ", style=title_style)
            return Panel(
                Group(*tool_parts) if tool_parts else Text("\u25cb running\u2026", style="italic #7B8CFF"),
                title=title_text,
                title_align="left",
                border_style=border_style,
                padding=(0, 1),
            )

        elif entry.role == "question":
            q_parts: list[tp.Any] = []
            q_parts.append(Text(entry.content or "", style="#c9d1d9"))
            if entry.meta:
                q_parts.append(Text(""))
                q_parts.append(Text(entry.meta, style="italic #484f58"))
            return Panel(
                Group(*q_parts),
                title=Text(f" \u2753 {entry.title or 'question'} ", style="bold #B07FFF"),
                title_align="left",
                border_style="#B07FFF",
                padding=(0, 1),
            )

        elif entry.role == "note":
            renderables.append(Text(f"   {entry.content or ''}", style="#484f58"))

        else:
            renderables.append(Text(""))
            err_parts: list[tp.Any] = []
            err_parts.append(Text(entry.content or "", style="#f85149"))
            return Panel(
                Group(*err_parts),
                title=Text(" \u2717 error ", style="bold #f85149"),
                title_align="left",
                border_style="#da3633",
                padding=(0, 1),
            )

        return Group(*renderables)

    def _start_tool_entry(
        self,
        function_id: str,
        function_name: str,
        progress: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        activity = self.tool_activity.get(function_id)
        if activity is None:
            activity = ToolActivity(
                function_id=function_id,
                function_name=function_name,
                progress=progress,
                agent_id=agent_id,
                started_at=time.perf_counter(),
            )
            self.tool_activity[function_id] = activity
            self.tool_activity_order = [function_id, *[key for key in self.tool_activity_order if key != function_id]][
                :10
            ]
        else:
            activity.function_name = function_name
            activity.status = "running"
            activity.progress = progress
            activity.agent_id = agent_id
            activity.preview = None
            activity.error = None
            activity.finished_at = None
        entry = self._find_tool_chat_entry(function_id)
        progress_text = f" {progress}" if progress else ""
        status_text = f"running {progress}" if progress else "running..."
        if entry is None:
            self.chat_history.append(
                ChatEntry(
                    role="tool",
                    title=f"{function_name}{progress_text}",
                    meta=activity.arguments_preview,
                    content=status_text,
                    streaming=True,
                    key=function_id,
                )
            )
        else:
            entry.title = f"{function_name}{progress_text}"
            entry.meta = activity.arguments_preview
            entry.content = status_text
            entry.streaming = True
        self._refresh_transcript()

    def _complete_tool_entry(
        self,
        function_id: str,
        function_name: str,
        status: str,
        result: tp.Any,
        error: str | None,
        agent_id: str | None = None,
    ) -> None:
        activity = self.tool_activity.get(function_id)
        if activity is None:
            activity = ToolActivity(
                function_id=function_id,
                function_name=function_name,
                agent_id=agent_id,
                started_at=time.perf_counter(),
            )
            self.tool_activity[function_id] = activity
            self.tool_activity_order = [function_id, *[key for key in self.tool_activity_order if key != function_id]][
                :10
            ]

        activity.function_name = function_name
        activity.agent_id = agent_id
        activity.status = "success" if status == "success" else "failed"
        activity.progress = None
        result_summary = self._summarize_tool_result(result) if status == "success" else None
        error_summary = self._summarize_tool_error(error) if status != "success" else None
        activity.preview = result_summary
        activity.error = error_summary
        activity.finished_at = time.perf_counter()
        for entry in reversed(self.chat_history):
            if entry.role == "tool" and entry.key == function_id:
                if status == "success":
                    entry.title = f"{function_name} \u2713"
                    entry.meta = activity.arguments_preview
                    entry.content = result_summary or "completed"
                else:
                    entry.title = f"{function_name} \u2717"
                    entry.meta = activity.arguments_preview
                    entry.content = error_summary or "unknown error"
                entry.streaming = False
                break
        self._refresh_transcript()

    def _sanitize_display_text(self, text: str) -> str:
        """Strip tool-call markup from user-visible transcript/reasoning text."""
        if not text:
            return ""
        cleaned = text
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"```tool_call.*?```", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<(\w+)>\s*<arguments>.*?</arguments>\s*</\1>", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def _begin_streaming_assistant_phase(self) -> None:
        """Append a fresh assistant slot for the next reasoning/tool phase."""
        self.chat_history.append(ChatEntry(role="assistant", content="", streaming=True))
        self.current_draft = ""
        self.current_reasoning = ""
        self._refresh_transcript()

    def _close_streaming_assistant_phase(self, content: str, reasoning: str) -> None:
        """Freeze the current assistant phase before tool execution/reinvoke."""
        final_content = self._sanitize_display_text(content)
        final_reasoning = self._sanitize_display_text(reasoning) or None
        if final_content or final_reasoning:
            self._finalize_streaming_assistant(final_content, final_reasoning)
        else:
            self._discard_streaming_assistant()
        self.current_draft = ""
        self.current_reasoning = ""
        self._refresh_transcript()

    def _scroll_transcript_to_end(self) -> None:
        self.call_after_refresh(lambda: self.query_one("#transcript-scroll", VerticalScroll).scroll_end(animate=False))

    def _update_streaming_assistant(self, content: str, reasoning: str) -> None:
        for entry in reversed(self.chat_history):
            if entry.role == "assistant" and entry.streaming:
                entry.content = content
                entry.meta = reasoning or None
                self._refresh_transcript()
                return

    def _finalize_streaming_assistant(self, content: str, reasoning: str | None) -> None:
        for entry in reversed(self.chat_history):
            if entry.role == "assistant" and entry.streaming:
                entry.content = content
                entry.meta = reasoning
                entry.streaming = False
                return
        self.chat_history.append(ChatEntry(role="assistant", content=content, meta=reasoning))

    def _discard_streaming_assistant(self) -> None:
        if self.chat_history and self.chat_history[-1].role == "assistant" and self.chat_history[-1].streaming:
            self.chat_history.pop()

    async def _refresh_available_models(self) -> None:
        """Fetch available models from the configured endpoint and cache them."""
        provider = self._provider_name()
        missing = self._provider_missing_requirements(
            provider=provider,
            base_url=self._base_url(),
            api_key=self._api_key(),
        )
        if missing is not None:
            self._append_event(f"Model discovery skipped: {missing}", "bold yellow")
            self._set_status(missing)
            return
        try:
            models = await asyncio.to_thread(
                discover_available_models,
                provider,
                model=self._current_model_name(),
                api_key=self._api_key(),
                base_url=self._base_url(),
            )
        except Exception as exc:
            self._append_event(
                f"Model discovery failed: {exc}. Use /model <name> to set one manually.",
                "bold red",
            )
            self._set_status(f"Model discovery failed: {exc}")
            return

        self.available_models = models
        self._persist_profile(available_models=models)
        if models:
            if self._current_model_name() not in models:
                self._set_active_model(models[0])
            self._append_event(f"Discovered models: {', '.join(models)}", "bold cyan")
            self._set_status(f"Loaded {len(models)} model(s).")
        else:
            self._set_status("No models returned by endpoint.")
        self._refresh_header_bar()

    def _current_agent(self) -> Agent | None:
        if not self.current_agent_id:
            return None
        return self.executor.orchestrator.agents.get(self.current_agent_id)

    def _current_sampling_params(self) -> dict[str, float | int]:
        agent = self._current_agent()
        if agent is not None:
            return {name: getattr(agent, name, DEFAULT_SAMPLING_PARAMS[name]) for name in SAMPLING_PARAM_KEYS}

        if self.profile is not None and self.profile.sampling_params:
            sampling = dict(DEFAULT_SAMPLING_PARAMS)
            sampling.update(self.profile.sampling_params)
            return sampling

        return dict(DEFAULT_SAMPLING_PARAMS)

    def _operator_state(self):
        runtime_state = self.executor._runtime_features_state
        if runtime_state is None:
            return None
        return runtime_state.operator_state

    def _pending_user_prompt(self) -> dict[str, tp.Any] | None:
        """Return the current pending user question from operator state."""
        operator_state = self._operator_state()
        if operator_state is None:
            return None
        return operator_state.user_prompt_manager.get_pending()

    def _sync_pending_user_prompt(self) -> None:
        """Mirror the pending user question into transcript and input state."""
        pending = self._pending_user_prompt()
        try:
            input_widget = self._input_widget()
        except NoMatches:
            return

        if pending is None:
            if self._last_pending_question_id is not None:
                self._last_pending_question_id = None
                if self.is_busy:
                    input_widget.disabled = True
                self._set_input_hint(self.DEFAULT_INPUT_PLACEHOLDER)
                self._update_status_from_input(input_widget.text)
            return

        self._last_pending_question_id = pending["request_id"]
        input_widget.disabled = False
        self._set_input_hint(
            pending.get("placeholder") or "Answer the question above. Cmd+Enter to send.",
            title="answer",
        )
        self._upsert_pending_question_entry(pending)
        self._update_status_from_input(input_widget.text)
        self._set_status("Waiting for your answer.")

    def _upsert_pending_question_entry(self, pending: dict[str, tp.Any]) -> None:
        """Create or update the visible transcript card for a pending question."""
        content_lines = [pending["question"]]
        options = pending.get("options") or []
        if options:
            content_lines.append("")
            content_lines.extend(f"{index}. {option['label']}" for index, option in enumerate(options, start=1))

        meta = (
            "Reply with a number, option label, or custom answer."
            if pending.get("allow_freeform", True)
            else "Reply with one of the listed options."
        )

        for entry in reversed(self.chat_history):
            if entry.role == "question" and entry.key == pending["request_id"]:
                entry.content = "\n".join(content_lines)
                entry.meta = meta
                self._refresh_transcript()
                return

        self.chat_history.append(
            ChatEntry(
                role="question",
                title="Needs Input",
                meta=meta,
                content="\n".join(content_lines),
                key=pending["request_id"],
            )
        )
        self._refresh_transcript()

    def _submit_pending_user_answer(self, text: str) -> None:
        """Resolve the current pending question from user input."""
        operator_state = self._operator_state()
        if operator_state is None:
            self._set_status("Operator tooling is not enabled for this runtime.")
            return

        try:
            result = operator_state.user_prompt_manager.answer(text)
        except ValueError as exc:
            self._set_status(str(exc))
            return

        answer_text = result["answer"]
        for entry in reversed(self.chat_history):
            if entry.role == "question" and entry.key == result["request_id"]:
                entry.meta = f"Answered: {answer_text}"
                break

        self.chat_history.append(ChatEntry(role="user", content=answer_text))
        self._refresh_transcript()
        self._refresh_footer_bar()
        self._set_status("Answer sent. Continuing...")
        input_widget = self._input_widget()
        input_widget.disabled = True
        self._set_input_hint(self.DEFAULT_INPUT_PLACEHOLDER)

    def _power_tools_enabled(self) -> bool:
        operator_state = self._operator_state()
        if operator_state is not None:
            return operator_state.config.power_tools_enabled
        if self.profile is not None:
            return self.profile.power_tools_enabled
        return False

    def _sampling_summary(self) -> str:
        sampling = self._current_sampling_params()
        ordered = [f"{name}={sampling[name]}" for name in SAMPLING_PARAM_KEYS]
        return "Sampling: " + ", ".join(ordered)

    def _parse_sampling_value(self, name: str, raw: str) -> float | int:
        lower, upper, expected_type = SAMPLING_PARAM_RANGES[name]
        cleaned = raw.strip()
        try:
            if expected_type is int:
                if any(marker in cleaned for marker in (".", "e", "E")):
                    raise ValueError
                value = int(cleaned)
            else:
                value = float(cleaned)
        except ValueError as exc:
            raise ValueError(f"{name} expects a valid {expected_type.__name__} value.") from exc

        if lower is not None and value < lower:
            raise ValueError(f"{name} must be >= {lower}.")
        if upper is not None and value > upper:
            raise ValueError(f"{name} must be <= {upper}.")
        return value

    def _set_sampling_from_command(self, argument: str) -> None:
        parts = argument.split(None, 1)
        if len(parts) != 2:
            self._set_status("Usage: /set <param> <value>")
            return

        raw_name, raw_value = parts
        canonical = SAMPLING_PARAM_ALIASES.get(raw_name.strip().lower())
        if canonical is None:
            self._set_status(
                "Unknown sampling parameter. Use temperature, top_p, max_tokens, top_k, min_p,"
                " presence_penalty, frequency_penalty, or repetition_penalty."
            )
            return

        try:
            value = self._parse_sampling_value(canonical, raw_value)
        except ValueError as exc:
            self._set_status(str(exc))
            return

        agent = self._current_agent()
        if agent is None:
            self._set_status("No active agent to update.")
            return

        agent.set_sampling_params(**{canonical: value})
        self._persist_profile(sampling_params=self._current_sampling_params())
        self._refresh_footer_bar()
        self._add_note(f"Updated sampling: {canonical}={value}")
        self._set_status(f"{canonical} set to {value}")

    def _reset_sampling_params(self) -> None:
        agent = self._current_agent()
        if agent is None:
            self._set_status("No active agent to update.")
            return

        agent.set_sampling_params(**DEFAULT_SAMPLING_PARAMS)
        self._persist_profile(sampling_params={})
        self._refresh_footer_bar()
        self._add_note("Sampling reset to Agent defaults.")
        self._set_status("Sampling parameters reset.")

    def _supported_provider_names(self) -> list[str]:
        return list(SUPPORTED_TUI_PROVIDERS)

    def _canonical_provider_name(self, provider_name: str | None) -> str:
        provider = (provider_name or "").strip().lower()
        return PROVIDER_ALIASES.get(provider, provider)

    def _provider_command_names(self) -> list[str]:
        return self._supported_provider_names() + sorted(PROVIDER_ALIASES.keys())

    def _current_model_name(self) -> str | None:
        agent = self._current_agent()
        if agent and agent.model:
            return agent.model
        if self.profile is not None:
            if self.profile.model is not None:
                return self.profile.model
            if self._canonical_provider_name(self.profile.provider) != self._canonical_provider_name(
                self._live_provider_name()
            ):
                return None
        return getattr(getattr(self.executor.llm_client, "config", None), "model", None)

    def _provider_name(self) -> str:
        if self.profile and self.profile.provider:
            return self.profile.provider
        name = self.executor.llm_client.__class__.__name__.replace("LLM", "").lower()
        return name if name not in {"none", "nonetype", ""} else "not set"

    def _live_provider_name(self) -> str:
        return self.executor.llm_client.__class__.__name__.replace("LLM", "").lower()

    def _base_url(self) -> str | None:
        if self.profile and self.profile.base_url:
            return self.profile.base_url
        return getattr(getattr(self.executor.llm_client, "config", None), "base_url", None)

    def _api_key(self) -> str | None:
        if self.profile and self.profile.api_key:
            return self.profile.api_key
        return getattr(getattr(self.executor.llm_client, "config", None), "api_key", None)

    def _endpoint_host(self) -> str:
        base_url = self._base_url()
        if not base_url:
            return "-"
        parsed = urlparse(base_url)
        return parsed.netloc or base_url

    def _masked_secret(self, value: str | None) -> str | None:
        if not value:
            return None
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}...{value[-4:]}"

    def _format_runtime_error(self, exc: Exception) -> str:
        text = str(exc).strip()
        if not text:
            return exc.__class__.__name__
        lines = [line for line in text.splitlines() if "developer.mozilla.org" not in line]
        text = "\n".join(lines).strip()
        return text or exc.__class__.__name__

    def _normalize_optional_value(self, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def _provider_missing_requirements(
        self,
        *,
        provider: str,
        base_url: str | None,
        api_key: str | None,
    ) -> str | None:
        provider = self._canonical_provider_name(provider)
        if provider == "openai":
            if not api_key and not base_url:
                return "OpenAI needs /endpoint or /apikey before it can connect."
            return None
        if provider == "anthropic":
            if not api_key:
                return "Anthropic needs /apikey before it can connect."
            return None
        if provider == "gemini":
            if not api_key:
                return "Gemini needs /apikey before it can connect."
            return None
        return None

    def _is_live_client_synced(self) -> bool:
        configured_provider = self._canonical_provider_name(self._provider_name())
        live_provider = self._canonical_provider_name(self._live_provider_name())
        if configured_provider != live_provider:
            return False

        config = getattr(self.executor.llm_client, "config", None)
        if config is None:
            return False

        return (
            getattr(config, "model", None) == self._current_model_name()
            and getattr(config, "base_url", None) == self._base_url()
            and getattr(config, "api_key", None) == self._api_key()
        )

    def _rebuild_llm_client(
        self,
        *,
        provider: str | object = _UNSET,
        model: str | None | object = _UNSET,
        base_url: str | None | object = _UNSET,
        api_key: str | None | object = _UNSET,
    ) -> tuple[str, str | None, str | None, str | None]:
        resolved_provider = self._provider_name() if provider is _UNSET else str(provider).strip().lower()
        resolved_model = self._current_model_name() if model is _UNSET else self._normalize_optional_value(model)
        resolved_base_url = self._base_url() if base_url is _UNSET else self._normalize_optional_value(base_url)
        resolved_api_key = self._api_key() if api_key is _UNSET else self._normalize_optional_value(api_key)

        llm_kwargs: dict[str, tp.Any] = {}
        if resolved_model:
            llm_kwargs["model"] = resolved_model
        if resolved_base_url:
            llm_kwargs["base_url"] = resolved_base_url
        if resolved_api_key:
            llm_kwargs["api_key"] = resolved_api_key

        llm = create_llm(resolved_provider, **llm_kwargs)
        self.executor.llm_client = llm

        effective_model = resolved_model or getattr(getattr(llm, "config", None), "model", None)
        agent = self._current_agent()
        if agent is not None:
            agent.model = effective_model
        if getattr(llm, "config", None) is not None and effective_model:
            llm.config.model = effective_model

        return resolved_provider, effective_model, resolved_base_url, resolved_api_key

    def _ensure_live_client_ready(self) -> None:
        provider = self._provider_name()
        model = self._current_model_name()
        base_url = self._base_url()
        api_key = self._api_key()
        missing = self._provider_missing_requirements(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
        )
        if missing is not None:
            raise ValueError(missing)
        if self._is_live_client_synced():
            return
        self._rebuild_llm_client(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    def _schedule_model_refresh(self) -> None:
        self.available_models = []
        self._persist_profile(available_models=[])
        self.run_worker(
            self._refresh_available_models(),
            name="model-discovery",
            group="metadata",
            description="Fetch available models",
        )

    def _set_provider(self, provider_name: str) -> None:
        raw_provider = provider_name.strip().lower()
        clean_provider = self._canonical_provider_name(raw_provider)
        if clean_provider not in self._supported_provider_names():
            message = "Unknown provider. Use openai/oai, anthropic/claude, gemini/google, or ollama/local."
            self._append_event(message, style="bold red")
            self._set_status(message)
            return
        self._persist_profile(provider=clean_provider, model=None, available_models=[])
        agent = self._current_agent()
        if agent is not None:
            agent.model = None
        self._refresh_header_bar()
        if raw_provider != clean_provider:
            self._append_event(
                f"Provider alias [bold]{raw_provider}[/] -> [bold]{clean_provider}[/].",
                style="bold cyan",
            )
        missing = self._provider_missing_requirements(
            provider=clean_provider,
            base_url=self._base_url(),
            api_key=self._api_key(),
        )
        if missing is not None:
            self._append_event(
                f"Provider set to [bold]{clean_provider}[/]. {missing}",
                style="bold yellow",
            )
            self._set_status(f"Provider: {clean_provider}. {missing}")
            return
        try:
            provider, model, _, _ = self._rebuild_llm_client(provider=clean_provider, model=None)
        except Exception as exc:
            self._append_event(f"Provider switch failed: {exc}", style="bold red")
            self._set_status(f"Provider switch failed: {exc}")
            return

        self._persist_profile(provider=provider, model=model, available_models=[])
        self._refresh_header_bar()
        self._append_event(f"Provider set to [bold]{provider}[/].", style="bold cyan")
        self._set_status(f"Provider: {provider}. Refreshing models...")
        self._schedule_model_refresh()

    def _set_endpoint(self, endpoint: str) -> None:
        clean_endpoint = (
            None
            if endpoint.strip().lower() in {"clear", "none", "unset", "off", "-"}
            else self._normalize_optional_value(endpoint)
        )
        if clean_endpoint and "://" not in clean_endpoint:
            self._set_status("Endpoint must include a scheme like http:// or https://")
            return
        provider_name = self._provider_name()
        self._persist_profile(base_url=clean_endpoint, available_models=[])
        self._refresh_header_bar()
        missing = self._provider_missing_requirements(
            provider=provider_name,
            base_url=clean_endpoint,
            api_key=self._api_key(),
        )
        if missing is not None:
            self._append_event(
                f"Endpoint saved as [bold]{clean_endpoint or '(default)'}[/]. {missing}",
                style="bold yellow",
            )
            self._set_status(f"Endpoint saved. {missing}")
            return
        try:
            provider, model, base_url, _ = self._rebuild_llm_client(base_url=clean_endpoint)
        except Exception as exc:
            self._append_event(f"Endpoint update failed: {exc}", style="bold red")
            self._set_status(f"Endpoint update failed: {exc}")
            return

        self._persist_profile(provider=provider, model=model, base_url=base_url, available_models=[])
        self._refresh_header_bar()
        self._append_event(
            f"Endpoint set to [bold]{base_url or '(default)'}[/].",
            style="bold cyan",
        )
        self._set_status("Endpoint updated. Refreshing models...")
        self._schedule_model_refresh()

    def _set_api_key(self, api_key: str) -> None:
        clean_key = None if api_key.strip().lower() in {"clear", "none", "unset", "off", "-"} else api_key.strip()
        provider_name = self._provider_name()
        self._persist_profile(api_key=clean_key)
        self._refresh_header_bar()
        missing = self._provider_missing_requirements(
            provider=provider_name,
            base_url=self._base_url(),
            api_key=clean_key,
        )
        if missing is not None:
            masked = self._masked_secret(clean_key) or "(cleared)"
            self._append_event(
                f"API key updated: [bold]{masked}[/]. {missing}",
                style="bold yellow",
            )
            self._set_status(f"API key saved. {missing}")
            return
        try:
            provider, model, _, resolved_key = self._rebuild_llm_client(api_key=clean_key)
        except Exception as exc:
            self._append_event(f"API key update failed: {exc}", style="bold red")
            self._set_status(f"API key update failed: {exc}")
            return

        self._persist_profile(provider=provider, model=model, api_key=resolved_key)
        self._refresh_header_bar()
        masked = self._masked_secret(resolved_key) or "(cleared)"
        self._append_event(f"API key updated: [bold]{masked}[/].", style="bold cyan")
        self._set_status("API key updated. Refreshing models...")
        self._schedule_model_refresh()

    def _set_active_model(self, model_name: str) -> None:
        clean_model = model_name.strip()
        if not clean_model:
            self._set_status("Model name cannot be empty.")
            return
        agent = self._current_agent()
        if agent is not None:
            agent.model = clean_model
        if getattr(self.executor.llm_client, "config", None) is not None:
            self.executor.llm_client.config.model = clean_model
        self._persist_profile(model=clean_model)
        self._refresh_header_bar()

    def _persist_profile(
        self,
        *,
        provider: str | object = _UNSET,
        model: str | None | object = _UNSET,
        base_url: str | None | object = _UNSET,
        api_key: str | None | object = _UNSET,
        prompt_profile: str | object = _UNSET,
        power_tools_enabled: bool | object = _UNSET,
        available_models: list[str] | None | object = _UNSET,
        sampling_params: dict[str, float | int] | object = _UNSET,
    ) -> None:
        if self.profile is None:
            return
        if provider is not _UNSET:
            self.profile.provider = provider
        if model is not _UNSET:
            self.profile.model = model
        if base_url is not _UNSET:
            self.profile.base_url = base_url
        if api_key is not _UNSET:
            self.profile.api_key = api_key
        if prompt_profile is not _UNSET:
            self.profile.prompt_profile = prompt_profile
        if power_tools_enabled is not _UNSET:
            self.profile.power_tools_enabled = bool(power_tools_enabled)
        if available_models is not _UNSET:
            self.profile.available_models = list(available_models)
        if sampling_params is not _UNSET:
            self.profile.sampling_params = dict(sampling_params)
        if self.config_store is not None:
            self.config_store.upsert_profile(self.profile, make_default=True)


def launch_tui(
    executor: Calute,
    agent: Agent | str | None = None,
    profile: TerminalProfile | None = None,
    config_store: TerminalConfigStore | None = None,
) -> TextualLauncher:
    """Create a deferred launcher for the Textual app."""
    return TextualLauncher(executor=executor, agent=agent, profile=profile, config_store=config_store)


__all__ = ["CaluteTUI", "ChatEntry", "TextualLauncher", "launch_tui", "parse_command", "preview_value"]
