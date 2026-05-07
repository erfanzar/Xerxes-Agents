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
"""Main TUI application orchestrator for Xerxes.

This module defines :class:`XerxesTUI`, which wires together the bridge client,
prompt toolkit UI, and event handlers to provide an interactive terminal
interface for the Xerxes agent system.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import uuid
from typing import Any

from .blocks import (
    ApprovalRequestPanel,
    QuestionRequestPanel,
    _ContentBlock,
    _NotificationBlock,
    _ThinkingBlock,
    _ToolCallBlock,
)
from .engine import BridgeClient
from .prompt import PersistentPrompt


def _git_branch(cwd: str | None = None) -> str:
    """Return the current Git branch name, or empty string on failure.

    Args:
        cwd (str | None): IN: Working directory for the Git command. OUT: Passed
            to ``subprocess.check_output`` as ``cwd``.

    Returns:
        str: OUT: Current branch name, or empty string if Git is unavailable or
            the command fails.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return ""


def _shorten_home(path: str) -> str:
    """Replace the user's home directory with ``~`` in a path.

    Args:
        path (str): IN: Absolute path. OUT: Shortened if it starts with the home
            directory.

    Returns:
        str: OUT: Path with home replaced by ``~``, or the original path.
    """
    home = os.path.expanduser("~")
    if path == home:
        return "~"
    if path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


_XERXES_LOGO = [
    "██╗  ██╗███████╗██████╗ ██╗  ██╗███████╗███████╗",
    "╚██╗██╔╝██╔════╝██╔══██╗╚██╗██╔╝██╔════╝██╔════╝",
    " ╚███╔╝ █████╗  ██████╔╝ ╚███╔╝ █████╗  ███████╗",
    " ██╔██╗ ██╔══╝  ██╔══██╗ ██╔██╗ ██╔══╝  ╚════██║",
    "██╔╝ ██╗███████╗██║  ██║██╔╝ ██╗███████╗███████║",
    "╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝",
]
_LOGO_WIDTH = 48


def _build_welcome_banner(model: str, session_id: str, cwd: str) -> str:
    """Build an ASCII welcome banner with the Xerxes logo and session info.

    Args:
        model (str): IN: Model identifier. OUT: Displayed in the banner.
        session_id (str): IN: Session identifier. OUT: Displayed in the banner.
        cwd (str): IN: Current working directory. OUT: Shortened and displayed.

    Returns:
        str: OUT: Multi-line boxed banner string with ANSI styling.
    """

    blue = "\x1b[34m"
    bold = "\x1b[1m"
    dim = "\x1b[2m"
    reset = "\x1b[0m"

    info = [
        f"{bold}Welcome to Xerxes!{reset}",
        f"{dim}Send /help for help information.{reset}",
        "",
        f"Directory: {_shorten_home(cwd)}",
        f"Session:   {session_id}",
        f"Model:     {model or '(not set — pick one with /provider)'}",
    ]

    ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def visible_len(s: str) -> int:
        """Visible len.

        Args:
            s (str): IN: s. OUT: Consumed during execution.
        Returns:
            int: OUT: Result of the operation."""
        return len(ansi_re.sub("", s))

    gap = "  "
    info_width = max((visible_len(line) for line in info), default=0)

    inner = _LOGO_WIDTH + len(gap) + info_width

    rows: list[str] = []
    for i, logo_row in enumerate(_XERXES_LOGO):
        info_row = info[i] if i < len(info) else ""
        right_pad = max(0, info_width - visible_len(info_row))
        rows.append(f"{blue}{logo_row}{reset}{gap}{info_row}{' ' * right_pad}")

    width = inner + 2
    top = f"╭{'─' * width}╮"
    bot = f"╰{'─' * width}╯"
    mid = [f"│ {row} │" for row in rows]
    return "\n".join([top, *mid, bot])


class XerxesTUI:
    """Interactive terminal UI for the Xerxes agent system.

    Manages the bridge subprocess, prompt toolkit UI, event processing,
    and turn lifecycle.
    """

    def __init__(
        self,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        permission_mode: str = "auto",
        python_executable: str | None = None,
        resume_session_id: str = "",
    ) -> None:
        """Initialize the TUI with configuration parameters.

        Args:
            model (str): IN: Default model identifier. OUT: Stored for initialization.
            base_url (str): IN: Provider base URL. OUT: Stored for initialization.
            api_key (str): IN: API key. OUT: Stored for initialization.
            permission_mode (str): IN: Permission mode such as ``"auto"``. OUT:
                Stored for initialization.
            python_executable (str | None): IN: Python executable for spawning the
                bridge. OUT: Stored for spawning.
            resume_session_id (str): IN: Session ID to resume. OUT: Stored for
                initialization.
        """
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._permission_mode = permission_mode
        self._python_executable = python_executable
        self._resume_session_id = resume_session_id

        self._session_id: str = ""

        self._banner_cwd: str = ""
        self._banner_index: int = -1

        self._client: BridgeClient | None = None
        self._prompt: PersistentPrompt | None = None
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._prompt_task: asyncio.Task[None] | None = None

        self._content_blocks: dict[str, _ContentBlock] = {}
        self._thinking_blocks: dict[str, _ThinkingBlock] = {}
        self._tool_blocks: dict[str, _ToolCallBlock] = {}
        self._notification_history: list[_NotificationBlock] = []
        self._live_notifications: asyncio.Queue[_NotificationBlock] = asyncio.Queue()

        self._active_content: _ContentBlock | None = None
        self._active_thinking: _ThinkingBlock | None = None
        self._active_tool: _ToolCallBlock | None = None

        self._approval_panel: ApprovalRequestPanel | None = None
        self._question_panel: QuestionRequestPanel | None = None
        self._pending_approval_request_id: str | None = None

        self._pending_question_panel: QuestionRequestPanel | None = None
        self._pending_question_request_id: str | None = None

        self._turn_done_event = asyncio.Event()
        self._current_request_id: str | None = None
        self._queued_inputs: list[str] = []
        self._turn_task: asyncio.Task[None] | None = None
        self._plan_mode = False
        self._activity_mode = "code"

    async def run(self) -> XerxesTUI:
        """Start the TUI, bridge client, and event loops.

        Spawns the bridge, initializes the prompt, displays the welcome banner,
        and begins consuming events and user input.

        Returns:
            XerxesTUI: OUT: Self for chaining or context management.
        """
        self._running = True

        self._client = BridgeClient(python_executable=self._python_executable)
        self._client.spawn()

        cwd = os.getcwd()
        provisional_session = uuid.uuid4().hex[:8]
        provisional_branch = _git_branch(cwd)

        self._prompt = PersistentPrompt(
            on_slash=self._handle_slash,
            on_submit=self._handle_submit,
        )
        self._prompt.set_session(
            agent_name="default",
            model=self._model,
            cwd=cwd,
            branch=provisional_branch,
        )

        self._prompt.append_line(
            _build_welcome_banner(
                model=self._model,
                session_id=provisional_session,
                cwd=cwd,
            )
        )
        self._banner_cwd = cwd
        self._banner_index = len(self._prompt._status._content_lines) - 1

        consumer = asyncio.create_task(self._event_consumer())

        await self._client.initialize(
            model=self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            permission_mode=self._permission_mode,
            resume_session_id=self._resume_session_id,
        )

        prompt_task = asyncio.create_task(self._run_prompt())
        self._prompt_task = prompt_task

        self._tasks = [consumer, prompt_task]

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_signal, sig)
            except NotImplementedError:
                pass

        return self

    async def _run_prompt(self) -> None:
        """Run the prompt UI and input processing concurrently.

        Waits for either the prompt application or input processor to finish,
        then cancels the remaining task.
        """
        if self._prompt is None or self._client is None:
            return

        prompt_task = asyncio.create_task(self._prompt.run())
        input_task = asyncio.create_task(self._process_prompt_input())
        try:
            done, pending = await asyncio.wait(
                {prompt_task, input_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                task.result()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except asyncio.CancelledError:
            self._prompt.stop()
            prompt_task.cancel()
            input_task.cancel()
            await asyncio.gather(prompt_task, input_task, return_exceptions=True)
            raise
        finally:
            self._running = False

    async def _process_prompt_input(self) -> None:
        """Consume user input from the prompt queue and dispatch it.

        Handles interrupt commands, question answers, slash commands, and
        normal turn starts.
        """
        if self._prompt is None or self._client is None:
            return

        while self._running:
            raw_input = await self._prompt.input_queue.get()
            if not raw_input:
                continue

            if raw_input == "/interrupt":
                await self._interrupt_current_turn()
                continue

            if raw_input == self._prompt.PLAN_TOGGLE_SENTINEL:
                await self._toggle_plan_mode()
                continue

            if self._approval_panel is not None:
                response = self._resolve_approval_input(raw_input)
                if response is None:
                    self._prompt.refresh_active_approval()
                    continue
                request_id = self._pending_approval_request_id or ""
                self._approval_panel = None
                self._current_request_id = None
                self._pending_approval_request_id = None
                self._prompt.clear_active_approval()
                await self._client.permission_response(
                    request_id=request_id,
                    response=response,
                )
                continue

            if self._pending_question_panel is not None:
                if raw_input == self._prompt.SELECT_SENTINEL:
                    panel = self._pending_question_panel
                    opts = panel.current_options
                    if opts and 0 <= panel._option_index < len(opts):
                        raw_input = opts[panel._option_index]
                    else:
                        continue
                answers = self._resolve_question_input(raw_input)
                if answers is None:
                    self._prompt.refresh_active_question()
                    continue
                request_id = self._pending_question_request_id or ""
                self._pending_question_panel = None
                self._pending_question_request_id = None
                self._prompt.clear_active_question()
                await self._client.question_response(
                    request_id=request_id,
                    answers=answers,
                )
                continue

            if self._prompt.is_running:
                await self._handle_running_input(raw_input)
                continue

            if raw_input.startswith("/"):
                await self._handle_slash(raw_input)
                continue

            self._start_turn(raw_input)

    async def _event_consumer(self) -> None:
        """Consume wire events from the bridge and dispatch them to handlers.

        Runs continuously while the bridge is active.
        """
        if self._client is None:
            return

        try:
            async for event in self._client.events():
                self._handle_event(event)
                if self._prompt and self._prompt.is_running:
                    self._prompt._invalidate()
        except asyncio.CancelledError:
            pass

    def _handle_event(self, event: Any) -> None:
        """Route a single wire event to its specific handler.

        Args:
            event (Any): IN: Wire event object from the bridge. OUT: Inspected
                for type and dispatched to the appropriate handler.
        """
        from ..streaming.wire_events import (
            ApprovalRequest,
            CompactionBegin,
            CompactionEnd,
            InitDone,
            Notification,
            QuestionRequest,
            StatusUpdate,
            StepBegin,
            SubagentEvent,
            TextPart,
            ThinkPart,
            ToolCall,
            ToolCallPart,
            ToolResult,
            TurnBegin,
            TurnEnd,
        )

        getattr(event, "event_type", "") or getattr(event, "type", "")

        if isinstance(event, InitDone):
            self._on_init_done(event)
            return

        if isinstance(event, TextPart):
            self._on_text_chunk(event.text)

        elif isinstance(event, ThinkPart):
            self._on_think_chunk(event.think)

        elif isinstance(event, TurnBegin):
            self._on_turn_begin()
        elif isinstance(event, TurnEnd):
            self._on_turn_end()
        elif isinstance(event, StepBegin):
            self._on_step_begin(event.n)
        elif isinstance(event, ToolCall):
            self._on_tool_call(event.id, event.name, event.arguments)
        elif isinstance(event, ToolCallPart):
            self._on_tool_call_part(event.arguments_part)
        elif isinstance(event, ToolResult):
            self._on_tool_result(event.tool_call_id, event.return_value, event.duration_ms)
        elif isinstance(event, ApprovalRequest):
            self._on_approval_request(event)
        elif isinstance(event, QuestionRequest):
            self._on_question_request(event)
        elif isinstance(event, Notification):
            self._on_notification(event)
        elif isinstance(event, StatusUpdate):
            self._on_status_update(event)
        elif isinstance(event, CompactionBegin):
            self._on_compaction_begin()
        elif isinstance(event, CompactionEnd):
            self._on_compaction_end()
        elif isinstance(event, SubagentEvent):
            self._on_subagent_event(event)
        else:
            if hasattr(event, "event_type"):
                pass

    def _on_turn_begin(self) -> None:
        """Handle the start of a new turn.

        Creates new content and thinking blocks and clears active tool state.
        """
        if self._prompt:
            self._prompt.set_running(True)

        import uuid

        block_id = str(uuid.uuid4())[:8]
        self._active_content = _ContentBlock(block_id=block_id)
        self._active_thinking = _ThinkingBlock(block_id=block_id)
        self._active_tool = None
        self._content_blocks[block_id] = self._active_content
        self._thinking_blocks[block_id] = self._active_thinking
        self._turn_done_event.clear()

    def _on_turn_end(self) -> None:
        """Handle the end of a turn.

        Finalizes blocks, commits streaming content, and signals completion.
        """
        if self._active_content:
            self._active_content.finalize()
        if self._active_thinking:
            self._active_thinking.finalize()
        if self._prompt:
            self._prompt.commit_streaming()
            self._prompt.clear_thinking()
            if self._approval_panel is not None:
                self._prompt.clear_active_approval()
            self._prompt.set_running(False)
        self._approval_panel = None
        self._current_request_id = None
        self._pending_approval_request_id = None
        self._turn_done_event.set()

    def _on_step_begin(self, n: int) -> None:
        """Handle the beginning of a step within a turn.

        Args:
            n (int): IN: Step number. OUT: Currently triggers streaming commit.
        """
        if self._prompt:
            self._prompt.commit_streaming()

    def _on_text_chunk(self, text: str) -> None:
        """Handle an incoming text content chunk.

        Args:
            text (str): IN: Text fragment. OUT: Appended to the active content block.
        """
        if self._active_content:
            self._active_content.append(text)
        if self._prompt:
            self._prompt.set_spinner_label("Generating")
            self._prompt.append_streaming(text)

    def _on_think_chunk(self, think: str) -> None:
        """Handle an incoming thinking trace chunk.

        Args:
            think (str): IN: Thinking fragment. OUT: Appended to the active thinking block.
        """
        if self._active_thinking:
            self._active_thinking.append(think)
        if self._prompt:
            self._prompt.set_spinner_label("Thinking")
            self._prompt.append_thinking(think)

    def _on_tool_call(self, tool_call_id: str, name: str, arguments: str | None) -> None:
        """Handle a tool call event.

        Args:
            tool_call_id (str): IN: Tool call identifier. OUT: Used as block key.
            name (str): IN: Tool name. OUT: Stored in the tool block.
            arguments (str | None): IN: Raw tool arguments JSON. OUT: Stored in the tool block.
        """
        if self._prompt:
            self._prompt.commit_streaming()
        self._set_activity_mode(self._infer_activity_mode(name, arguments))
        block = _ToolCallBlock(
            block_id=tool_call_id,
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
        )
        self._active_tool = block
        self._tool_blocks[tool_call_id] = block
        if self._prompt:
            self._prompt.set_spinner_label(f"Running {name}")
            self._prompt.reset_spinner_timer()
            self._prompt.set_active_tool(tool_call_id, block.compose)

    def _on_tool_call_part(self, arguments_part: str) -> None:
        """Handle a fragment of tool call arguments.

        Args:
            arguments_part (str): IN: Partial JSON arguments. OUT: Appended to the
                active tool block.
        """
        if self._active_tool:
            self._active_tool.append_args_part(arguments_part)

    def _on_tool_result(self, tool_call_id: str, return_value: str, duration_ms: float = 0.0) -> None:
        """Handle a tool result event.

        Args:
            tool_call_id (str): IN: Tool call identifier. OUT: Used to look up the
                matching tool block.
            return_value (str): IN: Tool result string. OUT: Stored in the tool block.
            duration_ms (float): IN: Tool execution duration. OUT: Displayed in
                the completed tool block.
        """
        block = self._tool_blocks.get(tool_call_id)
        if block:
            display_value = "" if block.name in {"AgentTool", "TaskCreateTool", "SpawnAgents"} else return_value
            block.set_result(display_value, duration_ms=duration_ms)
            if self._prompt:
                self._prompt.commit_active_tool(tool_call_id, block.compose())
                self._prompt.set_spinner_label("Thinking")

    def _on_subagent_event(self, event: Any) -> None:
        """Handle a nested subagent event associated with a parent tool call.

        Args:
            event (Any): IN: Subagent event wrapping a tool call or result. OUT:
                Dispatched to the matching parent tool block.
        """
        from ..streaming.wire_events import SubagentEvent

        if not isinstance(event, SubagentEvent):
            return

        parent_id = event.parent_tool_call_id
        if not parent_id or parent_id not in self._tool_blocks:
            return

        parent = self._tool_blocks[parent_id]
        sub_event = event.event

        from ..streaming.wire_events import ToolCall, ToolResult

        if event.subagent_type:
            self._set_activity_mode(event.subagent_type)

        if isinstance(sub_event, ToolCall):
            key_arg = getattr(sub_event, "arguments", "") or ""
            parent.append_sub_tool_call(
                tool_call_id=getattr(sub_event, "id", "") or "",
                name=getattr(sub_event, "name", ""),
                key_arg=str(key_arg)[:80],
            )
        elif isinstance(sub_event, ToolResult):
            parent.finish_sub_tool_call(
                tool_call_id=sub_event.tool_call_id,
                result=sub_event.return_value,
                duration_ms=0.0,
            )

    def _on_approval_request(self, event: Any) -> None:
        """Handle an approval request event by displaying a panel and awaiting input.

        Args:
            event (Any): IN: Approval request event with id, tool_call_id, action,
                and description. OUT: Used to construct the approval panel.
        """

        panel = ApprovalRequestPanel(
            request_id=event.id,
            tool_call_id=event.tool_call_id,
            action=event.action,
            description=event.description,
        )
        self._approval_panel = panel
        self._current_request_id = event.id
        self._pending_approval_request_id = event.id
        if self._prompt:
            self._prompt.set_active_approval(panel)
        return

    def _resolve_approval_input(self, text: str) -> str | None:
        """Resolve prompt input against the pending approval panel.

        Args:
            text (str): IN: Raw user input or approval sentinel. OUT: Converted
                to a bridge approval response.

        Returns:
            str | None: OUT: ``"approve"``, ``"approve_for_session"``,
                ``"reject"``, or ``None`` if the panel should keep waiting.
        """
        panel = self._approval_panel
        if panel is None or self._prompt is None:
            return None

        stripped = text.strip().lower()
        if text == self._prompt.APPROVAL_SENTINEL or stripped in {"", "y", "yes", "approve"}:
            return panel.selected_response
        if stripped in {"a", "all", "approve_all", "approve-all", "session"}:
            return "approve_for_session"
        if stripped in {"r", "reject", "n", "no", "/cancel", "/abort"}:
            return "reject"
        if stripped in {"up", "k"}:
            panel.move_cursor_up()
            return None
        if stripped in {"down", "j"}:
            panel.move_cursor_down()
            return None
        self._prompt.append_line("\x1b[31mApproval response must be Enter, A, or R.\x1b[0m")
        return None

    def _wait_for_approval_response(self, panel: ApprovalRequestPanel) -> str:
        """Block synchronously for an approval response from the user.

        Args:
            panel (ApprovalRequestPanel): IN: Panel being displayed. OUT: Ignored
                except for context; the response is read from stdin.

        Returns:
            str: OUT: One of ``"approve"``, ``"approve_for_session"``, or ``"reject"``.
        """
        print("\nOptions: ENTER=approve, A=approve_all, R=reject")
        try:
            line = input("> ").strip().lower()
            if line == "a":
                return "approve_for_session"
            elif line == "r":
                return "reject"
            return "approve"
        except (EOFError, KeyboardInterrupt):
            return "reject"

    def _on_question_request(self, event: Any) -> None:
        """Handle a question request event by constructing a question panel.

        Args:
            event (Any): IN: Question request event with id and questions list.
                OUT: Used to build and display a :class:`QuestionRequestPanel`.
        """

        def _opt_label(o: Any) -> str:
            """Internal helper to opt label.

            Args:
                o (Any): IN: o. OUT: Consumed during execution.
            Returns:
                str: OUT: Result of the operation."""
            if isinstance(o, str):
                return o
            if isinstance(o, dict):
                return str(o.get("label") or o.get("name") or o.get("value") or "")
            return getattr(o, "label", None) or str(o)

        def _qfield(q: Any, name: str, default: Any = "") -> Any:
            """Internal helper to qfield.

            Args:
                q (Any): IN: q. OUT: Consumed during execution.
                name (str): IN: name. OUT: Consumed during execution.
                default (Any, optional): IN: default. Defaults to ''. OUT: Consumed during execution.
            Returns:
                Any: OUT: Result of the operation."""
            if isinstance(q, dict):
                return q.get(name, default)
            return getattr(q, name, default)

        question_dicts = [
            {
                "id": _qfield(q, "id", "") or "",
                "question": _qfield(q, "question", "") or "",
                "options": [_opt_label(o) for o in (_qfield(q, "options", []) or [])],
                "allow_free_form": bool(_qfield(q, "allow_free_form", False)),
            }
            for q in event.questions
        ]
        panel = QuestionRequestPanel(request_id=event.id, questions=question_dicts)
        self._pending_question_panel = panel
        self._pending_question_request_id = event.id
        if self._prompt:
            self._prompt.set_active_question(panel)

    def _resolve_question_input(self, text: str) -> dict[str, str] | None:
        """Resolve user input against the pending question panel.

        Args:
            text (str): IN: Raw user input. OUT: Parsed as an option index or free text.

        Returns:
            dict[str, str] | None: OUT: Answers dictionary if complete, ``None`` if
                more input is needed, or empty dict on cancel.
        """
        panel = self._pending_question_panel
        if panel is None:
            return None
        if text.lower() in {"/cancel", "/abort"}:
            return {}

        opts = panel.current_options
        stripped = text.strip()
        if stripped.isdigit() and opts:
            idx = int(stripped) - 1
            if 0 <= idx < len(opts):
                panel._option_index = idx
                panel.confirm_current()
            else:
                if self._prompt:
                    self._prompt.append_line(f"\x1b[31m(option {stripped} out of range; pick 1-{len(opts)})\x1b[0m")
                return None
        else:
            panel._free_text_mode = True
            panel._free_text = stripped
            panel.confirm_current()

        if panel.is_complete:
            return panel._answers

        if self._prompt:
            self._prompt.append_line(panel.compose())
        return None

    def _wait_for_question_answers(self, panel: QuestionRequestPanel) -> dict[str, str]:
        """Block synchronously until all questions in the panel are answered.

        Args:
            panel (QuestionRequestPanel): IN: Panel with questions. OUT: Modified
                as the user answers each question.

        Returns:
            dict[str, str]: OUT: Mapping from question ID to answer string.
        """

        while not panel.is_complete:
            from .console import print_markdown

            print_markdown(panel.compose())
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if line.isdigit():
                idx = int(line) - 1
                if 0 <= idx < len(panel.current_options):
                    panel._option_index = idx
                    panel.confirm_current()
            elif line == "":
                panel.confirm_current()

        return panel._answers

    def _on_notification(self, event: Any) -> None:
        """Handle a notification event.

        Notifications with ``category == "subagent_stream"`` are treated as
        transient live updates: the body replaces a per-task preview line that
        sits above the input bar (next to the spinner) and is cleared when the
        sub-agent finishes. Empty bodies signal "clear this preview". All other
        notifications are appended to the prompt history as before.

        Args:
            event (Any): IN: Notification event with id, category, severity, title,
                body, and payload. OUT: Used to build and display a notification block.
        """
        if event.category == "subagent_stream":
            payload = getattr(event, "payload", {}) or {}
            task_id = str(payload.get("task_id") or event.id)
            label = str(payload.get("label") or "")
            body = (event.body or "").strip()
            if self._prompt is None:
                return
            if not body:
                self._prompt.clear_subagent_preview(task_id)
            else:
                self._prompt.set_subagent_preview(task_id, label, body)
            return

        block = _NotificationBlock(
            notification_id=event.id,
            category=event.category,
            severity=event.severity,
            title=event.title,
            body=event.body,
        )
        self._notification_history.append(block)
        if self._prompt:
            self._prompt.append_line(block.compose())

    def _on_init_done(self, event: Any) -> None:
        """Handle the initialization-done event from the bridge.

        Updates session metadata, context, skills, and the welcome banner.

        Args:
            event (Any): IN: InitDone event with session_id, model, cwd, branch,
                context_limit, and skills. OUT: Used to update prompt state.
        """
        if self._prompt is None:
            return
        self._session_id = event.session_id or ""
        self._prompt.set_session(
            agent_name=event.agent_name,
            model=event.model,
            cwd=event.cwd,
            branch=event.git_branch,
        )
        if event.context_limit:
            self._prompt.set_context(0, event.context_limit)

        if getattr(event, "skills", None):
            self._prompt.set_skills(list(event.skills))

        if self._banner_index >= 0 and event.session_id:
            cwd = event.cwd or self._banner_cwd
            new_banner = _build_welcome_banner(
                model=event.model or self._model,
                session_id=event.session_id,
                cwd=cwd,
            )
            history = self._prompt._status._content_lines
            if 0 <= self._banner_index < len(history):
                history[self._banner_index] = new_banner
                self._prompt._invalidate()

    def _on_status_update(self, event: Any) -> None:
        """Handle a context status update event.

        Args:
            event (Any): IN: StatusUpdate event with context_tokens and max_context.
                OUT: Used to update the footer context display.
        """
        if self._prompt is None:
            return
        self._prompt.set_context(event.context_tokens, event.max_context)
        self._plan_mode = bool(getattr(event, "plan_mode", self._plan_mode))
        self._sync_prompt_mode()

    def _on_compaction_begin(self) -> None:
        """Handle the start of a context compaction event."""
        if self._prompt:
            self._prompt.append_line("[dim]Compacting context...[/dim]")

    def _on_compaction_end(self) -> None:
        """Handle the end of a context compaction event."""
        if self._prompt:
            self._prompt.append_line("[dim]Context compaction done.[/dim]")

    async def _handle_submit(self, text: str) -> None:
        """Handle a normal text submission from the prompt.

        Args:
            text (str): IN: Submitted user text. OUT: Starts a new turn.
        """
        self._start_turn(text)

    def _start_turn(self, text: str) -> None:
        """Queue or start a turn with the given user input.

        Args:
            text (str): IN: User input text. OUT: Displayed and either queued or
                used to start a new turn task.
        """
        if self._prompt:
            self._prompt.append_line(f"✨ {text}")

        if self._turn_task is not None and not self._turn_task.done():
            self._queued_inputs.append(text)
            if self._prompt:
                self._prompt.set_queue_count(len(self._queued_inputs))
            return
        self._turn_task = asyncio.create_task(self._run_turns(text))
        self._tasks.append(self._turn_task)

    async def _run_turns(self, text: str) -> None:
        """Run sequential turns, processing queued inputs after each.

        Args:
            text (str): IN: Initial user input. OUT: Starts the first turn; subsequent
                inputs are drained from the queue.
        """
        if self._client is None:
            return

        current: str | None = text
        while current is not None:
            if self._prompt:
                self._prompt.set_running(True)
                self._prompt.set_queue_count(len(self._queued_inputs))

            turn_plan_mode = self._plan_mode
            if not turn_plan_mode:
                self._set_activity_mode("code")
            self._turn_done_event.clear()
            await self._client.query(current, plan_mode=turn_plan_mode)

            try:
                await asyncio.wait_for(self._turn_done_event.wait(), timeout=900)
            except TimeoutError:
                await self._client.cancel()
                self._approval_panel = None
                self._current_request_id = None
                self._pending_approval_request_id = None
                if self._prompt:
                    self._prompt.clear_active_approval()
                    self._prompt.set_running(False)
                    self._prompt.append_line("\x1b[31mTurn timed out after 900s.\x1b[0m")

            if turn_plan_mode and self._plan_mode:
                await self._set_plan_mode(False, notify=True)

            if self._queued_inputs:
                current = self._queued_inputs.pop(0)
                if self._prompt:
                    self._prompt.set_queue_count(len(self._queued_inputs))
            else:
                current = None

        if self._prompt:
            self._prompt.set_queue_count(0)

    async def _handle_running_input(self, text: str) -> None:
        """Handle user input while a turn is actively running.

        Args:
            text (str): IN: User input text. OUT: Interpreted as an interrupt,
                steer command, or queued input.
        """
        if self._client is None:
            return

        cmd = text.split()[0].lower() if text.startswith("/") else ""
        if cmd == "/interrupt":
            await self._interrupt_current_turn()
        elif cmd == "/btw":
            content = text[len("/btw") :].strip()
            if content:
                await self._client.steer(content)
        elif cmd == "/steer":
            content = text[len("/steer") :].strip()
            if content:
                await self._client.steer(content)
        elif text.startswith("/"):
            if self._prompt:
                self._prompt.append_line(f"\x1b[2m{text} is not available during streaming\x1b[0m")
        else:
            self._queued_inputs.append(text)
            if self._prompt:
                self._prompt.append_line(f"✨ {text}")
                self._prompt.set_queue_count(len(self._queued_inputs))

    async def _handle_slash(self, text: str) -> None:
        """Handle a slash command from the user.

        Args:
            text (str): IN: Full slash command string. OUT: Parsed and dispatched
                to the bridge or handled locally.
        """
        if self._client is None:
            return

        if self._prompt:
            self._prompt.append_line(f"\x1b[2m› {text}\x1b[0m")  # noqa: RUF001

        cmd = text.split()[0].lower() if text else ""

        if cmd == "/cancel":
            await self._interrupt_current_turn()
        elif cmd == "/cancel-all":
            await self._interrupt_current_turn(cancel_all=True)
        elif cmd == "/btw":
            content = text[len("/btw") :].strip()
            await self._client.steer(content)
        elif cmd == "/plan":
            content = text[len("/plan") :].strip()
            await self._set_plan_mode(not self._plan_mode if not content else True)
            if content:
                await self._client.steer(f"/plan {content}")
        else:
            await self._client._send_jsonrpc(
                method="slash",
                params={"command": text},
            )

    async def _toggle_plan_mode(self) -> None:
        """Toggle plan mode for subsequent turns."""
        await self._set_plan_mode(not self._plan_mode)

    async def _set_plan_mode(self, enabled: bool, *, notify: bool = True) -> None:
        """Set plan mode and synchronize the bridge runtime config.

        Args:
            enabled (bool): IN: Desired plan mode state. OUT: Reflected in the
                prompt UI and sent to the bridge.
            notify (bool): IN: Whether to append a visible mode-change line.
                OUT: Controls prompt history noise.
        """
        self._plan_mode = enabled
        if self._prompt:
            self._sync_prompt_mode()
            if notify:
                self._prompt.append_line(
                    "\x1b[36mPlan mode ON\x1b[0m" if enabled else "\x1b[2mCode mode ON\x1b[0m"
                )
        if self._client:
            await self._client._send_jsonrpc(
                method="set_plan_mode",
                params={"enabled": self._plan_mode},
            )

    def _set_activity_mode(self, mode: str) -> None:
        """Update inferred non-plan activity mode.

        Args:
            mode (str): IN: Inferred mode label. OUT: Displayed in the footer
                when plan mode is inactive.
        """
        normalized = (mode or "code").strip().lower()
        aliases = {
            "research": "researcher",
            "coding": "code",
            "agent": "code",
            "general-purpose": "code",
            "plan": "planner",
        }
        self._activity_mode = aliases.get(normalized, normalized)
        self._sync_prompt_mode()

    def _sync_prompt_mode(self) -> None:
        """Synchronize plan/activity mode state into the prompt renderers."""
        if not self._prompt:
            return
        self._prompt.set_plan_mode(self._plan_mode)
        self._prompt.set_activity_mode("plan" if self._plan_mode else self._activity_mode)

    @staticmethod
    def _infer_activity_mode(tool_name: str, arguments: str | None = None) -> str:
        """Infer a user-visible mode from a tool or sub-agent call."""
        if tool_name in {"AgentTool", "TaskCreateTool"}:
            try:
                args = json.loads(arguments or "{}")
            except Exception:
                args = {}
            subagent_type = str(args.get("subagent_type") or args.get("agent_type") or "").strip()
            return subagent_type or "agents"

        if tool_name == "SpawnAgents":
            try:
                args = json.loads(arguments or "{}")
            except Exception:
                args = {}
            agents = args.get("agents") if isinstance(args, dict) else None
            types = {
                str(agent.get("subagent_type") or agent.get("agent_type") or "").strip()
                for agent in agents or []
                if isinstance(agent, dict)
            }
            types.discard("")
            return next(iter(types)) if len(types) == 1 else "agents"

        research_tools = {
            "ReadFile",
            "GlobTool",
            "GrepTool",
            "ListDir",
            "DuckDuckGoSearch",
            "URLAnalyzer",
            "APIClient",
            "RSSReader",
        }
        code_tools = {
            "WriteFile",
            "AppendFile",
            "FileEditTool",
            "ExecuteShell",
            "ExecutePythonCode",
        }
        if tool_name in research_tools:
            return "researcher"
        if tool_name in code_tools:
            return "code"
        return "code"

    def _handle_signal(self, sig: int) -> None:
        """Handle OS signals by shutting down the running turn.

        Args:
            sig (int): IN: Signal number (e.g., ``signal.SIGINT``). OUT: Triggers
                cancellation of the current turn.
        """
        if self._running:
            self._pending_cancel_task = asyncio.create_task(self._interrupt_current_turn())

    async def _interrupt_current_turn(self, *, cancel_all: bool = True) -> None:
        """Cancel the active turn and clear local running UI immediately."""
        if self._client:
            if cancel_all:
                await self._client.cancel_all()
            else:
                await self._client.cancel()

        self._approval_panel = None
        self._current_request_id = None
        self._pending_approval_request_id = None
        self._pending_question_panel = None
        self._pending_question_request_id = None
        self._active_tool = None

        if self._prompt:
            self._prompt.clear_active_approval()
            self._prompt.clear_active_question()
            self._prompt.clear_active_tools()
            self._prompt.clear_subagent_previews()
            self._prompt.clear_thinking()
            self._prompt.set_running(False)
            self._prompt.append_line("\x1b[2mInterrupted.\x1b[0m")

        self._turn_done_event.set()
        await self._restart_bridge_after_interrupt()

    async def _restart_bridge_after_interrupt(self) -> None:
        """Restart the bridge subprocess so blocked provider streams are aborted."""
        old_client = self._client
        if old_client is not None:
            await asyncio.to_thread(old_client.close)

        self._client = BridgeClient(python_executable=self._python_executable)
        self._client.spawn()
        consumer = asyncio.create_task(self._event_consumer())
        self._tasks.append(consumer)
        await self._client.initialize(
            model=self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            permission_mode=self._permission_mode,
            resume_session_id=self._resume_session_id,
        )

    async def wait_until_done(self) -> None:
        """Wait for all background tasks to complete.

        Cancels any remaining tasks after the prompt task finishes.
        """
        if self._prompt_task is None:
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                pass
            return

        try:
            await self._prompt_task
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            others = [t for t in self._tasks if t is not self._prompt_task and not t.done()]
            for task in others:
                task.cancel()
            await asyncio.gather(*others, return_exceptions=True)

    async def __aenter__(self) -> XerxesTUI:
        """Enter the async runtime context, starting the TUI.

        Returns:
            XerxesTUI: OUT: Self after :meth:`run` completes.
        """
        return await self.run()

    async def __aexit__(self, *args: Any) -> None:
        """Exit the async runtime context, shutting down cleanly.

        Cancels tasks, shuts down the bridge, and prints a resume hint if a
        session ID was assigned.
        """
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._client:
            try:
                await self._client.shutdown()
            except Exception:
                pass
            self._client.close()

        if self._session_id:
            sys.stdout.write(f"\nTo resume this session: \x1b[1mxerxes -r {self._session_id}\x1b[0m\n")
            sys.stdout.flush()
