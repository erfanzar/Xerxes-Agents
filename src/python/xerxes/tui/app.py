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
"""Top-level TUI application orchestrator for Xerxes.

:class:`XerxesTUI` is the long-running asyncio object that owns the
:class:`~xerxes.tui.engine.BridgeClient`, the :class:`PersistentPrompt`,
the turn task, and the block/notification state. It maps inbound
:mod:`xerxes.streaming.wire_events` onto prompt mutations and routes
outbound user input back to the daemon."""

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
    """Return the active git branch under ``cwd`` (empty on any failure)."""
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
    """Collapse ``$HOME`` prefix in ``path`` to ``~`` for compact display."""
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
    """Return a multi-line ANSI banner with the Xerxes logo + session metadata.

    Mutated in place once the daemon reports the real session id via the
    ``InitDone`` event — :class:`XerxesTUI` stores ``_banner_index`` to
    rewrite the slot rather than appending a second banner."""

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
        """Return the printable-character length of ``s`` (ANSI escapes stripped)."""
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
    """Asyncio-driven full-screen TUI for the Xerxes agent system.

    Owns the :class:`BridgeClient`, the :class:`PersistentPrompt`, the
    turn-runner task, and the in-memory block state (content,
    thinking, tool calls, notifications, approval / question panels).
    Use as an async context manager: ``async with XerxesTUI(...) as tui``.
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
        """Stash configuration; no daemon contact happens until :meth:`run`.

        Args:
            model: Default model id forwarded to ``initialize``.
            base_url: Provider base URL forwarded to ``initialize``.
            api_key: Provider API key forwarded to ``initialize``.
            permission_mode: Initial permission mode (``"auto"`` /
                ``"manual"`` / ``"accept-all"``).
            python_executable: Interpreter the bridge uses when it needs
                to launch the daemon.
            resume_session_id: When non-empty, the daemon rehydrates
                this session instead of starting a fresh one.
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
        self._user_activity_mode: str | None = None

    async def run(self) -> XerxesTUI:
        """Spawn the bridge, build the prompt, paint the banner, and start loops.

        Concurrently kicks off the event consumer, the prompt UI, and
        the input processor; installs SIGINT/SIGTERM handlers that map
        to a turn cancellation. Returns ``self`` so callers can write
        ``await tui.run()`` and continue chaining."""
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
        """Run the prompt application and input pump until either finishes."""
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
        """Pump prompt input forever and route each submission appropriately.

        Routing precedence: interrupt sentinel → plan-toggle sentinel →
        pending approval panel → pending question panel → in-flight turn
        steer/cancel/btw → slash command → new turn start."""
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
                await self._cycle_interaction_mode()
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
        """Forever loop pulling :class:`WireEvent` instances into :meth:`_handle_event`."""
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
        """Dispatch ``event`` to the matching ``_on_*`` handler based on its class."""
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
        """Allocate fresh content/thinking blocks and flip the prompt to running."""
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
        """Finalize blocks, commit streamed text, dismiss panels, signal done."""
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
        """Commit accumulated streaming text on each new step (``n`` is informational)."""
        if self._prompt:
            self._prompt.commit_streaming()

    def _on_text_chunk(self, text: str) -> None:
        """Append a streaming text fragment to the active content block + prompt."""
        if self._active_content:
            self._active_content.append(text)
        if self._prompt:
            self._prompt.set_spinner_label("Generating")
            self._prompt.append_streaming(text)

    def _on_think_chunk(self, think: str) -> None:
        """Append a streaming reasoning fragment to the thinking pane."""
        if self._active_thinking:
            self._active_thinking.append(think)
        if self._prompt:
            self._prompt.set_spinner_label("Thinking")
            self._prompt.append_thinking(think)

    def _on_tool_call(self, tool_call_id: str, name: str, arguments: str | None) -> None:
        """Open a new :class:`_ToolCallBlock` and wire it into the live status area."""
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
        """Append a streamed JSON fragment to the active tool call block."""
        if self._active_tool:
            self._active_tool.append_args_part(arguments_part)

    def _on_tool_result(self, tool_call_id: str, return_value: str, duration_ms: float = 0.0) -> None:
        """Close the matching tool block, render its result, and commit it to history.

        Sub-agent / spawn tools intentionally hide their raw return value;
        the visible block just shows status + duration."""
        block = self._tool_blocks.get(tool_call_id)
        if block:
            display_value = "" if block.name in {"AgentTool", "TaskCreateTool", "SpawnAgents"} else return_value
            block.set_result(display_value, duration_ms=duration_ms)
            if self._prompt:
                self._prompt.commit_active_tool(tool_call_id, block.compose())
                self._prompt.set_spinner_label("Thinking")

    def _on_subagent_event(self, event: Any) -> None:
        """Attach a nested subagent ToolCall/ToolResult onto its parent block."""
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
        """Show an :class:`ApprovalRequestPanel` for ``event`` and stash request ids."""

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
        """Map a raw input line onto an approval verdict.

        Returns ``None`` when the input only moves the cursor (and the
        panel should keep waiting). Returns one of ``"approve"``,
        ``"approve_for_session"``, or ``"reject"`` once decided."""
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
        """Synchronous stdin fallback used when the prompt loop is unavailable.

        Currently retained for headless test paths. Blocks until the
        user types ``a``, ``r``, or hits Enter."""
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
        """Build and surface a :class:`QuestionRequestPanel` from ``event``."""

        def _opt_label(o: Any) -> str:
            """Return the best display label for an option, regardless of source shape."""
            if isinstance(o, str):
                return o
            if isinstance(o, dict):
                return str(o.get("label") or o.get("name") or o.get("value") or "")
            return getattr(o, "label", None) or str(o)

        def _qfield(q: Any, name: str, default: Any = "") -> Any:
            """Read ``q.name`` whether ``q`` is a dict-like or attribute-like object."""
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
        """Advance the question panel with ``text`` and return collected answers when done.

        Numeric input picks an option by 1-based index; any other text
        is captured as a free-form answer. ``/cancel`` / ``/abort``
        return an empty dict to short-circuit the conversation."""
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
        """Synchronous stdin fallback for question panels used in headless mode."""

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
        """Render an inbound notification, choosing transient vs persistent display.

        ``category == "subagent_stream"`` is treated as a transient live
        update: the body replaces a per-task preview line beside the
        spinner (and an empty body clears it). All other notifications
        are committed to the prompt history as :class:`_NotificationBlock`s."""
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
        """Apply daemon-confirmed session metadata to the prompt and welcome banner.

        Rewrites the placeholder banner in place once the real
        ``session_id`` is known so the user sees one definitive banner."""
        if self._prompt is None:
            return
        if event.session_id or event.model or event.cwd or event.agent_name:
            self._session_id = event.session_id or self._session_id
            self._prompt.set_session(
                agent_name=event.agent_name or "default",
                model=event.model or self._model,
                cwd=event.cwd or self._banner_cwd,
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
        """Apply a daemon status push (context tokens, plan/research mode) to the footer."""
        if self._prompt is None:
            return
        self._prompt.set_context(event.context_tokens, event.max_context)
        self._plan_mode = bool(getattr(event, "plan_mode", self._plan_mode))
        mode = str(getattr(event, "mode", "") or "")
        if mode and mode != "code" and not self._plan_mode:
            self._activity_mode = "researcher" if mode == "research" else mode
        self._sync_prompt_mode()

    def _on_compaction_begin(self) -> None:
        """Announce the start of a context compaction pass in the prompt history."""
        if self._prompt:
            self._prompt.append_line("[dim]Compacting context...[/dim]")

    def _on_compaction_end(self) -> None:
        """Announce the end of a context compaction pass."""
        if self._prompt:
            self._prompt.append_line("[dim]Context compaction done.[/dim]")

    async def _handle_submit(self, text: str) -> None:
        """prompt_toolkit submit callback: forward ``text`` into :meth:`_start_turn`."""
        self._start_turn(text)

    def _start_turn(self, text: str) -> None:
        """Display ``text`` and either queue it or kick off a new turn task."""
        if self._prompt:
            self._prompt.append_line(f"✨ {text}")

        if self._turn_task is not None and not self._turn_task.done():
            self._queued_inputs.append(text)
            if self._prompt:
                self._prompt.set_queue_count(len(self._queued_inputs))
            return
        self._turn_task = asyncio.create_task(self._run_turns(text))
        self._turn_task.add_done_callback(self._on_turn_task_done)
        self._tasks.append(self._turn_task)

    async def _run_turns(self, text: str) -> None:
        """Drive turns serially, draining the queued inputs FIFO after each one.

        Each turn waits up to 900 s on ``_turn_done_event``; a timeout
        triggers a cancel + visible error so the TUI never wedges."""
        if self._client is None:
            return

        current: str | None = text
        while current is not None:
            try:
                if self._prompt:
                    self._prompt.set_running(True)
                    self._prompt.set_queue_count(len(self._queued_inputs))

                turn_plan_mode = self._plan_mode
                if not turn_plan_mode and self._user_activity_mode is None:
                    self._set_activity_mode("code")
                self._turn_done_event.clear()
                await self._client.query(
                    current,
                    plan_mode=turn_plan_mode,
                    mode=self._current_interaction_mode(),
                )
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
            except Exception as exc:
                self._approval_panel = None
                self._current_request_id = None
                self._pending_approval_request_id = None
                if self._prompt:
                    self._prompt.clear_active_approval()
                    self._prompt.set_running(False)
                    self._prompt.append_line(f"\x1b[31mTurn failed: {exc}\x1b[0m")
                break

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

    def _on_turn_task_done(self, task: asyncio.Task[Any]) -> None:
        """Done-callback: surface unexpected turn-task failures in the prompt history."""
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if self._prompt:
                self._prompt.set_running(False)
                self._prompt.append_line(f"\x1b[31mTurn task failed: {exc}\x1b[0m")

    async def _handle_running_input(self, text: str) -> None:
        """Route input that arrived while a turn is in flight (interrupt/steer/queue)."""
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
        """Dispatch a slash command, either locally or via the daemon ``slash`` RPC."""
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
        """Flip ``_plan_mode`` and inform the daemon."""
        await self._set_plan_mode(not self._plan_mode)

    def _current_interaction_mode(self) -> str:
        """Return ``"plan"`` when plan mode is on; otherwise the activity mode."""
        if self._plan_mode:
            return "plan"
        return self._activity_mode or "code"

    async def _cycle_interaction_mode(self) -> None:
        """Shift+Tab cycle: code → plan → researcher → code."""
        if self._plan_mode:
            await self._set_plan_mode(False, notify=False, mode="researcher")
            self._set_activity_mode("researcher", user_selected=True)
            return

        if self._activity_mode == "researcher":
            self._set_activity_mode("code", user_selected=True)
            return

        await self._set_plan_mode(True, notify=False)

    async def _set_plan_mode(self, enabled: bool, *, notify: bool = True, mode: str | None = None) -> None:
        """Persist plan-mode state and sync it to the daemon.

        Args:
            enabled: Desired plan-mode flag; clears any sticky user
                activity mode when turning on.
            notify: Kept for API compatibility. Mode changes are surfaced
                in the footer instead of appended to prompt history.
            mode: Optional backend mode override sent with the RPC
                (defaults to :meth:`_current_interaction_mode`).
        """
        _ = notify
        self._plan_mode = enabled
        if enabled:
            self._user_activity_mode = None
        if self._prompt:
            self._sync_prompt_mode()
        if self._client:
            await self._client._send_jsonrpc(
                method="set_plan_mode",
                params={"enabled": self._plan_mode, "mode": mode or self._current_interaction_mode()},
            )

    def _set_activity_mode(self, mode: str, *, user_selected: bool = False) -> None:
        """Update the inferred non-plan activity mode shown in the footer.

        Args:
            mode: Inferred mode label (``"code"`` / ``"researcher"`` /
                an agent type for subagent tools). Aliases such as
                ``"research"`` → ``"researcher"`` are normalized here.
            user_selected: When ``True``, sticks across turns; otherwise
                a new turn may reset back to ``"code"``.
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
        if user_selected:
            self._user_activity_mode = None if self._activity_mode == "code" else self._activity_mode
        self._sync_prompt_mode()

    def _sync_prompt_mode(self) -> None:
        """Push plan / activity state down to the status and footer renderers."""
        if not self._prompt:
            return
        self._prompt.set_plan_mode(self._plan_mode)
        self._prompt.set_activity_mode("plan" if self._plan_mode else self._activity_mode)

    @staticmethod
    def _infer_activity_mode(tool_name: str, arguments: str | None = None) -> str:
        """Map a tool name + arguments to a user-visible activity label.

        Subagent-launch tools surface the requested subagent type;
        read-only tools (``ReadFile``, ``GrepTool``, ...) map to
        ``"researcher"``; write/exec tools map to ``"code"``."""
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
        """OS signal handler: schedule an interrupt of the active turn."""
        if self._running:
            self._pending_cancel_task = asyncio.create_task(self._interrupt_current_turn())

    async def _interrupt_current_turn(self, *, cancel_all: bool = True) -> None:
        """Cancel the active turn, clear panel state, and respawn the bridge.

        Restart-after-interrupt is necessary because long provider
        streams can wedge inside the daemon; tearing the connection
        forces them to abort cleanly."""
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
        """Replace ``_client`` with a fresh :class:`BridgeClient` and re-initialize it."""
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
        """Block until the prompt task exits, then cancel the rest of ``_tasks``."""
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
        """Async context entry: delegate to :meth:`run`."""
        return await self.run()

    async def __aexit__(self, *args: Any) -> None:
        """Async context exit: cancel tasks, shut down the bridge, print resume hint."""
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
