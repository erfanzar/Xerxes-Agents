"""Helper utilities for processing chat messages and managing UI state in Calute.

This module provides core functionality for handling message streaming, tool execution,
thinking panels, and UI updates in the Gradio-based chat interface. It manages the
complex state transitions during message processing and formats responses for display.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from gradio import ChatMessage

from calute.calute import Calute
from calute.cortex import Cortex, CortexAgent, CortexTask
from calute.cortex.dynamic import DynamicCortex
from calute.cortex.task_creator import TaskCreator
from calute.cortex.universal_agent import UniversalAgent
from calute.llms.base import BaseLLM
from calute.streamer_buffer import StreamerBuffer
from calute.types import (
    Completion,
    FunctionCallsExtracted,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    ReinvokeSignal,
    StreamChunk,
)
from calute.types.agent_types import Agent
from calute.types.messages import AssistantMessage, MessagesHistory, UserMessage


def normalize_history(history: list[ChatMessage | dict] | None) -> list[ChatMessage]:
    norm: list[ChatMessage] = []
    for m in history or []:
        if isinstance(m, ChatMessage):
            norm.append(m)
        elif isinstance(m, dict):
            norm.append(
                ChatMessage(
                    role=m.get("role", "assistant"),
                    content=m.get("content", ""),
                    metadata=m.get("metadata"),
                )
            )
    return norm


def _render_collapsible_panel(
    title: str, body_md: str | None, *, status: str = "pending", icon: str = "🧠", opened: bool = True
) -> str:
    # Body can be markdown. We wrap it into a <details> so it can collapse.
    open_attr = " open" if opened else ""
    status_class = "status-pending" if status == "pending" else "status-done"
    safe_body = (body_md or "").strip()
    # Keep an empty <div> to avoid collapsing height to zero when no body yet
    if not safe_body:
        safe_body = "<div class='panel-empty'>Initializing…</div>"

    return f"""
<details class="calute-panel" {open_attr}>
  <summary>
    <span class="dot {status_class}"></span>
    <span class="title">{icon} {title}</span>
    <span class="chevron">▾</span>
  </summary>
  <div class="panel-body">
    {safe_body}
  </div>
</details>
""".strip()


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        # Show tenths under 1s
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{round(seconds)}s"
    m, s = divmod(round(seconds), 60)
    return f"{m}m {s}s"


def process_message(
    message: str,
    history: list[ChatMessage | dict],
    calute_msgs: MessagesHistory | None,
    *,
    executor: Calute | CortexAgent | CortexTask | Cortex | TaskCreator,
    agent: Agent | Cortex | DynamicCortex | None,
):
    history = normalize_history(history)
    calute_msgs = calute_msgs or MessagesHistory(messages=[])

    history.append(ChatMessage(role="user", content=message))
    calute_msgs.messages.append(UserMessage(content=message))

    history.append(ChatMessage(role="assistant", content=""))
    assistant_idx = len(history) - 1

    # State for auxiliary panels
    in_think = False
    think_idx_current = None
    think_seq = 0
    tool_msg_idx: dict[str, int] = {}
    tool_arg_bufs: dict[str, str] = {}
    tool_name_by_id: dict[str, str] = {}
    reinvoke_idx = None
    reinvoking = False
    prev_buf = ""

    def append_main(text: str):
        if not text:
            return
        cur = history[assistant_idx].content or ""
        history[assistant_idx] = ChatMessage(role="assistant", content=cur + text)

    # ---- Thinking panel helpers ----
    def _set_panel_content(idx: int, *, title: str, body: str, status: str, icon: str, opened: bool):
        md = dict(history[idx].metadata or {})
        md["title"] = title
        md["status"] = status
        md["panel_type"] = md.get("panel_type", "thinking")
        md["icon"] = icon
        md["body"] = body
        content = _render_collapsible_panel(title, body, status=status, icon=icon, opened=opened)
        history[idx] = ChatMessage(role="assistant", content=content, metadata=md)

    def open_thinking_panel():
        nonlocal think_idx_current, think_seq, assistant_idx
        think_seq += 1
        title = "Thinking…" if think_seq == 1 else f"Thinking… {think_seq}"
        md = {
            "title": title,
            "status": "pending",
            "panel_type": "thinking",
            "icon": "🧠",
            "body": "",
            "t_start": time.perf_counter(),
        }
        history.insert(
            assistant_idx,
            ChatMessage(
                role="assistant",
                content=_render_collapsible_panel(title, "", status="pending", icon="🧠", opened=True),
                metadata=md,
            ),
        )
        think_idx_current = assistant_idx
        assistant_idx += 1

    def append_think(text: str):
        nonlocal think_idx_current
        if think_idx_current is None:
            open_thinking_panel()
        md = dict(history[think_idx_current].metadata or {})
        body = (md.get("body") or "") + (text or "")
        _set_panel_content(
            think_idx_current,
            title=md.get("title", "Thinking…"),
            body=body,
            status="pending",
            icon=md.get("icon", "🧠"),
            opened=True,
        )

    def close_thinking_panel():
        nonlocal think_idx_current
        if think_idx_current is not None:
            md = dict(history[think_idx_current].metadata or {})
            t0 = md.get("t_start")
            elapsed = _format_seconds(time.perf_counter() - t0) if t0 else ""
            new_title = f"Thought for {elapsed}" if elapsed else "Thoughts (completed)"
            body = md.get("body") or ""
            _set_panel_content(
                think_idx_current, title=new_title, body=body, status="done", icon=md.get("icon", "🧠"), opened=False
            )
            think_idx_current = None

    # ---- Tool panels ----
    def get_or_create_tool_panel(tool_id: str, tool_name: str | None) -> int:
        nonlocal assistant_idx
        if tool_id in tool_msg_idx:
            return tool_msg_idx[tool_id]

        tool_name_by_id[tool_id] = tool_name or "Tool"
        title = f"Tool: {tool_name_by_id[tool_id]}"
        md = {"title": title, "status": "pending", "panel_type": "tool", "icon": "🛠️", "body": ""}
        history.insert(
            assistant_idx,
            ChatMessage(
                role="assistant",
                content=_render_collapsible_panel(title, "", status="pending", icon="🛠️", opened=True),
                metadata=md,
            ),
        )
        idx = assistant_idx
        tool_msg_idx[tool_id] = idx
        assistant_idx += 1
        return idx

    def update_tool_args(tool_id: str, delta: str):
        idx = get_or_create_tool_panel(tool_id, tool_name_by_id.get(tool_id))
        tool_arg_bufs[tool_id] = tool_arg_bufs.get(tool_id, "") + (delta or "")
        raw = tool_arg_bufs[tool_id].strip()

        if raw:
            try:
                parsed = json.loads(raw)
                body = f"**Parameters:**\n```json\n{json.dumps(parsed, indent=2)}\n```"
            except Exception:
                body = f"**Parameters:**\n```json\n{raw}\n```"
        else:
            body = "Initializing…"

        md = dict(history[idx].metadata or {})
        title = f"Tool: {tool_name_by_id.get(tool_id) or 'Tool'}"
        _set_panel_content(idx, title=title, body=body, status="pending", icon=md.get("icon", "🛠️"), opened=True)

    def mark_tool_running(tool_id: str, progress: str | None = None):
        idx = get_or_create_tool_panel(tool_id, tool_name_by_id.get(tool_id))
        md = dict(history[idx].metadata or {})
        title = f"Tool: {tool_name_by_id.get(tool_id) or 'Tool'}" + (f" — {progress}" if progress else "")
        body = md.get("body") or ""
        _set_panel_content(idx, title=title, body=body, status="pending", icon=md.get("icon", "🛠️"), opened=True)

    def mark_tool_done(tool_id: str, result: Any = None, error: str | None = None):
        idx = get_or_create_tool_panel(tool_id, tool_name_by_id.get(tool_id))
        if error:
            title = f"Tool Failed: {tool_name_by_id.get(tool_id) or 'Tool'}"
            body = f"**Error:**\n```\n{error}\n```"
            _set_panel_content(idx, title=title, body=body, status="done", icon="⚠️", opened=False)
        else:
            title = f"Tool Completed: {tool_name_by_id.get(tool_id) or 'Tool'}"
            if isinstance(result, dict | list):
                body = f"**Result:**\n```json\n{json.dumps(result, indent=2)}\n```"
            elif result is None or str(result).strip() == "":
                body = "**Result:** Operation completed successfully."
            else:
                body = f"**Result:**\n{result!s}"
            _set_panel_content(idx, title=title, body=body, status="done", icon="✅", opened=False)

    # ---- Reinvoke panel ----
    def open_reinvoke_panel():
        nonlocal reinvoke_idx, reinvoking, assistant_idx, in_think, prev_buf
        if in_think:
            close_thinking_panel()
        if reinvoke_idx is None:
            md = {
                "title": "Reinvoking Model",
                "status": "pending",
                "panel_type": "reinvoke",
                "icon": "🔁",
                "body": "Processing tool results and continuing…",
            }
            history.insert(
                assistant_idx,
                ChatMessage(
                    role="assistant",
                    content=_render_collapsible_panel(md["title"], md["body"], status="pending", icon="🔁", opened=True),
                    metadata=md,
                ),
            )
            reinvoke_idx = assistant_idx
            assistant_idx += 1
        prev_buf = ""
        reinvoking = True

    def maybe_close_reinvoke_panel():
        nonlocal reinvoke_idx, reinvoking
        if reinvoking and reinvoke_idx is not None:
            md = dict(history[reinvoke_idx].metadata or {})
            _set_panel_content(
                reinvoke_idx,
                title=md.get("title", "Reinvoking Model"),
                body="Tool results processed.",
                status="done",
                icon="🔁",
                opened=False,
            )
            reinvoke_idx = None
            reinvoking = False

    if isinstance(executor, Calute):
        buffer, thread = executor.thread_run(messages=calute_msgs, agent_id=agent)
    elif isinstance(executor, CortexAgent):
        buffer, thread = executor.execute(task_description=calute_msgs.messages[-1].content, use_thread=True)
    elif isinstance(executor, CortexTask):
        buffer, thread = executor.execute(use_streaming=True)
    elif isinstance(executor, Cortex):
        buffer, thread = executor.kickoff(use_streaming=True)
    elif isinstance(executor, DynamicCortex):
        buffer, thread = executor.execute_prompt(prompt=calute_msgs.messages[-1].content, stream=True)
    elif isinstance(executor, TaskCreator):
        assert isinstance(agent, Cortex) or isinstance(agent, DynamicCortex) or isinstance(agent, BaseLLM), (
            "TaskCreator requires a Cortex or DynamicCortex agent"
        )
        if isinstance(agent, BaseLLM):
            buffer = StreamerBuffer()

            def fn():
                plan, tasks = TaskCreator(llm=agent).create_tasks_from_prompt(
                    prompt=calute_msgs.messages[-1].content,
                    available_agents=[UniversalAgent(llm=agent, allow_delegation=True)],
                    streamer_buffer=buffer,
                    stream=True,
                )

                Cortex.from_task_creator(tasks).kickoff(use_streaming=True, streamer_buffer=buffer)[-1].join()

            thread = threading.Thread(target=fn)
            thread.start()
        else:
            buffer, thread = executor.create_and_execute(
                prompt=calute_msgs.messages[-1].content,
                background=None,
                cortex=agent,
            )

    for ev in buffer.stream():
        if isinstance(ev, StreamChunk):
            if reinvoking:
                maybe_close_reinvoke_panel()

            if ev.streaming_tool_calls:
                for tc in ev.streaming_tool_calls:
                    if tc.id:
                        if tc.function_name and tc.id not in tool_name_by_id:
                            tool_name_by_id[tc.id] = tc.function_name
                        update_tool_args(tc.id, tc.arguments or "")
                yield history, calute_msgs

            buf = ev.buffered_content or ""
            if not buf:
                continue

            delta = buf[len(prev_buf) :] if len(buf) >= len(prev_buf) else buf
            prev_buf = buf

            if not delta:
                continue

            while delta:
                if in_think:
                    m = re.search(r"</(think|thinking|reason|reasoning)>", delta, flags=re.I)
                    if m:
                        append_think(delta[: m.start()])
                        delta = delta[m.end() :]
                        in_think = False
                        close_thinking_panel()
                        yield history, calute_msgs
                    else:
                        append_think(delta)
                        delta = ""
                        yield history, calute_msgs
                else:
                    m = re.search(r"<(think|thinking|reason|reasoning)>", delta, flags=re.I)
                    if m:
                        append_main(delta[: m.start()])
                        delta = delta[m.end() :]
                        in_think = True
                        open_thinking_panel()
                        yield history, calute_msgs
                    else:
                        append_main(delta)
                        delta = ""
                        yield history, calute_msgs

        elif isinstance(ev, FunctionCallsExtracted):
            for fc in ev.function_calls:
                tool_name_by_id[fc.id] = fc.name
                get_or_create_tool_panel(fc.id, fc.name)
            yield history, calute_msgs

        elif isinstance(ev, FunctionExecutionStart):
            mark_tool_running(ev.function_id, progress=ev.progress)
            yield history, calute_msgs

        elif isinstance(ev, FunctionExecutionComplete):
            mark_tool_done(ev.function_id, result=ev.result, error=ev.error)
            yield history, calute_msgs

        elif isinstance(ev, ReinvokeSignal):
            open_reinvoke_panel()
            yield history, calute_msgs

        elif isinstance(ev, Completion):
            if in_think:
                close_thinking_panel()
                in_think = False
            maybe_close_reinvoke_panel()

            for idx in list(tool_msg_idx.values()):
                md = dict(history[idx].metadata or {})
                if md.get("status") == "pending":
                    # Complete but keep the rendered (collapsed) panel
                    _set_panel_content(
                        idx,
                        title=md.get("title", "Tool"),
                        body=md.get("body", ""),
                        status="done",
                        icon=md.get("icon", "🛠️"),
                        opened=False,
                    )

        if not thread.is_alive():
            buffer.close()

    # Strip any private <think> tags that may have leaked into the main assistant message
    main_text = history[assistant_idx].content or ""
    main_text = re.sub(
        r"<(?:think|thinking|reason|reasoning)>.*?</(?:think|thinking|reason|reasoning)>",
        "",
        main_text,
        flags=re.S | re.I,
    ).strip()
    history[assistant_idx] = ChatMessage(role="assistant", content=main_text)

    calute_msgs.messages.append(AssistantMessage(content=main_text))
    yield history, calute_msgs


def clear_session():
    return [], MessagesHistory(messages=[])
