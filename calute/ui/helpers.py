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


"""Helper utilities for processing chat messages in Calute Chainlit UI.

This module provides core functionality for handling message streaming, tool execution,
thinking panels, and UI updates in the Chainlit-based chat interface.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import AsyncIterator
from queue import Empty
from typing import Any

import chainlit as cl

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

# Regex patterns for thinking tags
THINK_OPEN_PATTERN = re.compile(r"<(think|thinking|reason|reasoning)>", re.IGNORECASE)
THINK_CLOSE_PATTERN = re.compile(r"</(think|thinking|reason|reasoning)>", re.IGNORECASE)


def get_mcp_tools_for_llm() -> list[dict[str, Any]]:
    """Get all MCP tools in OpenAI function calling format.

    Returns:
        List of tool definitions in OpenAI format.
    """
    mcp_tools = cl.user_session.get("mcp_tools") or {}
    all_tools = []
    for server_name, server_data in mcp_tools.items():
        for tool in server_data.get("tools", []):
            # Add server prefix to avoid name collisions
            tool_copy = tool.copy()
            tool_copy["function"] = tool["function"].copy()
            tool_copy["function"]["name"] = f"mcp_{server_name}_{tool['function']['name']}"
            all_tools.append(tool_copy)
    return all_tools


def find_mcp_session_for_tool(tool_name: str) -> tuple[Any, str] | None:
    """Find the MCP session that provides a given tool.

    Args:
        tool_name: The tool name (may include mcp_servername_ prefix)

    Returns:
        Tuple of (session, original_tool_name) or None if not found.
    """
    mcp_tools = cl.user_session.get("mcp_tools") or {}

    # Handle prefixed tool names (mcp_servername_toolname)
    if tool_name.startswith("mcp_"):
        parts = tool_name[4:].split("_", 1)
        if len(parts) == 2:
            server_name, original_name = parts
            if server_name in mcp_tools:
                return mcp_tools[server_name].get("session"), original_name

    # Fallback: search all servers for the tool
    for _server_name, server_data in mcp_tools.items():
        for tool in server_data.get("tools", []):
            if tool["function"]["name"] == tool_name:
                return server_data.get("session"), tool_name

    return None


async def call_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call an MCP tool and return the result.

    Args:
        tool_name: Name of the tool to call
        arguments: Tool arguments

    Returns:
        Tool execution result
    """
    result = find_mcp_session_for_tool(tool_name)
    if not result:
        raise ValueError(f"MCP tool not found: {tool_name}")

    session, original_name = result

    # Call the tool via MCP session
    tool_result = await session.call_tool(original_name, arguments)

    # Extract content from result
    if hasattr(tool_result, "content"):
        content = tool_result.content
        if isinstance(content, list) and len(content) > 0:
            first_item = content[0]
            if hasattr(first_item, "text"):
                return first_item.text
            return str(first_item)
        return str(content)
    return str(tool_result)


_MCP_EVENT_LOOP = None
_MCP_LOOP_LOCK = threading.Lock()


def _get_mcp_event_loop():
    """Get or create a dedicated event loop for MCP calls."""
    global _MCP_EVENT_LOOP
    with _MCP_LOOP_LOCK:
        if _MCP_EVENT_LOOP is None or not _MCP_EVENT_LOOP.is_running():
            import threading as _threading

            def _run_loop(loop):
                asyncio.set_event_loop(loop)
                loop.run_forever()

            _MCP_EVENT_LOOP = asyncio.new_event_loop()
            thread = _threading.Thread(target=_run_loop, args=(_MCP_EVENT_LOOP,), daemon=True)
            thread.start()
            # Give loop time to start
            import time
            time.sleep(0.1)
        return _MCP_EVENT_LOOP


def create_mcp_tool_function(server_name: str, tool_def: dict[str, Any], session: Any) -> callable:
    """Create a callable function wrapper for an MCP tool.

    This creates a function that can be registered with Calute agents
    and called during function execution.

    Args:
        server_name: Name of the MCP server providing the tool
        tool_def: Tool definition in OpenAI format
        session: MCP session for calling the tool

    Returns:
        A callable function that wraps the MCP tool call
    """
    func_info = tool_def.get("function", tool_def)
    original_name = func_info["name"]
    description = func_info.get("description", "")
    parameters = func_info.get("parameters", {})

    # Create sync wrapper that runs async call
    def mcp_tool_wrapper(**kwargs):
        """MCP tool wrapper - executes tool call via MCP session."""
        import asyncio

        async def _call():
            tool_result = await session.call_tool(original_name, kwargs)
            # Extract content from result
            if hasattr(tool_result, "content"):
                content = tool_result.content
                if isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if hasattr(first_item, "text"):
                        return first_item.text
                    return str(first_item)
                return str(content)
            return str(tool_result)

        # Use dedicated MCP event loop to avoid conflicts
        loop = _get_mcp_event_loop()
        future = asyncio.run_coroutine_threadsafe(_call(), loop)
        # Wait for result with timeout
        try:
            return future.result(timeout=60.0)
        except Exception as e:
            return f"Error calling MCP tool: {e}"

    # Set function metadata for Calute's function_to_json
    mcp_tool_wrapper.__name__ = f"mcp_{server_name}_{original_name}"
    mcp_tool_wrapper.__doc__ = f"{description}\n\nMCP Tool from server: {server_name}"

    # Add schema annotation for Calute
    mcp_tool_wrapper.__calute_schema__ = {
        "name": f"mcp_{server_name}_{original_name}",
        "description": description,
        "parameters": parameters,
    }

    return mcp_tool_wrapper


def get_mcp_tool_functions() -> list[callable]:
    """Get all MCP tools as callable functions for Calute agents.

    Returns:
        List of callable functions wrapping MCP tools.
    """
    mcp_tools = cl.user_session.get("mcp_tools") or {}
    functions = []

    for server_name, server_data in mcp_tools.items():
        session = server_data.get("session")
        if not session:
            continue

        for tool_def in server_data.get("tools", []):
            func = create_mcp_tool_function(server_name, tool_def, session)
            functions.append(func)

    return functions


def inject_mcp_tools_to_agent(executor: Any, agent_id: str | None) -> list[callable]:
    """Inject MCP tools into an agent's function list.

    This temporarily adds MCP tool functions to the agent so they
    can be used during message processing.

    Args:
        executor: The Calute executor instance
        agent_id: The agent ID to inject tools into

    Returns:
        List of original functions (for restoration later)
    """
    mcp_functions = get_mcp_tool_functions()
    if not mcp_functions:
        return []

    # Get the agent
    agent = None
    if hasattr(executor, "orchestrator") and hasattr(executor.orchestrator, "agents"):
        agents = executor.orchestrator.agents
        if agent_id and agent_id in agents:
            agent = agents[agent_id]
        elif hasattr(executor.orchestrator, "get_current_agent"):
            agent = executor.orchestrator.get_current_agent()

    if not agent:
        return []

    # Store original functions
    original_functions = list(agent.functions) if agent.functions else []

    # Add MCP functions
    if agent.functions is None:
        agent.functions = []

    for mcp_func in mcp_functions:
        # Check if already added (avoid duplicates)
        if not any(f.__name__ == mcp_func.__name__ for f in agent.functions):
            agent.functions.append(mcp_func)

    return original_functions


def restore_agent_functions(executor: Any, agent_id: str | None, original_functions: list[callable]) -> None:
    """Restore an agent's original function list after MCP injection.

    Args:
        executor: The Calute executor instance
        agent_id: The agent ID to restore functions for
        original_functions: The original function list to restore
    """
    # Get the agent
    agent = None
    if hasattr(executor, "orchestrator") and hasattr(executor.orchestrator, "agents"):
        agents = executor.orchestrator.agents
        if agent_id and agent_id in agents:
            agent = agents[agent_id]
        elif hasattr(executor.orchestrator, "get_current_agent"):
            agent = executor.orchestrator.get_current_agent()

    if agent:
        agent.functions = original_functions


async def async_stream(buffer: StreamerBuffer) -> AsyncIterator:
    """Convert synchronous StreamerBuffer to async generator.

    Args:
        buffer: The StreamerBuffer to stream from.

    Yields:
        Events from the buffer.
    """
    loop = asyncio.get_event_loop()

    def get_next():
        try:
            return buffer.get(timeout=0.1)
        except Empty:
            return None

    while True:
        event = await loop.run_in_executor(None, get_next)
        if event is None:
            if buffer.closed:
                break
            await asyncio.sleep(0.01)
            continue
        yield event


def format_result(result: Any) -> str:
    """Format a tool result for display.

    Args:
        result: The result to format.

    Returns:
        Formatted string representation.
    """
    if isinstance(result, (dict, list)):
        try:
            return json.dumps(result, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(result)
    return str(result) if result is not None else "Done"


async def process_message_chainlit(
    message: str,
    calute_msgs: MessagesHistory | None,
    *,
    executor: Calute | CortexAgent | CortexTask | Cortex | TaskCreator | DynamicCortex,
    agent: Agent | Cortex | DynamicCortex | None,
) -> MessagesHistory:
    """Process a user message with Chainlit streaming support.

    Args:
        message: The user's message text.
        calute_msgs: Existing message history.
        executor: The executor to use for processing.
        agent: Optional agent configuration.

    Returns:
        Updated MessagesHistory with the new exchange.
    """
    calute_msgs = calute_msgs or MessagesHistory(messages=[])
    calute_msgs.messages.append(UserMessage(content=message))

    # Inject MCP tools into the agent before processing
    original_functions = inject_mcp_tools_to_agent(executor, agent)

    # Create the main assistant message for streaming
    msg = cl.Message(content="")
    await msg.send()

    # Add action buttons after initial send
    msg.actions = [
        cl.Action(name="regenerate", payload={}, label="Regenerate", description="Regenerate this response"),
        cl.Action(name="clear_history", payload={}, label="Clear History", description="Clear conversation"),
    ]

    # State tracking
    tool_steps: dict[str, cl.Step] = {}
    think_step: cl.Step | None = None
    in_thinking = False
    think_count = 0
    main_content = ""

    # Get buffer and thread from executor
    buffer, thread = _start_executor(executor, calute_msgs, agent)

    # Process stream events
    async for event in async_stream(buffer):
        if isinstance(event, StreamChunk):
            # Handle streaming tool call arguments
            if event.streaming_tool_calls:
                for tc in event.streaming_tool_calls:
                    if tc.id and tc.function_name:
                        await _handle_tool_args_stream(tc.id, tc.function_name, tc.arguments or "", tool_steps)

            # Handle text content with thinking detection
            # Use content or buffered_content - chunk is the raw ChatCompletionChunk object
            content = event.content or ""
            if content:
                main_content, in_thinking, think_step, think_count = await _process_content(
                    content, main_content, in_thinking, think_step, think_count, msg
                )

        elif isinstance(event, FunctionCallsExtracted):
            # Create steps for extracted function calls
            for fc in event.function_calls:
                if fc.id not in tool_steps:
                    step = cl.Step(name=fc.name, type="tool", show_input="json")
                    await step.__aenter__()
                    step.input = "Preparing..."
                    tool_steps[fc.id] = step

        elif isinstance(event, FunctionExecutionStart):
            await _handle_tool_start(event, tool_steps)

        elif isinstance(event, FunctionExecutionComplete):
            await _handle_tool_complete(event, tool_steps)

        elif isinstance(event, ReinvokeSignal):
            # Close any open thinking panel
            if in_thinking and think_step:
                await _close_think_step(think_step)
                think_step = None
                in_thinking = False

        elif isinstance(event, Completion):
            # Close any open thinking step
            if in_thinking and think_step:
                await _close_think_step(think_step)
                think_step = None

            # Close all pending tool steps
            for step in list(tool_steps.values()):
                if not step.output:
                    step.output = "Completed"
                try:
                    await step.__aexit__(None, None, None)
                except Exception:
                    pass
            tool_steps.clear()

        # Check if thread is done
        if not thread.is_alive():
            buffer.close()

    # Clean final content - remove any thinking tags
    main_content = _remove_thinking_tags(main_content)
    msg.content = main_content
    await msg.update()

    # Restore original agent functions (remove MCP tools)
    if original_functions:
        restore_agent_functions(executor, agent, original_functions)

    calute_msgs.messages.append(AssistantMessage(content=main_content))
    return calute_msgs


async def _process_content(
    content: str,
    main_content: str,
    in_thinking: bool,
    think_step: cl.Step | None,
    think_count: int,
    msg: cl.Message,
) -> tuple[str, bool, cl.Step | None, int]:
    """Process content chunk, handling thinking tags.

    Args:
        content: The new content chunk.
        main_content: Accumulated main content.
        in_thinking: Whether currently inside thinking tags.
        think_step: Current thinking step if open.
        think_count: Counter for thinking panels.
        msg: Main message to stream to.

    Returns:
        Tuple of (updated_content, in_thinking, think_step, think_count).
    """
    remaining = content

    while remaining:
        if in_thinking:
            # Look for closing tag
            match = THINK_CLOSE_PATTERN.search(remaining)
            if match:
                # Add content before closing tag to think step
                if think_step:
                    think_step.output = (think_step.output or "") + remaining[: match.start()]
                remaining = remaining[match.end() :]
                in_thinking = False
                if think_step:
                    await _close_think_step(think_step)
                    think_step = None
            else:
                # All remaining is thinking content
                if think_step:
                    think_step.output = (think_step.output or "") + remaining
                remaining = ""
        else:
            # Look for opening tag
            match = THINK_OPEN_PATTERN.search(remaining)
            if match:
                # Stream content before tag to main message
                before = remaining[: match.start()]
                if before:
                    main_content += before
                    await msg.stream_token(before)
                remaining = remaining[match.end() :]
                in_thinking = True
                think_count += 1
                # Open a new thinking step
                title = "Thinking..." if think_count == 1 else f"Thinking... ({think_count})"
                think_step = cl.Step(name=title, type="llm")
                await think_step.__aenter__()
                think_step.output = ""
            else:
                # All remaining is main content
                main_content += remaining
                await msg.stream_token(remaining)
                remaining = ""

    return main_content, in_thinking, think_step, think_count


async def _close_think_step(step: cl.Step) -> None:
    """Close a thinking step properly.

    Args:
        step: The thinking step to close.
    """
    step.output = step.output or ""
    try:
        await step.__aexit__(None, None, None)
    except Exception:
        pass


async def _handle_tool_args_stream(
    tool_id: str,
    tool_name: str,
    args_delta: str,
    tool_steps: dict[str, cl.Step],
) -> None:
    """Handle streaming tool call arguments.

    Args:
        tool_id: Unique tool call ID.
        tool_name: Name of the tool.
        args_delta: New argument text chunk.
        tool_steps: Dictionary of tool steps.
    """
    if tool_id not in tool_steps:
        step = cl.Step(name=tool_name, type="tool", show_input="json")
        await step.__aenter__()
        step.input = ""
        step.output = ""
        tool_steps[tool_id] = step

    step = tool_steps[tool_id]
    step.input = (step.input or "") + args_delta

    # Try to format as JSON if complete
    raw = step.input.strip()
    if raw:
        try:
            parsed = json.loads(raw)
            step.input = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pass  # Keep raw for incomplete JSON


async def _handle_tool_start(
    event: FunctionExecutionStart,
    tool_steps: dict[str, cl.Step],
) -> None:
    """Handle tool execution start event.

    Args:
        event: The function execution start event.
        tool_steps: Dictionary of tool steps.
    """
    if event.function_id not in tool_steps:
        step = cl.Step(name=event.function_name, type="tool", show_input="json")
        await step.__aenter__()
        step.input = ""
        step.output = ""
        tool_steps[event.function_id] = step

    step = tool_steps[event.function_id]
    if event.progress:
        step.name = f"{event.function_name} ({event.progress})"


async def _handle_tool_complete(
    event: FunctionExecutionComplete,
    tool_steps: dict[str, cl.Step],
) -> None:
    """Handle tool execution complete event.

    Args:
        event: The function execution complete event.
        tool_steps: Dictionary of tool steps.
    """
    step = tool_steps.pop(event.function_id, None)
    if not step:
        return

    if event.error:
        step.output = f"Error: {event.error}"
    elif event.result is not None:
        step.output = format_result(event.result)
    else:
        step.output = "Done"

    try:
        await step.__aexit__(None, None, None)
    except Exception:
        pass


def _start_executor(
    executor: Calute | CortexAgent | CortexTask | Cortex | TaskCreator | DynamicCortex,
    calute_msgs: MessagesHistory,
    agent: Any,
) -> tuple[StreamerBuffer, threading.Thread]:
    """Start the appropriate executor and return buffer + thread.

    Args:
        executor: The executor instance.
        calute_msgs: Message history.
        agent: Optional agent configuration.

    Returns:
        Tuple of (StreamerBuffer, Thread).
    """
    if isinstance(executor, Calute):
        return executor.thread_run(messages=calute_msgs, agent_id=agent)

    elif isinstance(executor, CortexAgent):
        return executor.execute(task_description=calute_msgs.messages[-1].content, use_thread=True)

    elif isinstance(executor, CortexTask):
        return executor.execute(use_streaming=True)

    elif isinstance(executor, Cortex):
        return executor.kickoff(use_streaming=True)

    elif isinstance(executor, DynamicCortex):
        return executor.execute_prompt(prompt=calute_msgs.messages[-1].content, stream=True)

    elif isinstance(executor, TaskCreator):
        # Handle TaskCreator special case
        if isinstance(agent, BaseLLM):
            buffer = StreamerBuffer()

            def fn():
                _plan, tasks = TaskCreator(llm=agent).create_tasks_from_prompt(
                    prompt=calute_msgs.messages[-1].content,
                    available_agents=[UniversalAgent(llm=agent, allow_delegation=True)],
                    streamer_buffer=buffer,
                    stream=True,
                )
                Cortex.from_task_creator(tasks).kickoff(use_streaming=True, streamer_buffer=buffer)[-1].join()

            thread = threading.Thread(target=fn)
            thread.start()
            return buffer, thread
        else:
            return executor.create_and_execute(
                prompt=calute_msgs.messages[-1].content,
                background=None,
                cortex=agent,
            )

    # Fallback - should not reach here
    raise TypeError(f"Unsupported executor type: {type(executor)}")


def _remove_thinking_tags(content: str) -> str:
    """Remove thinking tags from final content.

    Args:
        content: Text that may contain thinking tags.

    Returns:
        Cleaned text without thinking tags.
    """
    return re.sub(
        r"<(?:think|thinking|reason|reasoning)>.*?</(?:think|thinking|reason|reasoning)>",
        "",
        content,
        flags=re.S | re.I,
    ).strip()
