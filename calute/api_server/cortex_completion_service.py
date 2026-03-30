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


"""Cortex completion service for handling multi-agent orchestration via API.

This module provides the Cortex-based completion service infrastructure,
including:
- Multi-agent orchestration for complex task execution
- Task mode with dynamic task creation and agent assignment
- Instruction mode for direct prompt execution
- Streaming and non-streaming response generation
- Integration with DynamicCortex for sophisticated agent workflows

The service supports both sequential and parallel execution strategies,
with configurable process types and real-time streaming of execution events.
"""

from __future__ import annotations

import asyncio
import json
import threading
import typing
from collections.abc import AsyncIterator

from calute.types.function_execution_types import (
    AgentSwitch,
    Completion,
    FunctionCallsExtracted,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    ReinvokeSignal,
    StreamChunk,
)

from ..core.streamer_buffer import StreamerBuffer
from ..cortex import CortexAgent, DynamicCortex, TaskCreator, UniversalAgent
from ..cortex.core.enums import ProcessType
from ..logging.console import get_logger
from ..types import MessagesHistory
from ..types.oai_protocols import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)

if typing.TYPE_CHECKING:
    from ..llms.base import BaseLLM

DONE_TAG = '/["DONE"]/'
"""Sentinel tag used to signal the end of a Cortex streaming response."""


class CortexCompletionService:
    """Service for handling Cortex-based chat completions with multi-agent orchestration.

    Provides advanced multi-agent orchestration capabilities through the Cortex
    system. Supports two primary execution modes:

    - **Task mode**: Uses ``TaskCreator`` to dynamically decompose prompts into
      discrete tasks, assign agents to each task, and execute them through
      ``DynamicCortex``. Activated when the model name contains ``"task"``.
    - **Instruction mode**: Directly executes prompts through ``DynamicCortex``
      without task decomposition. Used as the default mode.

    Both modes support sequential, parallel, and hierarchical process types,
    as well as streaming and non-streaming response generation.

    Attributes:
        llm: The ``BaseLLM`` instance used for agent interactions.
        verbose: Flag indicating whether verbose logging is enabled.
        logger: Logger instance for verbose output (``None`` if disabled).
        agents: List of ``CortexAgent`` instances available for task execution.
        universal_agent: ``UniversalAgent`` instance for fallback handling, or
            ``None`` if disabled.
        task_creator: ``TaskCreator`` instance for dynamic task generation with
            automatic agent assignment.

    Example:
        >>> from calute.api_server.cortex_completion_service import CortexCompletionService
        >>> service = CortexCompletionService(llm=my_llm, agents=[agent1, agent2])
    """

    def __init__(
        self,
        llm: BaseLLM,
        agents: list[CortexAgent] | None = None,
        use_universal_agent: bool = True,
        verbose: bool = True,
    ):
        """Initialize the Cortex completion service.

        Sets up the agent pool, optional ``UniversalAgent`` fallback, and
        the ``TaskCreator`` for dynamic task generation. If
        ``use_universal_agent`` is ``True``, a ``UniversalAgent`` is created
        and appended to the agents list (if not already present).

        Args:
            llm: The ``BaseLLM`` instance to use for powering all agents
                and the task creator.
            agents: Optional list of specialized ``CortexAgent`` instances
                for task handling. Defaults to an empty list if ``None``.
            use_universal_agent: Whether to include a ``UniversalAgent`` as
                a fallback for tasks that don't match any specialized agent.
                The universal agent is configured with delegation enabled,
                temperature 0.7, and max_tokens 4096. Defaults to ``True``.
            verbose: Whether to enable verbose logging of execution events
                via the console logger. Defaults to ``True``.
        """
        self.llm = llm
        self.verbose = verbose
        self.logger = get_logger() if verbose else None

        self.agents = agents or []
        if use_universal_agent:
            self.universal_agent = UniversalAgent(
                llm=llm,
                verbose=verbose,
                allow_delegation=True,
                temperature=0.7,
                max_tokens=4096,
            )
            if self.universal_agent not in self.agents:
                self.agents.append(self.universal_agent)
        else:
            self.universal_agent = None

        self.task_creator = TaskCreator(
            llm=llm,
            verbose=verbose,
            auto_assign_agents=True,
        )

    def _extract_task_config(self, request: ChatCompletionRequest) -> dict:
        """Extract task configuration from the request model name and metadata.

        Parses the request model name and optional metadata to determine the
        execution configuration. The model name is checked for keywords to
        infer the mode and process type:

        - Contains ``"task"`` -> enables task mode
        - Contains ``"parallel"`` -> ``ProcessType.PARALLEL``
        - Contains ``"hierarchical"`` -> ``ProcessType.HIERARCHICAL``
        - Otherwise -> ``ProcessType.SEQUENTIAL`` (default)

        If the request has a ``metadata`` attribute with a dictionary value,
        it can override these inferred values via ``task_mode``,
        ``process_type``, and ``background`` keys.

        Args:
            request: The ``ChatCompletionRequest`` containing the model name
                string and optional metadata dictionary.

        Returns:
            A dictionary with the following keys:

            - ``task_mode`` (bool): Whether to use task mode with dynamic
              task creation and agent assignment.
            - ``process_type`` (``ProcessType``): The execution strategy enum
              value (``SEQUENTIAL``, ``PARALLEL``, or ``HIERARCHICAL``).
            - ``background`` (str or None): Optional background/approach
              context string for task creation.

        Example:
            >>> config = service._extract_task_config(request)
            >>> config["task_mode"]
            True
            >>> config["process_type"]
            <ProcessType.PARALLEL: ...>
        """
        task_mode = False
        process_type = ProcessType.SEQUENTIAL
        background = None

        model = request.model.lower() if request.model else ""

        if "task" in model:
            task_mode = True

        if "parallel" in model:
            process_type = ProcessType.PARALLEL
        elif "hierarchical" in model:
            process_type = ProcessType.HIERARCHICAL

        if hasattr(request, "metadata") and request.metadata:
            metadata = request.metadata if isinstance(request.metadata, dict) else {}
            task_mode = metadata.get("task_mode", task_mode)
            process_type_str = metadata.get("process_type", "sequential")
            try:
                process_type = ProcessType[process_type_str.upper()]
            except KeyError:
                pass
            background = metadata.get("background", None)

        return {
            "task_mode": task_mode,
            "process_type": process_type,
            "background": background,
        }

    def _extract_prompt_from_messages(self, messages: MessagesHistory) -> str:
        """Extract the latest user prompt from the message history.

        Iterates through the message history in reverse chronological order
        to find the most recent user message. Detection uses both the
        ``role`` attribute (checking for ``"user"``) and the class name
        (checking for ``"UserMessage"``) to handle different message formats.

        If no user message is found, all messages are concatenated with
        newline separators as a fallback.

        Args:
            messages: The ``MessagesHistory`` instance containing user,
                assistant, and system messages.

        Returns:
            The content string of the most recent user message, or a
            newline-joined concatenation of all message contents if no
            user message exists in the history.
        """

        for msg in reversed(messages.messages):
            if hasattr(msg, "role") and msg.role == "user":
                return msg.content
            elif msg.__class__.__name__ == "UserMessage":
                return msg.content

        return "\n".join(str(msg.content) for msg in messages.messages)

    async def create_completion(
        self,
        messages: MessagesHistory,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Create a non-streaming Cortex completion.

        Extracts the task configuration from the request, determines the
        latest user prompt, and executes either task mode or instruction
        mode accordingly. The result is wrapped in a
        ``ChatCompletionResponse`` with word-count-based usage estimates.

        Args:
            messages: The ``MessagesHistory`` containing the conversation
                context to process.
            request: The ``ChatCompletionRequest`` containing the model
                name (used to determine execution mode) and optional
                metadata for fine-grained configuration.

        Returns:
            A ``ChatCompletionResponse`` with a single choice containing
            the Cortex execution result as assistant content, estimated
            usage information (based on word counts), and finish reason
            ``"stop"``. The model field defaults to ``"cortex"`` if no
            model name is specified in the request.
        """
        config = self._extract_task_config(request)
        prompt = self._extract_prompt_from_messages(messages)

        if config["task_mode"]:
            result = await self._execute_task_mode(
                prompt=prompt,
                background=config["background"],
                process_type=config["process_type"],
                stream=False,
            )
        else:
            result = await self._execute_instruction_mode(
                prompt=prompt,
                process_type=config["process_type"],
                stream=False,
            )

        content = str(result) if not isinstance(result, str) else result

        return ChatCompletionResponse(
            model=request.model or "cortex",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                completion_tokens=len(content.split()),
                prompt_tokens=len(prompt.split()),
                total_tokens=len(content.split()) + len(prompt.split()),
            ),
        )

    async def create_streaming_completion(
        self,
        messages: MessagesHistory,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Create a streaming Cortex completion with real-time event updates.

        Executes the Cortex system in a background daemon thread, reading
        events from a ``StreamerBuffer`` and yielding them as SSE-formatted
        strings. The stream includes multiple event types:

        - ``StreamChunk``: Content delta with optional tool call information.
        - ``FunctionDetection``: Notification that functions are being detected.
        - ``FunctionCallsExtracted``: List of functions identified for execution.
        - ``FunctionExecutionStart``: Start signal for a specific function.
        - ``FunctionExecutionComplete``: Completion signal with result or error.
        - ``AgentSwitch``: Notification of agent delegation with reason.
        - ``ReinvokeSignal``: Signal that the agent is being reinvoked.
        - ``Completion``: Final task completion signal with execution stats.

        Each event is serialized as a ``ChatCompletionStreamResponse`` JSON
        object with optional metadata. The stream terminates with a final
        ``finish_reason="stop"`` chunk followed by ``"data: [DONE]"``.

        Args:
            messages: The ``MessagesHistory`` containing the conversation
                context to process.
            request: The ``ChatCompletionRequest`` containing the model
                name (used to determine execution mode) and optional
                metadata for fine-grained configuration.

        Yields:
            SSE-formatted strings (``"data: {json}\\n\\n"``) containing
            streaming response chunks. Each chunk may include content
            deltas and metadata about execution events. The stream ends
            with a ``"data: [DONE]\\n\\n"`` sentinel.
        """
        config = self._extract_task_config(request)
        prompt = self._extract_prompt_from_messages(messages)

        streamer_buffer = StreamerBuffer()

        if config["task_mode"]:
            thread = threading.Thread(
                target=self._execute_task_mode_sync,
                args=(prompt, config["background"], config["process_type"], streamer_buffer),
                daemon=True,
            )
        else:
            thread = threading.Thread(
                target=self._execute_instruction_mode_sync,
                args=(prompt, config["process_type"], streamer_buffer),
                daemon=True,
            )

        thread.start()

        chunk_id = 0
        for chunk in streamer_buffer.stream():
            content = None
            metadata = {}

            if isinstance(chunk, StreamChunk):
                if chunk.content:
                    content = chunk.content
                if hasattr(chunk, "streaming_tool_calls") and chunk.streaming_tool_calls:
                    tool_info = []
                    for tc in chunk.streaming_tool_calls:
                        tool_info.append({"name": tc.function_name, "arguments": tc.arguments})
                    metadata["tool_calls"] = tool_info

            elif isinstance(chunk, FunctionDetection):
                content = f"\n**Detecting functions: {chunk.message}**\n"
                metadata["event"] = "function_detection"

            elif isinstance(chunk, FunctionCallsExtracted):
                funcs = ", ".join([fc.name for fc in chunk.function_calls])
                content = f"\n*Functions to execute: {funcs}*\n"
                metadata["event"] = "functions_extracted"
                metadata["functions"] = [fc.name for fc in chunk.function_calls]

            elif isinstance(chunk, FunctionExecutionStart):
                content = f"\n⚡ Executing {chunk.function_name}...\n"
                metadata["event"] = "function_start"
                metadata["function"] = chunk.function_name
                if hasattr(chunk, "progress"):
                    metadata["progress"] = chunk.progress

            elif isinstance(chunk, FunctionExecutionComplete):
                content = f"\n*{chunk.function_name} completed*\n"
                metadata["event"] = "function_complete"
                metadata["function"] = chunk.function_name
                metadata["status"] = chunk.status
                if hasattr(chunk, "result") and chunk.result:
                    result_str = str(chunk.result)
                    if len(result_str) > 100:
                        result_str = result_str[:100] + "..."
                    content += f"   Result: {result_str}\n"
                    metadata["has_result"] = True
                elif hasattr(chunk, "error") and chunk.error:
                    content += f"   Error: {chunk.error}\n"
                    metadata["error"] = chunk.error

            elif isinstance(chunk, AgentSwitch):
                content = f"\n*Switching from {chunk.from_agent} to {chunk.to_agent}*\n"
                metadata["event"] = "agent_switch"
                metadata["from_agent"] = chunk.from_agent
                metadata["to_agent"] = chunk.to_agent
                if hasattr(chunk, "reason"):
                    content += f"   Reason: {chunk.reason}\n"
                    metadata["reason"] = chunk.reason

            elif isinstance(chunk, ReinvokeSignal):
                content = f"\n*Reinvoke* {chunk.message}\n"
                metadata["event"] = "reinvoke"

            elif isinstance(chunk, Completion):
                content = "\n*Task completed*\n"
                metadata["event"] = "completion"
                metadata["functions_executed"] = getattr(chunk, "function_calls_executed", 0)

            if content:
                stream_response = ChatCompletionStreamResponse(
                    model=request.model or "cortex",
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta=DeltaMessage(
                                role="assistant" if chunk_id == 0 else None,
                                content=content,
                            ),
                            finish_reason=None,
                        )
                    ],
                    usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )

                if metadata:
                    stream_response.metadata = metadata  # type: ignore

                yield f"data: {json.dumps(stream_response.model_dump())}\n\n"
                chunk_id += 1

            if not thread.is_alive():
                streamer_buffer.close()
        final_response = ChatCompletionStreamResponse(
            model=request.model or "cortex",
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=DeltaMessage(content=""),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        yield f"data: {json.dumps(final_response.model_dump())}\n\n"
        yield "data: [DONE]\n\n"

    async def _execute_task_mode(
        self,
        prompt: str,
        background: str | None,
        process_type: ProcessType,
        stream: bool,
    ) -> str | tuple[StreamerBuffer, threading.Thread]:
        """Execute in task mode with dynamic task creation (async wrapper).

        Offloads ``_execute_task_mode_sync`` to a thread executor via
        ``loop.run_in_executor`` to avoid blocking the async event loop.
        The synchronous method uses ``TaskCreator`` to decompose the prompt
        into discrete tasks, assigns agents to each task, and executes them
        using ``DynamicCortex``.

        Args:
            prompt: The user prompt to decompose into tasks and execute.
            background: Optional background/approach information providing
                additional context to the ``TaskCreator`` for better task
                decomposition.
            process_type: The ``ProcessType`` enum value determining the
                execution strategy (``SEQUENTIAL``, ``PARALLEL``, or
                ``HIERARCHICAL``).
            stream: Whether to stream results during execution. Currently
                a fresh ``StreamerBuffer`` is always created internally.

        Returns:
            The result from ``_execute_task_mode_sync``, typically a string
            containing the execution output or an error message.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._execute_task_mode_sync,
            prompt,
            background,
            process_type,
            StreamerBuffer(),
        )

    def _execute_task_mode_sync(
        self,
        prompt: str,
        background: str | None,
        process_type: ProcessType,
        streamer_buffer: StreamerBuffer,
    ) -> str:
        """Synchronously execute in task mode with dynamic task creation.

        Performs the following steps:

        1. Uses ``TaskCreator.create_tasks_from_prompt`` to decompose the
           prompt into a plan and a list of tasks, streaming progress into
           the provided buffer.
        2. Creates a ``DynamicCortex`` instance with the available agents,
           generated tasks, and specified process type.
        3. Kicks off execution with streaming enabled, writing events to
           the ``StreamerBuffer``, and blocks until the execution thread
           completes.

        If any exception occurs, the error message is logged (when verbose),
        written to the streamer buffer, and the buffer is closed.

        Args:
            prompt: The user prompt to decompose into tasks.
            background: Optional background/approach information providing
                additional context to ``TaskCreator`` for more accurate
                task decomposition.
            process_type: The ``ProcessType`` enum value determining the
                execution strategy (``SEQUENTIAL``, ``PARALLEL``, or
                ``HIERARCHICAL``).
            streamer_buffer: The ``StreamerBuffer`` instance for streaming
                execution events and results back to the caller.

        Returns:
            ``None`` on success (results are streamed to the buffer), or
            an error message string if execution fails.
        """
        try:
            _plan, tasks = self.task_creator.create_tasks_from_prompt(
                prompt=prompt,
                background=background,
                available_agents=self.agents,
                stream=True,
                streamer_buffer=streamer_buffer,
            )
            cortex = DynamicCortex(
                agents=self.agents,
                tasks=tasks,
                llm=self.llm,
                process=process_type,
                verbose=self.verbose,
            )

            cortex.kickoff(use_streaming=True, streamer_buffer=streamer_buffer, log_process=False)[-1].join()

        except Exception as e:
            error_msg = f"Error in task mode execution: {e!s}"
            if self.verbose and self.logger:
                self.logger.error(error_msg)
            if streamer_buffer:
                streamer_buffer.put(error_msg)
                streamer_buffer.close()
            return error_msg

    async def _execute_instruction_mode(
        self,
        prompt: str,
        process_type: ProcessType,
        stream: bool,
    ) -> str | tuple[StreamerBuffer, threading.Thread]:
        """Execute in instruction mode with direct prompt execution (async wrapper).

        Offloads ``_execute_instruction_mode_sync`` to a thread executor via
        ``loop.run_in_executor`` to avoid blocking the async event loop.
        The synchronous method directly executes the prompt through
        ``DynamicCortex`` without task decomposition, using the first
        available agent.

        Args:
            prompt: The user prompt to execute directly without task
                decomposition.
            process_type: The ``ProcessType`` enum value determining the
                execution strategy (``SEQUENTIAL``, ``PARALLEL``, or
                ``HIERARCHICAL``).
            stream: Whether to create a ``StreamerBuffer`` for streaming
                results. If ``False``, ``None`` is passed as the streamer
                buffer to the synchronous method.

        Returns:
            The result from ``_execute_instruction_mode_sync``, typically
            a string containing the execution output or an error message.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._execute_instruction_mode_sync,
            prompt,
            process_type,
            None if not stream else StreamerBuffer(),
        )

    def _execute_instruction_mode_sync(
        self,
        prompt: str,
        process_type: ProcessType,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> str:
        """Synchronously execute in instruction mode without task decomposition.

        Performs the following steps:

        1. Creates a ``DynamicCortex`` instance with available agents, an
           empty task list, and the specified process type.
        2. Calls ``cortex.execute_prompt`` with the first available agent
           (or ``None`` if no agents are registered), streaming results to
           the provided buffer.
        3. Blocks until the execution thread completes.

        If any exception occurs, the error message is logged (when verbose),
        written to the streamer buffer (if provided), and the buffer is closed.

        Args:
            prompt: The user prompt to execute directly without task
                decomposition or agent assignment.
            process_type: The ``ProcessType`` enum value determining the
                execution strategy (``SEQUENTIAL``, ``PARALLEL``, or
                ``HIERARCHICAL``).
            streamer_buffer: Optional ``StreamerBuffer`` instance for
                streaming execution events and results back to the caller.
                If ``None``, results are not streamed.

        Returns:
            ``None`` on success (results are streamed to the buffer), or
            an error message string if execution fails.
        """
        try:
            cortex = DynamicCortex(
                agents=self.agents,
                tasks=[],
                llm=self.llm,
                process=process_type,
                verbose=self.verbose,
            )

            cortex.execute_prompt(
                prompt=prompt,
                agent=self.agents[0] if self.agents else None,
                stream=True,
                streamer_buffer=streamer_buffer,
            )[-1].join()

        except Exception as e:
            error_msg = f"Error in instruction mode execution: {e!s}"
            if self.verbose and self.logger:
                self.logger.error(error_msg)
            if streamer_buffer:
                streamer_buffer.put(error_msg)
                streamer_buffer.close()
            return error_msg
