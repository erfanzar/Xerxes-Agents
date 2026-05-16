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
"""Cortex-backed completion service for the API server.

Translates OpenAI chat-completion requests into Cortex executions
in either task mode (plan + execute, optionally background) or
instruction mode (single-agent dispatch). Process type
(sequential / parallel / hierarchical) is derived from the model
name and request metadata.

Streaming runs Cortex on a background thread and bridges to SSE via
:class:`StreamerBuffer`; each Cortex streaming event is mapped onto a
chat-completion delta with optional ``metadata`` describing the event
kind (function detection, function execution, agent switch, etc.).
"""

from __future__ import annotations

import asyncio
import json
import threading
import typing
from collections.abc import AsyncIterator

from xerxes.types.function_execution_types import (
    AgentSwitch,
    Completion,
    FunctionCallsExtracted,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    ReinvokeSignal,
    StreamChunk,
    StreamingResponseType,
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


class CortexCompletionService:
    """OpenAI completion backend powered by the Cortex framework.

    Wires an LLM, an agent roster, and a :class:`TaskCreator`. The
    appropriate Cortex process is built per request based on the
    model name and metadata (``cortex-task`` enters task mode;
    ``parallel`` / ``hierarchical`` set the process type).
    """

    def __init__(
        self,
        llm: BaseLLM,
        agents: list[CortexAgent] | None = None,
        use_universal_agent: bool = True,
        verbose: bool = True,
    ):
        """Configure the LLM, agent roster, and task creator.

        Args:
            llm: backing model used by agents and the task creator.
            agents: explicit agent list (mutated to include the universal
                agent when requested).
            use_universal_agent: append a :class:`UniversalAgent` capable
                of delegation; ``False`` keeps ``agents`` exactly as given.
            verbose: enable verbose Cortex/console logging.
        """
        self.llm = llm
        self.verbose = verbose
        self.logger = get_logger() if verbose else None

        self.agents = agents or []
        if use_universal_agent:
            self.universal_agent: UniversalAgent | None = UniversalAgent(
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
        """Return ``{task_mode, process_type, background}`` derived from ``request``.

        Inspects the model name (``"task"``, ``"parallel"``,
        ``"hierarchical"``) and an optional ``metadata`` dict that may
        override either flag or provide a ``background`` string.
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
        """Return the most recent user message, or all messages joined."""
        for msg in reversed(messages.messages):
            if hasattr(msg, "role") and msg.role == "user":
                content = msg.content
                return content if isinstance(content, str) else str(content)
            elif msg.__class__.__name__ == "UserMessage":
                content = msg.content
                return content if isinstance(content, str) else str(content)

        return "\n".join(str(msg.content) for msg in messages.messages)

    async def create_completion(
        self,
        messages: MessagesHistory,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Run a Cortex flow and wrap the result in OpenAI's response shape.

        Token usage is approximated by whitespace splitting since
        Cortex doesn't surface provider-side usage counters.
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
        """Stream a Cortex flow as SSE chat completion deltas.

        Cortex runs synchronously on a background thread; this method
        consumes its :class:`StreamerBuffer` and converts each event
        into a ``ChatCompletionStreamResponse``. Each non-text Cortex
        event (function detection, agent switch, ...) is rendered as
        Markdown content plus a ``metadata`` field describing the
        event class. The generator terminates with a stop delta and
        ``data: [DONE]``.
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
            metadata: dict[str, typing.Any] = {}

            if isinstance(chunk, StreamChunk):
                if chunk.content:
                    content = chunk.content
                if hasattr(chunk, "streaming_tool_calls") and chunk.streaming_tool_calls:
                    tool_info: list[dict[str, typing.Any]] = []
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
                    setattr(stream_response, "metadata", metadata)

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
        """Run task-mode synchronously in an executor; return its result."""
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
        """Plan + execute via :class:`TaskCreator` and :class:`DynamicCortex`.

        Returns ``""`` on success or an error message. The caller is
        expected to drain ``streamer_buffer`` for streaming output.
        """
        try:
            _plan, tasks = self.task_creator.create_tasks_from_prompt(
                prompt=prompt,
                background=background,
                available_agents=self.agents,
                stream=True,
                streamer_buffer=streamer_buffer,
            )
            tasks = tasks or []
            cortex = DynamicCortex(
                agents=self.agents,
                tasks=tasks,
                llm=self.llm,
                process=process_type,
                verbose=self.verbose,
            )

            result = cortex.kickoff(use_streaming=True, streamer_buffer=streamer_buffer, log_process=False)
            if isinstance(result, tuple):
                result[-1].join()
            return ""

        except Exception as e:
            error_msg = f"Error in task mode execution: {e!s}"
            if self.verbose and self.logger:
                self.logger.error(error_msg)
            if streamer_buffer:
                streamer_buffer.put(typing.cast(StreamingResponseType, error_msg))
                streamer_buffer.close()
            return error_msg

    async def _execute_instruction_mode(
        self,
        prompt: str,
        process_type: ProcessType,
        stream: bool,
    ) -> str | tuple[StreamerBuffer, threading.Thread]:
        """Run instruction-mode synchronously in an executor; return its result."""
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
        """Dispatch ``prompt`` to the first available Cortex agent.

        Returns ``""`` on success or an error message. ``streamer_buffer``
        receives events when streaming.
        """
        try:
            cortex = DynamicCortex(
                agents=self.agents,
                tasks=[],
                llm=self.llm,
                process=process_type,
                verbose=self.verbose,
            )

            result = cortex.execute_prompt(
                prompt=prompt,
                agent=self.agents[0] if self.agents else None,
                stream=True,
                streamer_buffer=streamer_buffer,
            )
            if isinstance(result, tuple):
                result[-1].join()
            return ""

        except Exception as e:
            error_msg = f"Error in instruction mode execution: {e!s}"
            if self.verbose and self.logger:
                self.logger.error(error_msg)
            if streamer_buffer:
                streamer_buffer.put(typing.cast(StreamingResponseType, error_msg))
                streamer_buffer.close()
            return error_msg
