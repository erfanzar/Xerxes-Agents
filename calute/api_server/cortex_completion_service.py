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


"""Cortex completion service for handling multi-agent orchestration via API."""

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

from ..cortex import CortexAgent, DynamicCortex, TaskCreator, UniversalAgent
from ..cortex.enums import ProcessType
from ..loggings import get_logger
from ..streamer_buffer import StreamerBuffer
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
    """Service for handling Cortex-based chat completions with multi-agent orchestration."""

    def __init__(
        self,
        llm: BaseLLM,
        agents: list[CortexAgent] | None = None,
        use_universal_agent: bool = True,
        verbose: bool = True,
    ):
        """Initialize the Cortex completion service.

        Args:
            llm: The LLM instance to use for agents
            agents: Optional list of specialized agents
            use_universal_agent: Whether to include UniversalAgent as fallback
            verbose: Whether to enable verbose logging
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
        """Extract task configuration from request.

        Args:
            request: The chat completion request

        Returns:
            Dictionary with task configuration including:
            - task_mode: Whether to use task mode
            - process_type: Type of process (sequential, parallel, etc.)
            - background: Optional background/approach for task creation
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
        """Extract the latest user prompt from messages.

        Args:
            messages: The message history

        Returns:
            The latest user prompt as a string
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

        Args:
            messages: Chat messages history
            request: The original chat completion request

        Returns:
            ChatCompletionResponse with the Cortex execution result
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
        """Create a streaming Cortex completion.

        Args:
            messages: Chat messages history
            request: The original chat completion request

        Yields:
            Server-sent events containing streaming response chunks
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
        """Execute in task mode with dynamic task creation.

        Args:
            prompt: The user prompt to create tasks for
            background: Optional background/approach
            process_type: The process type to use
            stream: Whether to stream results

        Returns:
            Execution result or streaming tuple
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
        """Synchronously execute in task mode.

        Args:
            prompt: The user prompt to create tasks for
            background: Optional background/approach
            process_type: The process type to use
            streamer_buffer: Optional buffer for streaming

        Returns:
            Execution result as string
        """
        try:
            plan, tasks = self.task_creator.create_tasks_from_prompt(
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
        """Execute in instruction mode with direct prompt execution.

        Args:
            prompt: The user prompt to execute directly
            process_type: The process type to use
            stream: Whether to stream results

        Returns:
            Execution result or streaming tuple
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
        """Synchronously execute in instruction mode.

        Args:
            prompt: The user prompt to execute directly
            process_type: The process type to use
            streamer_buffer: Optional buffer for streaming

        Returns:
            Execution result as string
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
