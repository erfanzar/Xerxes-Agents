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
"""Standard completion service for the Xerxes API server.

This module provides :class:`CompletionService`, which bridges Xerxes agent
execution to OpenAI-compatible chat completion requests, supporting both
synchronous and streaming responses.
"""

from __future__ import annotations

import asyncio
import typing
from collections.abc import AsyncIterator

from ..types import Agent, MessagesHistory, StreamChunk
from ..types.function_execution_types import ResponseResult
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
    from xerxes import Xerxes


class CompletionService:
    """Wraps a Xerxes instance to provide OpenAI-compatible completions."""

    def __init__(self, xerxes: Xerxes, can_overide_samplings: bool = False):
        """Initialize the completion service.

        Args:
            xerxes (Xerxes): IN: The Xerxes runtime instance. OUT: Used to
                execute agent runs.
            can_overide_samplings (bool): IN: Whether request sampling parameters
                may override agent defaults. OUT: Stored and checked during request
                processing.
        """
        self.xerxes = xerxes
        self.can_overide_samplings = can_overide_samplings

    def apply_request_parameters(self, agent: Agent, request: ChatCompletionRequest) -> Agent:
        """Override agent sampling parameters from a chat completion request.

        Args:
            agent (Agent): IN: Base agent configuration. OUT: Deep-copied and
                selectively mutated with request parameters.
            request (ChatCompletionRequest): IN: Request containing optional
                sampling overrides. OUT: Read for parameter values.

        Returns:
            Agent: OUT: Configured agent with applied overrides.
        """
        configured_agent = agent.model_copy(deep=True)
        if self.can_overide_samplings:
            if request.max_tokens:
                configured_agent.max_tokens = request.max_tokens
            if request.temperature is not None:
                configured_agent.temperature = request.temperature
            if request.top_p is not None:
                configured_agent.top_p = request.top_p
            if request.top_k is not None:
                configured_agent.top_k = request.top_k
            if request.min_p is not None:
                configured_agent.min_p = request.min_p
            if request.stop:
                configured_agent.stop = request.stop
            if request.presence_penalty is not None:
                configured_agent.presence_penalty = request.presence_penalty
            if request.frequency_penalty is not None:
                configured_agent.frequency_penalty = request.frequency_penalty
            if request.repetition_penalty is not None:
                configured_agent.repetition_penalty = request.repetition_penalty
        return configured_agent

    async def create_completion(
        self,
        agent: Agent,
        messages: MessagesHistory,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion.

        Args:
            agent (Agent): IN: Configured agent to run. OUT: Passed to Xerxes.
            messages (MessagesHistory): IN: Conversation messages. OUT: Passed to
                Xerxes as the message history.
            request (ChatCompletionRequest): IN: Request metadata. OUT: Used for
                model name and to construct the response.

        Returns:
            ChatCompletionResponse: OUT: The completed assistant message with usage.
        """
        loop = asyncio.get_event_loop()
        response = typing.cast(
            ResponseResult,
            await loop.run_in_executor(
                None,
                self.xerxes.run,
                None,
                None,
                messages,
                agent,
                False,
                True,
            ),
        )
        usage_info = response.response.usage
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response.content or ""),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo.model_construct(
                completion_tokens=getattr(usage_info, "completion_tokens", 0) or 0,
                completion_tokens_details=getattr(usage_info, "completion_tokens_details", None),
                processing_time=getattr(usage_info, "processing_time", 0.0) or 0.0,
                prompt_tokens=getattr(usage_info, "prompt_tokens", 0) or 0,
                prompt_tokens_details=getattr(usage_info, "prompt_tokens_details", None),
                tokens_per_second=getattr(usage_info, "tokens_per_second", 0.0) or 0.0,
                total_tokens=getattr(usage_info, "total_tokens", 0) or 0,
            ),
        )

    async def create_streaming_completion(
        self,
        agent: Agent,
        messages: MessagesHistory,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str | bytes]:
        """Create a streaming chat completion.

        Args:
            agent (Agent): IN: Configured agent to run. OUT: Passed to Xerxes as
                the agent identifier.
            messages (MessagesHistory): IN: Conversation messages. OUT: Passed to
                Xerxes.
            request (ChatCompletionRequest): IN: Request metadata. OUT: Used for
                model name in streamed chunks.

        Yields:
            str | bytes: OUT: SSE-formatted JSON chunks and the final ``[DONE]`` marker.
        """
        usage_info = None
        stream_result = self.xerxes.run(
            messages=messages,
            agent_id=agent,
            stream=True,
            apply_functions=True,
        )
        if isinstance(stream_result, ResponseResult):
            return
        for chunk in stream_result:
            if isinstance(chunk, StreamChunk):
                usage_info = getattr(chunk.chunk, "usage", None) if chunk.chunk is not None else None

                stream_response = ChatCompletionStreamResponse(
                    model=request.model,
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant", content=chunk.content),
                            finish_reason=None,
                        )
                    ],
                    usage=UsageInfo.model_construct(
                        completion_tokens=getattr(usage_info, "completion_tokens", 0) or 0,
                        completion_tokens_details=getattr(usage_info, "completion_tokens_details", None),
                        processing_time=getattr(usage_info, "processing_time", 0.0) or 0.0,
                        prompt_tokens=getattr(usage_info, "prompt_tokens", 0) or 0,
                        prompt_tokens_details=getattr(usage_info, "prompt_tokens_details", None),
                        tokens_per_second=getattr(usage_info, "tokens_per_second", 0.0) or 0.0,
                        total_tokens=getattr(usage_info, "total_tokens", 0) or 0,
                    ),
                )
                yield f"data: {stream_response.model_dump_json(exclude_unset=True, exclude_none=True)}\n\n".encode()
                await asyncio.sleep(0)

        final_response = ChatCompletionStreamResponse(
            model=request.model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo.model_construct(
                completion_tokens=getattr(usage_info, "completion_tokens", 0) or 0,
                completion_tokens_details=getattr(usage_info, "completion_tokens_details", None),
                processing_time=getattr(usage_info, "processing_time", 0.0) or 0.0,
                prompt_tokens=getattr(usage_info, "prompt_tokens", 0) or 0,
                prompt_tokens_details=getattr(usage_info, "prompt_tokens_details", None),
                tokens_per_second=getattr(usage_info, "tokens_per_second", 0.0) or 0.0,
                total_tokens=getattr(usage_info, "total_tokens", 0) or 0,
            ),
        )
        yield f"data: {final_response.model_dump_json(exclude_unset=True, exclude_none=True)}\n\n".encode()
        yield "data: [DONE]\n\n"
