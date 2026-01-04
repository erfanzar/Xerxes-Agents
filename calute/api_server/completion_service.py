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


"""Chat completion service for handling Calute agent interactions.

This module provides the completion service infrastructure for Calute,
including:
- Non-streaming chat completions with full response generation
- Streaming chat completions with server-sent events
- Request parameter application to agents
- Integration with Calute's agent execution system

The service follows the OpenAI-compatible API format for chat completions
and supports both synchronous and asynchronous response generation.
"""

from __future__ import annotations

import asyncio
import typing
from collections.abc import AsyncIterator

from ..types import Agent, MessagesHistory, StreamChunk
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
    from calute import Calute


class CompletionService:
    """Service for handling chat completions with Calute agents.

    Provides the core functionality for processing chat completion requests,
    including both streaming and non-streaming responses. This service wraps
    the Calute agent execution system and formats responses according to the
    OpenAI-compatible API specification.

    Attributes:
        calute: The Calute instance used for running agent completions.
        can_overide_samplings: Flag indicating whether request parameters
            can override agent sampling settings.
    """

    def __init__(self, calute: Calute, can_overide_samplings: bool = False):
        """Initialize the completion service.

        Args:
            calute: The Calute instance to use for completions.
            can_overide_samplings: Whether to allow request parameters to
                override agent sampling settings (temperature, top_p, etc.).
        """
        self.calute = calute
        self.can_overide_samplings = can_overide_samplings

    def apply_request_parameters(self, agent: Agent, request: ChatCompletionRequest) -> None:
        """Apply request parameters to the agent.

        Conditionally applies sampling parameters from the request to the
        agent if `can_overide_samplings` is enabled. Parameters include
        max_tokens, temperature, top_p, stop sequences, and penalty values.

        Args:
            agent: The agent to modify with request parameters.
            request: The request containing parameters to apply.

        Returns:
            None
        """
        if self.can_overide_samplings:
            if request.max_tokens:
                agent.max_tokens = request.max_tokens
            if request.temperature is not None:
                agent.temperature = request.temperature
            if request.top_p is not None:
                agent.top_p = request.top_p
            if request.stop:
                agent.stop = request.stop
            if request.presence_penalty is not None:
                agent.presence_penalty = request.presence_penalty
            if request.frequency_penalty is not None:
                agent.frequency_penalty = request.frequency_penalty

    async def create_completion(
        self,
        agent: Agent,
        messages: MessagesHistory,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion.

        Executes the Calute agent with the provided messages and returns
        a complete response. The execution is run in a thread executor
        to avoid blocking the event loop.

        Args:
            agent: The agent to use for completion.
            messages: Chat messages history to process.
            request: The original chat completion request for model info.

        Returns:
            ChatCompletionResponse containing the agent's full response,
            usage information, and finish reason.
        """

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.calute.run,
            None,
            None,
            messages,
            agent,
            False,
            True,
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
            usage=UsageInfo(
                completion_tokens=usage_info.completion_tokens,
                completion_tokens_details=usage_info.completion_tokens_details,
                processing_time=usage_info.processing_time,
                prompt_tokens=usage_info.prompt_tokens,
                prompt_tokens_details=usage_info.prompt_tokens_details,
                tokens_per_second=usage_info.tokens_per_second,
                total_tokens=usage_info.total_tokens,
            ),
        )

    async def create_streaming_completion(
        self,
        agent: Agent,
        messages: MessagesHistory,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Create a streaming chat completion.

        Executes the Calute agent in streaming mode, yielding response
        chunks as server-sent events (SSE). Each chunk is formatted as
        a ChatCompletionStreamResponse and encoded for SSE transmission.

        Args:
            agent: The agent to use for completion.
            messages: Chat messages history to process.
            request: The original chat completion request for model info.

        Yields:
            Server-sent events containing streaming response chunks in SSE
            format. Each event contains delta content and usage information.
            The stream ends with a final chunk indicating completion and a
            "[DONE]" message.
        """

        usage_info = None
        for chunk in self.calute.run(
            messages=messages,
            agent_id=agent,
            stream=True,
            apply_functions=True,
        ):
            if isinstance(chunk, StreamChunk):
                usage_info = chunk.chunk.usage

                stream_response = ChatCompletionStreamResponse(
                    model=request.model,
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant", content=chunk.content),
                            finish_reason=None,
                        )
                    ],
                    usage=UsageInfo(
                        completion_tokens=usage_info.completion_tokens,
                        completion_tokens_details=usage_info.completion_tokens_details,
                        processing_time=usage_info.processing_time,
                        prompt_tokens=usage_info.prompt_tokens,
                        prompt_tokens_details=usage_info.prompt_tokens_details,
                        tokens_per_second=usage_info.tokens_per_second,
                        total_tokens=usage_info.total_tokens,
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
            usage=UsageInfo(
                completion_tokens=usage_info.completion_tokens,
                completion_tokens_details=usage_info.completion_tokens_details,
                processing_time=usage_info.processing_time,
                prompt_tokens=usage_info.prompt_tokens,
                prompt_tokens_details=usage_info.prompt_tokens_details,
                tokens_per_second=usage_info.tokens_per_second,
                total_tokens=usage_info.total_tokens,
            ),
        )
        yield f"data: {final_response.model_dump_json(exclude_unset=True, exclude_none=True)}\n\n".encode()
        yield "data: [DONE]\n\n"
