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

"""Chat completion service for handling Calute agent interactions."""

from __future__ import annotations

import time
import typing
import uuid
from collections.abc import AsyncIterator

from ..types import Agent, MessagesHistory, StreamChunk
from ..types.chat_completion_types import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    DeltaMessage,
    RequestChatMessage,
    UsageInfo,
)
from .models import ChatCompletionRequest

if typing.TYPE_CHECKING:
    from calute import Calute


class CompletionService:
    """Service for handling chat completions with Calute agents."""

    def __init__(self, calute: Calute, can_overide_samplings: bool = False):
        """Initialize the completion service.

        Args:
            calute: The Calute instance to use for completions
        """
        self.calute = calute
        self.can_overide_samplings = can_overide_samplings

    def apply_request_parameters(self, agent: Agent, request: ChatCompletionRequest) -> None:
        """Apply request parameters to the agent.

        Args:
            agent: The agent to modify
            request: The request containing parameters to apply
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

        Args:
            agent: The agent to use for completion
            messages: Chat messages history
            request: The original chat completion request

        Returns:
            ChatCompletionResponse with the agent's response
        """
        response = await self.calute.create_response(
            messages=messages,
            agent_id=agent,
            stream=False,
            apply_functions=True,
        )

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=RequestChatMessage(role="assistant", content=response.content or ""),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )

    async def create_streaming_completion(
        self, agent: Agent, messages: MessagesHistory, request: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        """Create a streaming chat completion.

        Args:
            agent: The agent to use for completion
            messages: Chat messages history
            request: The original chat completion request

        Yields:
            Server-sent events containing streaming response chunks
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())

        response_stream = await self.calute.create_response(
            messages=messages,
            agent_id=agent,
            stream=True,
            apply_functions=True,
        )

        usage_info = None
        async for chunk in response_stream:
            if isinstance(chunk, StreamChunk):
                usage_info = chunk.chunk.usage

                stream_response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant", content=chunk.content),
                            finish_reason=None,
                        )
                    ],
                    usage=UsageInfo(
                        completion_tokens=getattr(usage_info, "completion_tokens", 0),
                        completion_tokens_details=getattr(usage_info, "completion_tokens_details", None),
                        processing_time=getattr(usage_info, "processing_time", 0),
                        prompt_tokens=getattr(usage_info, "prompt_tokens", 0),
                        prompt_tokens_details=getattr(usage_info, "prompt_tokens_details", None),
                        tokens_per_second=getattr(usage_info, "tokens_per_second", 0),
                        total_tokens=getattr(usage_info, "total_tokens", 0),
                    ),
                )
                yield f"data: {stream_response.model_dump_json()}\n\n"

        # Send final chunk
        final_response = ChatCompletionStreamResponse(
            id=completion_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                completion_tokens=getattr(usage_info, "completion_tokens", 0) if usage_info else 0,
                completion_tokens_details=getattr(usage_info, "completion_tokens_details", None) if usage_info else None,
                processing_time=getattr(usage_info, "processing_time", 0) if usage_info else 0,
                prompt_tokens=getattr(usage_info, "prompt_tokens", 0) if usage_info else 0,
                prompt_tokens_details=getattr(usage_info, "prompt_tokens_details", None) if usage_info else None,
                tokens_per_second=getattr(usage_info, "tokens_per_second", 0) if usage_info else 0,
                total_tokens=getattr(usage_info, "total_tokens", 0) if usage_info else 0,
            ),
        )
        yield f"data: {final_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
