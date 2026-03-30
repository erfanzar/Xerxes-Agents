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

    The service is used internally by the ``ChatRouter`` and
    ``UnifiedChatRouter`` to delegate completion logic away from the
    HTTP routing layer.

    Attributes:
        calute: The ``Calute`` instance used for running agent completions.
        can_overide_samplings: Flag indicating whether request parameters
            can override agent sampling settings. When ``True``, parameters
            like ``temperature``, ``top_p``, ``max_tokens``, ``stop``,
            ``presence_penalty``, ``frequency_penalty``, ``repetition_penalty``,
            ``top_k``, and ``min_p`` from the request will be applied to the
            agent before execution.

    Example:
        >>> from calute.api_server.completion_service import CompletionService
        >>> service = CompletionService(calute_instance, can_overide_samplings=True)
    """

    def __init__(self, calute: Calute, can_overide_samplings: bool = False):
        """Initialize the completion service.

        Args:
            calute: The ``Calute`` instance to use for running agent
                completions. Must be fully initialized with a client.
            can_overide_samplings: Whether to allow request parameters to
                override agent sampling settings (temperature, top_p, etc.).
                Defaults to ``False``.
        """
        self.calute = calute
        self.can_overide_samplings = can_overide_samplings

    def apply_request_parameters(self, agent: Agent, request: ChatCompletionRequest) -> None:
        """Apply sampling parameters from the request to the agent.

        Conditionally transfers sampling parameters from the incoming request
        to the agent configuration. This only takes effect when
        ``can_overide_samplings`` is ``True``. Each parameter is applied only
        if it is explicitly set (not ``None``) in the request.

        The following parameters are supported:
            - ``max_tokens``: Maximum number of tokens to generate.
            - ``temperature``: Sampling temperature.
            - ``top_p``: Nucleus sampling threshold.
            - ``top_k``: Top-k sampling parameter.
            - ``min_p``: Minimum probability threshold.
            - ``stop``: Stop sequences for generation.
            - ``presence_penalty``: Presence penalty value.
            - ``frequency_penalty``: Frequency penalty value.
            - ``repetition_penalty``: Repetition penalty value.

        Args:
            agent: The ``Agent`` instance whose sampling settings will be
                modified in-place.
            request: The ``ChatCompletionRequest`` containing the sampling
                parameters to apply.
        """
        if self.can_overide_samplings:
            if request.max_tokens:
                agent.max_tokens = request.max_tokens
            if request.temperature is not None:
                agent.temperature = request.temperature
            if request.top_p is not None:
                agent.top_p = request.top_p
            if request.top_k is not None:
                agent.top_k = request.top_k
            if request.min_p is not None:
                agent.min_p = request.min_p
            if request.stop:
                agent.stop = request.stop
            if request.presence_penalty is not None:
                agent.presence_penalty = request.presence_penalty
            if request.frequency_penalty is not None:
                agent.frequency_penalty = request.frequency_penalty
            if request.repetition_penalty is not None:
                agent.repetition_penalty = request.repetition_penalty

    async def create_completion(
        self,
        agent: Agent,
        messages: MessagesHistory,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion.

        Executes the Calute agent with the provided messages and returns
        a complete response. The synchronous ``calute.run()`` call is
        offloaded to a thread executor via ``loop.run_in_executor`` to
        avoid blocking the async event loop.

        The response includes the full generated text, usage statistics
        (prompt tokens, completion tokens, processing time, etc.), and
        a finish reason of ``"stop"``.

        Args:
            agent: The ``Agent`` instance to use for generating the
                completion.
            messages: The ``MessagesHistory`` containing the conversation
                context to process.
            request: The original ``ChatCompletionRequest``, used to
                extract the model name for the response object.

        Returns:
            A ``ChatCompletionResponse`` containing a single choice with
            the assistant's full response message, usage information
            (token counts, processing time, tokens per second), and
            finish reason ``"stop"``.
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
        """Create a streaming chat completion using server-sent events.

        Executes the Calute agent in streaming mode and yields response
        chunks as SSE-formatted strings. Each chunk is serialized as a
        ``ChatCompletionStreamResponse`` JSON object prefixed with
        ``"data: "`` and followed by double newlines, conforming to the
        SSE protocol.

        The method yields an ``asyncio.sleep(0)`` after each chunk to
        allow the event loop to process other tasks, enabling cooperative
        multitasking during long-running generations.

        After all content chunks have been yielded, a final chunk with
        an empty delta and ``finish_reason="stop"`` is emitted, followed
        by a ``"data: [DONE]"`` sentinel to signal stream completion.

        Args:
            agent: The ``Agent`` instance to use for generating the
                streaming completion.
            messages: The ``MessagesHistory`` containing the conversation
                context to process.
            request: The original ``ChatCompletionRequest``, used to
                extract the model name for each streamed response chunk.

        Yields:
            Byte-encoded and plain string SSE events. Each content event
            is a ``bytes`` object containing ``"data: {json}\\n\\n"``.
            The final ``"[DONE]"`` event is a plain string.
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
