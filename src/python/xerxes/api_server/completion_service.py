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
"""Single-agent completion service for the Xerxes API server.

:class:`CompletionService` adapts the synchronous :meth:`Xerxes.run`
entry point to the FastAPI request handlers. ``create_completion``
returns a :class:`ChatCompletionResponse`; ``create_streaming_completion``
yields SSE ``data:`` frames followed by the OpenAI ``[DONE]`` marker.
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
    """Adapter from a :class:`Xerxes` instance to OpenAI completions."""

    def __init__(self, xerxes: Xerxes, can_overide_samplings: bool = False):
        """Bind to ``xerxes`` and remember whether requests may override sampling.

        When ``can_overide_samplings`` is ``False`` (the default), the
        agent's configured sampling parameters always win regardless of
        what the chat completion request supplies.
        """
        self.xerxes = xerxes
        self.can_overide_samplings = can_overide_samplings

    def apply_request_parameters(self, agent: Agent, request: ChatCompletionRequest) -> Agent:
        """Return a deep copy of ``agent`` with request overrides applied.

        Overrides only take effect when ``can_overide_samplings`` is
        ``True``. Each sampling parameter is copied only when present
        on the request so missing values keep the agent's defaults.
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
        """Run ``agent`` once and wrap the result in OpenAI's response shape.

        ``Xerxes.run`` is synchronous, so it executes in the default
        thread pool to keep the event loop unblocked. The returned
        usage info copies whatever counters the provider supplied,
        defaulting missing values to ``0`` / ``0.0``.
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
        """Yield SSE-formatted streaming chunks for one chat completion.

        Each delta is emitted as a ``data: {json}`` line followed by
        ``\\n\\n``. A final ``finish_reason="stop"`` chunk and the
        OpenAI ``data: [DONE]`` sentinel terminate the stream. The
        function yields nothing when :meth:`Xerxes.run` short-circuits
        to a :class:`ResponseResult` (no streaming available).
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
