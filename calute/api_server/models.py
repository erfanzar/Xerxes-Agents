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

"""Request and response models for the OpenAI-compatible API."""

from pydantic import BaseModel

from calute.types.chat_completion_types import RequestChatMessage


class ChatCompletionRequest(BaseModel):
    """Request model for OpenAI-compatible chat completions.

    Attributes:
        model: The model/agent ID to use for completion
        messages: List of chat messages in the conversation
        max_tokens: Maximum tokens to generate (default: 2048)
        temperature: Sampling temperature 0-2 (default: 0.7)
        top_p: Nucleus sampling parameter (default: 0.95)
        n: Number of completions to generate (default: 1)
        stream: Whether to stream the response (default: False)
        stop: Stop sequences to end generation
        presence_penalty: Penalty for new tokens (-2 to 2, default: 0.0)
        frequency_penalty: Penalty for repeated tokens (-2 to 2, default: 0.0)
    """

    model: str
    messages: list[RequestChatMessage]
    max_tokens: int | None = 2048
    temperature: float | None = 0.7
    top_p: float | None = 0.95
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0


class ModelInfo(BaseModel):
    """Information about an available model/agent.

    Attributes:
        id: Unique identifier for the model/agent
        object: Always "model" for OpenAI compatibility
        created: Unix timestamp when model was created
        owned_by: Owner of the model (always "calute")
    """

    id: str
    object: str = "model"
    created: int
    owned_by: str = "calute"


class ModelsResponse(BaseModel):
    """Response containing list of available models/agents.

    Attributes:
        object: Always "list" for OpenAI compatibility
        data: List of ModelInfo objects
    """

    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response model.

    Attributes:
        status: Health status string
        agents: Number of registered agents
    """

    status: str
    agents: int
