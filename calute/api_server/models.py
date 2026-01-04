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


"""Request and response models for the OpenAI-compatible API.

This module provides Pydantic models for the API server's request
and response handling. It includes:
- Model information and listing responses
- Health check response models
- OpenAI-compatible data structures

All models follow the OpenAI API specification for compatibility
with existing OpenAI client libraries and tools.

Example:
    >>> from calute.api_server.models import ModelInfo, ModelsResponse
    >>> model = ModelInfo(id="my-agent", created=1234567890)
    >>> response = ModelsResponse(data=[model])
"""

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Information about an available model/agent.

    Represents metadata for a single model or agent registered
    with the Calute API server. Follows the OpenAI model object
    specification for compatibility.

    Attributes:
        id: Unique identifier for the model/agent.
        object: Object type, always "model" for OpenAI compatibility.
        created: Unix timestamp when model was created.
        owned_by: Owner of the model (always "calute").
    """

    id: str
    object: str = "model"
    created: int
    owned_by: str = "calute"


class ModelsResponse(BaseModel):
    """Response containing list of available models/agents.

    Standard response format for the /v1/models endpoint,
    providing a list of all registered agents. Follows the
    OpenAI list response specification.

    Attributes:
        object: Object type, always "list" for OpenAI compatibility.
        data: List of ModelInfo objects representing available agents.
    """

    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response model.

    Response format for the /health endpoint, providing basic
    health status information about the API server and its
    registered agents.

    Attributes:
        status: Health status string (e.g., "healthy", "degraded").
        agents: Number of registered agents currently available.
    """

    status: str
    agents: int
