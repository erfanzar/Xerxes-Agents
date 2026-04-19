# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
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
    >>> from xerxes.api_server.models import ModelInfo, ModelsResponse
    >>> model = ModelInfo(id="my-agent", created=1234567890)
    >>> response = ModelsResponse(data=[model])
"""

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Information about an available model/agent.

    Represents metadata for a single model or agent registered
    with the Xerxes API server. Follows the OpenAI model object
    specification for compatibility with the ``/v1/models`` endpoint.

    Attributes:
        id: Unique identifier for the model/agent. This is the value
            clients use in the ``model`` field of chat completion requests.
        object: Object type, always ``"model"`` for OpenAI compatibility.
        created: Unix timestamp (seconds since epoch) indicating when the
            model entry was created.
        owned_by: Owner identifier for the model. Defaults to ``"xerxes"``.

    Example:
        >>> from xerxes.api_server.models import ModelInfo
        >>> info = ModelInfo(id="my-agent", created=1700000000)
        >>> info.object
        'model'
        >>> info.owned_by
        'xerxes'
    """

    id: str
    object: str = "model"
    created: int
    owned_by: str = "xerxes"


class ModelsResponse(BaseModel):
    """Response containing a list of available models/agents.

    Standard response format for the ``/v1/models`` endpoint,
    providing a list of all registered agents. Follows the
    OpenAI list response specification.

    Attributes:
        object: Object type, always ``"list"`` for OpenAI compatibility.
        data: List of ``ModelInfo`` objects representing all available
            agents and models registered with the server.

    Example:
        >>> from xerxes.api_server.models import ModelInfo, ModelsResponse
        >>> model = ModelInfo(id="assistant", created=1700000000)
        >>> response = ModelsResponse(data=[model])
        >>> response.object
        'list'
        >>> len(response.data)
        1
    """

    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response model.

    Response format for the ``/health`` endpoint, providing basic
    health status information about the API server and its
    registered agents. Used by load balancers and monitoring systems
    to verify server availability.

    Attributes:
        status: Health status string (e.g., ``"healthy"``, ``"degraded"``).
        agents: Number of registered agents currently available for
            serving chat completion requests.

    Example:
        >>> from xerxes.api_server.models import HealthResponse
        >>> health = HealthResponse(status="healthy", agents=3)
        >>> health.status
        'healthy'
    """

    status: str
    agents: int
