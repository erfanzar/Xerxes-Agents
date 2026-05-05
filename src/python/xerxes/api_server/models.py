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
"""Pydantic models for API server responses.

This module defines lightweight response schemas for the ``/models`` and
``/health`` endpoints.
"""

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Represents a single model available on the API server.

    Attributes:
        id (str): Unique model identifier.
        object (str): Object type (default ``"model"``).
        created (int): Unix timestamp of model creation/registration.
        owned_by (str): Owner identifier (default ``"xerxes"``).
    """

    id: str
    object: str = "model"
    created: int
    owned_by: str = "xerxes"


class ModelsResponse(BaseModel):
    """Response wrapper for the ``/v1/models`` endpoint.

    Attributes:
        object (str): Object type (default ``"list"``).
        data (list[ModelInfo]): List of available models.
    """

    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Response for the ``/health`` endpoint.

    Attributes:
        status (str): Server health status.
        agents (int): Number of registered agents.
    """

    status: str
    agents: int
