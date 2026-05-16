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
"""Pydantic response schemas for ``/v1/models`` and ``/health``.

The chat-completion request and response schemas live in
:mod:`xerxes.types.oai_protocols`; only the small list/health
envelopes are defined here.
"""

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """One row in the ``/v1/models`` list (OpenAI-compatible shape).

    Attributes:
        id: unique model identifier (agent id or Cortex variant name).
        object: always ``"model"``.
        created: unix epoch second when the agent/variant was registered.
        owned_by: namespace/owner; defaults to ``"xerxes"``.
    """

    id: str
    object: str = "model"
    created: int
    owned_by: str = "xerxes"


class ModelsResponse(BaseModel):
    """Envelope returned by ``GET /v1/models``.

    Attributes:
        object: always ``"list"``.
        data: one :class:`ModelInfo` per registered agent or variant.
    """

    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Body of ``GET /health``.

    Attributes:
        status: liveness sentinel (typically ``"ok"``).
        agents: number of registered agents at the moment of the check.
    """

    status: str
    agents: int
