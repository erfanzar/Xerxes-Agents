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
"""FastAPI routers for the Xerxes API server.

Provides individual routers for the standard chat backend, Cortex
chat backend, models listing, and health check, plus a
:class:`UnifiedChatRouter` that picks the right backend based on the
``model`` field of the incoming request.

Endpoints:

* ``POST /v1/chat/completions`` — chat completions (sync or SSE stream).
* ``GET  /v1/models`` — list available models.
* ``GET  /health`` — server health and registered agent count.
"""

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from xerxes.types import Agent

from ..types.oai_protocols import ChatCompletionRequest
from .completion_service import CompletionService
from .converters import MessageConverter
from .cortex_completion_service import CortexCompletionService
from .models import HealthResponse, ModelInfo, ModelsResponse


class ChatRouter:
    """Single-agent ``/v1/chat/completions`` router.

    Looks up the requested model in ``agents`` and dispatches to
    :class:`CompletionService`. Returns either a ``ChatCompletionResponse``
    or an SSE ``StreamingResponse`` depending on ``request.stream``.
    """

    def __init__(self, agents: dict[str, Agent], completion_service: CompletionService):
        """Bind the router to the agent registry and completion backend."""
        self.agents = agents
        self.completion_service = completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Attach the chat completions endpoint to the FastAPI router."""

        @self.router.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Run one chat completion for the requested agent model.

            Raises HTTP 404 if the model is not registered and HTTP 500
            on any other exception. Streaming responses use
            ``text/event-stream``.
            """
            try:
                agent = self.agents.get(request.model)
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

                messages_history = MessageConverter.convert_openai_to_xerxes(request.messages)

                agent = self.completion_service.apply_request_parameters(agent, request)

                if request.stream:
                    return StreamingResponse(
                        self.completion_service.create_streaming_completion(agent, messages_history, request),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                else:
                    return await self.completion_service.create_completion(agent, messages_history, request)

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e


class ModelsRouter:
    """Router that serves ``GET /v1/models``."""

    def __init__(self, agents: dict[str, Agent]):
        """Bind to the agent registry the response is derived from."""
        self.agents = agents
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Attach the models listing endpoint."""

        @self.router.get("/v1/models")
        async def list_models() -> ModelsResponse:
            """Return one :class:`ModelInfo` per registered agent."""
            models = []
            for agent_id, _ in self.agents.items():
                models.append(ModelInfo(id=agent_id, created=int(time.time())))
            return ModelsResponse(data=models)


class HealthRouter:
    """Router that serves ``GET /health``."""

    def __init__(self, agents: dict[str, Agent]):
        """Bind to the agent registry whose size populates ``agents``."""
        self.agents = agents
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Attach the health endpoint."""

        @self.router.get("/health")
        async def health_check() -> HealthResponse:
            """Return ``status="healthy"`` and the current agent count."""
            return HealthResponse(status="healthy", agents=len(self.agents))


class CortexChatRouter:
    """Cortex-only ``/v1/chat/completions`` router.

    Skips agent-id lookup; the Cortex service resolves the requested
    variant from the model name.
    """

    def __init__(self, cortex_completion_service: CortexCompletionService):
        """Bind the router to the Cortex completion backend."""
        self.cortex_completion_service = cortex_completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Attach the Cortex chat completions endpoint."""

        @self.router.post("/v1/chat/completions")
        async def cortex_chat_completions(request: ChatCompletionRequest):
            """Run one Cortex chat completion (streaming or sync)."""
            try:
                messages_history = MessageConverter.convert_openai_to_xerxes(request.messages)

                if request.stream:
                    return StreamingResponse(
                        self.cortex_completion_service.create_streaming_completion(messages_history, request),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                else:
                    return await self.cortex_completion_service.create_completion(messages_history, request)

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e


class UnifiedChatRouter:
    """Dispatches ``/v1/chat/completions`` between standard and Cortex backends.

    Routing is decided by the ``model`` field: a name containing
    ``"cortex"`` goes to :class:`CortexCompletionService`; anything
    else is looked up in ``agents`` and run through
    :class:`CompletionService`.
    """

    def __init__(
        self,
        agents: dict[str, Agent] | None = None,
        completion_service: CompletionService | None = None,
        cortex_completion_service: CortexCompletionService | None = None,
    ):
        """Bind to whichever backends this server has available."""
        self.agents = agents or {}
        self.completion_service = completion_service
        self.cortex_completion_service = cortex_completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _is_cortex_model(self, model_name: str) -> bool:
        """Return ``True`` when ``model_name`` should route to Cortex."""
        if not model_name:
            return False

        return "cortex" in model_name.lower()

    def _normalize_cortex_model(self, model_name: str) -> str:
        """Strip prefixes / separators and return the canonical Cortex name."""
        normalized = model_name.lower()
        for sep in [":", ".", "_"]:
            normalized = normalized.replace(sep, "-")

        cortex_index = normalized.find("cortex")
        if cortex_index >= 0:
            cortex_part = normalized[cortex_index:]

            cortex_part = cortex_part.rstrip("-")
            return cortex_part

        return normalized

    def _setup_routes(self) -> None:
        """Attach the unified chat completions endpoint."""

        @self.router.post("/v1/chat/completions")
        async def unified_chat_completions(request: ChatCompletionRequest):
            """Route to Cortex or single-agent backend based on ``model``.

            Raises HTTP 404 when the chosen backend is not configured
            or when a non-Cortex model id is not registered.
            """
            try:
                original_model = request.model
                if self._is_cortex_model(original_model):
                    if not self.cortex_completion_service:
                        raise HTTPException(status_code=404, detail="Cortex is not enabled on this server")

                    request.model = self._normalize_cortex_model(original_model)

                    messages_history = MessageConverter.convert_openai_to_xerxes(request.messages)

                    if request.stream:
                        return StreamingResponse(
                            self.cortex_completion_service.create_streaming_completion(messages_history, request),
                            media_type="text/event-stream",
                            headers={
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                            },
                        )
                    else:
                        return await self.cortex_completion_service.create_completion(messages_history, request)

                else:
                    if not self.completion_service:
                        raise HTTPException(status_code=404, detail="Standard agents are not available on this server")

                    agent = self.agents.get(request.model)
                    if not agent:
                        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

                    messages_history = MessageConverter.convert_openai_to_xerxes(request.messages)

                    agent = self.completion_service.apply_request_parameters(agent, request)

                    if request.stream:
                        return StreamingResponse(
                            self.completion_service.create_streaming_completion(agent, messages_history, request),
                            media_type="text/event-stream",
                            headers={
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                            },
                        )
                    else:
                        return await self.completion_service.create_completion(agent, messages_history, request)

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e
