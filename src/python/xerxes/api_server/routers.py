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

This module defines route handlers for chat completions, model listing, health
checks, and a unified router that dispatches between standard and Cortex backends.
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
    """Router for standard agent chat completions."""

    def __init__(self, agents: dict[str, Agent], completion_service: CompletionService):
        """Initialize the chat router.

        Args:
            agents (dict[str, Agent]): IN: Registered agents by model ID. OUT:
                Used for model lookup.
            completion_service (CompletionService): IN: Standard completion backend.
                OUT: Used to process chat requests.
        """
        self.agents = agents
        self.completion_service = completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register FastAPI routes on the internal router."""

        @self.router.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Handle a chat completion request for a standard agent.

            Args:
                request (ChatCompletionRequest): IN: The incoming chat request.
                    OUT: Validated and dispatched to the completion service.

            Returns:
                ChatCompletionResponse | StreamingResponse: OUT: Synchronous or
                    streaming response depending on ``request.stream``.

            Raises:
                HTTPException: If the model is not found or an internal error occurs.
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
    """Router for the ``/v1/models`` endpoint."""

    def __init__(self, agents: dict[str, Agent]):
        """Initialize the models router.

        Args:
            agents (dict[str, Agent]): IN: Registered agents by model ID. OUT:
                Enumerated in the models list.
        """
        self.agents = agents
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register FastAPI routes on the internal router."""

        @self.router.get("/v1/models")
        async def list_models() -> ModelsResponse:
            """List available models.

            Returns:
                ModelsResponse: OUT: Response containing all registered agent models.
            """
            models = []
            for agent_id, _ in self.agents.items():
                models.append(ModelInfo(id=agent_id, created=int(time.time())))
            return ModelsResponse(data=models)


class HealthRouter:
    """Router for the ``/health`` endpoint."""

    def __init__(self, agents: dict[str, Agent]):
        """Initialize the health router.

        Args:
            agents (dict[str, Agent]): IN: Registered agents. OUT: Counted for
                the health response.
        """
        self.agents = agents
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register FastAPI routes on the internal router."""

        @self.router.get("/health")
        async def health_check() -> HealthResponse:
            """Return server health status.

            Returns:
                HealthResponse: OUT: Status and registered agent count.
            """
            return HealthResponse(status="healthy", agents=len(self.agents))


class CortexChatRouter:
    """Router for Cortex-backed chat completions."""

    def __init__(self, cortex_completion_service: CortexCompletionService):
        """Initialize the Cortex chat router.

        Args:
            cortex_completion_service (CortexCompletionService): IN: Cortex
                completion backend. OUT: Used to process chat requests.
        """
        self.cortex_completion_service = cortex_completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register FastAPI routes on the internal router."""

        @self.router.post("/v1/chat/completions")
        async def cortex_chat_completions(request: ChatCompletionRequest):
            """Handle a chat completion request via the Cortex backend.

            Args:
                request (ChatCompletionRequest): IN: The incoming chat request.
                    OUT: Dispatched to the Cortex completion service.

            Returns:
                ChatCompletionResponse | StreamingResponse: OUT: Synchronous or
                    streaming response.

            Raises:
                HTTPException: On internal errors.
            """
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
    """Unified router that dispatches to standard or Cortex backends by model name."""

    def __init__(
        self,
        agents: dict[str, Agent] | None = None,
        completion_service: CompletionService | None = None,
        cortex_completion_service: CortexCompletionService | None = None,
    ):
        """Initialize the unified chat router.

        Args:
            agents (dict[str, Agent] | None): IN: Standard registered agents. OUT:
                Used when the requested model is not a Cortex model.
            completion_service (CompletionService | None): IN: Standard completion
                backend. OUT: Used for non-Cortex model requests.
            cortex_completion_service (CortexCompletionService | None): IN: Cortex
                completion backend. OUT: Used when the model name contains ``"cortex"``.
        """
        self.agents = agents or {}
        self.completion_service = completion_service
        self.cortex_completion_service = cortex_completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _is_cortex_model(self, model_name: str) -> bool:
        """Check whether a model name refers to the Cortex backend.

        Args:
            model_name (str): IN: Requested model identifier. OUT: Checked for
                the substring ``"cortex"``.

        Returns:
            bool: OUT: ``True`` if the model is a Cortex model.
        """
        if not model_name:
            return False

        return "cortex" in model_name.lower()

    def _normalize_cortex_model(self, model_name: str) -> str:
        """Normalize a Cortex model name for internal routing.

        Args:
            model_name (str): IN: Raw model name. OUT: Lowercased and stripped
                of prefix separators.

        Returns:
            str: OUT: Normalized model name starting with ``"cortex"``.
        """
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
        """Register FastAPI routes on the internal router."""

        @self.router.post("/v1/chat/completions")
        async def unified_chat_completions(request: ChatCompletionRequest):
            """Handle a chat completion request, routing to Cortex or standard backend.

            Args:
                request (ChatCompletionRequest): IN: The incoming chat request.
                    OUT: Routed based on model name.

            Returns:
                ChatCompletionResponse | StreamingResponse: OUT: Response from
                    the selected backend.

            Raises:
                HTTPException: If the requested backend is unavailable or the
                    model is not found.
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
