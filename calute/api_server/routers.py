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

"""FastAPI routers for the OpenAI-compatible API endpoints."""

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from calute.types import Agent

from ..types.oai_proxies import ChatCompletionRequest
from .completion_service import CompletionService
from .converters import MessageConverter
from .models import HealthResponse, ModelInfo, ModelsResponse


class ChatRouter:
    """Router for chat completion endpoints."""

    def __init__(self, agents: dict[str, Agent], completion_service: CompletionService):
        """Initialize the chat router.

        Args:
            agents: Dictionary of registered agents
            completion_service: Service for handling completions
        """
        self.agents = agents
        self.completion_service = completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Set up the chat completion routes."""

        @self.router.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Handle chat completion requests (OpenAI compatible)."""
            try:
                # Find the agent by model name
                agent = self.agents.get(request.model)
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

                # Convert OpenAI messages to Calute format
                messages_history = MessageConverter.convert_openai_to_calute(request.messages)

                # Apply request parameters to agent
                self.completion_service.apply_request_parameters(agent, request)

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

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e


class ModelsRouter:
    """Router for models endpoints."""

    def __init__(self, agents: dict[str, Agent]):
        """Initialize the models router.

        Args:
            agents: Dictionary of registered agents
        """
        self.agents = agents
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Set up the models routes."""

        @self.router.get("/v1/models")
        async def list_models() -> ModelsResponse:
            """List available models (agents) (OpenAI compatible)."""
            models = []
            for agent_id, _ in self.agents.items():
                models.append(ModelInfo(id=agent_id, created=int(time.time())))
            return ModelsResponse(data=models)


class HealthRouter:
    """Router for health check endpoints."""

    def __init__(self, agents: dict[str, Agent]):
        """Initialize the health router.

        Args:
            agents: Dictionary of registered agents
        """
        self.agents = agents
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Set up the health check routes."""

        @self.router.get("/health")
        async def health_check() -> HealthResponse:
            """Health check endpoint."""
            return HealthResponse(status="healthy", agents=len(self.agents))
