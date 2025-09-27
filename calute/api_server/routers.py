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

from ..types.oai_protocols import ChatCompletionRequest
from .completion_service import CompletionService
from .converters import MessageConverter
from .cortex_completion_service import CortexCompletionService
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
                agent = self.agents.get(request.model)
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

                messages_history = MessageConverter.convert_openai_to_calute(request.messages)

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


class CortexChatRouter:
    """Router for Cortex chat completion endpoints with multi-agent orchestration."""

    def __init__(self, cortex_completion_service: CortexCompletionService):
        """Initialize the Cortex chat router.

        Args:
            cortex_completion_service: Service for handling Cortex completions
        """
        self.cortex_completion_service = cortex_completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Set up the Cortex chat completion routes."""

        @self.router.post("/v1/chat/completions")
        async def cortex_chat_completions(request: ChatCompletionRequest):
            """Handle Cortex chat completion requests with multi-agent orchestration.

            Fully OpenAI-compatible endpoint that routes to Cortex when model starts with "cortex".

            Supports two modes:
            1. Task Mode: Dynamically creates tasks from prompt and executes them
               - Use model name "cortex-task" or "cortex:task"
            2. Instruction Mode: Executes prompt directly with agents
               - Use model name "cortex" or "cortex-instruct"

            Process types can be specified:
            - "cortex-task-parallel" for parallel execution
            - "cortex-task-hierarchical" for hierarchical execution
            - Default is sequential

            Examples:
                {"model": "cortex-task", "messages": [...]}
                {"model": "cortex", "messages": [...]}
                {"model": "cortex-task-parallel", "messages": [...]}
            """
            try:
                messages_history = MessageConverter.convert_openai_to_calute(request.messages)

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
    """Unified router that handles both standard and Cortex chat completions."""

    def __init__(
        self,
        agents: dict[str, Agent] | None = None,
        completion_service: CompletionService | None = None,
        cortex_completion_service: CortexCompletionService | None = None,
    ):
        """Initialize the unified chat router.

        Args:
            agents: Dictionary of registered standard agents
            completion_service: Service for standard completions
            cortex_completion_service: Service for Cortex completions
        """
        self.agents = agents or {}
        self.completion_service = completion_service
        self.cortex_completion_service = cortex_completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _is_cortex_model(self, model_name: str) -> bool:
        """Check if the model name indicates a Cortex request.

        Supports various prefix patterns:
        - Direct: "cortex", "cortex-task"
        - With custom prefix: "calute-cortex", "myapp-cortex-task"
        - Any model containing "cortex" is considered a Cortex model

        Args:
            model_name: The model name from the request

        Returns:
            True if this should be handled by Cortex
        """
        if not model_name:
            return False

        return "cortex" in model_name.lower()

    def _normalize_cortex_model(self, model_name: str) -> str:
        """Normalize Cortex model name for compatibility.

        Extracts and normalizes the Cortex-specific part from various formats:
        - "cortex:task" -> "cortex-task"
        - "calute-cortex-task" -> "cortex-task"
        - "myapp-cortex:task:parallel" -> "cortex-task-parallel"
        - "custom.cortex.task" -> "cortex-task"

        Args:
            model_name: The original model name

        Returns:
            Normalized model name containing only the cortex-relevant parts
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

    def _setup_routes(self):
        """Set up the unified chat completion routes."""

        @self.router.post("/v1/chat/completions")
        async def unified_chat_completions(request: ChatCompletionRequest):
            """Handle both standard and Cortex chat completions.

            This endpoint is fully OpenAI-compatible and automatically routes
            requests based on the model name:

            Standard agents:
            - Use the agent's registered name/ID

            Cortex modes:
            - "cortex" or "cortex-instruct": Instruction mode
            - "cortex-task": Task mode with dynamic task creation
            - "cortex-task-parallel": Task mode with parallel execution
            - "cortex-task-hierarchical": Task mode with hierarchical execution

            The endpoint maintains full compatibility with OpenAI client libraries.
            """
            try:
                original_model = request.model
                if self._is_cortex_model(original_model):
                    if not self.cortex_completion_service:
                        raise HTTPException(status_code=404, detail="Cortex is not enabled on this server")

                    request.model = self._normalize_cortex_model(original_model)

                    messages_history = MessageConverter.convert_openai_to_calute(request.messages)

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

                    messages_history = MessageConverter.convert_openai_to_calute(request.messages)

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

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e
