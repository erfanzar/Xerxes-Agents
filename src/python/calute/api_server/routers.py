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


"""FastAPI routers for the OpenAI-compatible API endpoints.

This module provides the routing infrastructure for the Calute API server,
including:
- Chat completion endpoints (standard and Cortex)
- Models listing endpoint
- Health check endpoint
- Unified routing for mixed agent types

Each router class encapsulates the endpoint logic for a specific
functionality group, following the modular architecture pattern.
The routers support both streaming and non-streaming responses,
with full OpenAI API compatibility.
"""

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
    """Router for chat completion endpoints.

    Provides the /v1/chat/completions endpoint for standard Calute agents,
    supporting both streaming and non-streaming responses with full
    OpenAI API compatibility.

    Attributes:
        agents: Dictionary mapping agent IDs to Agent objects.
        completion_service: Service for handling chat completions.
        router: FastAPI APIRouter instance with configured routes.
    """

    def __init__(self, agents: dict[str, Agent], completion_service: CompletionService):
        """Initialize the chat router.

        Creates a new ``APIRouter`` and registers the chat completion
        endpoint for handling standard agent requests.

        Args:
            agents: Dictionary mapping agent IDs (strings) to their
                corresponding ``Agent`` objects.
            completion_service: The ``CompletionService`` instance used
                to process chat completion requests.
        """
        self.agents = agents
        self.completion_service = completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Register the ``/v1/chat/completions`` POST endpoint on the router.

        The registered endpoint looks up the requested model in the agents
        dictionary, converts messages from OpenAI format to Calute format,
        applies any sampling parameter overrides, and delegates to the
        completion service for either streaming or non-streaming responses.

        Raises:
            HTTPException: 404 if the requested model is not found among
                registered agents; 500 for any unexpected processing error.
        """

        @self.router.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Handle a chat completion request (OpenAI-compatible).

            Resolves the agent by model name, converts messages, and
            returns either a streaming ``StreamingResponse`` or a
            complete ``ChatCompletionResponse``.

            Args:
                request: The incoming ``ChatCompletionRequest`` with
                    model name, messages, and optional parameters.

            Returns:
                A ``ChatCompletionResponse`` for non-streaming requests,
                or a ``StreamingResponse`` with SSE for streaming requests.

            Raises:
                HTTPException: 404 if model not found; 500 on internal error.
            """
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
    """Router for models listing endpoints.

    Provides the /v1/models endpoint that lists all available agents
    and models, following the OpenAI API specification.

    Attributes:
        agents: Dictionary mapping agent IDs to Agent objects or model configs.
        router: FastAPI APIRouter instance with configured routes.
    """

    def __init__(self, agents: dict[str, Agent]):
        """Initialize the models router.

        Creates a new ``APIRouter`` and registers the models listing
        endpoint for enumerating available agents.

        Args:
            agents: Dictionary mapping agent IDs (strings) to their
                corresponding ``Agent`` objects or model configurations.
        """
        self.agents = agents
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Register the ``/v1/models`` GET endpoint on the router.

        The registered endpoint iterates over all registered agents and
        returns their metadata in OpenAI-compatible ``ModelsResponse`` format.
        """

        @self.router.get("/v1/models")
        async def list_models() -> ModelsResponse:
            """List all available models and agents (OpenAI-compatible).

            Builds a list of ``ModelInfo`` objects from the registered agents,
            each stamped with the current Unix timestamp as the creation time.

            Returns:
                A ``ModelsResponse`` containing a list of ``ModelInfo`` objects
                for every registered agent.
            """
            models = []
            for agent_id, _ in self.agents.items():
                models.append(ModelInfo(id=agent_id, created=int(time.time())))
            return ModelsResponse(data=models)


class HealthRouter:
    """Router for health check endpoints.

    Provides the ``/health`` endpoint for server health monitoring,
    returning the server status and the number of registered agents.
    This is typically used by load balancers, container orchestrators,
    or monitoring systems to verify the server is operational.

    Attributes:
        agents: Dictionary mapping agent IDs to ``Agent`` objects.
            The length of this dictionary is reported in health responses.
        router: FastAPI ``APIRouter`` instance with configured routes.
    """

    def __init__(self, agents: dict[str, Agent]):
        """Initialize the health router.

        Creates a new ``APIRouter`` and registers the health check
        endpoint.

        Args:
            agents: Dictionary mapping agent IDs (strings) to their
                corresponding ``Agent`` objects. Used to report the
                number of available agents in the health response.
        """
        self.agents = agents
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Register the ``/health`` GET endpoint on the router.

        The registered endpoint returns a ``HealthResponse`` with
        the current server status and agent count.
        """

        @self.router.get("/health")
        async def health_check() -> HealthResponse:
            """Return the current health status of the API server.

            Returns:
                A ``HealthResponse`` with status ``"healthy"`` and
                the count of currently registered agents.
            """
            return HealthResponse(status="healthy", agents=len(self.agents))


class CortexChatRouter:
    """Router for Cortex chat completion endpoints with multi-agent orchestration.

    Provides the /v1/chat/completions endpoint for Cortex-based multi-agent
    orchestration, supporting both task mode and instruction mode with
    various execution strategies (sequential, parallel, hierarchical).

    Attributes:
        cortex_completion_service: Service for handling Cortex completions.
        router: FastAPI APIRouter instance with configured routes.
    """

    def __init__(self, cortex_completion_service: CortexCompletionService):
        """Initialize the Cortex chat router.

        Creates a new ``APIRouter`` and registers the Cortex chat
        completion endpoint for multi-agent orchestration requests.

        Args:
            cortex_completion_service: The ``CortexCompletionService``
                instance used to process multi-agent orchestration requests.
        """
        self.cortex_completion_service = cortex_completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """Register the ``/v1/chat/completions`` POST endpoint for Cortex on the router.

        The registered endpoint converts messages from OpenAI format to Calute
        format and delegates to the Cortex completion service. The model name
        in the request determines the execution mode (task vs. instruction) and
        process type (sequential, parallel, hierarchical).

        Raises:
            HTTPException: 500 for any unexpected processing error.
        """

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
    """Unified router that handles both standard and Cortex chat completions.

    Routes incoming requests to either standard Calute agents or Cortex
    multi-agent orchestration based on the model name in the request.
    This provides a single endpoint that supports all agent types.

    Attributes:
        agents: Dictionary mapping agent IDs to Agent objects.
        completion_service: Service for standard agent completions.
        cortex_completion_service: Service for Cortex completions.
        router: FastAPI APIRouter instance with configured routes.
    """

    def __init__(
        self,
        agents: dict[str, Agent] | None = None,
        completion_service: CompletionService | None = None,
        cortex_completion_service: CortexCompletionService | None = None,
    ):
        """Initialize the unified chat router.

        Creates a single router that can dispatch to either standard Calute
        agents or Cortex multi-agent orchestration based on the model name
        in each incoming request.

        Args:
            agents: Optional dictionary mapping agent IDs (strings) to their
                corresponding ``Agent`` objects for standard completions.
                Defaults to an empty dictionary if ``None``.
            completion_service: Optional ``CompletionService`` instance for
                handling standard agent completions. If ``None``, standard
                agent requests will return a 404 error.
            cortex_completion_service: Optional ``CortexCompletionService``
                instance for handling Cortex multi-agent completions. If
                ``None``, Cortex requests will return a 404 error.
        """
        self.agents = agents or {}
        self.completion_service = completion_service
        self.cortex_completion_service = cortex_completion_service
        self.router = APIRouter()
        self._setup_routes()

    def _is_cortex_model(self, model_name: str) -> bool:
        """Check if the model name indicates a Cortex request.

        Performs a case-insensitive check for the substring ``"cortex"``
        anywhere in the model name. This supports various naming patterns:

        - Direct: ``"cortex"``, ``"cortex-task"``
        - With custom prefix: ``"calute-cortex"``, ``"myapp-cortex-task"``
        - Any model containing ``"cortex"`` is considered a Cortex model

        Args:
            model_name: The model name string from the incoming request.

        Returns:
            ``True`` if the model name contains ``"cortex"`` (case-insensitive)
            and should be routed to the Cortex completion service;
            ``False`` otherwise or if the model name is empty/falsy.

        Example:
            >>> router = UnifiedChatRouter()
            >>> router._is_cortex_model("cortex-task-parallel")
            True
            >>> router._is_cortex_model("gpt-4")
            False
        """
        if not model_name:
            return False

        return "cortex" in model_name.lower()

    def _normalize_cortex_model(self, model_name: str) -> str:
        """Normalize a Cortex model name into a canonical hyphen-separated format.

        Extracts the Cortex-specific portion of the model name and replaces
        all non-hyphen separators (``:`` ``.`` ``_``) with hyphens. Any
        prefix before ``"cortex"`` is stripped, and trailing hyphens are removed.

        Args:
            model_name: The original model name string, potentially with
                custom prefixes and mixed separators.

        Returns:
            A normalized model name string containing only the cortex-relevant
            parts, using hyphens as separators. If ``"cortex"`` is not found
            in the name, the entire lowercased and separator-normalized string
            is returned.

        Example:
            >>> router = UnifiedChatRouter()
            >>> router._normalize_cortex_model("cortex:task")
            'cortex-task'
            >>> router._normalize_cortex_model("calute-cortex-task")
            'cortex-task'
            >>> router._normalize_cortex_model("myapp-cortex:task:parallel")
            'cortex-task-parallel'
            >>> router._normalize_cortex_model("custom.cortex.task")
            'cortex-task'
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
        """Register the unified ``/v1/chat/completions`` POST endpoint on the router.

        The registered endpoint inspects the model name in each request to
        determine whether it should be handled by the Cortex completion service
        or the standard completion service. Cortex model names are normalized
        before being passed to the service layer.

        Raises:
            HTTPException: 404 if the requested service or model is not
                available; 500 for any unexpected processing error.
        """

        @self.router.post("/v1/chat/completions")
        async def unified_chat_completions(request: ChatCompletionRequest):
            """Handle both standard and Cortex chat completions.

            This endpoint is fully OpenAI-compatible and automatically routes
            requests based on the model name. If the model name contains
            ``"cortex"``, the request is forwarded to the Cortex completion
            service; otherwise, it is handled by the standard completion service.

            Standard agents:
                Use the agent's registered name/ID as the model.

            Cortex modes:
                - ``"cortex"`` or ``"cortex-instruct"``: Instruction mode
                - ``"cortex-task"``: Task mode with dynamic task creation
                - ``"cortex-task-parallel"``: Task mode with parallel execution
                - ``"cortex-task-hierarchical"``: Hierarchical execution

            Args:
                request: The incoming ``ChatCompletionRequest`` with model
                    name, messages, and optional streaming/sampling parameters.

            Returns:
                A ``ChatCompletionResponse`` for non-streaming requests,
                or a ``StreamingResponse`` with SSE for streaming requests.

            Raises:
                HTTPException: 404 if the requested model or service is not
                    available; 500 for any unexpected processing error.
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
