# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Main API server for the modular Xerxes API server.

This module provides the core API server infrastructure for Xerxes,
including:
- FastAPI-based HTTP server with OpenAI-compatible endpoints
- Agent registration and management
- Cortex multi-agent orchestration support
- Modular router architecture for different endpoint groups
- Completion services for both standard and Cortex agents

The server supports both standard Xerxes agents and Cortex agents
for multi-agent orchestration, with full compatibility with OpenAI
client libraries.
"""

from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI

from xerxes import Xerxes
from xerxes.cortex import CortexAgent
from xerxes.llms.base import BaseLLM
from xerxes.types import Agent

from .completion_service import CompletionService
from .cortex_completion_service import CortexCompletionService
from .routers import ChatRouter, HealthRouter, ModelsRouter


class XerxesAPIServer:
    """Modular FastAPI server that provides OpenAI-compatible API for Xerxes agents.

    This server exposes registered Xerxes agents through HTTP endpoints that follow
    the OpenAI API specification, allowing seamless integration with OpenAI client libraries.

    The server is designed with a modular architecture:
    - Separate routers for different endpoint groups
    - Dedicated service for completion logic
    - Message conversion utilities
    - Centralized models for request/response handling

    Attributes:
        xerxes: The Xerxes instance managing agents.
        llm: LLM instance for Cortex agents.
        agents: Dictionary mapping agent IDs to Agent objects.
        cortex_agents: List of registered CortexAgent instances.
        enable_cortex: Whether Cortex endpoints are enabled.
        app: FastAPI application instance.
        completion_service: Service for handling standard chat completions.
        cortex_completion_service: Service for handling Cortex completions.

    Example:
        >>> from xerxes import Xerxes
        >>> from xerxes.api_server import XerxesAPIServer
        >>>
        >>> xerxes = Xerxes(client=openai_client)
        >>> server = XerxesAPIServer(xerxes)
        >>> server.register_agent(my_agent)
        >>> server.run(port=8000)
    """

    def __init__(
        self,
        xerxes_instance: Xerxes | None = None,
        llm: BaseLLM | None = None,
        can_overide_samplings: bool = False,
        enable_cortex: bool = False,
        use_universal_agent: bool = True,
    ):
        """Initialize the API server.

        Sets up the FastAPI application, completion services, and optionally
        the Cortex multi-agent orchestration layer. If Cortex is enabled and
        an LLM is provided, the Cortex completion service and routers are
        initialized immediately. Otherwise, routers are deferred until the
        first agent is registered.

        Args:
            xerxes_instance: Optional ``Xerxes`` instance to use for standard
                agent management and execution. Required for registering
                standard agents via ``register_agent``.
            llm: Optional ``BaseLLM`` instance for powering Cortex agents.
                Required when ``enable_cortex`` is ``True``.
            can_overide_samplings: Whether to allow incoming request parameters
                (temperature, top_p, max_tokens, etc.) to override the agent's
                default sampling settings. Defaults to ``False``.
            enable_cortex: Whether to enable Cortex multi-agent orchestration
                endpoints. When ``True``, the server supports model names
                containing ``"cortex"`` for multi-agent workflows.
                Defaults to ``False``.
            use_universal_agent: Whether to include a ``UniversalAgent`` as a
                fallback agent in the Cortex agent pool. Only relevant when
                ``enable_cortex`` is ``True``. Defaults to ``True``.
        """
        self.xerxes = xerxes_instance
        self.llm = llm
        self.agents: dict[str, Agent] = {}
        self.cortex_agents: list[CortexAgent] = []
        self.enable_cortex = enable_cortex

        title = "Xerxes API Server"
        if enable_cortex:
            title += " with Cortex"

        self.app = FastAPI(
            title=title,
            description="OpenAI-compatible API server for Xerxes agents with optional Cortex support",
            version="2.0.0",
        )

        if self.xerxes:
            self.completion_service = CompletionService(self.xerxes, can_overide_samplings=can_overide_samplings)
        else:
            self.completion_service = None

        if enable_cortex and llm:
            self.cortex_completion_service = CortexCompletionService(
                llm=llm,
                agents=self.cortex_agents,
                use_universal_agent=use_universal_agent,
                verbose=True,
            )
        else:
            self.cortex_completion_service = None

        self._routers_initialized = False

        if self.enable_cortex and self.cortex_completion_service:
            self._setup_routers()
            self._routers_initialized = True

    def register_agent(self, agent: Agent) -> None:
        """Register a standard agent to be available via the API.

        Adds the agent to both the Xerxes instance and the server's internal
        agent registry. The agent becomes accessible through the chat
        completions endpoint using its ID, name, or model as the ``model``
        parameter in requests. If routers have not yet been initialized,
        this method triggers router setup.

        Args:
            agent: The ``Agent`` instance to register. Must have at least
                one of ``id``, ``name``, or ``model`` set to serve as the
                lookup key in the agent registry.

        Raises:
            ValueError: If no ``Xerxes`` instance was provided during server
                initialization, since standard agents require Xerxes for
                execution.

        Example:
            >>> server = XerxesAPIServer(xerxes_instance=xerxes)
            >>> agent = Agent(id="assistant", model="gpt-4", instructions="Help users")
            >>> server.register_agent(agent)
        """
        if not self.xerxes:
            raise ValueError("Xerxes instance required for registering regular agents")

        self.xerxes.register_agent(agent)
        agent_key = agent.id or agent.name or agent.model
        self.agents[agent_key] = agent

        if not self._routers_initialized:
            self._setup_routers()
            self._routers_initialized = True

    def register_cortex_agent(self, agent: CortexAgent) -> None:
        """Register a ``CortexAgent`` for multi-agent orchestration.

        Adds the agent to the Cortex agent pool. If the Cortex completion
        service has already been initialized, its agent list is updated
        immediately. If routers have not yet been initialized, this method
        triggers router setup.

        Args:
            agent: The ``CortexAgent`` instance to register. This agent
                will be available for task assignment and orchestration
                through the Cortex completion service.

        Raises:
            ValueError: If Cortex was not enabled during server initialization
                (i.e., ``enable_cortex=False``).

        Example:
            >>> server = XerxesAPIServer(llm=my_llm, enable_cortex=True)
            >>> cortex_agent = CortexAgent(name="researcher", llm=my_llm)
            >>> server.register_cortex_agent(cortex_agent)
        """
        if not self.enable_cortex:
            raise ValueError("Cortex must be enabled to register CortexAgents")

        self.cortex_agents.append(agent)

        if self.cortex_completion_service:
            self.cortex_completion_service.agents = self.cortex_agents

        if not self._routers_initialized:
            self._setup_routers()
            self._routers_initialized = True

    def _setup_routers(self) -> None:
        """Set up and include FastAPI routers for the API endpoints.

        Configures the appropriate routers based on which services are
        available and includes them in the FastAPI application:

        - ``UnifiedChatRouter``: Used when Cortex is enabled. Handles both
          standard and Cortex requests through a single endpoint.
        - ``ChatRouter``: Used when only standard agents are available.
        - ``ModelsRouter``: Lists all available models/agents. Included
          whenever at least one completion service is active.
        - ``HealthRouter``: Provides the health check endpoint. Included
          whenever at least one completion service is active.

        This method is called automatically when the first agent is
        registered or during ``__init__`` if Cortex is pre-configured.
        """
        from .routers import UnifiedChatRouter

        if self.enable_cortex and self.cortex_completion_service:
            unified_router = UnifiedChatRouter(
                agents=self.agents,
                completion_service=self.completion_service,
                cortex_completion_service=self.cortex_completion_service,
            )
            self.app.include_router(unified_router.router, tags=["chat"])
        elif self.completion_service and self.agents:
            chat_router = ChatRouter(self.agents, self.completion_service)
            self.app.include_router(chat_router.router, tags=["chat"])

        if self.completion_service or self.cortex_completion_service:
            all_models = self._get_all_models()
            models_router = ModelsRouter(all_models)
            health_router = HealthRouter(all_models)
            self.app.include_router(models_router.router, tags=["models"])
            self.app.include_router(health_router.router, tags=["health"])

    def _get_all_models(self) -> dict[str, Any]:
        """Get all available models including Cortex virtual models.

        Builds a combined dictionary of all registered standard agents and,
        if Cortex is enabled, adds virtual model entries for each supported
        Cortex mode and process type. Virtual Cortex models are generated
        with multiple common prefixes (empty, ``"xerxes-"``, ``"api-"``,
        ``"v1-"``) to support flexible model naming in client requests.

        Returns:
            Dictionary mapping model name strings to either ``Agent`` objects
            (for standard agents) or configuration dictionaries (for Cortex
            virtual models) containing ``type``, ``mode``, and optionally
            ``process`` keys.
        """
        models = dict(self.agents)

        if self.enable_cortex:
            cortex_base_models = {
                "cortex": {"type": "cortex", "mode": "instruction"},
                "cortex-instruct": {"type": "cortex", "mode": "instruction"},
                "cortex-task": {"type": "cortex", "mode": "task"},
                "cortex-task-parallel": {"type": "cortex", "mode": "task", "process": "parallel"},
                "cortex-task-hierarchical": {"type": "cortex", "mode": "task", "process": "hierarchical"},
            }

            prefixes = ["", "xerxes-", "api-", "v1-"]
            for prefix in prefixes:
                for model_name, config in cortex_base_models.items():
                    full_name = f"{prefix}{model_name}" if prefix else model_name
                    models[full_name] = config

        return models

    def run(self, host: str = "0.0.0.0", port: int = 11881, **kwargs) -> None:
        """Run the API server using uvicorn.

        Starts the uvicorn ASGI server with the configured FastAPI application.
        If routers have not been initialized yet, this method attempts to set
        them up. Raises an error if no agents have been registered and Cortex
        is not enabled.

        Args:
            host: The hostname or IP address to bind the server to.
                Defaults to ``"0.0.0.0"`` (all interfaces).
            port: The TCP port number to bind the server to.
                Defaults to ``11881``.
            **kwargs: Additional keyword arguments passed directly to
                ``uvicorn.run()``, such as ``log_level``, ``workers``,
                ``ssl_keyfile``, etc.

        Raises:
            RuntimeError: If no agents are registered and Cortex is not
                enabled, since the server would have no endpoints to serve.

        Example:
            >>> server = XerxesAPIServer(xerxes_instance=xerxes)
            >>> server.register_agent(agent)
            >>> server.run(host="127.0.0.1", port=8000, log_level="info")
        """
        if not self._routers_initialized:
            if self.enable_cortex and self.cortex_completion_service:
                self._setup_routers()
                self._routers_initialized = True
            else:
                raise RuntimeError(
                    "No agents registered. Please register at least one agent before starting the server."
                )

        uvicorn.run(self.app, host=host, port=port, **kwargs)

    @classmethod
    def create_server(
        cls,
        client: Any,
        agents: list[Agent] | None | Agent = None,
        can_overide_samplings: bool = False,
        **xerxes_kwargs,
    ) -> XerxesAPIServer:
        """Create a Xerxes API server with the given client and agents.

        This is a convenience factory method that handles the full setup
        sequence: creating a ``Xerxes`` instance, wrapping it in a
        ``XerxesAPIServer``, and registering all provided agents. The
        returned server is ready to be started with ``run()``.

        Args:
            client: An OpenAI-compatible client instance (e.g.,
                ``openai.OpenAI(...)``). Passed to the ``Xerxes`` constructor.
            agents: A single ``Agent`` instance or a list of ``Agent``
                instances to register with the server. If ``None``, no agents
                are registered and they must be added later via
                ``register_agent()``.
            can_overide_samplings: Whether to allow incoming request parameters
                (temperature, top_p, max_tokens, etc.) to override the agent's
                default sampling settings. Defaults to ``False``.
            **xerxes_kwargs: Additional keyword arguments passed directly to
                the ``Xerxes`` constructor (e.g., ``max_history_length``,
                ``system_prompt``).

        Returns:
            A fully configured ``XerxesAPIServer`` instance with all provided
            agents registered and ready to serve requests.

        Example:
            >>> import openai
            >>> from xerxes.types import Agent
            >>> from xerxes.api_server import XerxesAPIServer
            >>>
            >>> client = openai.OpenAI(api_key="key", base_url="url")
            >>> agent = Agent(id="assistant", model="gpt-4", instructions="Help users")
            >>> server = XerxesAPIServer.create_server(client, agents=[agent])
            >>> server.run(port=8000)
        """
        xerxes = Xerxes(llm=client, **xerxes_kwargs)
        server = XerxesAPIServer(xerxes_instance=xerxes, can_overide_samplings=can_overide_samplings)
        if isinstance(agents, Agent):
            agents = [agents]
        if agents:
            for agent in agents:
                server.register_agent(agent)

        return server
