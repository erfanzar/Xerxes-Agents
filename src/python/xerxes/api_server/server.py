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
"""Xerxes API server bootstrap and agent registration.

This module provides :class:`XerxesAPIServer`, a FastAPI-based server that
exposes an OpenAI-compatible HTTP API for both standard agents and Cortex
multi-agent execution.
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
    """FastAPI server for Xerxes agents with optional Cortex support.

    Handles agent registration, router setup, and server execution via uvicorn.
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

        Args:
            xerxes_instance (Xerxes | None): IN: Xerxes runtime for standard agents.
                OUT: Used to create the completion service.
            llm (BaseLLM | None): IN: LLM backend for Cortex mode. OUT: Passed to
                :class:`CortexCompletionService` when ``enable_cortex`` is ``True``.
            can_overide_samplings (bool): IN: Whether request sampling parameters
                may override agent defaults. OUT: Passed to :class:`CompletionService`.
            enable_cortex (bool): IN: Whether Cortex multi-agent mode is enabled.
                OUT: Controls backend and router initialization.
            use_universal_agent (bool): IN: Whether to include a universal agent
                in the Cortex backend. OUT: Passed to :class:`CortexCompletionService`.
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

        self.completion_service: CompletionService | None = None
        if self.xerxes:
            self.completion_service = CompletionService(self.xerxes, can_overide_samplings=can_overide_samplings)

        self.cortex_completion_service: CortexCompletionService | None = None
        if enable_cortex and llm:
            self.cortex_completion_service = CortexCompletionService(
                llm=llm,
                agents=self.cortex_agents,
                use_universal_agent=use_universal_agent,
                verbose=True,
            )

        self._routers_initialized = False

        if self.enable_cortex and self.cortex_completion_service:
            self._setup_routers()
            self._routers_initialized = True

    def register_agent(self, agent: Agent) -> None:
        """Register a standard agent with the server.

        Args:
            agent (Agent): IN: Agent to register. OUT: Added to the internal agent
                dictionary and the Xerxes instance.

        Raises:
            ValueError: If no Xerxes instance was provided during initialization.
        """
        if not self.xerxes:
            raise ValueError("Xerxes instance required for registering regular agents")

        self.xerxes.register_agent(agent)
        agent_key = agent.id or agent.name or agent.model
        assert agent_key is not None
        self.agents[agent_key] = agent

        if not self._routers_initialized:
            self._setup_routers()
            self._routers_initialized = True

    def register_cortex_agent(self, agent: CortexAgent) -> None:
        """Register a Cortex agent with the server.

        Args:
            agent (CortexAgent): IN: Cortex agent to register. OUT: Appended to
                the agent list and reflected in the completion service.

        Raises:
            ValueError: If Cortex mode is not enabled.
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
        """Configure and include FastAPI routers based on available backends."""
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
        """Build a combined model dictionary including Cortex variants.

        Returns:
            dict[str, Any]: OUT: Mapping of model IDs to agent objects or Cortex
                configuration dicts.
        """
        models: dict[str, Any] = dict(self.agents)

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
        """Start the uvicorn server.

        Args:
            host (str): IN: Bind host. OUT: Passed to ``uvicorn.run``.
            port (int): IN: Bind port. OUT: Passed to ``uvicorn.run``.
            **kwargs (Any): IN: Additional uvicorn options. OUT: Forwarded to
                ``uvicorn.run``.

        Raises:
            RuntimeError: If no agents are registered and routers were not initialized.
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
        """Factory method to create and configure a server from an LLM client.

        Args:
            client (Any): IN: LLM client passed to :class:`Xerxes`. OUT: Used as
                the ``llm`` argument.
            agents (list[Agent] | None | Agent): IN: Agent(s) to register. OUT:
                Normalized to a list and registered.
            can_overide_samplings (bool): IN: Whether sampling overrides are allowed.
                OUT: Passed to the server constructor.
            **xerxes_kwargs (Any): IN: Extra keyword arguments for :class:`Xerxes`.
                OUT: Forwarded to the Xerxes constructor.

        Returns:
            XerxesAPIServer: OUT: Configured and ready-to-run server instance.
        """
        xerxes = Xerxes(llm=client, **xerxes_kwargs)
        server = XerxesAPIServer(xerxes_instance=xerxes, can_overide_samplings=can_overide_samplings)
        if isinstance(agents, Agent):
            agents = [agents]
        if agents:
            for agent in agents:
                server.register_agent(agent)

        return server
