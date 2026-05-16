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
"""FastAPI bootstrap for the OpenAI-compatible Xerxes API.

:class:`XerxesAPIServer` owns the :class:`FastAPI` app, registers
standard agents and Cortex agents, wires routers lazily on first
registration, and runs uvicorn in :meth:`run`.
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
    """OpenAI-compatible HTTP server for one or more Xerxes runtimes.

    Two backends share the same FastAPI app: a single-agent path
    powered by :class:`CompletionService` and an optional multi-agent
    path powered by :class:`CortexCompletionService`. Routers are set
    up the first time an agent of either kind is registered, so the
    object is usable before any agents exist.
    """

    def __init__(
        self,
        xerxes_instance: Xerxes | None = None,
        llm: BaseLLM | None = None,
        can_overide_samplings: bool = False,
        enable_cortex: bool = False,
        use_universal_agent: bool = True,
    ):
        """Build the FastAPI app and any completion services.

        Args:
            xerxes_instance: runtime for standard single-agent completions.
            llm: LLM backend used to construct the Cortex service.
            can_overide_samplings: forwarded to :class:`CompletionService`.
            enable_cortex: when ``True`` and ``llm`` is set, enable the
                Cortex multi-agent backend and unified routers.
            use_universal_agent: forwarded to :class:`CortexCompletionService`.
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
        """Register ``agent`` with the Xerxes runtime and the API surface.

        Initialises routers on the first registration. Raises
        ``ValueError`` if no :class:`Xerxes` instance was supplied at
        construction time.
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
        """Register ``agent`` with the Cortex completion service.

        Requires the server to have been built with ``enable_cortex=True``.
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
        """Mount the chat / models / health routers appropriate to this server."""
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
        """Return the merged map of standard agents and Cortex variants.

        For Cortex, this enumerates several model identifiers (with
        ``""``, ``"xerxes-"``, ``"api-"``, ``"v1-"`` prefixes) so that
        the OpenAI ``model=`` field can disambiguate the desired
        process type (instruction / task / parallel / hierarchical).
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
        """Run uvicorn on ``host:port`` (default ``0.0.0.0:11881``).

        Raises ``RuntimeError`` if neither a Cortex-only server nor a
        registered standard agent is available — routers must be set
        up before serving requests.
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
        """One-shot factory: build a Xerxes runtime and register agents.

        ``agents`` may be a single :class:`Agent` or a list; either way
        each is passed through :meth:`register_agent`.
        """
        xerxes = Xerxes(llm=client, **xerxes_kwargs)
        server = XerxesAPIServer(xerxes_instance=xerxes, can_overide_samplings=can_overide_samplings)
        if isinstance(agents, Agent):
            agents = [agents]
        if agents:
            for agent in agents:
                server.register_agent(agent)

        return server
