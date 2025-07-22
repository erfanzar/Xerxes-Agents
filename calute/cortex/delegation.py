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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .chains import AgentChain, ChainType
from .types import CortexAgent, CortexTask, CortexTaskOutput


class DelegationStrategy(Enum):
    """Strategies for task delegation"""

    CAPABILITY_BASED = "capability"
    LOAD_BALANCED = "load_balanced"
    EXPERTISE_MATCH = "expertise"
    AVAILABILITY_BASED = "availability"
    PERFORMANCE_BASED = "performance"
    COST_OPTIMIZED = "cost"


@dataclass
class AgentMetrics:
    """Performance metrics for an agent"""

    tasks_completed: int = 0
    success_rate: float = 1.0
    average_completion_time: float = 0.0
    current_load: int = 0
    expertise_areas: dict[str, float] = field(default_factory=dict)
    cost_per_task: float = 1.0
    availability: float = 1.0  # 0-1 scale

    def update_metrics(self, task_result: CortexTaskOutput, execution_time: float):
        """Update metrics based on task completion"""
        self.tasks_completed += 1
        # Update success rate (simple moving average)
        success = 1.0 if task_result.status == "success" else 0.0
        self.success_rate = (self.success_rate * (self.tasks_completed - 1) + success) / self.tasks_completed
        # Update average completion time
        self.average_completion_time = (
            self.average_completion_time * (self.tasks_completed - 1) + execution_time
        ) / self.tasks_completed


class DelegationEngine:
    """Advanced delegation engine for task assignment"""

    def __init__(self):
        self.agent_metrics: dict[str, AgentMetrics] = {}
        self.delegation_history: list[dict[str, Any]] = []

    def register_agent(self, agent_id: str, expertise_areas: dict[str, float] | None = None):
        """Register an agent with the delegation engine"""
        self.agent_metrics[agent_id] = AgentMetrics(expertise_areas=expertise_areas or {})

    def delegate_task(
        self,
        task: "CortexTask",
        available_agents: list["CortexAgent"],
        strategy: DelegationStrategy = DelegationStrategy.CAPABILITY_BASED,
        context: dict[str, Any] | None = None,
    ) -> "CortexAgent":
        """Delegate a task to the most suitable agent"""

        if strategy == DelegationStrategy.CAPABILITY_BASED:
            return self._delegate_by_capability(task, available_agents)
        elif strategy == DelegationStrategy.LOAD_BALANCED:
            return self._delegate_by_load(task, available_agents)
        elif strategy == DelegationStrategy.EXPERTISE_MATCH:
            return self._delegate_by_expertise(task, available_agents, context)
        elif strategy == DelegationStrategy.PERFORMANCE_BASED:
            return self._delegate_by_performance(task, available_agents)
        elif strategy == DelegationStrategy.COST_OPTIMIZED:
            return self._delegate_by_cost(task, available_agents)
        else:
            return self._delegate_by_availability(task, available_agents)

    def _delegate_by_capability(self, task: "CortexTask", agents: list["CortexAgent"]) -> "CortexAgent":
        """Delegate based on agent capabilities"""
        best_agent = None
        best_score = -1

        task_keywords = set(task.description.lower().split())

        for agent in agents:
            # Calculate capability match score
            agent_keywords = set(agent.goal.lower().split() + agent.role.lower().split())
            overlap = len(task_keywords & agent_keywords)

            # Check tool compatibility
            tool_score = 0
            if task.tools and agent.tools:
                task_tool_names = {t.name for t in task.tools}  # type:ignore
                agent_tool_names = {t.name for t in agent.tools}  # type:ignore
                tool_score = len(task_tool_names & agent_tool_names)

            total_score = overlap + tool_score * 2  # Weight tool compatibility higher

            if total_score > best_score:
                best_score = total_score
                best_agent = agent

        return best_agent or agents[0]

    def _delegate_by_expertise(
        self,
        task: "CortexTask",
        agents: list["CortexAgent"],
        context: dict[str, Any] | None = None,  # type:ignore
    ) -> "CortexAgent":
        """Delegate based on expertise matching"""

        task_domain = self._extract_domain(task.description)

        best_agent = None
        best_expertise = 0

        for agent in agents:
            metrics = self.agent_metrics.get(agent.id, AgentMetrics())
            expertise_score = metrics.expertise_areas.get(task_domain, 0)

            if expertise_score > best_expertise:
                best_expertise = expertise_score
                best_agent = agent

        return best_agent or agents[0]

    def _extract_domain(self, description: str) -> str:
        """Extract domain from task description"""
        # Simple keyword-based domain extraction
        domains = {
            "research": ["research", "investigate", "analyze", "study"],
            "writing": ["write", "draft", "compose", "create content"],
            "coding": ["code", "program", "implement", "develop"],
            "analysis": ["analyze", "evaluate", "assess", "examine"],
            "design": ["design", "create", "prototype", "mockup"],
        }

        description_lower = description.lower()
        for domain, keywords in domains.items():
            if any(keyword in description_lower for keyword in keywords):
                return domain

        return "general"


class CollaborationPattern:
    """Patterns for agent collaboration"""

    @staticmethod
    def pair_programming(driver: "CortexAgent", navigator: "CortexAgent", task: "CortexTask") -> AgentChain:  # type:ignore
        """Create a pair programming pattern"""
        chain = AgentChain("pair_programming", ChainType.SEQUENTIAL)

        # Navigator reviews and plans
        chain.add_link(navigator, transform=lambda x: {"plan": x, "review_notes": "Initial review"})

        # Driver implements
        chain.add_link(driver, transform=lambda x: {"implementation": x, "plan": x.get("plan")})

        # Navigator reviews implementation
        chain.add_link(navigator, condition=lambda ctx: ctx.get("needs_review", True))

        return chain

    @staticmethod
    def expert_consultation(primary: "CortexAgent", experts: list["CortexAgent"], task: "CortexTask") -> "AgentChain":  # type:ignore
        """Create an expert consultation pattern"""
        chain = AgentChain("expert_consultation", ChainType.SEQUENTIAL)

        # Primary agent initial attempt
        chain.add_link(primary, transform=lambda x: {"initial_attempt": x, "needs_expert_input": True})

        # Consult each expert
        for expert in experts:
            chain.add_link(
                expert,
                condition=lambda ctx: ctx.get("needs_expert_input", False),
                transform=lambda x: {"expert_feedback": x, "expert": expert.role},  # noqa
            )

        # Primary agent synthesizes feedback
        chain.add_link(primary, transform=lambda x: {"final_output": x, "incorporated_feedback": True})

        return chain

    @staticmethod
    def quality_assurance(developer: "CortexAgent", tester: "CortexAgent", reviewer: "CortexAgent") -> "AgentChain":
        """Create a QA pattern"""
        chain = AgentChain("quality_assurance", ChainType.LOOP)

        # Developer creates
        chain.add_link(developer, transform=lambda x: {"artifact": x, "version": 1})

        # Tester tests
        chain.add_link(
            tester,
            transform=lambda x: {
                "test_results": x,
                "issues_found": x.get("issues", []),
                "passed": len(x.get("issues", [])) == 0,
            },
        )

        chain.add_link(
            reviewer,
            condition=lambda ctx: ctx.get("passed", False),
            transform=lambda x: {"review_complete": True, "approved": x.get("approved", False)},
        )

        chain.add_link(
            developer,
            condition=lambda ctx: not ctx.get("approved", False) and ctx.get("version", 0) < 3,
            transform=lambda x: {"artifact": x, "version": x.get("version", 1) + 1},
        )

        return chain
