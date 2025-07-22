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

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .types import CortexAgent, CortexTask


class ChainType(Enum):
    """Types of agent chains"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    MAP_REDUCE = "map_reduce"
    PIPELINE = "pipeline"
    ROUTER = "router"


@dataclass
class ChainLink:
    """Single link in an agent chain"""

    agent: CortexAgent
    task: CortexTask | None = None
    condition: Callable[[dict[str, Any]], bool] | None = None
    transform: Callable[[Any], Any] | None = None
    retry_on_failure: int = 0
    timeout: float | None = None

    def should_execute(self, context: dict[str, Any]) -> bool:
        """Check if this link should execute"""
        if self.condition:
            return self.condition(context)
        return True


class AgentChain:
    """Advanced agent chaining system"""

    def __init__(self, name: str, chain_type: ChainType = ChainType.SEQUENTIAL, max_iterations: int = 10):
        self.name = name
        self.chain_type = chain_type
        self.links: list[ChainLink] = []
        self.max_iterations = max_iterations
        self.execution_history: list[dict[str, Any]] = []

    def add_link(
        self,
        agent: CortexAgent,
        task: CortexTask | None = None,
        condition: Callable | None = None,
        transform: Callable | None = None,
    ) -> "AgentChain":
        """Add a link to the chain"""
        link = ChainLink(agent=agent, task=task, condition=condition, transform=transform)
        self.links.append(link)
        return self

    async def execute(self, initial_input: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the agent chain"""
        context = context or {}
        start_time = datetime.now()

        try:
            if self.chain_type == ChainType.SEQUENTIAL:
                result = await self._execute_sequential(initial_input, context)
            elif self.chain_type == ChainType.PARALLEL:
                result = await self._execute_parallel(initial_input, context)
            elif self.chain_type == ChainType.CONDITIONAL:
                result = await self._execute_conditional(initial_input, context)
            elif self.chain_type == ChainType.LOOP:
                result = await self._execute_loop(initial_input, context)
            elif self.chain_type == ChainType.MAP_REDUCE:
                result = await self._execute_map_reduce(initial_input, context)
            elif self.chain_type == ChainType.PIPELINE:
                result = await self._execute_pipeline(initial_input, context)
            elif self.chain_type == ChainType.ROUTER:
                result = await self._execute_router(initial_input, context)
            else:
                raise ValueError(f"Unknown chain type: {self.chain_type}")

            # Add execution metadata
            execution_time = (datetime.now() - start_time).total_seconds()
            result["execution_time"] = execution_time
            result["chain_name"] = self.name

            return result

        except Exception as e:
            return {
                "error": str(e),
                "chain_type": self.chain_type.value,
                "chain_name": self.name,
                "execution_time": (datetime.now() - start_time).total_seconds(),
            }

    async def _execute_sequential(self, initial_input: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Execute links sequentially"""
        current_input = initial_input
        results = []

        for link in self.links:
            if link.should_execute(context):
                result = await self._execute_link(link, current_input, context)
                results.append(result)

                # Transform output if needed
                if link.transform:
                    current_input = link.transform(result)
                else:
                    current_input = result

                # Update context
                context[f"{link.agent.role}_output"] = result

        return {"final_output": current_input, "intermediate_results": results, "chain_type": "sequential"}

    async def _execute_parallel(self, initial_input: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Execute links in parallel"""
        tasks = []

        for link in self.links:
            if link.should_execute(context):
                task = self._execute_link(link, initial_input, context.copy())
                tasks.append(task)

        results = await asyncio.gather(*tasks)

        return {"outputs": results, "chain_type": "parallel", "execution_count": len(results)}

    async def _execute_loop(self, initial_input: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Execute in a loop until condition is met"""
        current_input = initial_input
        iterations = 0
        results = []

        while iterations < self.max_iterations:
            for link in self.links:
                if link.should_execute(context):
                    result = await self._execute_link(link, current_input, context)
                    results.append(result)
                    current_input = result

                    # Check loop termination condition
                    if context.get("terminate_loop", False):
                        return {
                            "final_output": current_input,
                            "iterations": iterations + 1,
                            "all_results": results,
                            "chain_type": "loop",
                        }

            iterations += 1

        return {
            "final_output": current_input,
            "iterations": iterations,
            "all_results": results,
            "chain_type": "loop",
            "terminated_by": "max_iterations",
        }

    async def _execute_map_reduce(self, initial_input: list[Any], context: dict[str, Any]) -> dict[str, Any]:
        """Map operation across inputs, then reduce"""
        if len(self.links) < 2:
            raise ValueError("Map-reduce requires at least 2 links (mapper and reducer)")

        mapper = self.links[0]
        reducer = self.links[1]

        # Map phase - parallel execution
        map_tasks = []
        for item in initial_input:
            task = self._execute_link(mapper, item, context.copy())
            map_tasks.append(task)

        map_results = await asyncio.gather(*map_tasks)

        # Reduce phase
        reduce_result = await self._execute_link(reducer, map_results, context)

        return {
            "map_results": map_results,
            "final_output": reduce_result,
            "chain_type": "map_reduce",
            "input_count": len(initial_input),
        }

    async def _execute_router(self, initial_input: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Route to different agents based on input"""
        # First link should be the router
        if not self.links:
            raise ValueError("Router chain requires at least one link")

        router_link = self.links[0]
        route_decision = await self._execute_link(router_link, initial_input, context)

        # Find the appropriate link based on routing decision
        selected_link = None
        for link in self.links[1:]:
            if link.agent.role == route_decision.get("route_to"):
                selected_link = link
                break

        if selected_link:
            result = await self._execute_link(selected_link, initial_input, context)
            return {
                "routing_decision": route_decision,
                "final_output": result,
                "selected_agent": selected_link.agent.role,
                "chain_type": "router",
            }

        return {"routing_decision": route_decision, "error": "No suitable agent found for route", "chain_type": "router"}

    async def _execute_sequential(self, initial_input: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Execute links sequentially"""
        current_input = initial_input
        results = []

        for i, link in enumerate(self.links):
            if link.should_execute(context):
                result = await self._execute_link(link, current_input, context)
                results.append(result)

                # Transform output if needed
                if link.transform:
                    current_input = link.transform(result)
                else:
                    current_input = result

                # Update context
                context[f"{link.agent.role}_output"] = result
                context[f"step_{i}_result"] = result

        return {
            "final_output": current_input,
            "intermediate_results": results,
            "chain_type": "sequential",
            "steps_executed": len(results),
        }

    async def _execute_parallel(self, initial_input: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Execute links in parallel"""
        tasks = []

        for link in self.links:
            if link.should_execute(context):
                task = self._execute_link(link, initial_input, context.copy())
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        return {
            "outputs": successful_results,
            "errors": [str(e) for e in failed_results],
            "chain_type": "parallel",
            "execution_count": len(results),
            "success_count": len(successful_results),
        }

    async def _execute_conditional(self, initial_input: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Execute links based on conditions"""
        current_input = initial_input
        results = []
        executed_links = []

        for i, link in enumerate(self.links):
            if link.should_execute(context):
                result = await self._execute_link(link, current_input, context)
                results.append(result)
                executed_links.append(i)

                # Update input for next link
                if link.transform:
                    current_input = link.transform(result)
                else:
                    current_input = result

                # Update context for next condition evaluation
                context[f"link_{i}_executed"] = True
                context[f"link_{i}_result"] = result

        return {
            "final_output": current_input,
            "intermediate_results": results,
            "chain_type": "conditional",
            "executed_links": executed_links,
            "total_links": len(self.links),
        }

    async def _execute_loop(self, initial_input: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Execute in a loop until condition is met"""
        current_input = initial_input
        iterations = 0
        all_results = []

        while iterations < self.max_iterations:
            iteration_results = []

            for link in self.links:
                if link.should_execute(context):
                    result = await self._execute_link(link, current_input, context)
                    iteration_results.append(result)

                    if link.transform:
                        current_input = link.transform(result)
                    else:
                        current_input = result

                    # Check loop termination condition
                    if context.get("terminate_loop", False):
                        all_results.extend(iteration_results)
                        return {
                            "final_output": current_input,
                            "iterations": iterations + 1,
                            "all_results": all_results,
                            "chain_type": "loop",
                            "terminated_by": "condition",
                        }

            all_results.extend(iteration_results)
            iterations += 1

            # Update context for next iteration
            context["iteration"] = iterations
            context["previous_iteration_results"] = iteration_results

        return {
            "final_output": current_input,
            "iterations": iterations,
            "all_results": all_results,
            "chain_type": "loop",
            "terminated_by": "max_iterations",
        }

    async def _execute_map_reduce(self, initial_input: list[Any], context: dict[str, Any]) -> dict[str, Any]:
        """Map operation across inputs, then reduce"""
        if len(self.links) < 2:
            raise ValueError("Map-reduce requires at least 2 links (mapper and reducer)")

        if not isinstance(initial_input, list):
            raise ValueError("Map-reduce requires list input")

        mapper = self.links[0]
        reducer = self.links[1]

        # Map phase - parallel execution
        map_tasks = []
        for item in initial_input:
            task = self._execute_link(mapper, item, context.copy())
            map_tasks.append(task)

        map_results = await asyncio.gather(*map_tasks, return_exceptions=True)

        # Filter successful results
        successful_maps = [r for r in map_results if not isinstance(r, Exception)]

        # Reduce phase
        reduce_input = successful_maps
        if reducer.transform:
            reduce_input = reducer.transform(successful_maps)

        reduce_result = await self._execute_link(reducer, reduce_input, context)

        return {
            "map_results": successful_maps,
            "map_errors": [str(e) for e in map_results if isinstance(e, Exception)],
            "final_output": reduce_result,
            "chain_type": "map_reduce",
            "input_count": len(initial_input),
            "successful_maps": len(successful_maps),
        }

    async def _execute_pipeline(self, initial_input: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Execute as a pipeline where output flows through each stage"""
        current_data = initial_input
        pipeline_stages = []

        for i, link in enumerate(self.links):
            stage_name = f"stage_{i}_{link.agent.role}"

            # Execute stage
            stage_result = await self._execute_link(link, current_data, context)

            # Transform for next stage
            if link.transform:
                current_data = link.transform(stage_result)
            else:
                current_data = stage_result

            # Record stage info
            pipeline_stages.append(
                {
                    "stage": stage_name,
                    "agent": link.agent.role,
                    "input_type": type(current_data).__name__,
                    "output_preview": str(current_data)[:100] + "..."
                    if len(str(current_data)) > 100
                    else str(current_data),
                }
            )

            # Update context
            context[stage_name] = stage_result

        return {
            "final_output": current_data,
            "pipeline_stages": pipeline_stages,
            "chain_type": "pipeline",
            "total_stages": len(self.links),
        }

    async def _execute_router(self, initial_input: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Route to different agents based on input"""
        if not self.links:
            raise ValueError("Router chain requires at least one link")

        # First link should be the router
        router_link = self.links[0]
        route_decision = await self._execute_link(router_link, initial_input, context)

        # Extract routing decision
        route_to = None
        if isinstance(route_decision, dict):
            route_to = route_decision.get("route_to") or route_decision.get("selected_agent")
        elif isinstance(route_decision, str):
            route_to = route_decision

        # Find the appropriate link based on routing decision
        selected_link = None
        for link in self.links[1:]:
            if link.agent.role == route_to or link.agent.id == route_to:
                selected_link = link
                break

        if selected_link:
            result = await self._execute_link(selected_link, initial_input, context)
            return {
                "routing_decision": route_decision,
                "final_output": result,
                "selected_agent": selected_link.agent.role,
                "chain_type": "router",
                "route_successful": True,
            }

        return {
            "routing_decision": route_decision,
            "error": f"No suitable agent found for route: {route_to}",
            "available_agents": [link.agent.role for link in self.links[1:]],
            "chain_type": "router",
            "route_successful": False,
        }

    async def _execute_link(self, link: ChainLink, input_data: Any, context: dict[str, Any]) -> Any:
        """Execute a single link in the chain"""
        execution_record = {
            "agent": link.agent.role,
            "input": str(input_data)[:100],
            "timestamp": datetime.now().isoformat(),
            "context_keys": list(context.keys()),
        }
        self.execution_history.append(execution_record)

        result = {
            "agent": link.agent.role,
            "agent_id": link.agent.id,
            "input": input_data,
            "output": f"Processed by {link.agent.role}: {str(input_data)[:50]}...",
            "timestamp": datetime.now().isoformat(),
            "success": True,
        }

        # If there's a specific task, include it
        if link.task:
            result["task"] = link.task.description[:100]

        return result
