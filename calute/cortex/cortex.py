# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from calute.types.function_execution_types import ResponseResult

from ..calute import Calute
from ..executors import AgentOrchestrator, FunctionExecutor
from ..memory import MemoryStore, MemoryType
from ..types import Agent
from .blackscreen import LearningEngine, PerformanceMetrics, PerformanceMonitor
from .chains import AgentChain, ChainType
from .delegation import DelegationEngine
from .tools import CortexTool
from .types import CortexAgent, CortexOutput, CortexTask, CortexTaskOutput, ProcessType


class CortexToolAdapter:
    """Adapter to integrate enhanced tools with CortexAI agents"""

    @staticmethod
    def convert_tools_for_agent(tools: list[CortexTool]) -> list[Callable | None]:
        """Convert enhanced tools to callables for CortexAgent"""
        return [tool.as_function() for tool in tools]

    @staticmethod
    def generate_tool_documentation(tools: list[CortexTool]) -> str:
        """Generate comprehensive documentation for tools"""
        doc_parts = ["# Available Tools\n"]

        for tool in tools:
            signature = tool.get_signature()

            doc_parts.append(f"\n## {tool.name}")
            doc_parts.append(f"\n{tool.description}\n")

            if signature.parameters:
                doc_parts.append("\n### Parameters:\n")
                for param in signature.parameters:
                    param_doc = f"- **{param.name}** ({param.type})"
                    if not param.required:
                        param_doc += " [optional]"
                    if param.description:
                        param_doc += f": {param.description}"
                    if param.default is not None:
                        param_doc += f" (default: {param.default})"
                    if param.enum:
                        param_doc += f" (choices: {', '.join(map(str, param.enum))})"
                    doc_parts.append(param_doc)

            if signature.returns:
                doc_parts.append(f"\n### Returns:\n{signature.returns}")

            if hasattr(tool, "examples") and tool.examples:
                doc_parts.append("\n### Examples:")
                for example in tool.examples:
                    doc_parts.append(f"\n```json\n{json.dumps(example, indent=2)}\n```")

        return "\n".join(doc_parts)

    @staticmethod
    def validate_tool_compatibility(agent: CortexAgent, tools: list[CortexTool]) -> dict[str, Any]:  # type:ignore
        """Validate tool compatibility with agent capabilities"""
        compatibility_report = {"compatible_tools": [], "incompatible_tools": [], "warnings": []}

        for tool in tools:
            signature = tool.get_signature()

            # Check if agent has required capabilities
            is_compatible = True
            warnings = []

            # Check parameter complexity
            param_count = len(signature.parameters)
            if param_count > 5:
                warnings.append(f"Tool {tool.name} has {param_count} parameters, which may be complex")

            # Check for required parameters without defaults
            required_params = [p for p in signature.parameters if p.required]
            if len(required_params) > 3:
                warnings.append(f"Tool {tool.name} has {len(required_params)} required parameters")

            if is_compatible:
                compatibility_report["compatible_tools"].append(tool.name)
            else:
                compatibility_report["incompatible_tools"].append(tool.name)

            if warnings:
                compatibility_report["warnings"].extend(warnings)

        return compatibility_report


class Cortex:
    def __init__(
        self,
        agents: list[CortexAgent],
        tasks: list[CortexTask],
        process: ProcessType = ProcessType.SEQUENTIAL,
        verbose: bool = False,
        memory: bool = True,
        cache: bool = True,
        max_rpm: int | None = None,
        share_cortex: bool = False,
        function_calling_llm: str | None = None,
        step_callback: Callable | None = None,
        task_callback: Callable | None = None,
        enable_agent_chains: bool = True,
        enable_delegation_engine: bool = True,
        enable_collaboration: bool = True,
        enable_learning: bool = True,
        enable_monitoring: bool = True,
        enable_enhanced_tools: bool = True,
        llm_client: Any = None,
    ):
        self.enable_agent_chains = enable_agent_chains
        self.enable_delegation_engine = enable_delegation_engine
        self.enable_collaboration = enable_collaboration
        self.enable_learning = enable_learning
        self.enable_monitoring = enable_monitoring
        self.enable_enhanced_tools = enable_enhanced_tools
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = verbose
        self.memory_enabled = memory
        self.cache = cache
        self.max_rpm = max_rpm
        self.share_cortex = share_cortex
        self.function_calling_llm = function_calling_llm
        self.step_callback = step_callback
        self.task_callback = task_callback
        self.llm_client = llm_client
        self.execution_log = []
        self.token_usage = {"total": 0, "prompt": 0, "completion": 0}
        self._start_time = None
        self._end_time = None
        self.memory_store = MemoryStore() if memory else None
        self.orchestrator = AgentOrchestrator()
        self.executor = FunctionExecutor(self.orchestrator)
        self._validate_inputs()
        self._setup_agents()
        if self.enable_delegation_engine:
            self.delegation_engine = DelegationEngine()
            self._register_agents_with_delegation_engine()

        self.agent_chains: dict[str, AgentChain] = {}
        self.collaboration_patterns: dict[str, Any] = {}
        self.tool_registry: dict[str, CortexTool] = {}
        self.tool_usage_stats: dict[str, dict[str, Any]] = {}
        self.performance_monitor = PerformanceMonitor() if self.enable_monitoring else None
        self.learning_engine = LearningEngine() if self.enable_learning else None
        self.tool_adapter = CortexToolAdapter() if self.enable_enhanced_tools else None

        if self.enable_enhanced_tools:
            self._process_enhanced_tools()
        if self.verbose:
            self._log("🚀 Cortex initialized successfully")

    def _validate_inputs(self):
        """Validate cortex inputs"""
        if not self.agents:
            raise ValueError("At least one agent is required")

        if not self.tasks:
            raise ValueError("At least one task is required")

        # Validate task assignments
        for task in self.tasks:
            if task.agent and task.agent not in self.agents:
                raise ValueError(f"CortexTask assigned to unknown agent: {task.agent.role}")

    def _process_enhanced_tools(self):
        """Process and register enhanced tools for all agents"""
        for agent in self.agents:
            processed_tools = []

            for tool in agent.tools:
                if isinstance(tool, CortexTool):
                    self.tool_registry[tool.name] = tool
                    self.tool_usage_stats[tool.name] = {
                        "calls": 0,
                        "successes": 0,
                        "failures": 0,
                        "total_execution_time": 0,
                        "last_used": None,
                    }
                    processed_tools.append(tool.as_function())

                    if self.verbose:
                        self._log(f"  📦 Registered enhanced tool: {tool.name}")
                elif callable(tool):
                    processed_tools.append(tool)
                else:
                    self._log(f"  ⚠️  Unknown tool type: {type(tool)}", level="WARNING")
            agent.tools = processed_tools

    def _register_agents_with_delegation_engine(self):
        """Register all agents with the delegation engine"""
        for agent in self.agents:
            expertise = self._extract_expertise(agent)
            self.delegation_engine.register_agent(agent.id, expertise)

    def _extract_expertise(self, agent: CortexAgent) -> dict[str, float]:
        """Extract expertise areas from agent description"""
        expertise = {}

        # Simple keyword-based expertise extraction
        keywords = {
            "research": ["research", "investigate", "analyze", "study"],
            "writing": ["write", "content", "blog", "article"],
            "coding": ["code", "program", "develop", "software"],
            "data": ["data", "statistics", "metrics", "analysis"],
            "design": ["design", "ui", "ux", "visual"],
        }

        agent_text = f"{agent.role} {agent.goal} {agent.backstory}".lower()

        for domain, domain_keywords in keywords.items():
            score = sum(1 for keyword in domain_keywords if keyword in agent_text)
            if score > 0:
                expertise[domain] = score / len(domain_keywords)

        return expertise

    def create_chain(self, name: str, chain_type: ChainType = ChainType.SEQUENTIAL) -> AgentChain:
        """Create a new agent chain"""
        chain = AgentChain(name, chain_type)
        self.agent_chains[name] = chain
        return chain

    def add_collaboration_pattern(self, name: str, pattern: Callable[..., AgentChain]):
        """Add a collaboration pattern"""
        self.collaboration_patterns[name] = pattern

    async def execute_chain(
        self,
        chain_name: str,
        initial_input: Any,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a named agent chain"""
        if chain_name not in self.agent_chains:
            raise ValueError(f"Chain '{chain_name}' not found")

        chain = self.agent_chains[chain_name]

        if self.verbose:
            self._log(f"🔗 Executing chain: {chain_name}")

        result = await chain.execute(initial_input, context)

        if self.enable_monitoring:
            self.performance_monitor.record_chain_execution(chain_name, result)

        return result

    async def kickoff_with_strategy(
        self, inputs: dict[str, Any] | None = None, execution_strategy: str = "standard"
    ) -> CortexOutput:
        """Execute cortex with different strategies"""

        if execution_strategy == "standard":
            return await self.kickoff(inputs)
        elif execution_strategy == "parallel_explore":
            return await self._parallel_exploration(inputs)
        elif execution_strategy == "iterative_refinement":
            return await self._iterative_refinement(inputs)
        elif execution_strategy == "consensus_building":
            return await self._consensus_building(inputs)
        else:
            raise ValueError(f"Unknown execution strategy: {execution_strategy}")

    async def _parallel_exploration(self, inputs: dict[str, Any] | None = None) -> CortexOutput:
        """Execute tasks with parallel exploration"""
        if self.verbose:
            self._log("🔀 Starting parallel exploration strategy")

        # Group tasks by independence
        independent_groups = self._identify_independent_task_groups()

        all_outputs = []

        for group_idx, task_group in enumerate(independent_groups):
            if self.verbose:
                self._log(f"  📦 Processing group {group_idx + 1}/{len(independent_groups)}")

            # Execute tasks in parallel within each group
            group_tasks = []
            for task in task_group:
                task_coro = self._execute_task(task, inputs or {})
                group_tasks.append(task_coro)

            group_outputs = await asyncio.gather(*group_tasks)
            all_outputs.extend(group_outputs)

            # Update context with group results
            if inputs is None:
                inputs = {}
            for task, output in zip(task_group, group_outputs, strict=False):
                inputs[f"{task.id}_result"] = output.summary

        return CortexOutput(
            raw=self._combine_outputs(all_outputs), tasks_output=all_outputs, token_usage=self.token_usage
        )

    async def _iterative_refinement(self, inputs: dict[str, Any] | None = None, max_iterations: int = 3) -> CortexOutput:
        """Execute with iterative refinement"""
        if self.verbose:
            self._log("🔄 Starting iterative refinement strategy")

        best_outputs = None
        best_score = 0

        for iteration in range(max_iterations):
            if self.verbose:
                self._log(f"  🔁 Iteration {iteration + 1}/{max_iterations}")

            # Execute all tasks
            outputs = await self._execute_sequential(inputs or {})

            # Evaluate quality
            score = await self._evaluate_outputs(outputs)

            if score > best_score:
                best_score = score
                best_outputs = outputs

            # Refine inputs based on outputs
            if iteration < max_iterations - 1:
                inputs = self._refine_inputs(inputs or {}, outputs)

        if self.verbose:
            self._log(f"✅ Best score achieved: {best_score:.2f}")

        return CortexOutput(
            raw=self._combine_outputs(best_outputs), tasks_output=best_outputs, token_usage=self.token_usage
        )

    def _identify_independent_task_groups(self) -> list[list["CortexTask"]]:
        """Identify groups of tasks that can be executed in parallel"""
        groups = []
        processed = set()

        for task in self.tasks:
            if task.id in processed:
                continue

            group = [task]
            processed.add(task.id)
            for other_task in self.tasks:
                if other_task.id in processed:
                    continue

                if set(task.context) == set(other_task.context):
                    group.append(other_task)
                    processed.add(other_task.id)

            groups.append(group)

        return groups

    async def _evaluate_outputs(self, outputs: list[CortexTaskOutput]) -> float:
        """Evaluate the quality of outputs"""
        total_score = 0

        for output in outputs:
            length_score = min(len(output.raw) / 1000, 1.0)
            completion_score = 1.0 if output.raw else 0.0
            total_score += (length_score + completion_score) / 2

        return total_score / len(outputs) if outputs else 0

    def _refine_inputs(self, current_inputs: dict[str, Any], outputs: list[CortexTaskOutput]) -> dict[str, Any]:
        """Refine inputs based on outputs"""
        refined = current_inputs.copy()

        # Add summaries of outputs
        refined["previous_iteration_summaries"] = [output.summary for output in outputs]

        # Add any identified issues or improvements
        issues = []
        for output in outputs:
            if len(output.raw) < 100:
                issues.append(f"Output too short for: {output.description}")

        if issues:
            refined["identified_issues"] = issues
            refined["instruction_modifier"] = "Please provide more detailed responses"

        return refined

    async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> CortexOutput:
        """
        Start the cortex execution

        Args:
            inputs: Optional initial context/inputs

        Returns:
            CortexOutput with results from all tasks
        """
        self._start_time = datetime.now()

        if self.verbose:
            self._log("=" * 50)
            self._log("🚀 Starting Cortex execution")
            self._log(f"📋 Process: {self.process.value}")
            self._log(f"👥 Agents: {', '.join([a.role for a in self.agents])}")
            self._log(f"📝 Tasks: {len(self.tasks)}")
            self._log("=" * 50)

        try:
            # Run async execution
            result = await self._execute_async(inputs)

            self._end_time = datetime.now()
            execution_time = (self._end_time - self._start_time).total_seconds()

            if self.verbose:
                self._log("=" * 50)
                self._log("✅ Cortex execution completed successfully")
                self._log(f"⏱️  Total execution time: {execution_time:.2f}s")
                self._log(f"📊 Tokens used: {self.token_usage['total']}")
                self._log("=" * 50)

            return result

        except Exception as e:
            self._log(f"❌ Cortex execution failed: {e!s}", level="ERROR")
            raise

    def kickoff(self, inputs: dict[str, Any] | None = None) -> CortexOutput:
        """
        Start the cortex execution

        Args:
            inputs: Optional initial context/inputs

        Returns:
            CortexOutput with results from all tasks
        """
        self._start_time = datetime.now()

        if self.verbose:
            self._log("=" * 50)
            self._log("🚀 Starting Cortex execution")
            self._log(f"📋 Process: {self.process.value}")
            self._log(f"👥 Agents: {', '.join([a.role for a in self.agents])}")
            self._log(f"📝 Tasks: {len(self.tasks)}")
            self._log("=" * 50)

        try:
            # Run async execution
            result = asyncio.run(self._execute_async(inputs))

            self._end_time = datetime.now()
            execution_time = (self._end_time - self._start_time).total_seconds()

            if self.verbose:
                self._log("=" * 50)
                self._log("✅ Cortex execution completed successfully")
                self._log(f"⏱️  Total execution time: {execution_time:.2f}s")
                self._log(f"📊 Tokens used: {self.token_usage['total']}")
                self._log("=" * 50)

            return result

        except Exception as e:
            self._log(f"❌ Cortex execution failed: {e!s}", level="ERROR")
            raise

    def _setup_agents(self):
        """Setup and register all agents"""
        for agent in self.agents:
            # Convert CortexAgent to internal Agent format
            internal_agent = self._convert_to_internal_agent(agent)
            self.orchestrator.register_agent(internal_agent)

            if self.verbose:
                self._log(f"🤖 Agent '{agent.role}' ready")

    def _convert_to_internal_agent(self, cortex_agent: CortexAgent) -> "Agent":
        """Convert CortexAgent to internal Agent format"""

        instructions = f"{cortex_agent.backstory}\n\nYour goal: {cortex_agent.goal}"

        functions = []
        for tool in cortex_agent.tools:
            if isinstance(tool, CortexTool):
                functions.append(tool.as_function())
            else:
                functions.append(tool)
        return Agent(
            id=cortex_agent.id,
            name=cortex_agent.role,
            model=cortex_agent.llm,
            instructions=instructions,
            functions=functions,
            rules=[
                f"You are a {cortex_agent.role}",
                f"Focus on: {cortex_agent.goal}",
                "Use available tools when they can help achieve your goal",
                "Provide detailed and actionable outputs",
            ],
            temperature=cortex_agent.temperature,
            max_tokens=cortex_agent.max_tokens,
            top_p=cortex_agent.top_p,
            stop=cortex_agent.stop,
        )

    async def _execute_async(self, inputs: dict[str, Any] | None = None) -> CortexOutput:
        """Async execution of cortex tasks"""
        context = inputs or {}
        task_outputs = []

        start_time = datetime.now()

        try:
            if self.process == ProcessType.SEQUENTIAL:
                task_outputs = await self._execute_sequential(context)
            elif self.process == ProcessType.HIERARCHICAL:
                task_outputs = await self._execute_hierarchical(context)
            elif self.process == ProcessType.CONSENSUAL:
                task_outputs = await self._execute_consensual(context)
            else:
                raise ValueError(f"Unknown process type: {self.process}")

        except Exception as e:
            self._log(f"❌ Cortex execution failed: {e!s}", level="ERROR")
            raise

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        if self.verbose:
            self._log(f"✅ Cortex execution completed in {execution_time:.2f}s")

        # Create final output
        return CortexOutput(
            raw=self._combine_outputs(task_outputs),
            tasks_output=task_outputs,
            token_usage=self.token_usage,
        )

    async def _execute_sequential(self, context: dict[str, Any]) -> list[CortexTaskOutput]:
        """Execute tasks sequentially"""
        outputs = []

        for i, task in enumerate(self.tasks):
            if self.verbose:
                self._log(f"\n📌 Starting CortexTask {i + 1}/{len(self.tasks)}: {task.description[:50]}...")

            task_context = context.copy()
            if task.context:
                for ctx_task in task.context:
                    if ctx_task.output:
                        task_context[f"task_{ctx_task.id}_output"] = ctx_task.output.raw
            output = await self._execute_task(task, task_context)
            outputs.append(output)

            task.output = output
            task.status = "completed"

            # Callback
            if self.task_callback:
                self.task_callback(output)

            if self.verbose:
                self._log(f"✓ CortexTask {i + 1} completed")

        return outputs

    async def _execute_hierarchical(self, context: dict[str, Any]) -> list[CortexTaskOutput]:
        """Execute tasks in hierarchical manner with manager delegation"""
        manager = next((a for a in self.agents if "manager" in a.role.lower()), self.agents[0])
        outputs = []

        if self.verbose:
            self._log(f"👔 Manager '{manager.role}' coordinating tasks...")

        for task in self.tasks:
            assigned_agent = self._delegate_task(task, manager)
            task.agent = assigned_agent

            output = await self._execute_task(task, context)
            outputs.append(output)

        return outputs

    async def _execute_consensual(self, context: dict[str, Any]) -> list[CortexTaskOutput]:
        """Execute tasks with consensus from multiple agents"""
        outputs = []

        for task in self.tasks:
            if self.verbose:
                self._log(f"\n🤝 Seeking consensus for: {task.description[:50]}...")

            proposals = []
            for agent in self.agents:
                if self._is_agent_relevant(agent, task):
                    proposal = await self._get_agent_proposal(agent, task, context)
                    proposals.append((agent, proposal))

            # Synthesize consensus
            output = await self._synthesize_consensus(task, proposals, context)
            outputs.append(output)

        return outputs

    async def _execute_task(self, task: CortexTask, context: dict[str, Any]) -> CortexTaskOutput:
        """Execute a single task"""
        task.start_time = datetime.now()

        if not task.agent:
            task.agent = self._select_best_agent(task)
        if self.enable_enhanced_tools and self.tool_adapter:
            agent_tools = [
                self.tool_registry[t.__name__]
                for t in task.agent.tools
                if hasattr(t, "__name__") and t.__name__ in self.tool_registry
            ]
            if agent_tools:
                compatibility = self.tool_adapter.validate_tool_compatibility(task.agent, agent_tools)

                if compatibility["warnings"] and self.verbose:
                    for warning in compatibility["warnings"]:
                        self._log(f"  ⚠️  {warning}", level="WARNING")

        prompt = self._create_task_prompt(task, context)

        if self.enable_enhanced_tools:
            tool_examples = self._get_relevant_tool_examples(task)
            if tool_examples:
                prompt += f"\n\n## Tool Usage Examples:\n{tool_examples}"

        # Execute with agent
        if self.verbose:
            self._log(f"  🤖 Agent '{task.agent.role}' working...")

        # Track tool usage during execution
        if self.enable_enhanced_tools:
            context["_tool_usage_callback"] = self._track_tool_usage

        calute = Calute(client=self.llm_client, enable_memory=self.memory_enabled)

        internal_agent = self._convert_to_internal_agent(task.agent)
        calute.register_agent(internal_agent)

        response = await calute.create_response(
            prompt=prompt,
            context_variables=context,
            agent_id=internal_agent.id,
            stream=False,
            apply_functions=True,
        )

        self._update_token_usage(response)

        output = CortexTaskOutput(
            description=task.expected_output,
            summary=self._extract_summary(response.content),
            raw=response.content,
            agent=task.agent.role,
            task_id=task.id,
        )

        if self.enable_monitoring:
            execution_time = (datetime.now() - task.start_time).total_seconds()

            metrics = PerformanceMetrics(
                execution_time=execution_time,
                token_usage=response.token_usage if hasattr(response, "token_usage") else 0,
                success=True,
                quality_score=self._calculate_quality_score(output),
                timestamp=datetime.now(),
            )

            self.performance_monitor.record_task_execution(task.id, task.agent.id, metrics)

        # Save to memory if enabled
        if self.memory_enabled:
            self.memory_store.add_memory(
                content=f"Task completed: {task.description}\nOutput: {output.summary}",
                memory_type=MemoryType.EPISODIC,
                agent_id=task.agent.id,
                context={"task_id": task.id, "tools_used": self._get_tools_used_in_task()},
                importance_score=0.8,
            )

        task.end_time = datetime.now()
        task.output = output

        # Step callback
        if self.step_callback:
            self.step_callback(task, output)

        return output

    def _create_task_prompt(self, task: CortexTask, context: dict[str, Any]) -> str:  # type:ignore
        """Create prompt for task execution"""
        prompt_parts = [f"# CortexTask\n{task.description}", f"\n# Expected Output\n{task.expected_output}"]
        if task.context:
            prompt_parts.append("\n# Context from Previous Tasks")
            for ctx_task in task.context:
                if ctx_task.output:
                    prompt_parts.append(f"\n## {ctx_task.description}")
                    prompt_parts.append(ctx_task.output.summary or ctx_task.output.raw[:500])

        if task.output_json:
            prompt_parts.append(
                f"\n# Output Format\nProvide output as JSON matching this schema: {task.output_json.model_json_schema()}"
            )

        if task.output_file:
            prompt_parts.append(f"\n# Note\nThe output will be saved to: {task.output_file}")
        return "\n".join(prompt_parts)

    def _select_best_agent(self, task: CortexTask) -> CortexAgent:
        """Select the best agent for a task based on role and capabilities"""
        # Simple selection based on keywords in task description
        task_lower = task.description.lower()

        for agent in self.agents:
            role_lower = agent.role.lower()
            goal_lower = agent.goal.lower()

            # Check if agent's role or goal matches task
            if any(keyword in task_lower for keyword in role_lower.split()):
                return agent
            if any(keyword in task_lower for keyword in goal_lower.split()):
                return agent

        # Default to first agent
        return self.agents[0]

    def _delegate_task(self, task: CortexTask, manager: CortexAgent) -> CortexAgent:  # type:ignore
        """Manager delegates task to appropriate agent"""
        # In a real implementation, this would use the LLM to decide
        # For now, use simple selection
        return self._select_best_agent(task)

    def _is_agent_relevant(self, agent: CortexAgent, task: CortexTask) -> bool:
        """Check if an agent is relevant for a task"""
        task_keywords = set(task.description.lower().split())
        agent_keywords = set(agent.role.lower().split() + agent.goal.lower().split())

        # Check for keyword overlap
        return len(task_keywords & agent_keywords) > 0

    async def _get_agent_proposal(self, agent: CortexAgent, task: CortexTask, context: dict[str, Any]) -> str:
        """Get a proposal from an agent for a task"""
        prompt = (
            f"As a {agent.role}, provide your approach for:\n"
            f"{task.description}\n\n"
            f"Expected output: {task.expected_output}\n\n"
            f"Provide a brief proposal (2-3 sentences)."
        )

        calute = Calute(client=self.llm_client)
        internal_agent = self._convert_to_internal_agent(agent)
        calute.register_agent(internal_agent)

        response = await calute.create_response(
            prompt=prompt,
            context_variables=context,
            agent_id=internal_agent.id,
            stream=False,
            apply_functions=False,  # No functions for proposals
        )

        return response.content

    async def _synthesize_consensus(
        self,
        task: CortexTask,
        proposals: list[tuple[CortexAgent, str]],
        context: dict[str, Any],
    ) -> CortexTaskOutput:
        """Synthesize consensus from multiple agent proposals"""
        synthesis_prompt = f"# CortexTask\n{task.description}\n\n# Expected Output\n{task.expected_output}\n\n"
        synthesis_prompt += "# Agent Proposals\n"

        for agent, proposal in proposals:
            synthesis_prompt += f"\n## {agent.role}\n{proposal}\n"

        synthesis_prompt += "\n# Instructions\nSynthesize these proposals into a unified approach and execute the task."

        synthesizer = self.agents[0]

        calute = Calute(client=self.llm_client)
        internal_agent = self._convert_to_internal_agent(synthesizer)
        calute.register_agent(internal_agent)

        response = await calute.create_response(
            prompt=synthesis_prompt,
            context_variables=context,
            agent_id=internal_agent.id,
            stream=False,
            apply_functions=True,
        )

        return CortexTaskOutput(
            description=task.expected_output,
            summary=f"Consensus approach from {len(proposals)} agents",
            raw=response.content,
            agent="consensus",
            task_id=task.id,
        )

    def _combine_outputs(self, outputs: list[CortexTaskOutput]) -> str:
        """Combine all task outputs into final raw output"""
        combined = []

        for i, output in enumerate(outputs):
            combined.append(f"=== CortexTask {i + 1} Output ===")
            combined.append(f"Description: {output.description}")
            if output.summary:
                combined.append(f"Summary: {output.summary}")
            combined.append(f"Agent: {output.agent}")
            combined.append(f"\n{output.raw}")
            combined.append("")

        return "\n".join(combined)

    def _extract_summary(self, content: str, max_length: int = 200) -> str:
        """Extract a summary from content"""
        # Take first paragraph or sentences up to max_length
        lines = content.strip().split("\n")
        summary = ""

        for line in lines:
            if line.strip():
                summary = line.strip()
                break

        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary

    def _update_token_usage(self, response: ResponseResult):
        """Update token usage tracking"""
        try:
            self.token_usage["total"] += response.response.usage.total_tokens
            self.token_usage["prompt"] += response.response.usage.prompt_tokens
            self.token_usage["completion"] += response.response.usage.completion_tokens
        except Exception:
            ...

    def _log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {"timestamp": timestamp, "level": level, "message": message}

        # Store in log
        self.execution_log.append(log_entry)

        # Print if verbose
        if self.verbose:
            # Color coding for different levels
            if level == "ERROR":
                print(f"\033[91m[{timestamp}] {message}\033[0m")  # Red
            elif level == "WARNING":
                print(f"\033[93m[{timestamp}] {message}\033[0m")  # Yellow
            else:
                print(f"[{timestamp}] {message}")

    def _track_tool_usage(self, tool_name: str, success: bool, execution_time: float = 0):
        """Track tool usage statistics"""
        if tool_name in self.tool_usage_stats:
            stats = self.tool_usage_stats[tool_name]
            stats["calls"] += 1
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            stats["total_execution_time"] += execution_time
            stats["last_used"] = datetime.now()

    def _get_relevant_tool_examples(self, task: CortexTask) -> str:
        """Get relevant tool examples for the task"""
        examples = []

        if not task.agent:
            return ""

        for tool in task.agent.tools:
            if hasattr(tool, "__name__") and tool.__name__ in self.tool_registry:
                enhanced_tool = self.tool_registry[tool.__name__]
                if hasattr(enhanced_tool, "examples") and enhanced_tool.examples:
                    examples.append(f"\n### {enhanced_tool.name}")
                    for example in enhanced_tool.examples[:2]:
                        examples.append(f"```json\n{json.dumps(example, indent=2)}\n```")

        return "\n".join(examples)

    def _get_tools_used_in_task(self) -> list[str]:
        """Get list of tools used in the current task"""
        # This would be populated during task execution
        # For now, return empty list
        return []

    def _calculate_quality_score(self, output: CortexTaskOutput) -> float:
        """Calculate quality score for task output"""
        score = 0.0

        # Length score
        if len(output.raw) > 100:
            score += 0.3

        # Has summary
        if output.summary:
            score += 0.2

        # Completeness (basic check)
        if output.raw and not any(word in output.raw.lower() for word in ["error", "failed", "unable"]):
            score += 0.3

        # Structure (has sections/formatting)
        if any(marker in output.raw for marker in ["##", "###", "1.", "2.", "- "]):
            score += 0.2

        return min(score, 1.0)

    def get_tool_usage_report(self) -> dict[str, Any]:
        """Generate comprehensive tool usage report"""
        if not self.enable_enhanced_tools:
            return {"error": "Enhanced tools not enabled"}

        report = {"timestamp": datetime.now().isoformat(), "total_tools": len(self.tool_registry), "tool_statistics": {}}

        for tool_name, stats in self.tool_usage_stats.items():
            if stats["calls"] > 0:
                report["tool_statistics"][tool_name] = {
                    "total_calls": stats["calls"],
                    "success_rate": stats["successes"] / stats["calls"] if stats["calls"] > 0 else 0,
                    "failure_rate": stats["failures"] / stats["calls"] if stats["calls"] > 0 else 0,
                    "average_execution_time": stats["total_execution_time"] / stats["calls"]
                    if stats["calls"] > 0
                    else 0,
                    "last_used": stats["last_used"].isoformat() if stats["last_used"] else None,
                }

                # Add tool signature information
                if tool_name in self.tool_registry:
                    tool = self.tool_registry[tool_name]
                    signature = tool.get_signature()
                    report["tool_statistics"][tool_name]["signature"] = {
                        "parameters": [
                            {"name": p.name, "type": p.type, "required": p.required} for p in signature.parameters
                        ]
                    }

        # Most used tools
        if self.tool_usage_stats:
            sorted_tools = sorted(self.tool_usage_stats.items(), key=lambda x: x[1]["calls"], reverse=True)
            report["most_used_tools"] = [{"name": name, "calls": stats["calls"]} for name, stats in sorted_tools[:5]]

        return report

    def validate_cortex_tools(self) -> dict[str, Any]:
        """Validate all tools in the cortex"""
        if not self.enable_enhanced_tools or not self.tool_adapter:
            return {"error": "Enhanced tools not enabled"}

        validation_report = {"timestamp": datetime.now().isoformat(), "agents": {}}

        for agent in self.agents:
            agent_tools = [
                self.tool_registry[t.__name__]
                for t in agent.tools
                if hasattr(t, "__name__") and t.__name__ in self.tool_registry
            ]

            if agent_tools:
                compatibility = self.tool_adapter.validate_tool_compatibility(agent, agent_tools)
                validation_report["agents"][agent.role] = compatibility

        return validation_report

    def export_tool_schemas(self, output_file: str = "cortex_tool_schemas.json"):
        """Export all tool schemas to a file"""
        if not self.enable_enhanced_tools:
            return {"error": "Enhanced tools not enabled"}

        schemas = {
            "cortex_name": getattr(self, "name", "unnamed_cortex"),
            "timestamp": datetime.now().isoformat(),
            "tools": {},
        }

        for tool_name, tool in self.tool_registry.items():
            schemas["tools"][tool_name] = tool.to_openai_schema()

        with open(output_file, "w") as f:
            json.dump(schemas, f, indent=2)

        if self.verbose:
            self._log(f"✅ Exported {len(schemas['tools'])} tool schemas to {output_file}")

        return schemas

    def train(self, n_iterations: int = 5, filename: str = "cortex_training_data.json"):
        """
        Train the cortex by running iterations and collecting feedback

        Args:
            n_iterations: Number of training iterations
            filename: File to save training data
        """
        if self.verbose:
            self._log(f"🎓 Starting cortex training for {n_iterations} iterations...")

        training_data = []

        for i in range(n_iterations):
            if self.verbose:
                self._log(f"\n📚 Training iteration {i + 1}/{n_iterations}")

            result = self.kickoff()
            feedback = self._collect_feedback(result)
            training_data.append(
                {
                    "iteration": i + 1,
                    "timestamp": datetime.now().isoformat(),
                    "result": result.model_dump(),
                    "feedback": feedback,
                    "execution_log": self.execution_log.copy(),
                }
            )

            # Update agents based on feedback
            self._update_agents_from_feedback(feedback)

            # Clear execution log for next iteration
            self.execution_log.clear()

        # Save training data
        with open(filename, "w") as f:
            json.dump(training_data, f, indent=2, default=str)

        if self.verbose:
            self._log(f"✅ Training completed. Data saved to {filename}")

    def _collect_feedback(self, result: CortexOutput) -> dict[str, Any]:
        """Collect feedback on cortex performance"""
        # In real implementation, this would be interactive
        return {
            "overall_quality": 0.8,
            "task_ratings": {task.task_id: 0.8 for task in result.tasks_output},
            "suggestions": ["Consider more detail in research phase"],
            "successful_patterns": ["Good collaboration between agents"],
        }

    def _update_agents_from_feedback(self, feedback: dict[str, Any]):
        """Update agent configurations based on feedback"""
        # This would implement learning logic
        # For now, just log
        if self.verbose:
            self._log(f"📈 Updating agents based on feedback (quality: {feedback['overall_quality']})")

    def replay(self, task_id: str, inputs: dict[str, Any] | None = None) -> CortexTaskOutput:
        """
        Replay a specific task

        Args:
            task_id: ID of task to replay
            inputs: Optional new inputs

        Returns:
            New CortexTaskOutput
        """
        task = next((t for t in self.tasks if t.id == task_id), None)
        if not task:
            raise ValueError(f"CortexTask {task_id} not found")

        if self.verbose:
            self._log(f"🔄 Replaying task: {task.description[:50]}...")

        context = inputs or {}
        return asyncio.run(self._execute_task(task, context))

    def test(self, n_iterations: int = 3, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Test the cortex with multiple iterations

        Args:
            n_iterations: Number of test iterations
            inputs: Optional test inputs

        Returns:
            Test results with statistics
        """
        if self.verbose:
            self._log(f"🧪 Testing cortex with {n_iterations} iterations...")

        results = []
        execution_times = []

        for i in range(n_iterations):
            start_time = datetime.now()

            try:
                result = self.kickoff(inputs)
                execution_time = (datetime.now() - start_time).total_seconds()

                results.append(
                    {
                        "iteration": i + 1,
                        "success": True,
                        "execution_time": execution_time,
                        "output_length": len(result.raw),
                        "tasks_completed": len(result.tasks_output),
                    }
                )
                execution_times.append(execution_time)

            except Exception as e:
                results.append({"iteration": i + 1, "success": False, "error": str(e)})

        # Calculate statistics
        successful_runs = [r for r in results if r["success"]]

        test_summary = {
            "total_iterations": n_iterations,
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs) / n_iterations,
            "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "results": results,
        }

        if self.verbose:
            self._log(f"✅ Testing completed. Success rate: {test_summary['success_rate'] * 100:.1f}%")

        return test_summary
