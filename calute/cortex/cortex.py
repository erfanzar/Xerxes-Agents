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


"""Main Cortex cortex orchestration"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from ..calute import Calute
from ..llms import BaseLLM
from ..loggings import get_logger, log_agent_start, log_success, log_task_start
from ..memory import MemoryStore, MemoryType
from ..streamer_buffer import StreamerBuffer
from ..types import Completion, StreamChunk
from .agent import CortexAgent
from .enums import ProcessType
from .memory_integration import CortexMemory
from .planner import CortexPlanner
from .task import CortexTask, CortexTaskOutput
from .templates import PromptTemplate


class MemoryConfig(TypedDict, total=False):
    """Configuration for memory system"""

    max_short_term: NotRequired[int]
    max_working: NotRequired[int]
    max_long_term: NotRequired[int]

    enable_short_term: NotRequired[bool]
    enable_long_term: NotRequired[bool]
    enable_entity: NotRequired[bool]
    enable_user: NotRequired[bool]
    persistence_path: NotRequired[str | None]
    short_term_capacity: NotRequired[int]
    long_term_capacity: NotRequired[int]


class Cortex:
    """Main orchestrator for multi-agent collaboration"""

    def __init__(
        self,
        agents: list[CortexAgent],
        tasks: list[CortexTask],
        llm: BaseLLM,
        process: ProcessType = ProcessType.SEQUENTIAL,
        manager_agent: CortexAgent | None = None,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        verbose: bool = True,
        max_iterations: int = 10,
        model: str = "gpt-4",
        memory: CortexMemory | None = None,
        memory_config: MemoryConfig | None = None,
        reinvoke_after_function: bool = True,
        enable_calute_memory: bool = False,
        cortex_name: str = "CorTex",
    ):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.manager_agent = manager_agent
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.reinvoke_after_function = reinvoke_after_function
        self.enable_calute_memory = enable_calute_memory
        self.cortex_name = cortex_name
        if memory:
            self.cortex_memory = memory
        else:
            config = memory_config or {}
            self.cortex_memory = CortexMemory(
                enable_short_term=config.get("enable_short_term", True),
                enable_long_term=config.get("enable_long_term", True),
                enable_entity=config.get("enable_entity", True),
                enable_user=config.get("enable_user", False),
                persistence_path=config.get("persistence_path", None),
                short_term_capacity=config.get("short_term_capacity", 50),
                long_term_capacity=config.get("long_term_capacity", 5000),
            )

        self.memory = MemoryStore()
        self.memory_type = memory_type
        self.task_outputs: list[CortexTaskOutput] = []

        self.logger = get_logger()
        self.template_engine = PromptTemplate()

        self.planner = CortexPlanner(cortex_instance=self, verbose=verbose) if process == ProcessType.PLANNED else None

        config = memory_config or {}
        calute_memory_config = {
            "max_short_term": config.get("max_short_term", 100),
            "max_working": config.get("max_working", 10),
            "max_long_term": config.get("max_long_term", 1000),
        }

        self.llm = llm
        self.calute = Calute(
            llm=self.llm,
            enable_memory=self.enable_calute_memory,
            memory_config=calute_memory_config,
        )

        for agent in self.agents:
            agent.calute_instance = self.calute
            agent.cortex_instance = self
            agent._logger = self.logger
            if not agent.model:
                agent.model = model

            agent.reinvoke_after_function = self.reinvoke_after_function

            self.calute.register_agent(agent._internal_agent)

            if agent.memory_enabled and not agent.memory:
                agent.memory = self.cortex_memory

        for task in self.tasks:
            if not task.agent and process != ProcessType.HIERARCHICAL:
                raise ValueError(f"Task '{task.description[:50]}...' has no assigned agent")

            if not task.memory:
                task.memory = self.cortex_memory

        if self.process == ProcessType.HIERARCHICAL:
            if not self.manager_agent:
                self.manager_agent = CortexAgent(
                    role="Cortex Manager",
                    goal="Efficiently delegate tasks to the right agents and ensure quality output",
                    backstory="You are an experienced manager who knows how to get the best out of your team",
                    model=model,
                    verbose=verbose,
                )
            self.manager_agent.calute_instance = self.calute
            self.manager_agent.cortex_instance = self
            self.manager_agent._logger = self.logger
            if not self.manager_agent.model:
                self.manager_agent.model = model

            self.manager_agent.reinvoke_after_function = self.reinvoke_after_function

            self.calute.register_agent(self.manager_agent._internal_agent)

            if self.manager_agent.memory_enabled and not self.manager_agent.memory:
                self.manager_agent.memory = self.cortex_memory

        if self.process == ProcessType.PLANNED and self.planner:
            self.planner.planner_agent.calute_instance = self.calute
            self.planner.planner_agent.cortex_instance = self
            self.planner.planner_agent._logger = self.logger
            if not self.planner.planner_agent.model:
                self.planner.planner_agent.model = model
            self.planner.planner_agent.reinvoke_after_function = self.reinvoke_after_function

            self.calute.register_agent(self.planner.planner_agent._internal_agent)

            if self.planner.planner_agent.memory_enabled and not self.planner.planner_agent.memory:
                self.planner.planner_agent.memory = self.cortex_memory

    def kickoff(
        self,
        use_streaming: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
        streamer_buffer: StreamerBuffer | None = None,
        log_process: bool = False,
    ) -> CortexOutput | tuple[StreamerBuffer, threading.Thread]:
        """Execute the cortex's tasks according to the specified process

        Args:
            use_streaming: If True, executes tasks with streaming in background threads
            stream_callback: Optional callback for streaming chunks

        Returns:
            If use_streaming=False: CortexOutput with results
            If use_streaming=True: tuple of (StreamerBuffer, Thread) for async streaming
        """
        self.logger.info(
            f"🚀 {self.cortex_name} Execution Started (Process: {self.process.value}, Agents: {len(self.agents)}, Tasks: {len(self.tasks)})"
        )

        if log_process and stream_callback is None:
            from calute.loggings import stream_callback

        if use_streaming:
            buffer_was_none = streamer_buffer is None
            buffer = streamer_buffer if streamer_buffer is not None else StreamerBuffer()

            def run_cortex():
                try:
                    start_time = time.time()

                    if self.process == ProcessType.SEQUENTIAL:
                        result = self._run_sequential_streaming(buffer, stream_callback)
                    elif self.process == ProcessType.PARALLEL:
                        result = self._run_parallel()
                        buffer.put(
                            StreamChunk(
                                chunk=None,
                                agent_id="cortex",
                                content=result,
                                buffered_content=result,
                                function_calls_detected=False,
                                reinvoked=False,
                            )
                        )
                    elif self.process == ProcessType.HIERARCHICAL:
                        result = self._run_hierarchical_streaming(buffer, stream_callback)
                    elif self.process == ProcessType.CONSENSUS:
                        result = self._run_consensus(streamer_buffer=buffer, stream_callback=stream_callback)
                    elif self.process == ProcessType.PLANNED:
                        result = self._run_planned_streaming(buffer, stream_callback)
                    else:
                        raise ValueError(f"Unknown process type: {self.process}")

                    execution_time = time.time() - start_time

                    buffer.put(
                        Completion(
                            final_content=result,
                            function_calls_executed=0,
                            agent_id="cortex",
                            execution_history=[],
                        )
                    )

                    buffer.cortex_output = CortexOutput(  # type: ignore
                        raw_output=result,
                        task_outputs=self.task_outputs,
                        execution_time=execution_time,
                    )

                    log_success(f"Cortex execution completed in {execution_time:.2f}s")

                    self.cortex_memory.save_cortex_decision(
                        decision=f"Completed {len(self.tasks)} tasks using {self.process.value} process",
                        context=f"Agents involved: {', '.join([a.role for a in self.agents])}",
                        outcome=f"Successfully completed in {execution_time:.2f} seconds",
                        importance=0.7,
                    )

                except Exception as e:
                    self.logger.error(f"❌ {e!s}")
                    raise
                finally:
                    if buffer_was_none:
                        buffer.close()

            thread = threading.Thread(target=run_cortex, daemon=True)
            thread.start()
            return buffer, thread

        start_time = time.time()

        try:
            if self.process == ProcessType.SEQUENTIAL:
                result = self._run_sequential()
            elif self.process == ProcessType.PARALLEL:
                result = self._run_parallel()
            elif self.process == ProcessType.HIERARCHICAL:
                result = self._run_hierarchical()
            elif self.process == ProcessType.CONSENSUS:
                result = self._run_consensus()
            elif self.process == ProcessType.PLANNED:
                result = self._run_planned()
            else:
                raise ValueError(f"Unknown process type: {self.process}")

            execution_time = time.time() - start_time
            log_success(f"Cortex execution completed in {execution_time:.2f}s")

            self.cortex_memory.save_cortex_decision(
                decision=f"Completed {len(self.tasks)} tasks using {self.process.value} process",
                context=f"Agents involved: {', '.join([a.role for a in self.agents])}",
                outcome=f"Successfully completed in {execution_time:.2f} seconds",
                importance=0.7,
            )

            return CortexOutput(
                raw_output=result,
                task_outputs=self.task_outputs,
                execution_time=execution_time,
            )

        except Exception as e:
            self.logger.error(f"❌ {e!s}")
            raise

    def _stream_agent_execution(
        self,
        agent: CortexAgent,
        task_description: str,
        context: str | None,
        main_buffer: StreamerBuffer,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> str:
        """Helper method to execute agent with streaming and collect output.

        This consolidates the repeated streaming logic used across different execution modes.
        """

        agent.execute(
            task_description=task_description,
            context=context,
            streamer_buffer=main_buffer,
            stream_callback=stream_callback,
        )

        collected_content = []
        streaming_complete = False

        while not streaming_complete:
            try:
                chunk = main_buffer.get(timeout=0.1)
                if chunk is None:
                    agent_thread = getattr(main_buffer, "agent_thread", None)
                    if agent_thread and hasattr(agent_thread, "is_alive"):
                        if not agent_thread.is_alive():
                            streaming_complete = True
                    else:
                        streaming_complete = True
                    continue

                main_buffer.put(chunk)
                if stream_callback:
                    stream_callback(chunk)

                if hasattr(chunk, "content") and chunk.content:
                    collected_content.append(chunk.content)

            except Exception:
                agent_thread = getattr(main_buffer, "agent_thread", None)
                if agent_thread and hasattr(agent_thread, "is_alive"):
                    if not agent_thread.is_alive():
                        streaming_complete = True
                else:
                    streaming_complete = True
                continue

        agent_thread = getattr(main_buffer, "agent_thread", None)
        thread = getattr(main_buffer, "thread", None)
        if agent_thread and hasattr(agent_thread, "join"):
            agent_thread.join(timeout=30)
        elif thread and hasattr(thread, "join"):
            thread.join(timeout=30)

        return "".join(collected_content) if collected_content else ""

    def _run_sequential(self) -> str:
        """Run tasks sequentially, passing context between them"""
        context_outputs = []

        for i, task in enumerate(self.tasks):
            if not hasattr(task, "task_id"):
                task.task_id = str(uuid.uuid4())[:18]
            log_task_start(f"Task {i + 1}/{len(self.tasks)}")

            task_context = []

            if hasattr(task, "dependencies") and task.dependencies:
                for dep_task in task.dependencies:
                    for completed_task in self.task_outputs:
                        if completed_task.task.description == dep_task.description:
                            task_context.append(f"Previous Task ({dep_task.agent.role}): {completed_task.output}")
                            break

            if task.context:
                if context_outputs:
                    for j, prev_output in enumerate(context_outputs, 1):
                        task_context.append(f"Task {j} Output: {prev_output}")

            task_output = task.execute(task_context if (task_context or task.context) else None)

            context_outputs.append(task_output.output)
            self.task_outputs.append(task_output)

            log_success(f"Task completed by {task.agent.role}")

            if task.chain:
                if task.chain.condition and task.chain.condition(task_output.output):
                    if task.chain.next_task:
                        self.tasks.insert(i + 1, task.chain.next_task)
                elif task.chain.fallback_task:
                    self.tasks.insert(i + 1, task.chain.fallback_task)

        return context_outputs[-1] if context_outputs else ""

    def _run_sequential_streaming(
        self,
        buffer: StreamerBuffer,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> str:
        """Run tasks sequentially with streaming support"""
        context_outputs = []
        all_content = []

        for i, task in enumerate(self.tasks):
            if not hasattr(task, "task_id"):
                task.task_id = str(uuid.uuid4())[:18]
            log_task_start(f"Task {i + 1}/{len(self.tasks)}")

            task_context = []

            if hasattr(task, "dependencies") and task.dependencies:
                for dep_task in task.dependencies:
                    for completed_task in self.task_outputs:
                        if completed_task.task.description == dep_task.description:
                            task_context.append(f"Previous Task ({dep_task.agent.role}): {completed_task.output}")
                            break

            if task.context:
                if context_outputs:
                    for j, prev_output in enumerate(context_outputs, 1):
                        task_context.append(f"Task {j} Output: {prev_output}")

            start_chunk = StreamChunk(
                chunk=None,
                agent_id=task.agent.role,
                content=f"\n\n[{task.agent.role}] Starting task {i + 1}/{len(self.tasks)}...\n",
                buffered_content="",
                function_calls_detected=False,
                reinvoked=False,
            )
            buffer.put(start_chunk)
            if stream_callback:
                stream_callback(start_chunk)

            task_description = f"{task.description}\n\nExpected Output: {task.expected_output}"
            context_str = "\n\n".join(task_context) if task_context else None

            output_content = self._stream_agent_execution(
                agent=task.agent,
                task_description=task_description,
                context=context_str,
                main_buffer=buffer,
                stream_callback=stream_callback,
            )

            all_content.append(output_content)

            task_output = CortexTaskOutput(
                task=task,
                output=output_content,
                agent=task.agent,
            )

            context_outputs.append(task_output.output)
            self.task_outputs.append(task_output)

            log_success(f"Task completed by {task.agent.role}")

            if task.chain:
                if task.chain.condition and task.chain.condition(task_output.output):
                    if task.chain.next_task:
                        self.tasks.insert(i + 1, task.chain.next_task)
                elif task.chain.fallback_task:
                    self.tasks.insert(i + 1, task.chain.fallback_task)

        return context_outputs[-1] if context_outputs else ""

    def _run_parallel(self, streamer_buffer: StreamerBuffer | None = None) -> str:
        """Run tasks in parallel using asyncio with optional streaming"""

        cortex_self = self

        async def run_task_async(
            task: CortexTask, context_outputs: list[str], streamer_buffer: StreamerBuffer | None = None
        ) -> CortexTaskOutput:
            if streamer_buffer:
                task_description = f"{task.description}\n\nExpected Output: {task.expected_output}"
                context_str = "\n\n".join(context_outputs) if context_outputs else None

                output_content = await asyncio.to_thread(
                    cortex_self._stream_agent_execution,
                    agent=task.agent,
                    task_description=task_description,
                    context=context_str,
                    main_buffer=streamer_buffer,
                )

                return CortexTaskOutput(
                    task=task,
                    output=output_content,
                    agent=task.agent,
                )
            else:
                task_output = await asyncio.to_thread(
                    task.execute,
                    context_outputs if task.context else None,
                )
                return task_output

        async def run_all_tasks():
            independent_tasks = [t for t in self.tasks if not t.context]
            dependent_tasks = [t for t in self.tasks if t.context]

            if independent_tasks:
                results = await asyncio.gather(
                    *[run_task_async(task, [], streamer_buffer) for task in independent_tasks]
                )
                self.task_outputs.extend(results)

            context_outputs = [r.output for r in self.task_outputs]
            for task in dependent_tasks:
                result = await run_task_async(task, context_outputs, streamer_buffer)
                self.task_outputs.append(result)
                context_outputs.append(result.output)

            return self.task_outputs[-1].output if self.task_outputs else ""

        return asyncio.run(run_all_tasks())

    def _run_hierarchical(self) -> str:
        """Run tasks with a manager agent delegating to workers"""
        if not self.manager_agent:
            raise ValueError("Hierarchical process requires a manager agent")

        self.logger.info("📝 Manager is creating execution plan...")
        manager_prompt = self.template_engine.render_manager_delegation(
            agents=self.agents,
            tasks=self.tasks,
        )

        plan_response = self.manager_agent.execute(
            task_description=manager_prompt,
            context=None,
        )

        try:
            json_match = re.search(r"\{[\s\S]*\}", plan_response)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                raise ValueError("Manager failed to produce a valid JSON execution plan")
        except Exception as e:
            self.logger.error(f"❌ Failed to parse manager plan: {e}")
            raise RuntimeError(f"Manager agent failed to create valid execution plan: {e}") from e

        completed_tasks = {}

        if "execution_plan" not in plan:
            raise ValueError("Manager plan missing 'execution_plan' key")

        for task_plan in plan["execution_plan"]:
            if "task_id" not in task_plan:
                raise ValueError("Task plan missing 'task_id'")
            task_id = task_plan["task_id"] - 1
            if task_id >= len(self.tasks):
                continue

            task = self.tasks[task_id]
            if "assigned_to" not in task_plan:
                raise ValueError(f"Task plan for task_id {task_plan['task_id']} missing 'assigned_to' field")
            assigned_agent_role = task_plan["assigned_to"]

            assigned_agent = None
            for agent in self.agents:
                if agent.role == assigned_agent_role:
                    assigned_agent = agent
                    break

            if not assigned_agent:
                raise ValueError(f"Manager assigned task to non-existent agent: {assigned_agent_role}")

            task.agent = assigned_agent

            self.logger.info(f"📌 Manager delegating Task {task_id + 1} to {assigned_agent.role}")

            context = []
            if "dependencies" in task_plan:
                for dep_id in task_plan["dependencies"]:
                    if dep_id not in completed_tasks:
                        raise ValueError(f"Task {task_id + 1} depends on task {dep_id} which hasn't been completed yet")
                    context.append(completed_tasks[dep_id])

            self.logger.agent_start(assigned_agent.role, task.description)
            task_output = task.execute(context if context else None)
            output = task_output.output
            completed_tasks[task_id + 1] = output

            self.logger.info(f"🔍 Manager reviewing output from {assigned_agent.role}")
            review_prompt = self.template_engine.render_manager_review(
                agent_role=assigned_agent.role,
                task_description=task.description,
                output=output,
            )

            review = self.manager_agent.execute(
                task_description=review_prompt,
                context=None,
            )

            try:
                review_json_match = re.search(r"\{[\s\S]*\}", review)
                if not review_json_match:
                    raise ValueError("Manager review did not contain valid JSON")

                review_data = json.loads(review_json_match.group())
                if "approved" not in review_data:
                    raise ValueError("Manager review missing 'approved' field")

                if not review_data["approved"]:
                    if "improvements_needed" not in review_data:
                        raise ValueError("Manager disapproved but provided no improvements")

                    improvements = review_data["improvements_needed"]
                    if not improvements:
                        raise ValueError("Manager disapproved but improvements list is empty")

                    self.logger.warning(f"⚠️ Manager requested improvements: {', '.join(improvements)}")

                    feedback = review_data.get("feedback", "")
                    improvement_prompt = (
                        f"Please improve your previous output based on this feedback:\n{feedback}\n\n"
                        f"Improvements needed:\n" + "\n".join([f"- {imp}" for imp in improvements])
                    )
                    output = assigned_agent.execute(
                        task_description=improvement_prompt,
                        context=output,
                    )
                    completed_tasks[task_id + 1] = output
            except Exception as e:
                self.logger.error(f"❌ Failed to parse manager review: {e}")
                raise RuntimeError(f"Manager review process failed: {e}") from e

            task_output = CortexTaskOutput(
                task=task,
                output=output,
                agent=assigned_agent,
            )
            self.task_outputs.append(task_output)

        final_summary = self.manager_agent.execute(
            task_description="Provide a final summary of all completed tasks and their outcomes",
            context="\n\n".join([o.output for o in self.task_outputs]),
        )

        return final_summary

    def _run_consensus(
        self,
        streamer_buffer: StreamerBuffer | None = None,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> str:
        """Run tasks with consensus among all agents with optional streaming.

        Each task is executed by all agents, then a consensus is reached.
        """
        final_outputs = []

        for i, task in enumerate(self.tasks, 1):
            task_description = task.description
            if task.expected_output:
                task_description += f"\n\nExpected Output: {task.expected_output}"

            context = "\n\n".join(final_outputs) if final_outputs else None

            self.logger.info(f"🤝 Task {i}/{len(self.tasks)}: Seeking consensus among {len(self.agents)} agents")

            agent_outputs = {}
            for agent in self.agents:
                log_agent_start(agent.role)

                if streamer_buffer:
                    output = self._stream_agent_execution(
                        agent=agent,
                        task_description=task_description,
                        context=context,
                        main_buffer=streamer_buffer,
                        stream_callback=stream_callback,
                    )
                else:
                    output = agent.execute(
                        task_description=task_description,
                        context=context,
                    )

                agent_outputs[agent.role] = output
                log_success(f"{agent.role} completed contribution")

            self.logger.info("🔮 Synthesizing consensus from all agent outputs...")
            consensus_prompt = self.template_engine.render_consensus(
                task_description=task_description,
                agent_outputs=agent_outputs,
            )

            lead_agent = task.agent if task.agent else self.agents[0]

            if streamer_buffer:
                consensus = self._stream_agent_execution(
                    agent=lead_agent,
                    task_description=consensus_prompt,
                    context=None,
                    main_buffer=streamer_buffer,
                    stream_callback=stream_callback,
                )
            else:
                consensus = lead_agent.execute(
                    task_description=consensus_prompt,
                    context=None,
                )

            final_outputs.append(consensus)

            task_output = CortexTaskOutput(
                task=task,
                output=consensus,
                agent=lead_agent,
            )
            self.task_outputs.append(task_output)

            log_success(f"Consensus reached for task {i}/{len(self.tasks)}")

        return final_outputs[-1] if final_outputs else ""

    def _run_planned(self) -> str:
        """Run tasks using XML-based planning"""
        if not self.planner:
            raise ValueError("Planner not initialized for PLANNED process type")

        if not self.tasks:
            raise ValueError("No tasks provided for planning")

        objective = "Complete the following objectives:\n"
        for i, task in enumerate(self.tasks, 1):
            objective += f"{i}. {task.description}\n"
            if task.expected_output:
                objective += f"   Expected output: {task.expected_output}\n"

        if self.verbose:
            self.logger.info("🧠 Creating execution plan for objective")

        execution_plan = self.planner.create_plan(
            objective=objective.strip(),
            available_agents=self.agents,
            context=f"Total tasks: {len(self.tasks)}, Agents available: {len(self.agents)}",
        )

        if self.verbose:
            self.logger.info(f"📋 Executing plan with {len(execution_plan.steps)} steps")

        step_results = self.planner.execute_plan(execution_plan, self.tasks)

        final_outputs = []
        for step_id, result in step_results.items():
            final_outputs.append(f"Step {step_id} result: {result}")

        for i, task in enumerate(self.tasks):
            if i < len(step_results):
                result_key = list(step_results.keys())[i]
                result = step_results[result_key]
            else:
                result = "Task completed as part of the execution plan"

            agent = task.agent if task.agent else self.agents[0]

            task_output = CortexTaskOutput(task=task, output=result, agent=agent)
            self.task_outputs.append(task_output)

        return final_outputs[-1] if final_outputs else "Planning execution completed"

    def _run_hierarchical_streaming(
        self,
        buffer: StreamerBuffer,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> str:
        """Run hierarchical process with proper streaming support"""
        if not self.manager_agent:
            raise ValueError("Hierarchical process requires a manager agent")

        self.logger.info("📝 Manager is creating execution plan...")
        manager_prompt = self.template_engine.render_manager_delegation(
            agents=self.agents,
            tasks=self.tasks,
        )

        plan_response = self._stream_agent_execution(
            agent=self.manager_agent,
            task_description=manager_prompt,
            context=None,
            main_buffer=buffer,
            stream_callback=stream_callback,
        )

        try:
            json_match = re.search(r"\{[\s\S]*\}", plan_response)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                raise ValueError("Manager failed to produce a valid JSON execution plan")
        except Exception as e:
            self.logger.error(f"❌ Failed to parse manager plan: {e}")
            raise RuntimeError(f"Manager agent failed to create valid execution plan: {e}") from e

        completed_tasks = {}

        for task_plan in plan.get("execution_plan", []):
            task_id = task_plan.get("task_id", 1) - 1
            if task_id >= len(self.tasks):
                continue

            task = self.tasks[task_id]
            assigned_agent_role = task_plan.get("assigned_to")

            assigned_agent = None
            for agent in self.agents:
                if agent.role == assigned_agent_role:
                    assigned_agent = agent
                    break

            if not assigned_agent:
                raise ValueError(f"Manager assigned task to non-existent agent: {assigned_agent_role}")

            task.agent = assigned_agent
            self.logger.info(f"📌 Manager delegating Task {task_id + 1} to {assigned_agent.role}")

            context = []
            if "dependencies" in task_plan:
                for dep_id in task_plan["dependencies"]:
                    if dep_id in completed_tasks:
                        context.append(completed_tasks[dep_id])

            output = self._stream_agent_execution(
                agent=task.agent,
                task_description=task.description,
                context="\n\n".join(context) if context else None,
                main_buffer=buffer,
                stream_callback=stream_callback,
            )
            completed_tasks[task_id + 1] = output

            task_output = CortexTaskOutput(
                task=task,
                output=output,
                agent=assigned_agent,
            )
            self.task_outputs.append(task_output)

        return completed_tasks.get(len(self.tasks), "")

    def _run_planned_streaming(
        self,
        buffer: StreamerBuffer,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> str:
        """Run planned process with proper streaming support"""
        if not self.planner:
            raise ValueError("Planner not initialized for PLANNED process type")

        if not self.tasks:
            raise ValueError("No tasks provided for planning")

        objective = "Complete the following objectives:\n"
        for i, task in enumerate(self.tasks, 1):
            objective += f"{i}. {task.description}\n"
            if task.expected_output:
                objective += f"   Expected output: {task.expected_output}\n"

        if self.verbose:
            self.logger.info("🧠 Creating execution plan for objective")

        execution_plan = self.planner.create_plan(
            objective=objective.strip(),
            available_agents=self.agents,
            context=f"Total tasks: {len(self.tasks)}, Agents available: {len(self.agents)}",
            streamer_buffer=buffer,
            stream_callback=stream_callback,
        )

        if self.verbose:
            self.logger.info(f"📋 Executing plan with {len(execution_plan.steps)} steps")

        for i, task in enumerate(self.tasks):
            if i >= len(execution_plan.steps):
                break

            step = execution_plan.steps[i]
            assigned_agent = None
            if hasattr(step, "assigned_agent"):
                for agent in self.agents:
                    if agent.role == step.assigned_agent:
                        assigned_agent = agent
                        break

            if not assigned_agent:
                assigned_agent = task.agent if task.agent else self.agents[0]

            task_context = []
            if i > 0 and self.task_outputs:
                for prev_output in self.task_outputs:
                    task_context.append(prev_output.output)

            task_description = f"{task.description}\n\nExpected Output: {task.expected_output}"
            context_str = "\n\n".join(task_context) if task_context else None

            output = self._stream_agent_execution(
                agent=assigned_agent,
                task_description=task_description,
                context=context_str,
                main_buffer=buffer,
                stream_callback=stream_callback,
            )

            task_output = CortexTaskOutput(task=task, output=output, agent=assigned_agent)
            self.task_outputs.append(task_output)

        return self.task_outputs[-1].output if self.task_outputs else "Planning execution completed"

    @classmethod
    def from_task_creator(
        cls,
        tasks: list[CortexTask],
        llm: BaseLLM | None = None,
        agents: list[CortexAgent] | None = None,
        process: ProcessType = ProcessType.SEQUENTIAL,
        manager_agent: CortexAgent | None = None,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        verbose: bool = True,
        max_iterations: int = 10,
        model: str = "gpt-4",
        memory: CortexMemory | None = None,
        memory_config: MemoryConfig | None = None,
        reinvoke_after_function: bool = True,
        enable_calute_memory: bool = False,
    ):
        if llm is None:
            agent = tasks[0].agent
            if isinstance(agent, list):
                agent = agent[0]
            llm = agent.llm
        _agents = []
        if agents is None:
            for task in tasks:
                if isinstance(task.agent, list):
                    _agents.extend(task.agent)
                else:
                    _agents.append(task.agent)
            agents = list(set(_agents))
        return Cortex(
            agents=agents,
            tasks=tasks,
            cortex_name="AutoCortex",
            llm=llm,
            enable_calute_memory=enable_calute_memory,
            manager_agent=manager_agent,
            max_iterations=max_iterations,
            memory=memory,
            memory_config=memory_config,
            memory_type=memory_type,
            model=model,
            process=process,
            reinvoke_after_function=reinvoke_after_function,
            verbose=verbose,
        )

    def get_memory_summary(self) -> str:
        """Get a summary of the cortex's memory"""
        return self.cortex_memory.get_summary()

    def save_memory(self, persistence_path: str | None = None) -> None:
        """Save the cortex's memory to disk"""
        if persistence_path and self.cortex_memory.storage:
            self.cortex_memory.storage.db_path = Path(persistence_path)

    def clear_short_term_memory(self) -> None:
        """Clear the cortex's short-term memory"""
        self.cortex_memory.reset_short_term()

    def clear_all_memory(self) -> None:
        """Clear all cortex memory"""
        self.cortex_memory.reset_all()

    def create_ui(self):
        from calute.ui import create_application

        return create_application(executor=self)


@dataclass
class CortexOutput:
    """Output from Cortex execution"""

    raw_output: str
    task_outputs: list[CortexTaskOutput]
    execution_time: float

    def __str__(self) -> str:
        return self.raw_output

    def to_dict(self) -> dict:
        """Convert output to dictionary"""
        return {
            "raw_output": self.raw_output,
            "task_outputs": [
                {
                    "task": t.task.description,
                    "output": t.output,
                    "agent": t.agent.role,
                    "timestamp": t.timestamp,
                }
                for t in self.task_outputs
            ],
            "execution_time": self.execution_time,
        }
