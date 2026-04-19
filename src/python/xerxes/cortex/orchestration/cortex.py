# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Main Cortex orchestration module for multi-agent collaboration.

This module provides the core Cortex orchestrator class that coordinates multiple
AI agents to accomplish complex tasks through various execution strategies. It serves
as the central hub for managing agent lifecycles, task execution, memory integration,
and output collection.

The Cortex orchestrator supports multiple process types:
    - SEQUENTIAL: Tasks are executed one after another, with context passing
    - PARALLEL: Independent tasks run concurrently for faster execution
    - HIERARCHICAL: A manager agent delegates tasks to worker agents
    - CONSENSUS: All agents contribute to each task, then reach agreement
    - PLANNED: An AI planner creates an optimized execution plan

Key Components:
    - Cortex: Main orchestrator class managing agents, tasks, and execution
    - CortexOutput: Structured output container with execution metadata
    - MemoryConfig: TypedDict for configuring memory system parameters

Typical usage example:
    >>> from xerxes.cortex import Cortex, CortexAgent, CortexTask
    >>> from xerxes.llms import OpenAILLM
    >>>
    >>> llm = OpenAILLM(api_key="your-key")
    >>> agent = CortexAgent(role="Writer", goal="Write content", backstory="Expert writer")
    >>> task = CortexTask(description="Write an article", expected_output="Article text", agent=agent)
    >>> cortex = Cortex(agents=[agent], tasks=[task], llm=llm)
    >>> result = cortex.kickoff()
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from ...core.streamer_buffer import StreamerBuffer
from ...llms import BaseLLM
from ...logging.console import get_logger, log_agent_start, log_success, log_task_start
from ...memory import MemoryStore, MemoryType
from ...types import Completion, StreamChunk
from ...xerxes import Xerxes
from ..agents.agent import CortexAgent
from ..agents.memory_integration import CortexMemory
from ..core.enums import ProcessType
from ..core.templates import PromptTemplate
from .planner import CortexPlanner
from .task import CortexTask, CortexTaskOutput


class MemoryConfig(TypedDict, total=False):
    """Configuration dictionary for the Cortex memory system.

    This TypedDict defines the configuration options for both CortexMemory
    and Xerxes's internal memory systems. All fields are optional and will
    use sensible defaults if not specified.

    Attributes:
        max_short_term: Maximum entries in Xerxes's short-term memory. Defaults to 100.
        max_working: Maximum entries in Xerxes's working memory. Defaults to 10.
        max_long_term: Maximum entries in Xerxes's long-term memory. Defaults to 1000.
        enable_short_term: Enable CortexMemory short-term storage. Defaults to True.
        enable_long_term: Enable CortexMemory long-term storage. Defaults to True.
        enable_entity: Enable entity tracking in CortexMemory. Defaults to True.
        enable_user: Enable user-specific memory in CortexMemory. Defaults to False.
        persistence_path: File path for persisting memory to disk. Defaults to None.
        short_term_capacity: Maximum entries in CortexMemory short-term. Defaults to 50.
        long_term_capacity: Maximum entries in CortexMemory long-term. Defaults to 5000.

    Example:
        >>> config: MemoryConfig = {
        ...     "enable_short_term": True,
        ...     "enable_long_term": True,
        ...     "short_term_capacity": 100,
        ...     "persistence_path": "./memory.db"
        ... }
    """

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
    """Main orchestrator for multi-agent collaboration and task execution.

    Cortex is the central coordination hub that manages multiple AI agents working
    together on complex tasks. It handles task distribution, execution flow, memory
    management, and result collection across different execution strategies.

    The orchestrator supports five distinct process types:
        - SEQUENTIAL: Execute tasks one after another, passing context between them
        - PARALLEL: Run independent tasks concurrently for improved performance
        - HIERARCHICAL: Use a manager agent to delegate and review worker tasks
        - CONSENSUS: Gather contributions from all agents and synthesize agreement
        - PLANNED: Use AI-powered planning to optimize task execution order

    Attributes:
        agents: List of CortexAgent instances available for task execution.
        tasks: List of CortexTask instances to be executed.
        process: The ProcessType determining execution strategy.
        manager_agent: Optional agent for hierarchical process management.
        verbose: Whether to enable detailed logging output.
        max_iterations: Maximum retry attempts for failed task executions.
        reinvoke_after_function: Whether to reinvoke LLM after tool execution.
        enable_xerxes_memory: Whether to enable Xerxes's internal memory system.
        cortex_name: Display name for the Cortex instance in logs.
        cortex_memory: CortexMemory instance for shared memory across agents.
        memory: MemoryStore instance for basic memory operations.
        memory_type: Type of memory to use for context management.
        task_outputs: List of CortexTaskOutput from completed tasks.
        logger: Logger instance for verbose output.
        template_engine: PromptTemplate instance for generating prompts.
        planner: CortexPlanner instance for PLANNED process type (if applicable).
        llm: BaseLLM instance for language model interactions.
        xerxes: Xerxes instance managing agent registrations and execution.

    Example:
        >>> from xerxes.cortex import Cortex, CortexAgent, CortexTask, ProcessType
        >>> from xerxes.llms import OpenAILLM
        >>>
        >>> llm = OpenAILLM(api_key="your-key")
        >>> writer = CortexAgent(
        ...     role="Content Writer",
        ...     goal="Create engaging content",
        ...     backstory="Professional writer with 10 years experience"
        ... )
        >>> editor = CortexAgent(
        ...     role="Editor",
        ...     goal="Polish and improve content",
        ...     backstory="Detail-oriented editor"
        ... )
        >>> write_task = CortexTask(
        ...     description="Write an article about AI",
        ...     expected_output="Draft article",
        ...     agent=writer
        ... )
        >>> edit_task = CortexTask(
        ...     description="Edit and improve the article",
        ...     expected_output="Polished article",
        ...     agent=editor,
        ...     context=True
        ... )
        >>> cortex = Cortex(
        ...     agents=[writer, editor],
        ...     tasks=[write_task, edit_task],
        ...     llm=llm,
        ...     process=ProcessType.SEQUENTIAL
        ... )
        >>> result = cortex.kickoff()
        >>> print(result.raw_output)
    """

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
        enable_xerxes_memory: bool = False,
        cortex_name: str = "CorTex",
        parallel_max_workers: int | None = None,
    ) -> None:
        """Initialize a new Cortex orchestrator instance.

        Sets up the orchestration environment including agent registration,
        memory configuration, and process-specific components like planners
        or manager agents.

        Args:
            agents: List of CortexAgent instances to participate in execution.
            tasks: List of CortexTask instances defining the work to be done.
            llm: BaseLLM instance for language model operations.
            process: Execution strategy to use. Defaults to SEQUENTIAL.
            manager_agent: Agent for hierarchical delegation. Auto-created if
                needed but not provided.
            memory_type: Type of memory for context management.
            verbose: Enable detailed logging. Defaults to True.
            max_iterations: Maximum retry attempts. Defaults to 10.
            model: Default model identifier for agents. Defaults to "gpt-4".
            memory: Pre-configured CortexMemory instance. Created if not provided.
            memory_config: Configuration dict for memory systems.
            reinvoke_after_function: Reinvoke LLM after tool calls. Defaults to True.
            enable_xerxes_memory: Enable Xerxes's memory. Defaults to False.
            cortex_name: Display name for logging. Defaults to "CorTex".
            parallel_max_workers: Optional maximum number of worker threads to use
                for `ProcessType.PARALLEL`. If not provided, Python's default thread
                pool sizing behavior is used.

        Raises:
            ValueError: If a task has no assigned agent and process is not HIERARCHICAL.
        """
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.manager_agent = manager_agent
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.reinvoke_after_function = reinvoke_after_function
        self.enable_xerxes_memory = enable_xerxes_memory
        self.cortex_name = cortex_name
        self.parallel_max_workers = parallel_max_workers
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
        xerxes_memory_config = {
            "max_short_term": config.get("max_short_term", 100),
            "max_working": config.get("max_working", 10),
            "max_long_term": config.get("max_long_term", 1000),
        }

        self.llm = llm
        self.xerxes = Xerxes(
            llm=self.llm,
            enable_memory=self.enable_xerxes_memory,
            memory_config=xerxes_memory_config,
        )

        for agent in self.agents:
            agent.xerxes_instance = self.xerxes
            agent.cortex_instance = self
            agent._logger = self.logger
            if not agent.model:
                agent.model = model

            agent.reinvoke_after_function = self.reinvoke_after_function

            self.xerxes.register_agent(agent._internal_agent)

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
            self.manager_agent.xerxes_instance = self.xerxes
            self.manager_agent.cortex_instance = self
            self.manager_agent._logger = self.logger
            if not self.manager_agent.model:
                self.manager_agent.model = model

            self.manager_agent.reinvoke_after_function = self.reinvoke_after_function

            self.xerxes.register_agent(self.manager_agent._internal_agent)

            if self.manager_agent.memory_enabled and not self.manager_agent.memory:
                self.manager_agent.memory = self.cortex_memory

        if self.process == ProcessType.PLANNED and self.planner:
            self.planner.planner_agent.xerxes_instance = self.xerxes
            self.planner.planner_agent.cortex_instance = self
            self.planner.planner_agent._logger = self.logger
            if not self.planner.planner_agent.model:
                self.planner.planner_agent.model = model
            self.planner.planner_agent.reinvoke_after_function = self.reinvoke_after_function

            self.xerxes.register_agent(self.planner.planner_agent._internal_agent)

            if self.planner.planner_agent.memory_enabled and not self.planner.planner_agent.memory:
                self.planner.planner_agent.memory = self.cortex_memory

    def _run_async_coro(self, coro):
        """Run an async coroutine safely from sync code, even inside an existing event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()

    def _interpolate_inputs(self, inputs: dict[str, Any]) -> None:
        """Interpolate template variables into all agents and tasks.

        Replaces ``{variable}`` placeholders in all registered agents' roles,
        goals, backstories, and instructions, as well as task descriptions,
        expected outputs, and other template-enabled fields.

        Args:
            inputs: Dictionary mapping template variable names to their values.
                Values are substituted into all agents (including manager and
                planner agents) and all tasks.

        Side Effects:
            - Modifies agent attributes (role, goal, backstory, instructions).
            - Modifies task attributes (description, expected_output, etc.).
            - Original values are preserved internally for re-interpolation.
        """

        for agent in self.agents:
            agent.interpolate_inputs(inputs)

        if self.manager_agent:
            self.manager_agent.interpolate_inputs(inputs)

        if self.planner and self.planner.planner_agent:
            self.planner.planner_agent.interpolate_inputs(inputs)

        for task in self.tasks:
            task.interpolate_inputs(inputs)

    def kickoff(
        self,
        inputs: dict[str, Any] | None = None,
        use_streaming: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
        streamer_buffer: StreamerBuffer | None = None,
        log_process: bool = False,
    ) -> CortexOutput | tuple[StreamerBuffer, threading.Thread]:
        """Execute the Cortex's tasks according to the configured process type.

        Main entry point for executing all registered tasks. Dispatches to the
        appropriate execution strategy (sequential, parallel, hierarchical,
        consensus, or planned) based on ``self.process``. Optionally interpolates
        template variables and supports streaming execution.

        Args:
            inputs: Optional dictionary of template variable values to interpolate
                into all agent and task templates before execution.
            use_streaming: If True, executes tasks in a background thread with
                real-time streaming output. Defaults to False (blocking execution).
            stream_callback: Optional callback function invoked with each stream
                chunk during execution. If ``log_process`` is True and no callback
                is provided, a default logging callback is used.
            streamer_buffer: Optional pre-existing StreamerBuffer for collecting
                streaming output. A new buffer is created if None.
            log_process: If True, enables logging of the execution process via a
                default stream callback when no custom callback is provided.
                Defaults to False.

        Returns:
            If ``use_streaming=False``: A CortexOutput containing the final result
            string, individual task outputs, and total execution time.
            If ``use_streaming=True``: A tuple of (StreamerBuffer, Thread) for
            asynchronous consumption of streaming output.

        Raises:
            ValueError: If an unknown process type is configured.
            Exception: Re-raises any exceptions from task execution.

        Side Effects:
            - Interpolates inputs into agents and tasks if ``inputs`` is provided.
            - Populates ``self.task_outputs`` with results from each task.
            - Saves a cortex-level decision to memory upon completion.
        """

        if inputs:
            self._interpolate_inputs(inputs)

        self.logger.info(
            f"🚀 {self.cortex_name} Execution Started (Process: {self.process.value}, Agents: {len(self.agents)}, Tasks: {len(self.tasks)})"
        )

        if log_process and stream_callback is None:
            from xerxes.logging.console import stream_callback as default_stream_callback

            stream_callback = default_stream_callback

        if use_streaming:
            buffer_was_none = streamer_buffer is None
            buffer = streamer_buffer if streamer_buffer is not None else StreamerBuffer()

            def run_cortex() -> None:
                """Execute the full Cortex workflow in a background thread and write results to buffer."""
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
        """Execute an agent with streaming support and collect output.

        This helper method consolidates the streaming logic used across different
        execution modes. It handles stream chunk collection, callback invocation,
        and thread synchronization for streaming agent execution.

        Args:
            agent: The CortexAgent to execute the task.
            task_description: The description of the task to perform.
            context: Optional context string from previous task outputs.
            main_buffer: StreamerBuffer for collecting streaming output.
            stream_callback: Optional callback invoked for each stream chunk.

        Returns:
            The collected output content as a concatenated string.

        Note:
            This method blocks until the agent's streaming execution completes
            and all output has been collected from the buffer.
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
        """Execute tasks sequentially with context passing between them.

        Tasks are executed one after another, with each task potentially
        receiving context from previously completed tasks. Supports task
        dependencies and conditional chaining.

        Returns:
            The output string from the last completed task.

        Note:
            Task dependencies are resolved before execution. If a task has
            a chain defined, conditional branching may insert additional tasks.
        """
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
        """Execute tasks sequentially with streaming output support.

        Similar to _run_sequential but with real-time streaming of agent
        outputs. Each task's output is streamed as it's generated.

        Args:
            buffer: StreamerBuffer for collecting streaming output chunks.
            stream_callback: Optional callback invoked for each stream chunk.

        Returns:
            The output string from the last completed task.
        """
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
        """Execute tasks in parallel using asyncio for concurrent processing.

        Independent tasks (those without context dependencies) run concurrently,
        while dependent tasks wait for their prerequisites to complete.

        Args:
            streamer_buffer: Optional StreamerBuffer for streaming output.

        Returns:
            The output string from the last completed task.

        Note:
            Uses asyncio.gather for concurrent execution. Tasks with context=True
            are executed after independent tasks complete.
        """

        cortex_self = self

        async def run_task_async(
            task: CortexTask,
            context_outputs: list[str],
            executor: ThreadPoolExecutor | None = None,
            streamer_buffer: StreamerBuffer | None = None,
        ) -> CortexTaskOutput:
            """Execute a single task asynchronously in a thread-pool executor.

            Args:
                task: The CortexTask to execute.
                context_outputs: Previous task outputs to pass as context.
                executor: Optional ThreadPoolExecutor for thread management.
                streamer_buffer: Optional buffer for streaming output.

            Returns:
                CortexTaskOutput containing the task result and metadata.
            """
            loop = asyncio.get_running_loop()

            if streamer_buffer:
                task_description = f"{task.description}\n\nExpected Output: {task.expected_output}"
                context_str = "\n\n".join(context_outputs) if context_outputs else None

                output_content = await loop.run_in_executor(
                    executor,
                    lambda: cortex_self._stream_agent_execution(
                        agent=task.agent,
                        task_description=task_description,
                        context=context_str,
                        main_buffer=streamer_buffer,
                    ),
                )

                return CortexTaskOutput(
                    task=task,
                    output=output_content,
                    agent=task.agent,
                )
            else:
                task_output = await loop.run_in_executor(
                    executor,
                    lambda: task.execute(context_outputs if task.context else None),
                )
                return task_output

        async def run_all_tasks() -> str:
            """Gather independent and dependent tasks, run them concurrently, and return final output."""
            independent_tasks = [t for t in self.tasks if not t.context]
            dependent_tasks = [t for t in self.tasks if t.context]

            max_workers = self.parallel_max_workers
            if max_workers is not None:
                max_workers = max(1, max_workers)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if independent_tasks:
                    results = await asyncio.gather(
                        *[run_task_async(task, [], executor, streamer_buffer) for task in independent_tasks]
                    )
                    self.task_outputs.extend(results)

                context_outputs = [r.output for r in self.task_outputs]
                for task in dependent_tasks:
                    result = await run_task_async(task, context_outputs, executor, streamer_buffer)
                    self.task_outputs.append(result)
                    context_outputs.append(result.output)

            return self.task_outputs[-1].output if self.task_outputs else ""

        return self._run_async_coro(run_all_tasks())

    def _run_hierarchical(self) -> str:
        """Execute tasks with a manager agent delegating to worker agents.

        The manager agent creates an execution plan, assigns tasks to appropriate
        worker agents, monitors progress, and reviews outputs for quality. The
        manager may request improvements if outputs don't meet expectations.

        Returns:
            A final summary from the manager agent after all tasks complete.

        Raises:
            ValueError: If no manager agent is configured or if the manager's
                execution plan is invalid.
            RuntimeError: If the manager fails to create a valid plan or review.
        """
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
            if task_id < 0 or task_id >= len(self.tasks):
                self.logger.warning(
                    f"⚠️ Skipping invalid task_id {task_plan['task_id']} (valid range: 1-{len(self.tasks)})"
                )
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

            log_agent_start(assigned_agent.role)
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
        """Execute tasks with consensus among all agents, then synthesize agreement.

        Each task is processed by every configured agent independently. After
        all agents contribute, a synthesis prompt is sent to the lead agent (or
        the first agent) to produce a unified consensus response. Results from
        previous tasks are passed as context to subsequent tasks.

        Args:
            streamer_buffer: Optional StreamerBuffer for streaming agent outputs.
            stream_callback: Optional callback invoked for each stream chunk.

        Returns:
            The consensus output from the final task, or an empty string if
            no tasks were executed.
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
        """Execute tasks using AI-powered planning for optimized sequencing.

        The CortexPlanner creates an intelligent execution plan based on task
        descriptions and available agents, then executes steps according to
        the plan's recommended order and agent assignments.

        Returns:
            The result from the final step in the execution plan.

        Raises:
            ValueError: If the planner is not initialized or no tasks are provided.
        """
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
        """Execute hierarchical process with streaming output support.

        Similar to _run_hierarchical but with real-time streaming of all
        agent outputs including the manager's planning and delegation.

        Args:
            buffer: StreamerBuffer for collecting streaming output chunks.
            stream_callback: Optional callback invoked for each stream chunk.

        Returns:
            The output from the last completed task.

        Raises:
            ValueError: If no manager agent is configured or agent assignment fails.
            RuntimeError: If the manager fails to create a valid execution plan.
        """
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
        """Execute planned process with streaming output support.

        Similar to _run_planned but with real-time streaming of all agent
        outputs as the execution plan is carried out.

        Args:
            buffer: StreamerBuffer for collecting streaming output chunks.
            stream_callback: Optional callback invoked for each stream chunk.

        Returns:
            The output from the last completed task, or a default message
            if no tasks were executed.

        Raises:
            ValueError: If the planner is not initialized or no tasks are provided.
        """
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
        enable_xerxes_memory: bool = False,
    ) -> Cortex:
        """Create a Cortex instance from tasks with auto-detected agents and LLM.

        Factory method that automatically extracts agents from the provided tasks
        and infers the LLM from the first task's agent. Useful when tasks have
        already been configured with their assigned agents.

        Args:
            tasks: List of CortexTask instances with agents already assigned.
            llm: Optional BaseLLM instance. If None, extracted from first task's agent.
            agents: Optional agent list. If None, extracted from all tasks.
            process: Execution strategy to use. Defaults to SEQUENTIAL.
            manager_agent: Optional agent for hierarchical delegation.
            memory_type: Type of memory for context management.
            verbose: Enable detailed logging. Defaults to True.
            max_iterations: Maximum retry attempts. Defaults to 10.
            model: Default model identifier. Defaults to "gpt-4".
            memory: Pre-configured CortexMemory instance.
            memory_config: Configuration dict for memory systems.
            reinvoke_after_function: Reinvoke LLM after tool calls. Defaults to True.
            enable_xerxes_memory: Enable Xerxes's memory. Defaults to False.

        Returns:
            A configured Cortex instance with name "AutoCortex".

        Example:
            >>> from xerxes.cortex import Cortex, CortexTask, TaskCreator
            >>> creator = TaskCreator(llm=llm, agents=[agent1, agent2])
            >>> tasks = creator.create_tasks("Build a website")
            >>> cortex = Cortex.from_task_creator(tasks)
            >>> result = cortex.kickoff()
        """
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
                elif task.agent is not None:
                    _agents.append(task.agent)

            seen: set = set()
            agents = []
            for a in _agents:
                if a not in seen:
                    seen.add(a)
                    agents.append(a)
        return Cortex(
            agents=agents,
            tasks=tasks,
            cortex_name="AutoCortex",
            llm=llm,
            enable_xerxes_memory=enable_xerxes_memory,
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
        """Get a human-readable summary of the Cortex's memory state.

        Returns:
            A formatted string summarizing short-term, long-term, and entity
            memories currently stored in the CortexMemory system.
        """
        return self.cortex_memory.get_summary()

    def save_memory(self, persistence_path: str | None = None) -> None:
        """Save the Cortex's memory to disk for later retrieval.

        Args:
            persistence_path: Optional file path for saving. If provided,
                updates the storage path before saving.
        """
        if persistence_path and self.cortex_memory.storage:
            self.cortex_memory.storage.db_path = Path(persistence_path)

    def clear_short_term_memory(self) -> None:
        """Clear the Cortex's short-term memory while preserving long-term memories.

        Useful for starting fresh between sessions while retaining
        important learned information.
        """
        self.cortex_memory.reset_short_term()

    def clear_all_memory(self) -> None:
        """Clear all Cortex memory including short-term, long-term, and entity memories.

        Warning:
            This operation is irreversible. All stored memories will be permanently deleted.
        """
        self.cortex_memory.reset_all()


@dataclass
class CortexOutput:
    """Structured output container from Cortex execution.

    Contains the final output from a Cortex execution along with metadata
    about each individual task's output, timing information, and agent
    assignments. Provides convenient methods for accessing and serializing
    the execution results.

    Attributes:
        raw_output: The final output string from the last completed task.
        task_outputs: List of CortexTaskOutput instances for each completed task.
        execution_time: Total execution time in seconds for the entire workflow.

    Example:
        >>> result = cortex.kickoff()
        >>> print(result.raw_output)
        >>> print(f"Completed in {result.execution_time:.2f}s")
        >>> for task_output in result.task_outputs:
        ...     print(f"{task_output.agent.role}: {task_output.output[:100]}...")
    """

    raw_output: str
    task_outputs: list[CortexTaskOutput]
    execution_time: float

    def __str__(self) -> str:
        """Return the raw output when converted to string.

        Returns:
            The raw_output string from the execution.
        """
        return self.raw_output

    def to_dict(self) -> dict:
        """Convert the output to a serializable dictionary format.

        Creates a dictionary representation of the execution results
        suitable for JSON serialization or API responses.

        Returns:
            Dictionary containing raw_output, task_outputs with task
            descriptions, outputs, agent roles, timestamps, and
            the total execution_time.
        """
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
