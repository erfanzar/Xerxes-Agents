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
"""Core orchestration engine for multi-agent Cortex runs.

Defines :class:`Cortex`, the top-level orchestrator that owns a set of
:class:`CortexAgent` instances and a list of :class:`CortexTask` items and
runs them under one of five :class:`ProcessType` topologies — SEQUENTIAL,
PARALLEL, HIERARCHICAL, CONSENSUS, or PLANNED. The module also exposes
:class:`CortexOutput`, the wire-friendly result envelope, and the
:class:`MemoryConfig` ``TypedDict`` for memory wiring.
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


def _resolve_execute_output(value: Any) -> str:
    """Coerce a :meth:`CortexAgent.execute` return value into plain text.

    ``execute`` may return either a string or a ``(buffer, ...)`` tuple
    whose first element exposes ``get_result(timeout)``. The historical
    ``is not None`` check on the bound method was always truthy and so
    the string fallback was unreachable; ``callable(getattr(...))``
    correctly distinguishes a streamer buffer from a plain object.
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, tuple) or not value:
        return str(value)
    head = value[0]
    getter = getattr(head, "get_result", None)
    if callable(getter):
        result = getter(1.0)
        return result if result is not None else str(head)
    return str(head)


class MemoryConfig(TypedDict, total=False):
    """Optional kwargs bundle for configuring a :class:`Cortex` memory stack.

    Two groups of keys exist:

    * ``max_short_term``, ``max_working``, ``max_long_term`` — sizing for
      the internal :class:`Xerxes` runner's memory subsystem.
    * ``enable_*``, ``persistence_path``, ``short_term_capacity``,
      ``long_term_capacity`` — sizing and toggles for :class:`CortexMemory`
      (short-term / long-term / entity / user tiers).
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
    """Top-level multi-agent orchestrator.

    A :class:`Cortex` owns a list of :class:`CortexAgent` and a list of
    :class:`CortexTask` items plus a shared :class:`Xerxes` LLM runner.
    :meth:`kickoff` executes every task under one of five topologies:

    * ``SEQUENTIAL`` — tasks run in order; each task receives the prior
      tasks' outputs as context.
    * ``PARALLEL`` — context-free tasks run concurrently on a thread
      pool; tasks that opt into context still run sequentially after
      the independent batch completes.
    * ``HIERARCHICAL`` — a manager agent produces a JSON delegation plan,
      every step runs through the assigned agent, and the manager
      reviews each output (with an improvement loop) before producing a
      final summary.
    * ``CONSENSUS`` — every agent attempts each task and a lead agent
      synthesises the answers into one consensus response.
    * ``PLANNED`` — a :class:`CortexPlanner` produces an XML execution
      plan; each step runs through the agent the plan named.

    The class also owns a :class:`CortexMemory` instance that backs every
    agent's recall and a :class:`MemoryStore` used by the runner.
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
        """Wire agents, tasks and the shared LLM runner together.

        Side effects:

        * Every agent has its ``xerxes_instance``, ``cortex_instance``,
          ``model`` and ``memory`` fields populated.
        * Every internal :class:`XerxesAgent` is registered with the
          shared :class:`Xerxes` runner.
        * In ``HIERARCHICAL`` mode a default manager agent is created if
          none was supplied; in ``PLANNED`` mode the planner agent is
          wired in the same way as the regular agents.
        * Tasks without an assigned agent are rejected outside of
          ``HIERARCHICAL`` mode (where the manager assigns at runtime).

        Args:
            agents: Agents available to the orchestrator.
            tasks: Ordered task list.
            llm: LLM backend used to build the internal :class:`Xerxes`.
            process: Topology applied by :meth:`kickoff`.
            manager_agent: Required for ``HIERARCHICAL``; auto-created if
                ``None``.
            memory_type: Memory tier used by the runner's
                :class:`MemoryStore`.
            verbose: Forward verbose logging into every agent.
            max_iterations: Per-agent retry budget.
            model: Default model id assigned to agents lacking one.
            memory: Pre-built :class:`CortexMemory`; overrides
                ``memory_config`` when supplied.
            memory_config: Optional dict of memory tier toggles and caps.
            reinvoke_after_function: Mirrored onto every agent.
            enable_xerxes_memory: Toggle the runner's internal memory.
            cortex_name: Display name used in log messages.
            parallel_max_workers: Cap for the ``PARALLEL`` thread pool;
                ``None`` lets :class:`ThreadPoolExecutor` decide.
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

            if agent._internal_agent is not None:
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

            if self.manager_agent._internal_agent is not None:
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

            if self.planner.planner_agent._internal_agent is not None:
                self.xerxes.register_agent(self.planner.planner_agent._internal_agent)

            if self.planner.planner_agent.memory_enabled and not self.planner.planner_agent.memory:
                self.planner.planner_agent.memory = self.cortex_memory

    def _run_async_coro(self, coro):
        """Run ``coro`` from synchronous code, even inside a live event loop.

        Uses :func:`asyncio.run` when no loop is active; otherwise dispatches
        the coroutine to a one-shot thread-pool so the outer loop is not
        disturbed.
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()

    def _interpolate_inputs(self, inputs: dict[str, Any]) -> None:
        """Apply template substitutions to every agent, the manager, and tasks."""

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
        """Execute every task under the configured :class:`ProcessType`.

        When ``use_streaming`` is ``False`` this returns a fully formed
        :class:`CortexOutput`. When ``True`` it returns a ``(buffer, thread)``
        pair: the caller can consume ``buffer`` directly and retrieve the
        final :class:`CortexOutput` via ``buffer.cortex_output`` once the
        thread completes.

        Args:
            inputs: Template variables applied to every agent and task
                before execution.
            use_streaming: Run the orchestrator on a background thread and
                stream chunks into ``streamer_buffer``.
            stream_callback: Invoked for every chunk while streaming.
            streamer_buffer: Caller-supplied buffer; one is allocated if
                ``None``.
            log_process: When ``True`` and ``stream_callback`` is ``None``,
                a default logger-backed callback is used.
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
                """Drive the configured topology and finalise the shared buffer."""
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

                    buffer.cortex_output = CortexOutput(
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
        """Drive one agent in streaming mode and return its concatenated text.

        Drains chunks from ``main_buffer``, re-emits them so downstream
        consumers also see the stream, optionally forwards them to
        ``stream_callback``, and returns the joined ``chunk.content`` once
        the underlying agent thread (if any) has terminated.
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
        """Run every task in declaration order, threading outputs as context.

        Each task can declare ``dependencies`` (looked up by description in
        ``self.task_outputs``) and a ``chain`` link that may insert a
        follow-up task based on the output. The final string is the last
        task's output.
        """

        context_outputs: list[str] = []

        for i, task in enumerate(self.tasks):
            if not hasattr(task, "task_id"):
                setattr(task, "task_id", str(uuid.uuid4())[:18])
            log_task_start(f"Task {i + 1}/{len(self.tasks)}")

            task_context: list[str] = []

            if hasattr(task, "dependencies") and task.dependencies:
                for dep_task in task.dependencies:
                    for completed_task in self.task_outputs:
                        if completed_task.task.description == dep_task.description:
                            if dep_task.agent is not None:
                                task_context.append(f"Previous Task ({dep_task.agent.role}): {completed_task.output}")
                            break

            if task.context:
                if context_outputs:
                    for j, prev_output in enumerate(context_outputs, 1):
                        task_context.append(f"Task {j} Output: {prev_output}")

            _exec_result = task.execute(task_context if (task_context or task.context) else None)
            if isinstance(_exec_result, tuple):
                task_output = CortexTaskOutput(
                    task=task,
                    output=_resolve_execute_output(_exec_result),
                    agent=task.agent if task.agent is not None else self.agents[0],
                )
            else:
                task_output = _exec_result

            context_outputs.append(task_output.output)
            self.task_outputs.append(task_output)

            if task.agent is not None:
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
        """Streaming counterpart to :meth:`_run_sequential`.

        Emits a synthetic start chunk per task before delegating to
        :meth:`_stream_agent_execution`.
        """

        context_outputs: list[str] = []
        all_content: list[str] = []

        for i, task in enumerate(self.tasks):
            if not hasattr(task, "task_id"):
                setattr(task, "task_id", str(uuid.uuid4())[:18])
            log_task_start(f"Task {i + 1}/{len(self.tasks)}")

            task_context: list[str] = []

            if hasattr(task, "dependencies") and task.dependencies:
                for dep_task in task.dependencies:
                    for completed_task in self.task_outputs:
                        if completed_task.task.description == dep_task.description:
                            if dep_task.agent is not None:
                                task_context.append(f"Previous Task ({dep_task.agent.role}): {completed_task.output}")
                            break

            if task.context:
                if context_outputs:
                    for j, prev_output in enumerate(context_outputs, 1):
                        task_context.append(f"Task {j} Output: {prev_output}")

            if task.agent is None:
                raise ValueError(f"Task '{task.description[:50]}...' has no assigned agent")

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
        """Run context-free tasks concurrently then dependent tasks in order.

        Tasks that opt into ``context`` are not parallelisable (they depend
        on every earlier output) so they run sequentially after the
        independent batch resolves. Concurrency is capped by
        :attr:`parallel_max_workers`.
        """

        cortex_self = self

        async def run_task_async(
            task: CortexTask,
            context_outputs: list[str],
            executor: ThreadPoolExecutor | None = None,
            streamer_buffer: StreamerBuffer | None = None,
        ) -> CortexTaskOutput:
            """Run one task on the supplied thread pool.

            When ``streamer_buffer`` is supplied the agent is driven in
            streaming mode through :meth:`_stream_agent_execution`;
            otherwise the task's blocking :meth:`execute` is dispatched
            into the pool.
            """
            loop = asyncio.get_running_loop()

            if streamer_buffer:
                if task.agent is None:
                    raise ValueError(f"Task '{task.description[:50]}...' has no assigned agent")
                _task_agent = task.agent
                task_description = f"{task.description}\n\nExpected Output: {task.expected_output}"
                context_str = "\n\n".join(context_outputs) if context_outputs else None

                output_content = await loop.run_in_executor(
                    executor,
                    lambda: cortex_self._stream_agent_execution(
                        agent=_task_agent,
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
                _task_output = await loop.run_in_executor(
                    executor,
                    lambda: task.execute(context_outputs if task.context else None),
                )
                if isinstance(_task_output, tuple):
                    return CortexTaskOutput(
                        task=task,
                        output=_resolve_execute_output(_task_output),
                        agent=task.agent if task.agent is not None else self.agents[0],
                    )
                return _task_output

        async def run_all_tasks() -> str:
            """Fan out independent tasks then walk dependent tasks in order."""
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
        """Run a manager-led delegation, execution and review loop.

        Steps:

        1. The manager produces a JSON execution plan (one entry per task,
           naming an agent role and optional dependencies).
        2. Each task is delegated to the assigned agent.
        3. The manager reviews the output; if ``approved`` is false, the
           agent is invoked again with the manager's improvement list.
        4. After every task is approved the manager generates a final
           summary.

        Raises:
            ValueError: When the manager omits required plan fields or
                assigns work to an unknown agent.
            RuntimeError: When the plan or review JSON cannot be parsed.
        """

        if not self.manager_agent:
            raise ValueError("Hierarchical process requires a manager agent")

        self.logger.info("📝 Manager is creating execution plan...")
        manager_prompt = self.template_engine.render_manager_delegation(
            agents=self.agents,
            tasks=self.tasks,
        )

        raw_plan_response = self.manager_agent.execute(
            task_description=manager_prompt,
            context=None,
        )
        if isinstance(raw_plan_response, tuple):
            plan_response = _resolve_execute_output(raw_plan_response)
        else:
            plan_response = raw_plan_response

        try:
            json_match = re.search(r"\{[\s\S]*\}", plan_response)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                raise ValueError("Manager failed to produce a valid JSON execution plan")
        except Exception as e:
            self.logger.error(f"❌ Failed to parse manager plan: {e}")
            raise RuntimeError(f"Manager agent failed to create valid execution plan: {e}") from e

        completed_tasks: dict[int, str] = {}

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

            raw_review = self.manager_agent.execute(
                task_description=review_prompt,
                context=None,
            )
            if isinstance(raw_review, tuple):
                review = _resolve_execute_output(raw_review)
            else:
                review = raw_review

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
                    improved = assigned_agent.execute(
                        task_description=improvement_prompt,
                        context=output,
                    )
                    output = _resolve_execute_output(improved)
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

        raw_summary = self.manager_agent.execute(
            task_description="Provide a final summary of all completed tasks and their outcomes",
            context="\n\n".join([o.output for o in self.task_outputs]),
        )
        if isinstance(raw_summary, tuple):
            return _resolve_execute_output(raw_summary)
        return raw_summary

    def _run_consensus(
        self,
        streamer_buffer: StreamerBuffer | None = None,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> str:
        """Run each task through every agent then synthesise via a lead agent.

        For each task: every agent answers independently, then either the
        task-bound agent (or :pyattr:`self.agents[0]` as fallback) is
        invoked with the consensus template to merge the answers. The
        final return is the consensus from the last task.
        """

        final_outputs: list[str] = []

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
                    _agent_output = agent.execute(
                        task_description=task_description,
                        context=context,
                    )
                    output = _resolve_execute_output(_agent_output)

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
                _consensus = lead_agent.execute(
                    task_description=consensus_prompt,
                    context=None,
                )
                consensus = _resolve_execute_output(_consensus)

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
        """Build an objective string, ask the planner for a plan, and execute it.

        The planner expands the tasks into ordered :class:`PlanStep` entries
        (each potentially assigned to a specific agent role). After
        execution, every task is paired with a step's result so
        :attr:`task_outputs` reflects the plan's progress.
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
        """Streaming counterpart to :meth:`_run_hierarchical`.

        Skips the manager-review improvement loop and ignores invalid
        ``task_id`` entries in the plan rather than raising.
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

        completed_tasks: dict[int, str] = {}

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

            context_parts: list[str] = []
            if "dependencies" in task_plan:
                for dep_id in task_plan["dependencies"]:
                    if dep_id in completed_tasks:
                        context_parts.append(completed_tasks[dep_id])

            output = self._stream_agent_execution(
                agent=task.agent,
                task_description=task.description,
                context="\n\n".join(context_parts) if context_parts else None,
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
        """Streaming counterpart to :meth:`_run_planned`.

        Streams both plan creation and per-step execution through the
        shared buffer.
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
        """Build a :class:`Cortex` from tasks, inferring agents and LLM.

        Pulls agents off each task (de-duplicating by identity) and
        defaults the LLM to ``tasks[0].agent.llm`` when not supplied.

        Raises:
            ValueError: When the LLM cannot be inferred or the first
                task has no agent.
        """

        if llm is None:
            agent = tasks[0].agent
            if isinstance(agent, list):
                agent = agent[0]
            if agent is None:
                raise ValueError("First task must have an assigned agent")
            llm = agent.llm
            if llm is None:
                raise ValueError("Agent must have an LLM configured")
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
        """Return :meth:`CortexMemory.get_summary` for the owned memory stack."""

        return self.cortex_memory.get_summary()

    def save_memory(self, persistence_path: str | None = None) -> None:
        """Repoint the underlying SQLite store at ``persistence_path``.

        No-op when memory persistence is disabled (``storage is None``).
        """

        if persistence_path and self.cortex_memory.storage:
            self.cortex_memory.storage.db_path = Path(persistence_path)

    def clear_short_term_memory(self) -> None:
        """Clear only the short-term memory buffer."""

        self.cortex_memory.reset_short_term()

    def clear_all_memory(self) -> None:
        """Clear all memory subsystems."""

        self.cortex_memory.reset_all()


@dataclass
class CortexOutput:
    """Aggregate result envelope returned by :meth:`Cortex.kickoff`.

    Attributes:
        raw_output: Primary textual output (typically the last task's text,
            or the manager summary for ``HIERARCHICAL``).
        task_outputs: Per-task results in execution order.
        execution_time: Wall-clock seconds spent inside :meth:`Cortex.kickoff`.
    """

    raw_output: str
    task_outputs: list[CortexTaskOutput]
    execution_time: float

    def __str__(self) -> str:
        """Return :attr:`raw_output` so the dataclass prints as plain text."""

        return self.raw_output

    def to_dict(self) -> dict:
        """Return a JSON-safe dict view of the run."""

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
