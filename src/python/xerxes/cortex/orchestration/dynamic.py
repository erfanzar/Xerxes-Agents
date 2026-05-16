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
"""Dynamic task and cortex builders for prompt-driven multi-agent runs.

Provides :class:`DynamicTaskBuilder` (static helpers that turn prompts into
:class:`CortexTask` objects), :class:`DynamicCortex` (a :class:`Cortex`
subclass that can decompose, route and execute prompts at runtime), and
:func:`create_dynamic_cortex` (a no-tasks factory).
"""

import threading
from collections.abc import Callable
from typing import Any

from xerxes.cortex.agents.memory_integration import CortexMemory
from xerxes.memory.compat import MemoryType

from ...core.streamer_buffer import StreamerBuffer
from ...llms import BaseLLM
from ..agents.agent import CortexAgent
from ..core.enums import ProcessType
from .cortex import Cortex, MemoryConfig
from .task import CortexTask
from .task_creator import TaskCreationPlan, TaskCreator


class DynamicTaskBuilder:
    """Static helper methods for constructing ``CortexTask`` objects on the fly."""

    @staticmethod
    def from_prompt(
        prompt: str,
        agent: CortexAgent | None = None,
        expected_output: str = "Complete the requested task",
        tools: list | None = None,
        **task_kwargs,
    ) -> CortexTask:
        """Build one :class:`CortexTask` from a free-form prompt string."""

        return CortexTask(
            description=prompt, expected_output=expected_output, agent=agent, tools=tools or [], **task_kwargs
        )

    @staticmethod
    def chain_prompts(
        prompts: list[str], agents: list[CortexAgent] | None = None, use_context: bool = True
    ) -> list[CortexTask]:
        """Build an ordered task chain from ``prompts``.

        Agents (if given) are assigned round-robin. With ``use_context``
        each task receives the prior task as its context source so outputs
        flow forward when run sequentially.
        """

        tasks: list[CortexTask] = []

        for i, prompt in enumerate(prompts):
            agent = None
            if agents:
                agent = agents[i % len(agents)]

            task = CortexTask(
                description=prompt,
                expected_output="Complete the requested task and provide detailed output",
                agent=agent,
                context=tasks[-1:] if use_context and tasks else None,
            )
            tasks.append(task)

        return tasks


class DynamicCortex(Cortex):
    """A ``Cortex`` variant that supports runtime task creation and prompt-based execution.

    Extends ``Cortex`` with methods to create tasks from natural language prompts,
    execute single prompts directly, and batch-process multiple prompts.
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
    ):
        """Forward every argument to :class:`Cortex` and start with no task creator.

        ``self.task_creator`` is allocated lazily on the first prompt-based
        method call so the dynamic cortex stays cheap until used.
        """

        super().__init__(
            agents=agents,
            tasks=tasks,
            llm=llm,
            process=process,
            manager_agent=manager_agent,
            memory_type=memory_type,
            verbose=verbose,
            max_iterations=max_iterations,
            model=model,
            memory=memory,
            memory_config=memory_config,
            reinvoke_after_function=reinvoke_after_function,
            enable_xerxes_memory=enable_xerxes_memory,
            cortex_name=cortex_name,
        )
        self.task_creator: TaskCreator | None = None

    def create_tasks_from_prompt(
        self,
        prompt: str,
        background: str | None = None,
        auto_assign: bool = True,
        stream: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> TaskCreationPlan | tuple[TaskCreationPlan, list[CortexTask]]:
        """Decompose ``prompt`` into tasks and optionally bind them to agents.

        When ``auto_assign`` is ``True`` and the task creator returns
        executable :class:`CortexTask` objects, :attr:`tasks` is replaced
        with the new list before returning.
        """

        if not self.task_creator:
            self.task_creator = TaskCreator(verbose=self.verbose, llm=self.llm, auto_assign_agents=auto_assign)

        result = self.task_creator.create_tasks_from_prompt(
            prompt=prompt,
            background=background,
            available_agents=self.agents if auto_assign else None,
            stream=stream,
            stream_callback=stream_callback,
        )

        if isinstance(result, tuple):
            plan, cortex_tasks = result
            assert cortex_tasks is not None
            self.tasks = cortex_tasks
            return plan, cortex_tasks
        else:
            return result

    def execute_with_task_creation(
        self,
        prompt: str,
        inputs: dict[str, Any] | None = None,
        background: str | None = None,
        process: ProcessType | None = None,
        stream: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> Any:
        """Run :meth:`create_tasks_from_prompt` then :meth:`kickoff` end-to-end.

        ``process`` temporarily overrides :attr:`process` for the duration
        of this call and is restored on exit (even on failure).
        """

        creation_result = self.create_tasks_from_prompt(
            prompt=prompt,
            background=background,
            auto_assign=True,
            stream=False,
        )
        assert isinstance(creation_result, tuple)
        _plan, cortex_tasks = creation_result

        self.tasks = cortex_tasks

        if process:
            original_process = self.process
            self.process = process

        try:
            if stream:
                result = self.kickoff(inputs=inputs, use_streaming=True, stream_callback=stream_callback)
            else:
                result = self.kickoff(inputs=inputs)
        finally:
            if process:
                self.process = original_process

        return result

    def execute_prompt(
        self,
        prompt: str,
        agent: CortexAgent | str | None = None,
        stream: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> str | tuple[StreamerBuffer, threading.Thread]:
        """Run one prompt directly through a chosen (or default) agent.

        ``agent`` may be the role name, a :class:`CortexAgent`, or ``None``
        (in which case ``agents[0]`` is used).

        Raises:
            ValueError: When ``agent`` is a role name with no match.
        """

        target_agent = None
        if isinstance(agent, str):
            for a in self.agents:
                if a.role.lower() == agent.lower():
                    target_agent = a
                    break
        elif isinstance(agent, CortexAgent):
            target_agent = agent
        else:
            target_agent = self.agents[0] if self.agents else None

        if not target_agent:
            raise ValueError(f"No agent found for: {agent}")

        task = DynamicTaskBuilder.from_prompt(prompt, target_agent)
        self.tasks = [task]

        if stream:
            buffer_was_none = streamer_buffer is None
            if streamer_buffer is None:
                streamer_buffer = StreamerBuffer()

            def execute_with_stream() -> None:
                """Drive ``target_agent`` and stash the final text on ``result_holder``."""
                try:
                    if stream_callback:
                        _stream_result = target_agent.execute_stream(task_description=prompt, callback=stream_callback)
                    else:
                        _exec_result = target_agent.execute(
                            task_description=prompt, streamer_buffer=streamer_buffer, use_thread=False
                        )
                        _stream_result = (
                            _exec_result
                            if isinstance(_exec_result, str)
                            else _exec_result[0].get_result(1.0)
                            if _exec_result[0].get_result is not None
                            else str(_exec_result[0])
                        )

                    if hasattr(streamer_buffer, "result_holder"):
                        streamer_buffer.result_holder = [_stream_result]
                finally:
                    if buffer_was_none:
                        streamer_buffer.close()

            thread = threading.Thread(target=execute_with_stream, daemon=True)
            thread.start()

            return streamer_buffer, thread
        else:
            result = self.kickoff()
            assert not isinstance(result, tuple)
            return result.raw_output

    def execute_prompts(
        self,
        prompts: list[str] | dict[str, str],
        process: ProcessType | None = None,
        stream: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> dict[str, str] | str | tuple[StreamerBuffer, threading.Thread]:
        """Execute multiple prompts in one Cortex run.

        ``prompts`` may be a ``{role: prompt}`` dict (each prompt routed to
        the named agent) or a list (chained through
        :meth:`DynamicTaskBuilder.chain_prompts`). The return shape mirrors
        the input: a ``{role: output}`` dict, a single raw output string,
        or a streaming tuple when ``stream`` is ``True``.

        Raises:
            ValueError: When a dict key references an unknown agent role.
        """

        if isinstance(prompts, dict):
            tasks = []
            for agent_role, prompt in prompts.items():
                agent = None
                for a in self.agents:
                    if a.role.lower() == agent_role.lower():
                        agent = a
                        break

                if not agent:
                    raise ValueError(f"Agent not found: {agent_role}")

                task = DynamicTaskBuilder.from_prompt(prompt, agent)
                tasks.append(task)
        else:
            tasks = DynamicTaskBuilder.chain_prompts(
                prompts, self.agents, use_context=(process == ProcessType.SEQUENTIAL)
            )

        self.tasks = tasks

        if process:
            original_process = self.process
            self.process = process

        if stream:
            if streamer_buffer is None:
                streamer_buffer = StreamerBuffer()

            kickoff_result = self.kickoff(use_streaming=True, stream_callback=stream_callback)
            assert isinstance(kickoff_result, tuple)
            buffer, thread = kickoff_result

            if process:
                self.process = original_process

            return buffer, thread
        else:
            result = self.kickoff(use_streaming=False)
            assert not isinstance(result, tuple)

            if process:
                self.process = original_process

            if isinstance(prompts, dict):
                outputs = {}
                for i, (role, _prompt) in enumerate(prompts.items()):
                    if i < len(result.task_outputs):
                        outputs[role] = result.task_outputs[i].output
                return outputs
            else:
                return result.raw_output


def create_dynamic_cortex(
    agents: list[CortexAgent], llm: BaseLLM, process: ProcessType = ProcessType.SEQUENTIAL, **cortex_kwargs
) -> DynamicCortex:
    """Build a :class:`DynamicCortex` with no initial tasks.

    Tasks are added later via :meth:`DynamicCortex.create_tasks_from_prompt`,
    :meth:`execute_prompt` or :meth:`execute_prompts`.
    """

    return DynamicCortex(agents=agents, tasks=[], llm=llm, process=process, **cortex_kwargs)
