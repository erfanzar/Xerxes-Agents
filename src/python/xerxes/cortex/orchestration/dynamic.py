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
"""Dynamic task and cortex builders for ad-hoc multi-agent execution."""

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
        """Create a single ``CortexTask`` from a natural language prompt.

        Args:
            prompt (str): The task description.
                IN: Becomes the ``description`` field of the created task.
                OUT: Directly assigned to ``CortexTask.description``.
            agent (CortexAgent | None): The agent to assign.
                IN: Optional agent responsible for execution.
                OUT: Set as the task's ``agent``.
            expected_output (str): Description of the desired result.
                IN: Sets expectations for the task output.
                OUT: Assigned to ``CortexTask.expected_output``.
            tools (list | None): Additional tools for this task.
                IN: Optional list of ``CortexTool`` or callable objects.
                OUT: Assigned to ``CortexTask.tools``.
            **task_kwargs: Arbitrary keyword arguments.
                IN: Forwarded to the ``CortexTask`` constructor.
                OUT: Allows customization of any ``CortexTask`` field.

        Returns:
            CortexTask: A ready-to-execute task instance.
                OUT: Configured with the provided prompt, agent, and options.
        """

        return CortexTask(
            description=prompt, expected_output=expected_output, agent=agent, tools=tools or [], **task_kwargs
        )

    @staticmethod
    def chain_prompts(
        prompts: list[str], agents: list[CortexAgent] | None = None, use_context: bool = True
    ) -> list[CortexTask]:
        """Create a sequential list of tasks from a list of prompts.

        Args:
            prompts (list[str]): The prompts to turn into tasks.
                IN: Each string becomes a ``CortexTask.description``.
                OUT: Tasks are created in the same order as the input list.
            agents (list | None): Agents to assign cyclically.
                IN: Agent ``i`` is assigned to prompt ``i % len(agents)``.
                OUT: Each task gets a rotating agent from this list.
            use_context (bool): Whether each task receives the prior task as context.
                IN: If ``True``, ``task.context`` is set to ``[previous_task]``.
                OUT: Enables sequential information flow.

        Returns:
            list[CortexTask]: Ordered list of tasks forming a chain.
                OUT: Ready for sequential execution.
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
        """Initialize a ``DynamicCortex``.

        Args:
            agents (list[CortexAgent]): Agents available for task execution.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Registered and wired into the Xerxes instance.
            tasks (list[CortexTask]): Initial task list.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: May be replaced by dynamic task creation methods.
            llm (BaseLLM): The language model backend.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Configures the underlying Xerxes LLM.
            process (ProcessType): The default execution process type.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Determines how tasks are orchestrated.
            manager_agent (CortexAgent | None): Manager agent for hierarchical mode.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Used when ``process`` is ``ProcessType.HIERARCHICAL``.
            memory_type (MemoryType): Default memory type.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Configures the memory store.
            verbose (bool): Whether to enable verbose logging.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Controls log verbosity.
            max_iterations (int): Maximum execution iterations.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Limits agent execution loops.
            model (str): Default model identifier.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Assigned to agents without an explicit model.
            memory (CortexMemory | None): Pre-configured memory instance.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Shared across agents and tasks.
            memory_config (MemoryConfig | None): TypedDict memory configuration.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Used to construct a ``CortexMemory`` if *memory* is ``None``.
            reinvoke_after_function (bool): Whether to reinvoke the agent after tool calls.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Controls agent execution behavior.
            enable_xerxes_memory (bool): Whether to enable Xerxes-level memory.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Configures the Xerxes instance memory.
            cortex_name (str): Display name for this cortex.
                IN: Passed to the base ``Cortex`` constructor.
                OUT: Used in log messages.
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
        """Decompose a prompt into tasks and optionally assign them to agents.

        Args:
            prompt (str): The high-level objective.
                IN: Passed to ``TaskCreator`` for decomposition.
                OUT: Drives the creation of ``TaskDefinition`` objects.
            background (str | None): Additional context for task creation.
                IN: Passed to ``TaskCreator``.
                OUT: Guides the decomposition strategy.
            auto_assign (bool): Whether to map tasks to available agents.
                IN: Passed to ``TaskCreator`` as ``auto_assign_agents``.
                OUT: Controls whether ``CortexTask`` objects are produced.
            stream (bool): Whether to stream the creation response.
                IN: Passed to ``TaskCreator``.
                OUT: Enables real-time observation.
            stream_callback (Callable | None): Callback for streamed chunks.
                IN: Passed to ``TaskCreator``.
                OUT: Invoked during streaming creation.

        Returns:
            TaskCreationPlan | tuple[TaskCreationPlan, list[CortexTask]]: The
                raw plan, or a tuple with executable tasks when auto-assignment
                succeeds. OUT: If a tuple is returned, ``self.tasks`` is updated
                with the new ``CortexTask`` list.
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
        """Create tasks from a prompt and immediately execute them.

        Args:
            prompt (str): The objective to decompose and execute.
                IN: Passed to ``create_tasks_from_prompt``.
                OUT: Drives both task creation and execution.
            inputs (dict | None): Template variables for interpolation.
                IN: Passed to ``kickoff`` for input substitution.
                OUT: Applied to agents and tasks before execution.
            background (str | None): Background context for task creation.
                IN: Passed to ``create_tasks_from_prompt``.
                OUT: Guides decomposition.
            process (ProcessType | None): Temporary process type override.
                IN: If provided, temporarily replaces ``self.process``.
                OUT: Restored after execution.
            stream (bool): Whether to stream execution output.
                IN: Passed to ``kickoff``.
                OUT: Determines if execution is streamed.
            stream_callback (Callable | None): Callback for streamed chunks.
                IN: Passed to ``kickoff`` when streaming.
                OUT: Invoked during streamed execution.

        Returns:
            Any: The execution result from ``kickoff``.
                OUT: ``CortexOutput`` or streaming tuple depending on *stream*.
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
        """Execute a single prompt against a specific agent.

        Args:
            prompt (str): The task description.
                IN: Executed directly by the target agent.
                OUT: Becomes the task description.
            agent (CortexAgent | str | None): The agent to use.
                IN: If a string, matched against agent roles; if ``None``,
                the first agent in ``self.agents`` is used.
                OUT: Becomes the executor of the prompt.
            stream (bool): Whether to stream the response.
                IN: Enables threaded streaming execution.
                OUT: Determines the return type and execution path.
            stream_callback (Callable | None): Callback for streamed chunks.
                IN: Invoked for each chunk during streaming.
                OUT: Forwarded to the agent's streaming methods.
            streamer_buffer (StreamerBuffer | None): Optional pre-existing buffer.
                IN: Used for streaming if provided; otherwise a new one is created.
                OUT: Receives and yields chunks during execution.

        Returns:
            str | tuple[StreamerBuffer, threading.Thread]: The result string, or
                a streaming tuple when *stream* is ``True``.
                OUT: Direct agent output or buffered stream handles.

        Raises:
            ValueError: If no matching agent is found.
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
                """Run the agent in streaming mode and store the result.

                Args:
                    None: Closure over *prompt*, *target_agent*, *stream_callback*,
                    and *streamer_buffer*.
                """
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
        """Execute multiple prompts, mapping each to an agent.

        Args:
            prompts (list[str] | dict[str, str]): Prompts to execute.
                IN: If a dict, keys are agent role names; if a list, prompts are
                chained sequentially.
                OUT: Converted into ``CortexTask`` objects and assigned to agents.
            process (ProcessType | None): Temporary process type override.
                IN: If provided, temporarily replaces ``self.process``.
                OUT: Restored after kickoff.
            stream (bool): Whether to stream the execution.
                IN: Passed to ``kickoff``.
                OUT: Determines if a streaming tuple is returned.
            stream_callback (Callable | None): Callback for streamed chunks.
                IN: Passed to ``kickoff`` when streaming.
                OUT: Invoked during streamed execution.
            streamer_buffer (StreamerBuffer | None): Optional pre-existing buffer.
                IN: Used for streaming if provided.
                OUT: Forwarded to ``kickoff``.

        Returns:
            dict[str, str] | str | tuple[StreamerBuffer, threading.Thread]:
                A mapping of role to output for dict inputs; the raw output for
                list inputs; or a streaming tuple when *stream* is ``True``.
                OUT: Derived from ``CortexOutput.task_outputs``.
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
    """Factory function to create a ``DynamicCortex`` with no initial tasks.

    Args:
        agents (list[CortexAgent]): Agents available for execution.
            IN: Passed to ``DynamicCortex``.
            OUT: Registered in the new cortex instance.
        llm (BaseLLM): The language model backend.
            IN: Passed to ``DynamicCortex``.
            OUT: Configures the underlying Xerxes instance.
        process (ProcessType): The default execution process.
            IN: Passed to ``DynamicCortex``.
            OUT: Determines task orchestration strategy.
        **cortex_kwargs: Additional keyword arguments.
            IN: Forwarded to ``DynamicCortex.__init__``.
            OUT: Allows customization of memory, manager, verbosity, etc.

    Returns:
        DynamicCortex: A ready-to-use dynamic cortex with an empty task list.
            OUT: Tasks can be added dynamically via creation methods.
    """

    return DynamicCortex(agents=agents, tasks=[], llm=llm, process=process, **cortex_kwargs)
