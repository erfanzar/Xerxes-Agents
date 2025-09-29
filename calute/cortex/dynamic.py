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


"""Dynamic prompt-based execution for Cortex framework"""

import threading
from collections.abc import Callable
from typing import Any

from calute.cortex.memory_integration import CortexMemory
from calute.memory.compat import MemoryType

from ..llms import BaseLLM
from ..streamer_buffer import StreamerBuffer
from .agent import CortexAgent
from .cortex import Cortex, MemoryConfig
from .enums import ProcessType
from .task import CortexTask
from .task_creator import TaskCreationPlan, TaskCreator


class DynamicTaskBuilder:
    """Utility for creating tasks dynamically from prompts"""

    @staticmethod
    def from_prompt(
        prompt: str,
        agent: CortexAgent | None = None,
        expected_output: str = "Complete the requested task",
        tools: list | None = None,
        **task_kwargs,
    ) -> CortexTask:
        """
        Create a CortexTask dynamically from a prompt.

        Args:
            prompt: The prompt/instruction for the task
            agent: Optional agent to assign (can be assigned later)
            expected_output: What output is expected
            tools: Optional list of tools for this specific task
            **task_kwargs: Additional CortexTask parameters

        Returns:
            CortexTask configured with the prompt
        """
        return CortexTask(
            description=prompt, expected_output=expected_output, agent=agent, tools=tools or [], **task_kwargs
        )

    @staticmethod
    def chain_prompts(
        prompts: list[str], agents: list[CortexAgent] | None = None, use_context: bool = True
    ) -> list[CortexTask]:
        """
        Create a chain of tasks from a list of prompts.

        Args:
            prompts: List of prompts to execute in sequence
            agents: Optional list of agents (matched by index, cycles if fewer agents)
            use_context: Whether each task should use previous outputs as context

        Returns:
            List of CortexTask objects with context dependencies
        """
        tasks = []

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
    """Extended Cortex with dynamic prompt execution capabilities"""

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
        """Initialize DynamicCortex with optional TaskCreator"""
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
            enable_calute_memory=enable_calute_memory,
            cortex_name=cortex_name,
        )
        self.task_creator = None

    def create_tasks_from_prompt(
        self,
        prompt: str,
        background: str | None = None,
        auto_assign: bool = True,
        stream: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> TaskCreationPlan | tuple[TaskCreationPlan, list[CortexTask]]:
        """
        Create tasks dynamically from a prompt using TaskCreator.

        Args:
            prompt: The objective to break down into tasks
            background: Optional approach/context to guide task creation
            auto_assign: Whether to automatically assign agents to tasks
            stream: Whether to stream the creation process
            stream_callback: Optional callback for streaming

        Returns:
            TaskCreationPlan or tuple of (plan, CortexTask list)
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
            self.tasks = cortex_tasks
            return plan, cortex_tasks
        else:
            return result

    def execute_with_task_creation(
        self,
        prompt: str,
        background: str | None = None,
        process: ProcessType | None = None,
        stream: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
    ) -> Any:
        """
        Create tasks from prompt and execute them immediately.

        Args:
            prompt: The objective to accomplish
            background: Optional approach/context
            process: Optional ProcessType override
            stream: Whether to stream execution
            stream_callback: Optional streaming callback

        Returns:
            Cortex execution result
        """

        plan, cortex_tasks = self.create_tasks_from_prompt(
            prompt=prompt,
            background=background,
            auto_assign=True,
            stream=False,
        )

        self.tasks = cortex_tasks

        if process:
            original_process = self.process
            self.process = process

        try:
            if stream:
                result = self.kickoff(use_streaming=True, stream_callback=stream_callback)
            else:
                result = self.kickoff()
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
        """
        Execute a single prompt with an agent dynamically.

        Args:
            prompt: The prompt to execute
            agent: Agent instance, agent role name, or None for first available
            stream: Whether to stream the response
            stream_callback: Optional callback for streaming
            streamer_buffer: Optional StreamerBuffer to use for streaming

        Returns:
            If stream=False: The agent's response string
            If stream=True: Tuple of (StreamerBuffer, Thread) for async consumption
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

            def execute_with_stream():
                try:
                    if stream_callback:
                        result = target_agent.execute_stream(task_description=prompt, callback=stream_callback)
                    else:
                        result = target_agent.execute(
                            task_description=prompt, streamer_buffer=streamer_buffer, use_thread=False
                        )

                    if hasattr(streamer_buffer, "result_holder"):
                        streamer_buffer.result_holder = [result]
                finally:
                    if buffer_was_none:
                        streamer_buffer.close()

            thread = threading.Thread(target=execute_with_stream, daemon=True)
            thread.start()

            return streamer_buffer, thread
        else:
            result = self.kickoff()
            return result.raw_output

    def execute_prompts(
        self,
        prompts: list[str] | dict[str, str],
        process: ProcessType | None = None,
        stream: bool = False,
        stream_callback: Callable[[Any], None] | None = None,
        streamer_buffer: StreamerBuffer | None = None,
    ) -> dict[str, str] | str | tuple[StreamerBuffer, threading.Thread]:
        """
        Execute multiple prompts with automatic agent assignment.

        Args:
            prompts: List of prompts or dict mapping agent roles to prompts
            process: Override default process type
            stream: Whether to stream responses
            stream_callback: Optional callback for streaming chunks
            streamer_buffer: Optional StreamerBuffer to use for streaming

        Returns:
            If stream=False: Dict mapping prompts to responses, or final output for sequential
            If stream=True: Tuple of (StreamerBuffer, Thread) for async consumption
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

            buffer, thread = self.kickoff(use_streaming=True, stream_callback=stream_callback)

            if process:
                self.process = original_process

            return buffer, thread
        else:
            result = self.kickoff(use_streaming=False)

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

    def create_ui(self):
        from calute.ui import create_application

        return create_application(executor=self)


def create_dynamic_cortex(
    agents: list[CortexAgent], llm: BaseLLM, process: ProcessType = ProcessType.SEQUENTIAL, **cortex_kwargs
) -> DynamicCortex:
    """
    Create a DynamicCortex instance for flexible prompt execution.

    Args:
        agents: List of agents to use
        llm: LLM instance
        process: Default process type
        **cortex_kwargs: Additional Cortex parameters

    Returns:
        DynamicCortex configured for dynamic execution
    """
    return DynamicCortex(agents=agents, tasks=[], llm=llm, process=process, **cortex_kwargs)
