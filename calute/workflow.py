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
import typing as tp
from dataclasses import dataclass, field
from enum import Enum

from calute.calute import Calute

from .types.function_execution_types import ExecutionStatus, FunctionCall


class WorkflowStepType(Enum):
    """Types of workflow steps"""

    FUNCTION = "function"  # Execute a function
    CONDITION = "condition"  # Check a condition to decide next step
    PARALLEL = "parallel"  # Execute multiple steps in parallel
    SEQUENCE = "sequence"  # Execute a sequence of steps
    MAP = "map"  # Map over a collection
    RETRY = "retry"  # Retry a step on failure
    AGENT_SWITCH = "agent_switch"  # Switch to a different agent


@dataclass
class WorkflowStep:
    """A step in a workflow"""

    id: str
    type: WorkflowStepType
    name: str
    config: dict = field(default_factory=dict)
    next_steps: list[str] = field(default_factory=list)
    condition: tp.Callable | None = None


@dataclass
class Workflow:
    """A workflow definition"""

    id: str
    name: str
    entry_point: str
    steps: dict[str, WorkflowStep]
    description: str = ""


class WorkflowEngine:
    """Engine for executing workflows"""

    def __init__(self, calute: Calute):
        self.calute = calute
        self.workflows: dict[str, Workflow] = {}
        self.active_workflows: dict[str, dict] = {}

    def register_workflow(self, workflow: Workflow):
        """Register a workflow with the engine"""
        self.workflows[workflow.id] = workflow

    def create_function_workflow(
        self,
        id: str,  # noqa:A002
        name: str,
        functions: list[str | tp.Callable],
        description: str = "",
    ) -> Workflow:
        """Helper to create a simple linear function workflow"""
        steps = {}
        next_step_id = None

        # Build the workflow backwards to establish the chain
        for i, func in reversed(list(enumerate(functions))):
            func_name = func if isinstance(func, str) else func.__name__
            step_id = f"{id}_step_{i}"

            steps[step_id] = WorkflowStep(
                id=step_id,
                type=WorkflowStepType.FUNCTION,
                name=func_name,
                next_steps=[next_step_id] if next_step_id else [],
            )
            next_step_id = step_id

        workflow = Workflow(
            id=id,
            name=name,
            description=description,
            entry_point=f"{id}_step_0",  # First step
            steps=steps,
        )

        self.register_workflow(workflow)
        return workflow

    async def execute_workflow(
        self,
        workflow_id: str,
        context_variables: dict | None = None,
        input_data: dict | None = None,
    ) -> dict:
        """Execute a workflow from start to finish"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        execution_id = f"exec_{workflow_id}_{id(object())}"
        self.active_workflows[execution_id] = {
            "workflow_id": workflow_id,
            "status": "running",
            "context": context_variables or {},
            "results": {},
            "current_step": workflow.entry_point,
        }

        if input_data:
            self.active_workflows[execution_id]["context"].update(input_data)

        try:
            await self._execute_step(
                workflow,
                workflow.entry_point,
                self.active_workflows[execution_id]["context"],
                self.active_workflows[execution_id]["results"],
            )
            self.active_workflows[execution_id]["status"] = "completed"
        except Exception as e:
            self.active_workflows[execution_id]["status"] = "failed"
            self.active_workflows[execution_id]["error"] = str(e)

        return self.active_workflows[execution_id]

    async def _execute_step(
        self,
        workflow: Workflow,
        step_id: str,
        context: dict,
        results: dict,
    ) -> tp.Any:
        """Execute a single workflow step"""
        step = workflow.steps.get(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found in workflow {workflow.id}")

        if step.type == WorkflowStepType.FUNCTION:
            result = await self._execute_function_step(step, context, results)
        elif step.type == WorkflowStepType.CONDITION:
            result = await self._execute_condition_step(step, context, results)
        elif step.type == WorkflowStepType.PARALLEL:
            result = await self._execute_parallel_step(workflow, step, context, results)
        elif step.type == WorkflowStepType.SEQUENCE:
            result = await self._execute_sequence_step(workflow, step, context, results)
        elif step.type == WorkflowStepType.MAP:
            result = await self._execute_map_step(workflow, step, context, results)
        elif step.type == WorkflowStepType.RETRY:
            result = await self._execute_retry_step(workflow, step, context, results)
        elif step.type == WorkflowStepType.AGENT_SWITCH:
            result = await self._execute_agent_switch_step(step, context, results)
        else:
            raise ValueError(f"Unknown step type {step.type}")

        results[step_id] = result

        for next_step_id in step.next_steps:
            await self._execute_step(workflow, next_step_id, context, results)

        return result

    async def _execute_function_step(
        self,
        step: WorkflowStep,
        context: dict,
        results: dict,
    ) -> tp.Any:
        """Execute a function step"""
        function_name = step.name
        args = step.config.get("arguments", {})

        processed_args = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("${results.") and value.endswith("}"):
                result_key = value[10:-1]
                if result_key in results:
                    processed_args[key] = results[result_key]
                else:
                    raise ValueError(f"Referenced result {result_key} not found")
            else:
                processed_args[key] = value

        if step.config.get("include_context", True):
            processed_args["context_variables"] = context

        function_call = FunctionCall(
            name=function_name,
            arguments=processed_args | context,
            timeout=step.config.get("timeout", 30.0),
            max_retries=step.config.get("max_retries", 3),
        )
        executor = self.calute.executor
        result = await executor._execute_single_call(function_call, context)
        if step.config.get("store_result_as"):
            result_key = step.config["store_result_as"]
            context[result_key] = result.result

        if result.status != ExecutionStatus.SUCCESS:
            if step.config.get("fail_on_error", True):
                raise RuntimeError(f"Function {function_name} failed: {result.error}")

        return result.result

    async def _execute_condition_step(
        self,
        step: WorkflowStep,
        context: dict,
        results: dict,
    ) -> bool:
        """Execute a condition step"""
        condition_func = step.condition
        if not condition_func:
            raise ValueError("Condition step missing condition function")

        result = condition_func(context, results)

        if result:
            step.next_steps = step.config.get("if_true", [])
        else:
            step.next_steps = step.config.get("if_false", [])

        return result

    async def _execute_parallel_step(
        self,
        workflow: Workflow,
        step: WorkflowStep,
        context: dict,
        results: dict,
    ) -> list[tp.Any]:
        """Execute multiple steps in parallel"""
        parallel_steps = step.config.get("steps", [])
        tasks = []

        for parallel_step_id in parallel_steps:
            branch_context = context.copy()
            branch_results = {}

            tasks.append(self._execute_step(workflow, parallel_step_id, branch_context, branch_results))

        parallel_results = await asyncio.gather(*tasks, return_exceptions=True)

        if step.config.get("merge_results", False):
            for i, parallel_step_id in enumerate(parallel_steps):
                results[parallel_step_id] = parallel_results[i]

        return parallel_results

    async def _execute_sequence_step(
        self,
        workflow: Workflow,
        step: WorkflowStep,
        context: dict,
        results: dict,
    ) -> list[tp.Any]:
        """Execute a sequence of steps in order"""
        sequence_steps = step.config.get("steps", [])
        sequence_results = []

        for seq_step_id in sequence_steps:
            result = await self._execute_step(workflow, seq_step_id, context, results)
            sequence_results.append(result)

        return sequence_results

    async def _execute_map_step(
        self, workflow: Workflow, step: WorkflowStep, context: dict, results: dict
    ) -> list[tp.Any]:
        """Execute a step for each item in a collection"""
        collection_key = step.config.get("collection")
        if not collection_key:
            raise ValueError("Map step missing collection key")

        if collection_key.startswith("${results.") and collection_key.endswith("}"):
            result_key = collection_key[10:-1]
            collection = results.get(result_key, [])
        else:
            collection = context.get(collection_key, [])

        if not isinstance(collection, list | tuple):
            raise TypeError(f"Map collection must be a list or tuple, got {type(collection)}")

        item_variable = step.config.get("item_variable", "item")
        index_variable = step.config.get("index_variable", "index")
        target_step = step.config.get("target_step")

        if not target_step:
            raise ValueError("Map step missing target step")

        map_results = []

        for i, item in enumerate(collection):
            item_context = context.copy()
            item_context[item_variable] = item
            item_context[index_variable] = i

            result = await self._execute_step(workflow, target_step, item_context, results)
            map_results.append(result)

        return map_results

    async def _execute_retry_step(self, workflow: Workflow, step: WorkflowStep, context: dict, results: dict) -> tp.Any:
        """Retry a step with backoff until success or max retries"""
        target_step = step.config.get("target_step")
        if not target_step:
            raise ValueError("Retry step missing target step")

        max_retries = step.config.get("max_retries", 3)
        backoff_factor = step.config.get("backoff_factor", 2)

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = await self._execute_step(workflow, target_step, context, results)
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(backoff_factor**attempt)

        if last_error:
            raise last_error
        raise RuntimeError("All retries failed")

    async def _execute_agent_switch_step(
        self,
        step: WorkflowStep,
        context: dict,
        results: dict,
    ) -> str:
        """Switch to a different agent"""
        target_agent = step.config.get("target_agent")
        if not target_agent:
            raise ValueError("Agent switch step missing target agent")

        reason = step.config.get("reason", "Workflow agent switch")

        self.calute.orchestrator.switch_agent(target_agent, reason)

        return target_agent
