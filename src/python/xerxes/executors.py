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
"""Function execution, agent orchestration, and call dispatch for Xerxes.

Provides ``AgentOrchestrator`` for multi-agent management,
``FunctionExecutor`` for executing tool calls with retry / sandbox / policy
support, and enhanced subclasses with metrics and stricter validation.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import re
import threading
import time
import traceback
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .core.utils import get_callable_public_name
from .runtime.loop_detection import LoopDetector, LoopSeverity, ToolLoopError
from .security.policy import PolicyAction, ToolPolicyViolation
from .security.sandbox import ExecutionContext, SandboxExecutionUnavailableError
from .types.agent_types import Result
from .types.function_execution_types import (
    AgentSwitchTrigger,
    ExecutionStatus,
    FunctionCallStrategy,
    RequestFunctionCall,
)

if tp.TYPE_CHECKING:
    from .runtime.features import RuntimeFeaturesState
    from .types import Agent

logger = logging.getLogger(__name__)

__CTX_VARS_NAME__ = "context_variables"
SEP = "  "
def add_depth(x, ep=False):
    return SEP + x.replace("\n", f"\n{SEP}") if ep else x.replace("\n", f"\n{SEP}")


class FunctionRegistry:
    """Index of callable functions grouped by name and owning agent.

    Each function is stored as ``(callable, agent_id)`` so that lookups can
    prefer the current agent's version.
    """

    def __init__(self):
        """Initialize empty function and metadata indexes."""

        self._functions: dict[str, list[tuple[tp.Callable, str]]] = {}
        self._function_metadata: dict[str, dict] = {}

    def register(self, func: tp.Callable, agent_id: str, metadata: dict | None = None):
        """Add a function to the registry.

        Args:
            func (tp.Callable): IN: Function to register. OUT: Stored by its
                public name.
            agent_id (str): IN: Owning agent identifier. OUT: Stored with the
                function.
            metadata (dict | None): IN: Optional metadata dict. OUT: Stored
                under the function name.

        Returns:
            None: OUT: Function is indexed.
        """

        func_name = get_callable_public_name(func)
        if func_name not in self._functions:
            self._functions[func_name] = []
        self._functions[func_name].append((func, agent_id))
        self._function_metadata[func_name] = metadata or {}

    def get_function(self, name: str, current_agent_id: str | None = None) -> tuple[tp.Callable | None, str | None]:
        """Retrieve a function by name, preferring the current agent's copy.

        Args:
            name (str): IN: Function identifier. OUT: Looked up in the index.
            current_agent_id (str | None): IN: Agent to prefer. OUT: If
                matched, that entry is returned first.

        Returns:
            tuple[tp.Callable | None, str | None]: OUT: Function and agent_id,
            or ``(None, None)``.
        """

        entries = self._functions.get(name, [])
        if not entries:
            return None, None
        if current_agent_id:
            for func, agent_id in entries:
                if agent_id == current_agent_id:
                    return func, agent_id
        return entries[0]

    def get_functions_by_agent(self, agent_id: str) -> list[tp.Callable]:
        """Return all functions owned by a specific agent.

        Args:
            agent_id (str): IN: Agent identifier. OUT: Filter key.

        Returns:
            list[tp.Callable]: OUT: Matching functions.
        """

        return [func for entries in self._functions.values() for func, aid in entries if aid == agent_id]


class AgentOrchestrator:
    """Manages a fleet of agents, routing function calls and handling switches.

    Args:
        max_agents (int): IN: Agent population limit. OUT: Enforced in
            ``register_agent``.
        enable_metrics (bool): IN: Whether to collect metrics. OUT: Stored.
    """

    def __init__(self, max_agents: int = 100, enable_metrics: bool = True):
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            max_agents (int, optional): IN: max agents. Defaults to 100. OUT: Consumed during execution.
            enable_metrics (bool, optional): IN: enable metrics. Defaults to True. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""
        self.agents: dict[str, Agent] = {}
        self.function_registry = FunctionRegistry()
        self.switch_triggers: dict[AgentSwitchTrigger, tp.Callable] = {}
        self.current_agent_id: str | None = None
        self.execution_history: list[dict] = []
        self.max_agents = max_agents
        self.enable_metrics = enable_metrics
        self._lock = threading.Lock()

    def register_agent(self, agent: Agent) -> None:
        """Add an agent to the orchestrator and index its functions.

        Args:
            agent (Agent): IN: Agent instance. OUT: Assigned an ID if absent,
                then stored.

        Returns:
            None: OUT: Agent is registered and its functions are indexed.

        Raises:
            ValueError: OUT: If the agent ID already exists or ``max_agents``
                is exceeded.
        """

        with self._lock:
            agent_id = agent.id
            if not agent_id:
                agent_id = f"agent_{len(self.agents)}"
                agent.id = agent_id

            if agent_id in self.agents:
                raise ValueError(f"Agent {agent_id} is already registered")
            if len(self.agents) >= self.max_agents:
                raise ValueError(f"Maximum number of agents ({self.max_agents}) reached")

            self.agents[agent_id] = agent

            for func in agent.functions:
                self.function_registry.register(func, agent_id)

            if self.current_agent_id is None:
                self.current_agent_id = agent_id

    def register_switch_trigger(self, trigger: AgentSwitchTrigger, handler: tp.Callable) -> None:
        """Register a callback that can propose agent switches.

        Args:
            trigger (AgentSwitchTrigger): IN: Trigger type. OUT: Used as key.
            handler (tp.Callable): IN: Function that returns a target agent ID.
                OUT: Stored.

        Returns:
            None: OUT: Trigger is registered.
        """

        self.switch_triggers[trigger] = handler

    def should_switch_agent(self, context: dict) -> str | None:
        """Evaluate switch triggers and return a target agent ID if any fire.

        Args:
            context (dict): IN: Runtime context. OUT: Passed to each trigger
                handler.

        Returns:
            str | None: OUT: Target agent ID, or ``None`` if no switch is
            needed.
        """

        current_agent = self.agents.get(self.current_agent_id) if self.current_agent_id else None
        if current_agent:
            for trigger in current_agent.switch_triggers:
                handler = self.switch_triggers.get(trigger)
                if handler:
                    try:
                        target_agent = handler(context, self.agents, self.current_agent_id)
                        if target_agent and target_agent != self.current_agent_id:
                            return target_agent
                    except Exception as e:
                        logger.error(f"Error in switch trigger {trigger}: {e}")

        for trigger, handler in self.switch_triggers.items():
            try:
                target_agent = handler(context, self.agents, self.current_agent_id)
                if target_agent and target_agent != self.current_agent_id:
                    return target_agent
            except Exception as e:
                logger.error(f"Error in switch trigger {trigger}: {e}")
        return None

    def switch_agent(self, target_agent_id: str, reason: str | None = None) -> None:
        """Change the active agent and log the transition.

        Args:
            target_agent_id (str): IN: Agent to activate. OUT: Looked up and
                set as ``current_agent_id``.
            reason (str | None): IN: Optional rationale. OUT: Logged.

        Returns:
            None: OUT: History is updated.

        Raises:
            ValueError: OUT: If ``target_agent_id`` is not registered.
        """

        with self._lock:
            if target_agent_id not in self.agents:
                raise ValueError(f"Agent {target_agent_id} not found")

            old_agent = self.current_agent_id
            self.current_agent_id = target_agent_id

            self.execution_history.append(
                {
                    "action": "agent_switch",
                    "type": "agent_switch",
                    "from": old_agent,
                    "to": target_agent_id,
                    "reason": reason,
                    "timestamp": self._get_timestamp(),
                }
            )

    def get_current_agent(self) -> Agent:
        """Return the currently active agent.

        Returns:
            Agent: OUT: Active agent instance.

        Raises:
            ValueError: OUT: If no agent is active.
        """

        if not self.current_agent_id:
            raise ValueError("No active agent")
        return self.agents[self.current_agent_id]

    def _get_timestamp(self) -> str:
        """Return the current ISO timestamp.

        Returns:
            str: OUT: ``datetime.now().isoformat()``.
        """

        return datetime.now().isoformat()


@dataclass
class FunctionExecutionHistory:
    """Record of completed function calls for context building.

    Attributes:
        executions (list[RequestFunctionCall]): IN: Empty initially. OUT:
            Appended by ``add_execution``.
        _execution_by_id (dict[str, RequestFunctionCall]): IN: Empty initially.
            OUT: Maps call ID to call.
        _executions_by_name (dict[str, list[RequestFunctionCall]]): IN: Empty
            initially. OUT: Maps function name to calls.
    """

    executions: list[RequestFunctionCall] = field(default_factory=list)
    _execution_by_id: dict[str, RequestFunctionCall] = field(default_factory=dict)
    _executions_by_name: dict[str, list[RequestFunctionCall]] = field(default_factory=dict)

    def add_execution(self, call: RequestFunctionCall) -> None:
        """Record a completed call.

        Args:
            call (RequestFunctionCall): IN: Completed call. OUT: Indexed by ID
                and name.

        Returns:
            None: OUT: Internal indexes are updated.
        """

        self.executions.append(call)
        self._execution_by_id[call.id] = call
        if call.name not in self._executions_by_name:
            self._executions_by_name[call.name] = []
        self._executions_by_name[call.name].append(call)

    def get_by_id(self, call_id: str) -> RequestFunctionCall | None:
        """Retrieve a call by its unique ID.

        Args:
            call_id (str): IN: Call identifier. OUT: Looked up in the index.

        Returns:
            RequestFunctionCall | None: OUT: Matching call or ``None``.
        """

        return self._execution_by_id.get(call_id)

    def get_by_name(self, name: str) -> RequestFunctionCall | None:
        """Return the most recent call for a given function name.

        Args:
            name (str): IN: Function name. OUT: Looked up in the index.

        Returns:
            RequestFunctionCall | None: OUT: Latest call or ``None``.
        """

        calls = self._executions_by_name.get(name)
        return calls[-1] if calls else None

    def get_successful_results(self) -> dict[str, tp.Any]:
        """Return results from all successful executions.

        Returns:
            dict[str, tp.Any]: OUT: Mapping from function name to result.
        """

        return {
            call.name: call.result
            for call in self.executions
            if call.status == ExecutionStatus.SUCCESS and call.result is not None
        }

    def as_context_dict(self) -> dict:
        """Build a serialisable context dictionary for prompt injection.

        Returns:
            dict: OUT: Contains ``function_history`` and ``latest_results``.
        """

        return {
            "function_history": [
                {
                    "name": call.name,
                    "id": call.id,
                    "status": call.status.value,
                    "result_summary": (
                        str(call.result)[:100] + "..."
                        if call.result and len(str(call.result)) > 100
                        else str(call.result)
                    ),
                }
                for call in self.executions
            ],
            "latest_results": {name: result for name, result in self.get_successful_results().items()},
        }


class FunctionExecutor:
    """Executes lists of ``RequestFunctionCall`` objects with strategies.

    Args:
        orchestrator (AgentOrchestrator): IN: Orchestrator providing agent and
            function registry. OUT: Stored.
    """

    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            orchestrator (AgentOrchestrator): IN: orchestrator. OUT: Consumed during execution."""
        self.orchestrator = orchestrator
        self.execution_queue: list[RequestFunctionCall] = []
        self.completed_calls: dict[str, RequestFunctionCall] = {}
        self.execution_history = FunctionExecutionHistory()

    async def execute_function_calls(
        self,
        calls: list[RequestFunctionCall],
        strategy: FunctionCallStrategy = FunctionCallStrategy.SEQUENTIAL,
        context_variables: dict | None = None,
        agent: Agent | None = None,
        runtime_features_state: RuntimeFeaturesState | None = None,
        loop_detector: LoopDetector | None = None,
    ) -> list[RequestFunctionCall]:
        """Execute a batch of function calls using the chosen strategy.

        Args:
            calls (list[RequestFunctionCall]): IN: Calls to execute. OUT:
                Routed to the appropriate strategy handler.
            strategy (FunctionCallStrategy): IN: Execution strategy. OUT:
                Selects sequential, parallel, pipeline, or conditional.
            context_variables (dict | None): IN: Shared context. OUT: Merged
                with execution history and passed to calls.
            agent (Agent | None): IN: Optional agent context. OUT: Passed to
                single-call execution.
            runtime_features_state (RuntimeFeaturesState | None): IN: Features
                like hooks, policy, sandbox. OUT: Passed to single-call
                execution.
            loop_detector (LoopDetector | None): IN: Loop detection state.
                OUT: Passed to single-call execution.

        Returns:
            list[RequestFunctionCall]: OUT: Executed calls with updated status
            and results.

        Raises:
            ValueError: OUT: If ``strategy`` is unrecognised.
        """

        context_variables = context_variables or {}
        context_variables.update(self.execution_history.as_context_dict())

        if strategy == FunctionCallStrategy.SEQUENTIAL:
            results = await self._execute_sequential(
                calls, context_variables, agent, runtime_features_state, loop_detector
            )
        elif strategy == FunctionCallStrategy.PARALLEL:
            results = await self._execute_parallel(
                calls, context_variables, agent, runtime_features_state, loop_detector
            )
        elif strategy == FunctionCallStrategy.PIPELINE:
            results = await self._execute_pipeline(
                calls, context_variables, agent, runtime_features_state, loop_detector
            )
        elif strategy == FunctionCallStrategy.CONDITIONAL:
            results = await self._execute_conditional(
                calls,
                context_variables,
                agent,
                runtime_features_state,
                loop_detector,
            )
        else:
            raise ValueError(f"Unknown execution strategy: {strategy}")

        for result in results:
            self.execution_history.add_execution(result)

        return results

    async def _execute_sequential(
        self,
        calls: list[RequestFunctionCall],
        context: dict,
        agent: Agent | None = None,
        runtime_features_state: RuntimeFeaturesState | None = None,
        loop_detector: LoopDetector | None = None,
    ) -> list[RequestFunctionCall]:
        """Execute calls one at a time, updating shared context on success.

        Args:
            calls (list[RequestFunctionCall]): IN: Calls to execute. OUT:
                Iterated in order.
            context (dict): IN: Shared context. OUT: Updated with result
                ``context_variables``.
            agent (Agent | None): IN: Agent context. OUT: Passed to
                ``_execute_single_call``.
            runtime_features_state (RuntimeFeaturesState | None): IN: Features.
                OUT: Passed to ``_execute_single_call``.
            loop_detector (LoopDetector | None): IN: Loop detector. OUT:
                Passed to ``_execute_single_call``.

        Returns:
            list[RequestFunctionCall]: OUT: Executed calls.
        """

        results = []
        for call in calls:
            try:
                result = await self._execute_single_call(call, context, agent, runtime_features_state, loop_detector)
                results.append(result)
                if hasattr(result.result, "context_variables"):
                    context.update(result.result.context_variables)
            except Exception as e:
                call.status = ExecutionStatus.FAILURE
                call.error = str(e)
                results.append(call)
        return results

    async def _execute_parallel(
        self,
        calls: list[RequestFunctionCall],
        context: dict,
        agent: Agent | None = None,
        runtime_features_state: RuntimeFeaturesState | None = None,
        loop_detector: LoopDetector | None = None,
    ) -> list[RequestFunctionCall]:
        """Execute calls concurrently with isolated context copies.

        Args:
            calls (list[RequestFunctionCall]): IN: Calls to execute. OUT:
                Gathered in parallel.
            context (dict): IN: Shared context. OUT: Copied for each call.
            agent (Agent | None): IN: Agent context. OUT: Passed to
                ``_execute_single_call``.
            runtime_features_state (RuntimeFeaturesState | None): IN: Features.
                OUT: Passed to ``_execute_single_call``.
            loop_detector (LoopDetector | None): IN: Loop detector. OUT:
                Passed to ``_execute_single_call``.

        Returns:
            list[RequestFunctionCall]: OUT: Executed calls; exceptions are
            converted to failure statuses.
        """

        context_dict = context if isinstance(context, dict) else {}
        tasks = [
            self._execute_single_call(call, context_dict.copy(), agent, runtime_features_state, loop_detector)
            for call in calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_results: list[RequestFunctionCall] = []
        for call, result in zip(calls, results, strict=False):
            if isinstance(result, Exception):
                call.status = ExecutionStatus.FAILURE
                call.error = str(result)
                final_results.append(call)
            elif isinstance(result, RequestFunctionCall):
                final_results.append(result)
            else:
                call.status = ExecutionStatus.FAILURE
                call.error = "Unexpected result type"
                final_results.append(call)
        return final_results

    async def _execute_pipeline(
        self,
        calls: list[RequestFunctionCall],
        context: dict,
        agent: Agent | None = None,
        runtime_features_state: RuntimeFeaturesState | None = None,
        loop_detector: LoopDetector | None = None,
    ) -> list[RequestFunctionCall]:
        """Execute calls sequentially, chaining ``previous_result`` through context.

        Args:
            calls (list[RequestFunctionCall]): IN: Calls to execute. OUT:
                Iterated in order.
            context (dict): IN: Shared context. OUT: Updated with each call's
                result.
            agent (Agent | None): IN: Agent context. OUT: Passed to
                ``_execute_single_call``.
            runtime_features_state (RuntimeFeaturesState | None): IN: Features.
                OUT: Passed to ``_execute_single_call``.
            loop_detector (LoopDetector | None): IN: Loop detector. OUT:
                Passed to ``_execute_single_call``.

        Returns:
            list[RequestFunctionCall]: OUT: Executed calls.
        """

        results = []

        context_dict = context if isinstance(context, dict) else {}
        current_context = context_dict.copy()

        for call in calls:
            result = await self._execute_single_call(
                call,
                current_context,
                agent,
                runtime_features_state,
                loop_detector,
            )
            results.append(result)

            if result.status == ExecutionStatus.SUCCESS and result.result:
                if hasattr(result.result, "value"):
                    current_context["previous_result"] = result.result.value
                if hasattr(result.result, "context_variables"):
                    current_context.update(result.result.context_variables)

        return results

    async def _execute_conditional(
        self,
        calls: list[RequestFunctionCall],
        context: dict,
        agent: Agent | None = None,
        runtime_features_state: RuntimeFeaturesState | None = None,
        loop_detector: LoopDetector | None = None,
    ) -> list[RequestFunctionCall]:
        """Execute calls in topological order, skipping those whose dependencies fail.

        Args:
            calls (list[RequestFunctionCall]): IN: Calls to execute. OUT:
                Sorted topologically.
            context (dict): IN: Shared context. OUT: Passed to each call.
            agent (Agent | None): IN: Agent context. OUT: Passed to
                ``_execute_single_call``.
            runtime_features_state (RuntimeFeaturesState | None): IN: Features.
                OUT: Passed to ``_execute_single_call``.
            loop_detector (LoopDetector | None): IN: Loop detector. OUT:
                Passed to ``_execute_single_call``.

        Returns:
            list[RequestFunctionCall]: OUT: Executed calls.
        """

        sorted_calls = self._topological_sort(calls)
        results: list[RequestFunctionCall] = []

        for call in sorted_calls:
            if self._dependencies_satisfied(call, results):
                result = await self._execute_single_call(
                    call,
                    context,
                    agent,
                    runtime_features_state,
                    loop_detector,
                )
                results.append(result)
                self.completed_calls[call.id] = result

        return results

    async def _execute_single_call(
        self,
        call: RequestFunctionCall,
        context: dict,
        agent: Agent | None = None,
        runtime_features_state: RuntimeFeaturesState | None = None,
        loop_detector: LoopDetector | None = None,
        audit_turn_id: str | None = None,
    ) -> RequestFunctionCall:
        """Execute one function call with full policy, sandbox, and hook support.

        Args:
            call (RequestFunctionCall): IN: Call to execute. OUT: Mutated with
                status, result, and error.
            context (dict): IN: Shared context. OUT: Passed to the function
                if it accepts ``context_variables``.
            agent (Agent | None): IN: Agent context. OUT: Passed to
                ``_resolve_function_and_agent``.
            runtime_features_state (RuntimeFeaturesState | None): IN: Features.
                OUT: Enables hooks, policy, sandbox routing, and audit.
            loop_detector (LoopDetector | None): IN: Loop detection. OUT:
                Checked before execution.
            audit_turn_id (str | None): IN: Turn ID for audit events. OUT:
                Passed to audit emitter.

        Returns:
            RequestFunctionCall: OUT: The same call object, updated.
        """

        call.status = ExecutionStatus.PENDING

        for attempt in range(call.max_retries + 1):
            agent_id = agent.id if agent is not None else None
            _audit = runtime_features_state.audit_emitter if runtime_features_state is not None else None
            try:
                func, agent_id = self._resolve_function_and_agent(call, agent, _audit, audit_turn_id)
                args = self._normalize_call_arguments(call)
                args = self._coerce_argument_types(args, func)
                args = self._resolve_argument_templates(args)

                try:
                    sig = inspect.signature(func)
                    func_accepts_context = __CTX_VARS_NAME__ in sig.parameters
                except (ValueError, TypeError):
                    func_accepts_context = False

                if func_accepts_context:
                    args[__CTX_VARS_NAME__] = context
                    if self.execution_history.executions:
                        args[__CTX_VARS_NAME__]["function_results"] = self.execution_history.get_successful_results()

                        if len(self.execution_history.executions) > 0:
                            previous_call = self.execution_history.executions[-1]
                            if previous_call.status == ExecutionStatus.SUCCESS:
                                args[__CTX_VARS_NAME__]["prior_result"] = previous_call.result

                if loop_detector is not None:
                    loop_event = loop_detector.record_call(call.name, args)
                    logger.info(
                        "loop_detection tool=%s severity=%s pattern=%s",
                        call.name,
                        loop_event.severity.value,
                        loop_event.pattern,
                    )
                    if _audit is not None and loop_event.severity.value != "none":
                        if loop_event.severity == LoopSeverity.CRITICAL:
                            _audit.emit_tool_loop_block(
                                call.name,
                                pattern=loop_event.pattern,
                                count=loop_event.call_count,
                                agent_id=agent_id,
                                turn_id=audit_turn_id,
                            )
                        else:
                            _audit.emit_loop_warning(
                                call.name,
                                pattern=loop_event.pattern,
                                severity=loop_event.severity.value,
                                count=loop_event.call_count,
                                agent_id=agent_id,
                                turn_id=audit_turn_id,
                            )
                    if loop_event.severity == LoopSeverity.CRITICAL:
                        raise ToolLoopError(loop_event)

                if runtime_features_state is not None:
                    policy_action = runtime_features_state.policy_engine.check(call.name, agent_id)
                    logger.info("tool_policy tool=%s agent=%s action=%s", call.name, agent_id, policy_action.value)
                    if _audit is not None:
                        _audit.emit_tool_policy_decision(
                            call.name,
                            agent_id=agent_id,
                            action=policy_action.value,
                            turn_id=audit_turn_id,
                        )
                    if policy_action == PolicyAction.DENY:
                        raise ToolPolicyViolation(call.name, agent_id)

                    if runtime_features_state.hook_runner.has_hooks("before_tool_call"):
                        original_args = args
                        args = runtime_features_state.hook_runner.run(
                            "before_tool_call",
                            tool_name=call.name,
                            arguments=args,
                            agent_id=agent_id,
                        )
                        if args != original_args:
                            logger.info("hook_mutation hook=before_tool_call tool=%s agent=%s", call.name, agent_id)
                            if _audit is not None:
                                _audit.emit_hook_mutation(
                                    "before_tool_call",
                                    tool_name=call.name,
                                    agent_id=agent_id,
                                    field="arguments",
                                    turn_id=audit_turn_id,
                                )

                if _audit is not None:
                    _audit.emit_tool_call_attempt(
                        call.name,
                        args=str(args)[:200],
                        agent_id=agent_id,
                        turn_id=audit_turn_id,
                    )
                _exec_start = time.perf_counter()

                router = (
                    runtime_features_state.get_sandbox_router(agent_id) if runtime_features_state is not None else None
                )
                if router is not None:
                    decision = router.decide(call.name)
                    logger.info(
                        "sandbox_routing tool=%s agent=%s context=%s reason=%s",
                        call.name,
                        agent_id,
                        decision.context.value,
                        decision.reason,
                    )
                    if _audit is not None:
                        _audit.emit_sandbox_decision(
                            call.name,
                            context=decision.context.value,
                            reason=decision.reason,
                            agent_id=agent_id,
                            turn_id=audit_turn_id,
                        )
                    if decision.context == ExecutionContext.SANDBOX:
                        result = await self._run_function_in_sandbox(router, call.name, func, args, call.timeout)
                    else:
                        result = await self._run_function_with_timeout(func, args, call.timeout)
                else:
                    result = await self._run_function_with_timeout(func, args, call.timeout)

                if runtime_features_state is not None and runtime_features_state.hook_runner.has_hooks(
                    "after_tool_call"
                ):
                    original_result = result
                    result = runtime_features_state.hook_runner.run(
                        "after_tool_call",
                        tool_name=call.name,
                        arguments=args,
                        result=result,
                        agent_id=agent_id,
                    )
                    if result != original_result:
                        logger.info("hook_mutation hook=after_tool_call tool=%s agent=%s", call.name, agent_id)
                        if _audit is not None:
                            _audit.emit_hook_mutation(
                                "after_tool_call",
                                tool_name=call.name,
                                agent_id=agent_id,
                                field="result",
                                turn_id=audit_turn_id,
                            )

                _exec_duration_ms = (time.perf_counter() - _exec_start) * 1000
                call.result = result

                if isinstance(result, Result) and result.agent is not None and result.agent.id:
                    if result.agent.id != self.orchestrator.current_agent_id:
                        if result.agent.id not in self.orchestrator.agents:
                            self.orchestrator.register_agent(result.agent)
                        old_agent = self.orchestrator.current_agent_id
                        self.orchestrator.switch_agent(
                            result.agent.id, f"Function {call.name} requested handoff to agent {result.agent.id}"
                        )
                        if _audit is not None:
                            _audit.emit_agent_switch(
                                from_agent=old_agent or "",
                                to_agent=result.agent.id,
                                reason=f"Function {call.name} requested handoff to agent {result.agent.id}",
                                agent_id=agent_id,
                                turn_id=audit_turn_id,
                            )
                call.status = ExecutionStatus.SUCCESS
                self.execution_history.add_execution(call)
                if _audit is not None:
                    _audit.emit_tool_call_complete(
                        call.name,
                        status="success",
                        duration_ms=_exec_duration_ms,
                        result=str(result)[:200],
                        agent_id=agent_id,
                        turn_id=audit_turn_id,
                    )

                _skill_meta_tools = {"skill_view", "skills_list", "skill_manage", "set_skill_registry"}
                if (
                    call.name in _skill_meta_tools
                    and runtime_features_state is not None
                    and runtime_features_state.audit_emitter is not None
                ):
                    skill_name = ""
                    if isinstance(result, dict):
                        skill_name = str(result.get("skill_name", result.get("name", "")) or "")
                    elif isinstance(result, str):
                        try:
                            parsed = json.loads(result)
                            skill_name = parsed.get("skill_name", parsed.get("name", ""))
                        except Exception:
                            pass
                    runtime_features_state.audit_emitter.emit_skill_used(
                        skill_name=skill_name,
                        version="",
                        outcome="success",
                        duration_ms=_exec_duration_ms,
                        triggered_automatically=False,
                        agent_id=agent_id,
                        turn_id=audit_turn_id,
                    )
                break

            except TimeoutError:
                call.retry_count += 1
                call.error = f"Function timed out after {call.timeout}s"
                if _audit is not None:
                    _audit.emit_tool_call_failure(
                        call.name,
                        error_type="TimeoutError",
                        error_msg=call.error,
                        agent_id=agent_id,
                        turn_id=audit_turn_id,
                    )
                if attempt < call.max_retries:
                    await asyncio.sleep(2**attempt)
            except (ToolLoopError, ToolPolicyViolation, SandboxExecutionUnavailableError) as e:
                call.retry_count += 1
                call.error = str(e)
                if _audit is not None:
                    _audit.emit_tool_call_failure(
                        call.name,
                        error_type=type(e).__name__,
                        error_msg=str(e),
                        agent_id=agent_id,
                        turn_id=audit_turn_id,
                    )
                break
            except Exception as e:
                traceback.print_exc()
                call.retry_count += 1
                call.error = str(e)
                if _audit is not None:
                    _audit.emit_tool_call_failure(
                        call.name,
                        error_type=type(e).__name__,
                        error_msg=str(e),
                        agent_id=agent_id,
                        turn_id=audit_turn_id,
                    )
                if attempt < call.max_retries:
                    await asyncio.sleep(2**attempt)

        if call.status != ExecutionStatus.SUCCESS:
            call.status = ExecutionStatus.FAILURE
            self.execution_history.add_execution(call)

        return call

    def _resolve_function_and_agent(
        self,
        call: RequestFunctionCall,
        agent: Agent | None,
        _audit,
        audit_turn_id,
    ) -> tuple[tp.Callable, str | None]:
        """Find the implementation and owning agent for a function call.

        Args:
            call (RequestFunctionCall): IN: Call to resolve. OUT: ``name`` is
                used for lookup.
            agent (Agent | None): IN: Preferred agent context. OUT: Checked
                first.
            _audit: IN: Audit emitter. OUT: Used to log agent switches.
            audit_turn_id: IN: Turn ID for audit. OUT: Passed to emitter.

        Returns:
            tuple[tp.Callable, str | None]: OUT: Function and agent_id.

        Raises:
            ValueError: OUT: If the function is not found.
        """

        if agent is not None:
            func = {get_callable_public_name(fn): fn for fn in agent.functions}.get(call.name, None)
            agent_id = agent.id
        else:
            func_result = self.orchestrator.function_registry.get_function(
                call.name, current_agent_id=self.orchestrator.current_agent_id
            )
            func, agent_id = func_result if func_result else (None, None)

            if agent_id and agent_id != self.orchestrator.current_agent_id:
                old_agent = self.orchestrator.current_agent_id
                self.orchestrator.switch_agent(agent_id, f"Function {call.name} requires agent {agent_id}")
                if _audit is not None:
                    _audit.emit_agent_switch(
                        from_agent=old_agent or "",
                        to_agent=agent_id,
                        reason=f"Function {call.name} requires agent {agent_id}",
                        agent_id=agent_id,
                        turn_id=audit_turn_id,
                    )

        if not func:
            raise ValueError(f"Function {call.name} not found")
        return func, agent_id

    @staticmethod
    def _normalize_call_arguments(call: RequestFunctionCall) -> dict:
        """Normalise a call's arguments into a plain dict.

        Args:
            call (RequestFunctionCall): IN: Call with ``arguments``. OUT:
                Parsed from dict, JSON string, or empty.

        Returns:
            dict: OUT: Clean argument mapping.
        """

        if isinstance(call.arguments, dict):
            return call.arguments.copy()
        if isinstance(call.arguments, str):
            if call.arguments == "":
                return {}
            try:
                return json.loads(call.arguments)
            except json.JSONDecodeError:
                try:
                    fixed = call.arguments.rstrip().rstrip("}").rstrip(",") + "}"
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    return {}
        return {}

    @staticmethod
    def _coerce_argument_types(args: dict, func: tp.Callable) -> dict:
        """Coerce string arguments to match the function's type annotations.

        Supports ``int``, ``float``, ``bool``, ``list``, and ``dict``.

        Args:
            args (dict): IN: Normalised arguments. OUT: Types may be adjusted.
            func (tp.Callable): IN: Target function. OUT: Signature is
                inspected.

        Returns:
            dict: OUT: Potentially modified arguments.
        """

        import typing

        coerced = dict(args)
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return coerced

        for param_name, param in sig.parameters.items():
            if param_name not in coerced:
                continue
            value = coerced[param_name]
            if not isinstance(value, str):
                continue
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                continue

            origin = getattr(ann, "__origin__", None)
            args_tuple = getattr(ann, "__args__", ())
            is_union = origin is typing.Union or type(ann).__name__ == "UnionType"
            if is_union and any(a is type(None) for a in args_tuple):
                ann = next(a for a in args_tuple if a is not type(None))
                origin = getattr(ann, "__origin__", None)
                args_tuple = getattr(ann, "__args__", ())

            if origin is list and len(args_tuple) == 1:
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        coerced[param_name] = parsed
                except Exception:
                    pass
                continue

            if origin is dict and len(args_tuple) == 2:
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        coerced[param_name] = parsed
                except Exception:
                    pass
                continue

            if ann is int:
                try:
                    coerced[param_name] = int(value)
                except ValueError:
                    pass
            elif ann is float:
                try:
                    coerced[param_name] = float(value)
                except ValueError:
                    pass
            elif ann is bool:
                coerced[param_name] = value.lower() in ("true", "1", "yes", "on")
        return coerced

    def _resolve_argument_templates(self, arguments: dict) -> dict:
        """Replace ``{call_id.attr}`` template references with actual values.

        Args:
            arguments (dict): IN: Arguments that may contain template strings.
                OUT: Templates are resolved against ``execution_history``.

        Returns:
            dict: OUT: Arguments with templates substituted.
        """

        pattern = re.compile(r"^\{([^{}]+)\}$")

        def _lookup(reference: str) -> tp.Any:
            """Resolve a ``call_id.attr`` reference from execution history.

            Args:
                reference (str): IN: Dotted reference. OUT: Split and looked
                    up.

            Returns:
                tp.Any: OUT: Attribute value or ``None``.
            """

            parts = reference.split(".")
            if len(parts) != 2:
                return None
            call_id, attr = parts
            call = self.execution_history.get_by_id(call_id)
            if call is None:
                return None
            return getattr(call, attr, None)

        def _resolve(value: tp.Any) -> tp.Any:
            """Recursively resolve templates in a value.

            Args:
                value (tp.Any): IN: Scalar, list, or dict. OUT: Templates are
                    replaced.

            Returns:
                tp.Any: OUT: Resolved value.
            """

            if isinstance(value, str):
                whole_match = pattern.match(value)
                if whole_match:
                    resolved = _lookup(whole_match.group(1))
                    return resolved if resolved is not None else value

                return re.sub(
                    r"\{([^{}]+)\}",
                    lambda match: str(
                        _lookup(match.group(1)) if _lookup(match.group(1)) is not None else match.group(0)
                    ),
                    value,
                )
            if isinstance(value, list):
                return [_resolve(item) for item in value]
            if isinstance(value, dict):
                return {key: _resolve(item) for key, item in value.items()}
            return value

        return {key: _resolve(value) for key, value in arguments.items()}

    async def _run_function(self, func: tp.Callable, args: dict) -> tp.Any:
        """Invoke ``func`` with ``args``, handling sync vs async.

        Args:
            func (tp.Callable): IN: Function to run. OUT: Invoked.
            args (dict): IN: Keyword arguments. OUT: Passed to ``func``.

        Returns:
            tp.Any: OUT: Function return value.
        """

        if asyncio.iscoroutinefunction(func):
            return await func(**args)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**args))

    async def _run_function_with_timeout(self, func: tp.Callable, args: dict, timeout: float | None) -> tp.Any:
        """Run a function with an optional timeout.

        Args:
            func (tp.Callable): IN: Function to run. OUT: Invoked.
            args (dict): IN: Keyword arguments. OUT: Passed to ``func``.
            timeout (float | None): IN: Seconds to wait. OUT: Passed to
                ``asyncio.wait_for``.

        Returns:
            tp.Any: OUT: Function return value.

        Raises:
            TimeoutError: OUT: If the timeout expires.
        """

        if timeout:
            return await asyncio.wait_for(self._run_function(func, args), timeout=timeout)
        return await self._run_function(func, args)

    async def _run_function_in_sandbox(
        self,
        router: tp.Any,
        tool_name: str,
        func: tp.Callable,
        args: dict,
        timeout: float | None,
    ) -> tp.Any:
        """Run a function inside a sandbox environment.

        Args:
            router (tp.Any): IN: Sandbox router. OUT: Used to decide and
                execute.
            tool_name (str): IN: Tool identifier. OUT: Passed to the router.
            func (tp.Callable): IN: Function to run. OUT: Passed to the
                router.
            args (dict): IN: Keyword arguments. OUT: Passed to the router.
            timeout (float | None): IN: Seconds to wait. OUT: Passed to
                ``asyncio.wait_for``.

        Returns:
            tp.Any: OUT: Sandbox execution result.

        Raises:
            SandboxExecutionUnavailableError: OUT: If the sandbox backend is
                missing.
            TimeoutError: OUT: If the timeout expires.
        """

        async def _sandbox_runner() -> tp.Any:
            """Asynchronously Internal helper to sandbox runner.

            Returns:
                tp.Any: OUT: Result of the operation."""
            if router.backend is None:
                raise SandboxExecutionUnavailableError(tool_name)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: router.execute_in_sandbox(tool_name, func, args))

        if timeout:
            return await asyncio.wait_for(_sandbox_runner(), timeout=timeout)
        return await _sandbox_runner()

    def _topological_sort(self, calls: list[RequestFunctionCall]) -> list[RequestFunctionCall]:
        """Sort calls so dependencies are satisfied before dependents.

        Args:
            calls (list[RequestFunctionCall]): IN: Calls with
                ``dependencies``. OUT: Reordered.

        Returns:
            list[RequestFunctionCall]: OUT: Topologically sorted calls.

        Raises:
            ValueError: OUT: If a circular dependency is detected.
        """

        sorted_calls = []
        remaining = calls.copy()

        while remaining:
            ready_calls = [call for call in remaining if all(dep in self.completed_calls for dep in call.dependencies)]

            if not ready_calls:
                remaining_names = [call.name for call in remaining]
                raise ValueError(f"Circular dependency detected in: {remaining_names}")

            sorted_calls.extend(ready_calls)
            for call in ready_calls:
                remaining.remove(call)

        return sorted_calls

    def _dependencies_satisfied(self, call: RequestFunctionCall, completed: list[RequestFunctionCall]) -> bool:
        """Check whether all of a call's dependencies have succeeded.

        Args:
            call (RequestFunctionCall): IN: Call to check. OUT:
                ``dependencies`` are inspected.
            completed (list[RequestFunctionCall]): IN: Already finished calls.
                OUT: Checked for success and ID membership.

        Returns:
            bool: OUT: ``True`` if every dependency is present and successful.
        """

        completed_ids = {c.id for c in completed if c.status == ExecutionStatus.SUCCESS}
        return all(dep in completed_ids for dep in call.dependencies)


if tp.TYPE_CHECKING:
    from .core.errors import (
        AgentError,
        FunctionExecutionError,
        ValidationError,
        XerxesTimeoutError,
    )
else:
    try:
        from .core.errors import (
            AgentError,
            FunctionExecutionError,
            ValidationError,
            XerxesTimeoutError,
        )
    except ImportError:

        class AgentError(Exception):
            """Base error for agent-related failures.

            Args:
                agent_id (str): IN: Agent identifier. OUT: Included in the
                    message.
                message (str): IN: Error description. OUT: Included in the
                    message.
            """

            def __init__(self, agent_id: str, message: str) -> None:
                """Initialize the instance.

                Args:
                    self: IN: The instance. OUT: Used for attribute access.
                    agent_id (str): IN: agent id. OUT: Consumed during execution.
                    message (str): IN: message. OUT: Consumed during execution."""
                super().__init__(f"Agent {agent_id}: {message}")

        class XerxesTimeoutError(Exception):
            """Raised when a function exceeds its timeout.

            Args:
                func_name (str): IN: Function identifier. OUT: Included in the
                    message.
                timeout (float): IN: Timeout seconds. OUT: Included in the
                    message.
            """

            def __init__(self, func_name: str, timeout: float) -> None:
                """Initialize the instance.

                Args:
                    self: IN: The instance. OUT: Used for attribute access.
                    func_name (str): IN: func name. OUT: Consumed during execution.
                    timeout (float): IN: timeout. OUT: Consumed during execution."""
                super().__init__(f"Function {func_name} timed out after {timeout}s")

        class FunctionExecutionError(Exception):
            """Raised when a function fails during execution.

            Args:
                func_name (str): IN: Function identifier. OUT: Included in
                    the message.
                message (str): IN: Error description. OUT: Included in the
                    message.
                original_error (BaseException | None): IN: Wrapped exception.
                    OUT: Stored.
            """

            def __init__(self, func_name: str, message: str, original_error: BaseException | None = None) -> None:
                """Initialize the instance.

                Args:
                    self: IN: The instance. OUT: Used for attribute access.
                    func_name (str): IN: func name. OUT: Consumed during execution.
                    message (str): IN: message. OUT: Consumed during execution.
                    original_error (BaseException | None, optional): IN: original error. Defaults to None. OUT: Consumed during execution."""
                super().__init__(f"Function {func_name}: {message}")
                self.original_error = original_error

        class ValidationError(Exception):
            """Raised when argument validation fails.

            Args:
                param_name (str): IN: Parameter identifier. OUT: Included in
                    the message.
                message (str): IN: Error description. OUT: Included in the
                    message.
            """

            def __init__(self, param_name: str, message: str) -> None:
                """Initialize the instance.

                Args:
                    self: IN: The instance. OUT: Used for attribute access.
                    param_name (str): IN: param name. OUT: Consumed during execution.
                    message (str): IN: message. OUT: Consumed during execution."""
                super().__init__(f"Validation error for {param_name}: {message}")


class RetryPolicy:
    """Exponential-backoff retry configuration.

    Args:
        max_retries (int): IN: Maximum retry attempts. OUT: Stored.
        initial_delay (float): IN: Starting delay in seconds. OUT: Stored.
        max_delay (float): IN: Delay ceiling. OUT: Stored.
        exponential_base (float): IN: Multiplier per attempt. OUT: Stored.
        jitter (bool): IN: Whether to randomise delay. OUT: Stored.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            max_retries (int, optional): IN: max retries. Defaults to 3. OUT: Consumed during execution.
            initial_delay (float, optional): IN: initial delay. Defaults to 1.0. OUT: Consumed during execution.
            max_delay (float, optional): IN: max delay. Defaults to 60.0. OUT: Consumed during execution.
            exponential_base (float, optional): IN: exponential base. Defaults to 2.0. OUT: Consumed during execution.
            jitter (bool, optional): IN: jitter. Defaults to True. OUT: Consumed during execution."""
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Compute the delay before a retry attempt.

        Args:
            attempt (int): IN: Zero-based attempt number. OUT: Used in the
                exponential formula.

        Returns:
            float: OUT: Delay in seconds, with optional jitter.
        """

        delay = min(self.initial_delay * (self.exponential_base**attempt), self.max_delay)
        if self.jitter:
            import random

            delay *= random.uniform(0.5, 1.5)
        return delay


@dataclass
class ExecutionMetrics:
    """Running statistics for function execution performance.

    Attributes:
        total_calls (int): IN: Counter. OUT: Incremented by
            ``record_execution``.
        successful_calls (int): IN: Counter. OUT: Incremented on success.
        failed_calls (int): IN: Counter. OUT: Incremented on failure.
        timeout_calls (int): IN: Counter. OUT: Currently unused.
        total_duration (float): IN: Accumulator. OUT: Incremented by
            ``record_execution``.
        average_duration (float): IN: Computed metric. OUT: Updated by
            ``record_execution``.
        max_duration (float): IN: Peak metric. OUT: Updated by
            ``record_execution``.
        min_duration (float): IN: Floor metric. OUT: Updated by
            ``record_execution``.
    """

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    max_duration: float = 0.0
    min_duration: float = float("inf")

    def record_execution(self, duration: float, status: ExecutionStatus) -> None:
        """Update metrics with a new observation.

        Args:
            duration (float): IN: Elapsed seconds. OUT: Incorporated.
            status (ExecutionStatus): IN: Outcome. OUT: Determines success vs
                failure increment.

        Returns:
            None: OUT: All counters and averages are updated.
        """

        self.total_calls += 1
        self.total_duration += duration

        if status == ExecutionStatus.SUCCESS:
            self.successful_calls += 1
        elif status == ExecutionStatus.FAILURE:
            self.failed_calls += 1

        self.max_duration = max(self.max_duration, duration)
        self.min_duration = min(self.min_duration, duration)
        self.average_duration = self.total_duration / self.total_calls


class EnhancedFunctionRegistry(FunctionRegistry):
    """Function registry extended with per-function validators and metrics."""

    def __init__(self) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""
        super().__init__()
        self._function_validators: dict[str, tp.Callable | None] = {}
        self._function_metrics: dict[str, ExecutionMetrics] = {}

    def register(
        self,
        func: tp.Callable,
        agent_id: str,
        metadata: dict | None = None,
        validator: tp.Callable | None = None,
    ) -> None:
        """Register a function with an optional validator and metrics slot.

        Args:
            func (tp.Callable): IN: Function to register. OUT: Passed to
                ``super().register``.
            agent_id (str): IN: Owning agent. OUT: Passed to
                ``super().register``.
            metadata (dict | None): IN: Optional metadata. OUT: Passed to
                ``super().register``.
            validator (tp.Callable | None): IN: Optional argument validator.
                OUT: Stored.

        Returns:
            None: OUT: Function, metadata, validator, and metrics are indexed.
        """

        super().register(func, agent_id, metadata)
        func_name = get_callable_public_name(func)
        self._function_validators[func_name] = validator
        self._function_metrics[func_name] = ExecutionMetrics()

    def validate_arguments(self, func_name: str, arguments: dict) -> None:
        """Validate that required parameters are present and pass custom validators.

        Args:
            func_name (str): IN: Function to validate. OUT: Looked up in the
                registry.
            arguments (dict): IN: Proposed arguments. OUT: Checked against
                signature and custom validator.

        Returns:
            None: OUT: Silent on success.

        Raises:
            ValidationError: OUT: If a required parameter is missing or the
                custom validator rejects the arguments.
        """

        entries = self._functions.get(func_name, [])
        if not entries:
            raise ValidationError(func_name, "Function not registered")

        func = entries[0][0]
        sig = inspect.signature(func)

        for param_name, param in sig.parameters.items():
            if param_name == __CTX_VARS_NAME__:
                continue

            if param.default == inspect.Parameter.empty and param_name not in arguments:
                raise ValidationError(param_name, f"Required parameter missing for {func_name}")

        validator = self._function_validators.get(func_name)
        if validator:
            validator(arguments)

    def get_metrics(self, func_name: str) -> ExecutionMetrics | None:
        """Return metrics for a registered function.

        Args:
            func_name (str): IN: Function identifier. OUT: Looked up.

        Returns:
            ExecutionMetrics | None: OUT: Metrics object or ``None``.
        """

        return self._function_metrics.get(func_name)


class EnhancedAgentOrchestrator(AgentOrchestrator):
    """Agent orchestrator that uses ``EnhancedFunctionRegistry``."""

    def __init__(self, max_agents: int = 100, enable_metrics: bool = True) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            max_agents (int, optional): IN: max agents. Defaults to 100. OUT: Consumed during execution.
            enable_metrics (bool, optional): IN: enable metrics. Defaults to True. OUT: Consumed during execution."""
        super().__init__(max_agents=max_agents, enable_metrics=enable_metrics)
        self.function_registry = EnhancedFunctionRegistry()


class EnhancedFunctionExecutor(FunctionExecutor):
    """Function executor with semaphore-based concurrency, retry, and timeout.

    Args:
        orchestrator (AgentOrchestrator): IN: Orchestrator instance. OUT:
            Passed to ``super().__init__``.
        default_timeout (float): IN: Default call timeout. OUT: Stored.
        retry_policy (RetryPolicy | None): IN: Retry configuration. OUT:
            Defaults to a new ``RetryPolicy``.
        max_concurrent_executions (int): IN: Semaphore size. OUT: Limits
            concurrent calls.
    """

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        default_timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        max_concurrent_executions: int = 10,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            orchestrator (AgentOrchestrator): IN: orchestrator. OUT: Consumed during execution.
            default_timeout (float, optional): IN: default timeout. Defaults to 30.0. OUT: Consumed during execution.
            retry_policy (RetryPolicy | None, optional): IN: retry policy. Defaults to None. OUT: Consumed during execution.
            max_concurrent_executions (int, optional): IN: max concurrent executions. Defaults to 10. OUT: Consumed during execution."""
        super().__init__(orchestrator)
        self.default_timeout = default_timeout
        self.retry_policy = retry_policy or RetryPolicy()
        self.max_concurrent = max_concurrent_executions
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_executions)

    async def _execute_single_call(
        self,
        call: RequestFunctionCall,
        context: dict,
        agent: Agent | None = None,
        runtime_features_state: Any = None,
        loop_detector: Any = None,
        audit_turn_id: str | None = None,
    ) -> RequestFunctionCall:
        """Delegate to the base implementation.

        Args:
            call (RequestFunctionCall): IN: Call to execute. OUT: Passed up.
            context (dict): IN: Shared context. OUT: Passed up.
            agent (Agent | None): IN: Agent context. OUT: Passed up.
            runtime_features_state (Any): IN: Features. OUT: Passed up.
            loop_detector (Any): IN: Loop detection. OUT: Passed up.
            audit_turn_id (str | None): IN: Audit turn ID. OUT: Passed up.

        Returns:
            RequestFunctionCall: OUT: Result from ``super()``.
        """

        return await super()._execute_single_call(
            call,
            context,
            agent,
            runtime_features_state=runtime_features_state,
            loop_detector=loop_detector,
            audit_turn_id=audit_turn_id,
        )

    async def execute_with_timeout(
        self,
        func: tp.Callable,
        arguments: dict,
        timeout: float | None = None,
    ) -> tp.Any:
        """Run a function with timeout, using the thread pool for sync functions.

        Args:
            func (tp.Callable): IN: Function to run. OUT: Invoked.
            arguments (dict): IN: Keyword arguments. OUT: Passed to ``func``.
            timeout (float | None): IN: Override timeout. OUT: Defaults to
                ``self.default_timeout``.

        Returns:
            tp.Any: OUT: Function result.

        Raises:
            XerxesTimeoutError: OUT: On timeout.
            FunctionExecutionError: OUT: On execution failure.
        """

        timeout = timeout or self.default_timeout

        try:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(func(**arguments), timeout=timeout)
            else:
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(self.thread_pool, functools.partial(func, **arguments))
                return await asyncio.wait_for(future, timeout=timeout)

        except TimeoutError:
            raise XerxesTimeoutError(get_callable_public_name(func), timeout) from None
        except Exception as e:
            raise FunctionExecutionError(get_callable_public_name(func), str(e), original_error=e) from e

    async def execute_with_retry(
        self,
        func: tp.Callable,
        arguments: dict,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> tp.Any:
        """Run a function with automatic retries on failure.

        Args:
            func (tp.Callable): IN: Function to run. OUT: Invoked repeatedly.
            arguments (dict): IN: Keyword arguments. OUT: Passed to ``func``.
            timeout (float | None): IN: Per-attempt timeout. OUT: Passed to
                ``execute_with_timeout``.
            retry_policy (RetryPolicy | None): IN: Override retry config.
                OUT: Defaults to ``self.retry_policy``.

        Returns:
            tp.Any: OUT: Function result on success.

        Raises:
            XerxesTimeoutError: OUT: On timeout (not retried).
            FunctionExecutionError: OUT: If all attempts fail.
        """

        policy = retry_policy or self.retry_policy
        last_error = None

        for attempt in range(policy.max_retries + 1):
            try:
                return await self.execute_with_timeout(func, arguments, timeout)

            except XerxesTimeoutError:
                raise

            except FunctionExecutionError as e:
                last_error = e
                if attempt < policy.max_retries:
                    delay = policy.get_delay(attempt)
                    logger.warning(
                        f"Function {get_callable_public_name(func)} failed (attempt {attempt + 1}), retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Function {get_callable_public_name(func)} failed after {policy.max_retries + 1} attempts"
                    )

        if last_error:
            raise last_error

    async def execute_single_call(
        self,
        call: RequestFunctionCall,
        context_variables: dict | None = None,
        agent: Agent | None = None,
    ) -> RequestFunctionCall:
        """Execute one call under the semaphore with metrics recording.

        Args:
            call (RequestFunctionCall): IN: Call to execute. OUT: Mutated.
            context_variables (dict | None): IN: Shared context. OUT: Passed to
                the function if accepted.
            agent (Agent | None): IN: Agent context. OUT: Used for timeout
                override.

        Returns:
            RequestFunctionCall: OUT: Updated call object.
        """

        async with self.execution_semaphore:
            start_time = time.time()
            func_name = call.name

            try:
                func_result = self.orchestrator.function_registry.get_function(
                    func_name, current_agent_id=self.orchestrator.current_agent_id
                )
                func = func_result[0] if func_result else None

                if not func:
                    raise FunctionExecutionError(func_name, "Function not found")

                registry = tp.cast(EnhancedFunctionRegistry, self.orchestrator.function_registry)
                registry.validate_arguments(func_name, call.arguments)

                if __CTX_VARS_NAME__ in inspect.signature(func).parameters:
                    call.arguments[__CTX_VARS_NAME__] = context_variables or {}

                timeout = (
                    agent.function_timeout if agent and hasattr(agent, "function_timeout") else self.default_timeout
                )

                result = await self.execute_with_retry(func, call.arguments, timeout)

                call.result = result

                if not hasattr(call, "status"):
                    call.status = ExecutionStatus.SUCCESS
                else:
                    call.status = ExecutionStatus.SUCCESS
                if not hasattr(call, "execution_time"):
                    setattr(call, "execution_time", time.time() - start_time)
                else:
                    setattr(call, "execution_time", time.time() - start_time)

                logger.info(f"Successfully executed {func_name} in {getattr(call, 'execution_time', 0):.2f}s")

            except XerxesTimeoutError as e:
                call.result = f"Function timed out: {e}"
                if hasattr(call, "status"):
                    call.status = ExecutionStatus.FAILURE
                if hasattr(call, "error"):
                    call.error = str(e)
                if hasattr(call, "execution_time"):
                    setattr(call, "execution_time", time.time() - start_time)
                logger.error(f"Function {func_name} timed out: {e}")

            except (FunctionExecutionError, ValidationError) as e:
                call.result = f"Function execution error: {e}"
                if hasattr(call, "status"):
                    call.status = ExecutionStatus.FAILURE
                if hasattr(call, "error"):
                    call.error = str(e)
                if hasattr(call, "execution_time"):
                    setattr(call, "execution_time", time.time() - start_time)
                logger.error(f"Function {func_name} failed: {e}")

            except Exception as e:
                call.result = f"Unexpected error: {e}"
                if hasattr(call, "status"):
                    call.status = ExecutionStatus.FAILURE
                if hasattr(call, "error"):
                    call.error = f"Unexpected error: {e!s}"
                if hasattr(call, "execution_time"):
                    setattr(call, "execution_time", time.time() - start_time)
                logger.error(f"Unexpected error in {func_name}: {e}", exc_info=True)

            finally:
                if self.orchestrator.enable_metrics:
                    registry = tp.cast(EnhancedFunctionRegistry, self.orchestrator.function_registry)
                    metrics = registry.get_metrics(func_name)
                    if metrics:
                        exec_time = getattr(call, "execution_time", 0)
                        status = getattr(call, "status", ExecutionStatus.SUCCESS)
                        metrics.record_execution(exec_time, status)

            return call

    async def execute_function_calls(
        self,
        calls: list[RequestFunctionCall],
        strategy: FunctionCallStrategy = FunctionCallStrategy.SEQUENTIAL,
        context_variables: dict | None = None,
        agent: Agent | None = None,
        runtime_features_state: RuntimeFeaturesState | None = None,
        loop_detector: LoopDetector | None = None,
    ) -> list[RequestFunctionCall]:
        """Execute a batch of calls, storing successful results in context.

        Supports ``SEQUENTIAL`` and ``PARALLEL`` only.

        Args:
            calls (list[RequestFunctionCall]): IN: Calls to execute. OUT:
                Executed.
            strategy (FunctionCallStrategy): IN: Execution mode. OUT:
                ``SEQUENTIAL`` or ``PARALLEL``.
            context_variables (dict | None): IN: Shared context. OUT:
                Enriched with ``{name}_result`` entries on sequential success.
            agent (Agent | None): IN: Agent context. OUT: Passed to
                ``execute_single_call``.
            runtime_features_state (RuntimeFeaturesState | None): IN: Features.
                OUT: Unused in this override.
            loop_detector (LoopDetector | None): IN: Loop detection. OUT:
                Unused in this override.

        Returns:
            list[RequestFunctionCall]: OUT: Executed calls.

        Raises:
            ValueError: OUT: If strategy is not supported.
        """

        context_variables = context_variables or {}

        if strategy == FunctionCallStrategy.SEQUENTIAL:
            results = []
            for call in calls:
                result = await self.execute_single_call(call, context_variables, agent)
                results.append(result)

                if result.status == ExecutionStatus.SUCCESS:
                    context_variables[f"{call.name}_result"] = result.result

        elif strategy == FunctionCallStrategy.PARALLEL:
            context_dict = context_variables if isinstance(context_variables, dict) else {}
            tasks = [self.execute_single_call(call, context_dict.copy(), agent) for call in calls]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        return results

    @asynccontextmanager
    async def batch_execution(self) -> tp.AsyncGenerator[EnhancedFunctionExecutor, None]:
        """Context manager for scoped batch execution.

        Returns:
            tp.AsyncGenerator[EnhancedFunctionExecutor, None]: OUT: Yields
            ``self``.
        """

        try:
            yield self
        finally:
            await asyncio.sleep(0)

    def __del__(self) -> None:
        """Clean up the thread pool on garbage collection.

        Returns:
            None: OUT: Thread pool is shut down without waiting.
        """

        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=False)
