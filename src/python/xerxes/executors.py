# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Function execution and agent orchestration system.

This module provides the core execution infrastructure for Xerxes,
including:
- Function registry and management
- Agent orchestration and switching
- Function execution with various strategies (sequential, parallel, pipeline)
- Retry policies and error handling
- Execution metrics and monitoring
- Enhanced versions with additional features

The module supports both synchronous and asynchronous function execution,
timeout management, and sophisticated error recovery mechanisms.
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
add_depth = lambda x, ep=False: SEP + x.replace("\n", f"\n{SEP}") if ep else x.replace("\n", f"\n{SEP}")  # noqa


class FunctionRegistry:
    """Registry for managing functions across agents.

    Maintains a central registry of all functions available in the system,
    tracking which agent owns each function and associated metadata.
    Multiple agents may register the same function name; lookups prefer
    the currently active agent's version.

    Attributes:
        _functions: Dictionary mapping function names to lists of
            (callable, agent_id) tuples.
        _function_metadata: Dictionary mapping function names to metadata.
    """

    def __init__(self):
        """Initialize an empty function registry."""
        self._functions: dict[str, list[tuple[tp.Callable, str]]] = {}
        self._function_metadata: dict[str, dict] = {}

    def register(self, func: tp.Callable, agent_id: str, metadata: dict | None = None):
        """Register a function with the registry.

        Args:
            func: The callable function to register.
            agent_id: ID of the agent that owns this function.
            metadata: Optional metadata about the function.
        """
        func_name = get_callable_public_name(func)
        if func_name not in self._functions:
            self._functions[func_name] = []
        self._functions[func_name].append((func, agent_id))
        self._function_metadata[func_name] = metadata or {}

    def get_function(self, name: str, current_agent_id: str | None = None) -> tuple[tp.Callable | None, str | None]:
        """Get function and its associated agent.

        Args:
            name: Name of the function to retrieve.
            current_agent_id: Optional ID of the currently active agent.
                If provided, the agent's own version is preferred.

        Returns:
            Tuple of (function, agent_id) or (None, None) if not found.
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
        """Get all functions for a specific agent.

        Args:
            agent_id: ID of the agent.

        Returns:
            List of functions registered to the agent.
        """
        return [func for entries in self._functions.values() for func, aid in entries if aid == agent_id]


class AgentOrchestrator:
    """Orchestrates multiple agents and handles switching logic.

    Manages a collection of agents, their functions, and the logic for
    switching between agents based on various triggers.

    Attributes:
        agents: Dictionary of registered agents by ID.
        function_registry: Registry of all available functions.
        switch_triggers: Dictionary of trigger handlers for agent switching.
        current_agent_id: ID of the currently active agent.
        execution_history: History of agent switches and executions.
        max_agents: Maximum number of agents that may be registered.
        enable_metrics: Whether to record per-function execution metrics.
    """

    def __init__(self, max_agents: int = 100, enable_metrics: bool = True):
        """Initialize the agent orchestrator."""
        self.agents: dict[str, Agent] = {}
        self.function_registry = FunctionRegistry()
        self.switch_triggers: dict[AgentSwitchTrigger, tp.Callable] = {}
        self.current_agent_id: str | None = None
        self.execution_history: list[dict] = []
        self.max_agents = max_agents
        self.enable_metrics = enable_metrics
        self._lock = threading.Lock()

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: The agent instance to register.

        Returns:
            None

        Raises:
            ValueError: If the agent is already registered or max agents reached.
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
        """Register a custom switch trigger handler.

        Args:
            trigger: The trigger type to register.
            handler: The callable handler for this trigger.

        Returns:
            None
        """
        self.switch_triggers[trigger] = handler

    def should_switch_agent(self, context: dict) -> str | None:
        """Determine if agent switching is needed.

        Evaluates both the current agent's own ``switch_triggers`` and the
        orchestrator-level trigger handlers.

        Args:
            context: The current execution context.

        Returns:
            The ID of the target agent if switching is needed, None otherwise.
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
        """Switch to a different agent.

        Args:
            target_agent_id: ID of the agent to switch to.
            reason: Optional reason for the switch.

        Returns:
            None

        Raises:
            ValueError: If the target agent is not found.
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
        """Get the currently active agent.

        Returns:
            The currently active Agent instance.

        Raises:
            ValueError: If no agent is currently active.
        """
        if not self.current_agent_id:
            raise ValueError("No active agent")
        return self.agents[self.current_agent_id]

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format.

        Returns:
            Current timestamp as an ISO formatted string.
        """
        return datetime.now().isoformat()


@dataclass
class FunctionExecutionHistory:
    """History of function executions and their results."""

    executions: list[RequestFunctionCall] = field(default_factory=list)
    _execution_by_id: dict[str, RequestFunctionCall] = field(default_factory=dict)
    _executions_by_name: dict[str, list[RequestFunctionCall]] = field(default_factory=dict)

    def add_execution(self, call: RequestFunctionCall) -> None:
        """Add a completed function call to the history, indexing it by ID and name."""
        self.executions.append(call)
        self._execution_by_id[call.id] = call
        if call.name not in self._executions_by_name:
            self._executions_by_name[call.name] = []
        self._executions_by_name[call.name].append(call)

    def get_by_id(self, call_id: str) -> RequestFunctionCall | None:
        """Return the function call matching call_id, or None if not found."""
        return self._execution_by_id.get(call_id)

    def get_by_name(self, name: str) -> RequestFunctionCall | None:
        """Return the most recently recorded function call with the given name, or None."""
        calls = self._executions_by_name.get(name)
        return calls[-1] if calls else None

    def get_successful_results(self) -> dict[str, tp.Any]:
        """Return a mapping of function_name to result for all successful calls."""
        return {
            call.name: call.result
            for call in self.executions
            if call.status == ExecutionStatus.SUCCESS and call.result is not None
        }

    def as_context_dict(self) -> dict:
        """Convert execution history to a context dictionary suitable for prompt generation."""
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
    """Handles function execution with various strategies."""

    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        """Initialize the FunctionExecutor with a backing AgentOrchestrator."""
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
        """Execute function calls using the specified strategy.

        Args:
            calls: List of function calls to execute.
            strategy: Execution strategy (SEQUENTIAL, PARALLEL, PIPELINE, CONDITIONAL).
            context_variables: Optional context variables to pass to functions.
            agent: Optional agent instance for function lookup.
            runtime_features_state: Optional runtime features for policy/hooks/audit.
            loop_detector: Optional loop detector to guard against repetitive tool calls.

        Returns:
            List of RequestFunctionCall instances with populated results and statuses.
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
        """Execute calls one after another, passing context updates between them."""
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
        """Execute calls in parallel using asyncio.gather."""

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
        """Execute calls in a pipeline where the output of one feeds into the next."""
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
        """Execute calls in topological dependency order, skipping unsatisfied deps."""
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
        """Execute a single function call with error handling and retries.

        Applies loop detection, policy checks, before/after hooks, sandbox routing,
        and emits audit events throughout the lifecycle of the call.

        Args:
            call: The function call to execute.
            context: Context variables for the call.
            agent: Optional agent used for direct function lookup.
            runtime_features_state: Runtime state supplying policy, hooks, and audit.
            loop_detector: Optional detector for repetitive tool-call patterns.
            audit_turn_id: Optional turn identifier for audit trail correlation.

        Returns:
            The updated RequestFunctionCall with result/status/error populated.
        """
        call.status = ExecutionStatus.PENDING

        for attempt in range(call.max_retries + 1):
            agent_id = agent.id if agent is not None else None
            _audit = runtime_features_state.audit_emitter if runtime_features_state is not None else None
            try:
                func, agent_id = self._resolve_function_and_agent(call, agent, _audit, audit_turn_id)
                args = self._normalize_call_arguments(call)
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
        """Resolve the callable and owning agent ID for a function call.

        Args:
            call: The function call whose name will be looked up.
            agent: Optional agent to search first; falls back to the orchestrator registry.
            _audit: Optional audit emitter for emitting agent switch events.
            audit_turn_id: Optional turn identifier for audit trail correlation.

        Returns:
            Tuple of (callable, agent_id). Triggers an agent switch if the function
            belongs to a different agent than the currently active one.

        Raises:
            ValueError: If the function name is not found in the agent or registry.
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
        """Normalize call arguments to a plain dict, parsing JSON strings if needed."""
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

    def _resolve_argument_templates(self, arguments: dict) -> dict:
        """Resolve template references like ``{call_id.attr}`` within argument values.

        Supports whole-value replacement (e.g. ``"{prev.result}"``) and
        inline interpolation (e.g. ``"prefix {prev.result} suffix"``).

        Args:
            arguments: Raw argument dictionary potentially containing template strings.

        Returns:
            A new dictionary with template references resolved to concrete values.
        """
        pattern = re.compile(r"^\{([^{}]+)\}$")

        def _lookup(reference: str) -> tp.Any:
            """Look up a dotted reference (call_id.attr) in execution history."""
            parts = reference.split(".")
            if len(parts) != 2:
                return None
            call_id, attr = parts
            call = self.execution_history.get_by_id(call_id)
            if call is None:
                return None
            return getattr(call, attr, None)

        def _resolve(value: tp.Any) -> tp.Any:
            """Recursively resolve template references in a value."""
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
        """Run a function, awaiting it if it is a coroutine function or offloading to a thread otherwise."""
        if asyncio.iscoroutinefunction(func):
            return await func(**args)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**args))

    async def _run_function_with_timeout(self, func: tp.Callable, args: dict, timeout: float | None) -> tp.Any:
        """Run a function with an optional timeout, raising asyncio.TimeoutError if exceeded."""
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
        """Execute a function via the sandbox router, applying the optional timeout.

        Args:
            router: SandboxRouter instance that provides the sandbox backend.
            tool_name: Name of the tool being executed (used for backend dispatch).
            func: The callable to execute inside or outside the sandbox.
            args: Keyword arguments to pass to the function.
            timeout: Optional wall-clock timeout in seconds.

        Returns:
            The function's return value.

        Raises:
            SandboxExecutionUnavailableError: If no sandbox backend is configured.
            asyncio.TimeoutError: If the execution exceeds the timeout.
        """

        async def _sandbox_runner() -> tp.Any:
            if router.backend is None:
                raise SandboxExecutionUnavailableError(tool_name)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: router.execute_in_sandbox(tool_name, func, args))

        if timeout:
            return await asyncio.wait_for(_sandbox_runner(), timeout=timeout)
        return await _sandbox_runner()

    def _topological_sort(self, calls: list[RequestFunctionCall]) -> list[RequestFunctionCall]:
        """Sort function calls into a dependency-safe execution order.

        Args:
            calls: List of function calls, each potentially listing other call IDs
                in their ``dependencies`` attribute.

        Returns:
            The same calls reordered so that dependencies come before dependents.

        Raises:
            ValueError: If a circular dependency is detected.
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
        """Return True if all dependencies of call have been successfully completed."""
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
            """Raised when an agent-level error occurs."""

            def __init__(self, agent_id: str, message: str) -> None:
                """Initialize with the agent ID and error message."""
                super().__init__(f"Agent {agent_id}: {message}")

        class XerxesTimeoutError(Exception):
            """Raised when a function execution exceeds its time limit."""

            def __init__(self, func_name: str, timeout: float) -> None:
                """Initialize with the function name and timeout duration."""
                super().__init__(f"Function {func_name} timed out after {timeout}s")

        class FunctionExecutionError(Exception):
            """Raised when a function call fails during execution."""

            def __init__(self, func_name: str, message: str, original_error: BaseException | None = None) -> None:
                """Initialize with the function name, error message, and optional original exception."""
                super().__init__(f"Function {func_name}: {message}")
                self.original_error = original_error

        class ValidationError(Exception):
            """Raised when function argument validation fails."""

            def __init__(self, param_name: str, message: str) -> None:
                """Initialize with the parameter name and validation message."""
                super().__init__(f"Validation error for {param_name}: {message}")


class RetryPolicy:
    """Configurable retry policy for function execution."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ) -> None:
        """Initialize the RetryPolicy with backoff parameters.

        Args:
            max_retries: Maximum number of retry attempts after the initial try.
            initial_delay: Delay in seconds before the first retry.
            max_delay: Maximum delay cap in seconds.
            exponential_base: Base for exponential backoff calculation.
            jitter: If True, apply random jitter to the delay to avoid thundering herd.
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt."""
        delay = min(self.initial_delay * (self.exponential_base**attempt), self.max_delay)
        if self.jitter:
            import random

            delay *= random.uniform(0.5, 1.5)
        return delay


@dataclass
class ExecutionMetrics:
    """Metrics for function execution.

    Tracks aggregate statistics across all recorded executions including
    success/failure counts, call counts, and duration statistics.
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
        """Record a single execution result into the running metrics.

        Args:
            duration: Wall-clock time for the execution in seconds.
            status: The terminal status of the execution.
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
    """Enhanced registry with validation, metrics, and metadata management.

    Extends :class:`FunctionRegistry` with per-function argument validators
    and execution metrics while inheriting multi-agent function storage.
    """

    def __init__(self) -> None:
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
        """Register a function with optional argument validator.

        Args:
            func: The callable to register.
            agent_id: ID of the owning agent.
            metadata: Optional metadata dictionary for the function.
            validator: Optional callable that receives the arguments dict and raises
                on invalid input.
        """
        super().register(func, agent_id, metadata)
        func_name = get_callable_public_name(func)
        self._function_validators[func_name] = validator
        self._function_metrics[func_name] = ExecutionMetrics()

    def validate_arguments(self, func_name: str, arguments: dict) -> None:
        """Validate that all required parameters are present and pass the custom validator.

        Args:
            func_name: Name of the registered function to validate against.
            arguments: Argument dictionary to check.

        Raises:
            ValidationError: If the function is not registered, a required parameter
                is missing, or the custom validator rejects the arguments.
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
        """Get execution metrics for a function."""
        return self._function_metrics.get(func_name)


class EnhancedAgentOrchestrator(AgentOrchestrator):
    """Enhanced orchestrator — alias for AgentOrchestrator.

    All enhancements have been merged into the base class; this subclass
    is kept for backward compatibility and explicit opt-in.
    """

    def __init__(self, max_agents: int = 100, enable_metrics: bool = True) -> None:
        super().__init__(max_agents=max_agents, enable_metrics=enable_metrics)
        self.function_registry = EnhancedFunctionRegistry()


class EnhancedFunctionExecutor(FunctionExecutor):
    """Enhanced function executor — extends FunctionExecutor with retry, timeout,
    concurrency limits, and batch execution helpers.
    """

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        default_timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        max_concurrent_executions: int = 10,
    ) -> None:
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
        """Compatibility wrapper for Xerxes internal usage.

        Delegates to the base :class:`FunctionExecutor` implementation while
        accepting the same signature so ``Xerxes`` can use either executor
        interchangeably.
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
        """Execute a function with a timeout, wrapping exceptions in framework types."""
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
        """Execute a function with automatic retries on FunctionExecutionError."""
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
        """Execute a single function call with full error handling."""
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
        """Execute multiple function calls with specified strategy."""
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
        """Async context manager for a batch execution session with guaranteed cleanup."""
        try:
            yield self
        finally:
            await asyncio.sleep(0)

    def __del__(self) -> None:
        """Shut down the thread pool executor on garbage collection."""
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=False)
