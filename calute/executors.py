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
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime

from .types.function_execution_types import (
    AgentSwitchTrigger,
    ExecutionStatus,
    FunctionCallStrategy,
    RequestFunctionCall,
)

if tp.TYPE_CHECKING:
    from .types import Agent

logger = logging.getLogger(__name__)

__CTX_VARS_NAME__ = "context_variables"
SEP = "  "
add_depth = (  # noqa
    lambda x, ep=False: SEP + x.replace("\n", f"\n{SEP}") if ep else x.replace("\n", f"\n{SEP}")
)


class FunctionRegistry:
    """Registry for managing functions across agents"""

    def __init__(self):
        self._functions: dict[str, tp.Callable] = {}
        self._function_agents: dict[str, str] = {}  # function_name -> agent_id
        self._function_metadata: dict[str, dict] = {}

    def register(self, func: tp.Callable, agent_id: str, metadata: dict | None = None):
        """Register a function with the registry"""
        func_name = func.__name__
        self._functions[func_name] = func
        self._function_agents[func_name] = agent_id
        self._function_metadata[func_name] = metadata or {}

    def get_function(self, name: str) -> tuple[tp.Callable | None, str | None]:
        """Get function and its associated agent"""
        func = self._functions.get(name)
        agent_id = self._function_agents.get(name)
        return func, agent_id

    def get_functions_by_agent(self, agent_id: str) -> list[tp.Callable]:
        """Get all functions for a specific agent"""
        return [func for func_name, func in self._functions.items() if self._function_agents[func_name] == agent_id]


class AgentOrchestrator:
    """Orchestrates multiple agents and handles switching logic"""

    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self.function_registry = FunctionRegistry()
        self.switch_triggers: dict[AgentSwitchTrigger, tp.Callable] = {}
        self.current_agent_id: str | None = None
        self.execution_history: list[dict] = []

    def register_agent(self, agent: Agent):
        """Register an agent with the orchestrator"""
        agent_id = agent.id or f"agent_{len(self.agents)}"
        agent.id = agent_id
        self.agents[agent_id] = agent

        for func in agent.functions:
            self.function_registry.register(func, agent_id)

        if self.current_agent_id is None:
            self.current_agent_id = agent_id

    def register_switch_trigger(self, trigger: AgentSwitchTrigger, handler: tp.Callable):
        """Register a custom switch trigger handler"""
        self.switch_triggers[trigger] = handler

    def should_switch_agent(self, context: dict) -> str | None:
        """Determine if agent switching is needed"""
        for _, handler in self.switch_triggers.items():
            target_agent = handler(context, self.agents, self.current_agent_id)
            if target_agent and target_agent != self.current_agent_id:
                return target_agent
        return None

    def switch_agent(self, target_agent_id: str, reason: str | None = None):
        """Switch to a different agent"""
        if target_agent_id not in self.agents:
            raise ValueError(f"Agent {target_agent_id} not found")

        old_agent = self.current_agent_id
        self.current_agent_id = target_agent_id

        self.execution_history.append(
            {
                "type": "agent_switch",
                "from": old_agent,
                "to": target_agent_id,
                "reason": reason,
                "timestamp": self._get_timestamp(),
            }
        )

    def get_current_agent(self) -> Agent:
        """Get the currently active agent"""
        if not self.current_agent_id:
            raise ValueError("No active agent")
        return self.agents[self.current_agent_id]

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime

        return datetime.datetime.now().isoformat()


@dataclass
class FunctionExecutionHistory:
    """History of function executions and their results"""

    executions: list[RequestFunctionCall] = field(default_factory=list)
    execution_map: dict[str, RequestFunctionCall] = field(default_factory=dict)

    def add_execution(self, call: RequestFunctionCall):
        """Add an execution to the history"""
        self.executions.append(call)
        self.execution_map[call.id] = call
        self.execution_map[call.name] = call

    def get_by_id(self, call_id: str) -> RequestFunctionCall | None:
        """Get function call by ID"""
        return self.execution_map.get(call_id)

    def get_by_name(self, name: str) -> RequestFunctionCall | None:
        """Get latest function call by name"""
        return self.execution_map.get(name)

    def get_successful_results(self) -> dict[str, tp.Any]:
        """Get all successful results as a dictionary of function_name -> result"""
        return {
            call.name: call.result
            for call in self.executions
            if call.status == ExecutionStatus.SUCCESS and call.result is not None
        }

    def as_context_dict(self) -> dict:
        """Convert execution history to a context dictionary for prompt generation"""
        return {
            "function_history": [
                {
                    "name": call.name,
                    "id": call.id,
                    "status": call.status.value,
                    "result_summary": str(call.result)[:100] + "..."
                    if call.result and len(str(call.result)) > 100
                    else str(call.result),
                }
                for call in self.executions
            ],
            "latest_results": {name: result for name, result in self.get_successful_results().items()},
        }


class FunctionExecutor:
    """Handles function execution with various strategies"""

    def __init__(self, orchestrator: AgentOrchestrator):
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
    ) -> list[RequestFunctionCall]:
        """Execute function calls using the specified strategy"""
        context_variables = context_variables or {}
        context_variables.update(self.execution_history.as_context_dict())

        if strategy == FunctionCallStrategy.SEQUENTIAL:
            results = await self._execute_sequential(calls, context_variables, agent)
        elif strategy == FunctionCallStrategy.PARALLEL:
            results = await self._execute_parallel(calls, context_variables, agent)
        elif strategy == FunctionCallStrategy.PIPELINE:
            results = await self._execute_pipeline(calls, context_variables, agent)
        elif strategy == FunctionCallStrategy.CONDITIONAL:
            results = await self._execute_conditional(calls, context_variables, agent)
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
    ) -> list[RequestFunctionCall]:
        """Execute calls one after another"""
        results = []
        for call in calls:
            try:
                result = await self._execute_single_call(call, context, agent)
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
    ) -> list[RequestFunctionCall]:
        """Execute calls in parallel"""
        tasks = [self._execute_single_call(call, context.copy(), agent) for call in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_results = []
        for call, result in zip(calls, results, strict=False):
            if isinstance(result, Exception):
                call.status = ExecutionStatus.FAILURE
                call.error = str(result)
                final_results.append(call)
            else:
                final_results.append(result)
        return final_results

    async def _execute_pipeline(
        self,
        calls: list[RequestFunctionCall],
        context: dict,
        agent: Agent | None = None,
    ) -> list[RequestFunctionCall]:
        """Execute calls in a pipeline where output of one feeds into next"""
        results = []
        current_context = context.copy()

        for call in calls:
            result = await self._execute_single_call(call, current_context, agent)
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
    ) -> list[RequestFunctionCall]:
        """Execute calls based on conditions and dependencies"""
        sorted_calls = self._topological_sort(calls)
        results = []

        for call in sorted_calls:
            if self._dependencies_satisfied(call, results):
                result = await self._execute_single_call(call, context, agent)
                results.append(result)
                self.completed_calls[call.id] = result

        return results

    async def _execute_single_call(
        self,
        call: RequestFunctionCall,
        context: dict,
        agent: Agent | None = None,
    ) -> RequestFunctionCall:
        """Execute a single function call with error handling and retries"""
        call.status = ExecutionStatus.PENDING

        for attempt in range(call.max_retries + 1):
            try:
                if agent is not None:
                    func, agent_id = {fn.__name__: fn for fn in agent.functions}.get(call.name, None), agent.id

                else:
                    func, agent_id = self.orchestrator.function_registry.get_function(call.name)

                    if agent_id != self.orchestrator.current_agent_id:
                        self.orchestrator.switch_agent(agent_id, f"Function {call.name} requires agent {agent_id}")

                if not func:
                    raise ValueError(f"Function {call.name} not found")
                args = call.arguments.copy()
                if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                    args[__CTX_VARS_NAME__] = context
                    if self.execution_history.executions:
                        args[__CTX_VARS_NAME__]["function_results"] = self.execution_history.get_successful_results()

                        if len(self.execution_history.executions) > 0:
                            previous_call = self.execution_history.executions[-1]
                            if previous_call.status == ExecutionStatus.SUCCESS:
                                args[__CTX_VARS_NAME__]["prior_result"] = previous_call.result

                if call.timeout:
                    result = await asyncio.wait_for(self._run_function(func, args), timeout=call.timeout)
                else:
                    result = await self._run_function(func, args)

                call.result = result
                call.status = ExecutionStatus.SUCCESS
                self.execution_history.add_execution(call)
                break

            except asyncio.TimeoutError:
                call.retry_count += 1
                call.error = f"Function timed out after {call.timeout}s"
                if attempt < call.max_retries:
                    await asyncio.sleep(2**attempt)
            except Exception as e:
                call.retry_count += 1
                call.error = str(e)
                if attempt < call.max_retries:
                    await asyncio.sleep(2**attempt)

        if call.status != ExecutionStatus.SUCCESS:
            call.status = ExecutionStatus.FAILURE
            self.execution_history.add_execution(call)

        return call

    async def _run_function(self, func: tp.Callable, args: dict):
        """Run function async or sync"""
        if asyncio.iscoroutinefunction(func):
            return await func(**args)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**args))

    def _topological_sort(self, calls: list[RequestFunctionCall]) -> list[RequestFunctionCall]:
        """Sort function calls based on dependencies"""
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
        """Check if call's dependencies are satisfied"""
        completed_ids = {c.id for c in completed if c.status == ExecutionStatus.SUCCESS}
        return all(dep in completed_ids for dep in call.dependencies)


# ===========================
# ENHANCED CLASSES BELOW
# ===========================

try:
    from .errors import (
        AgentError,
        CaluteTimeoutError,
        FunctionExecutionError,
        ValidationError,
    )
except ImportError:
    # Define stub errors if errors module is not available
    class AgentError(Exception):
        def __init__(self, agent_id: str, message: str):
            super().__init__(f"Agent {agent_id}: {message}")
    
    class CaluteTimeoutError(Exception):
        def __init__(self, func_name: str, timeout: float):
            super().__init__(f"Function {func_name} timed out after {timeout}s")
    
    class FunctionExecutionError(Exception):
        def __init__(self, func_name: str, message: str, original_error=None):
            super().__init__(f"Function {func_name}: {message}")
            self.original_error = original_error
    
    class ValidationError(Exception):
        def __init__(self, param_name: str, message: str):
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
    ):
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
    """Metrics for function execution."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    max_duration: float = 0.0
    min_duration: float = float("inf")

    def record_execution(self, duration: float, status: ExecutionStatus):
        """Record execution metrics."""
        self.total_calls += 1
        self.total_duration += duration

        if status == ExecutionStatus.SUCCESS:
            self.successful_calls += 1
        elif status == ExecutionStatus.FAILURE:
            self.failed_calls += 1
        # Note: We count timeouts as failures since there's no TIMEOUT status

        self.max_duration = max(self.max_duration, duration)
        self.min_duration = min(self.min_duration, duration)
        self.average_duration = self.total_duration / self.total_calls


class EnhancedFunctionRegistry:
    """Enhanced registry with validation and metadata management."""

    def __init__(self):
        self._functions: dict[str, tp.Callable] = {}
        self._function_agents: dict[str, str] = {}
        self._function_metadata: dict[str, dict] = {}
        self._function_validators: dict[str, tp.Callable] = {}
        self._function_metrics: dict[str, ExecutionMetrics] = {}

    def register(
        self,
        func: tp.Callable,
        agent_id: str,
        metadata: dict | None = None,
        validator: tp.Callable | None = None,
    ):
        """Register a function with validation."""
        func_name = func.__name__

        # Validate function signature
        sig = inspect.signature(func)
        if not sig.parameters:
            logger.warning(f"Function {func_name} has no parameters")

        self._functions[func_name] = func
        self._function_agents[func_name] = agent_id
        self._function_metadata[func_name] = metadata or {}
        self._function_validators[func_name] = validator
        self._function_metrics[func_name] = ExecutionMetrics()

        logger.info(f"Registered function {func_name} for agent {agent_id}")

    def validate_arguments(self, func_name: str, arguments: dict) -> None:
        """Validate function arguments."""
        if func_name not in self._functions:
            raise ValidationError(func_name, "Function not registered")

        func = self._functions[func_name]
        sig = inspect.signature(func)

        # Check required parameters
        for param_name, param in sig.parameters.items():
            if param_name == __CTX_VARS_NAME__:
                continue

            if param.default == inspect.Parameter.empty and param_name not in arguments:
                raise ValidationError(param_name, f"Required parameter missing for {func_name}")

        # Run custom validator if provided
        validator = self._function_validators.get(func_name)
        if validator:
            validator(arguments)

    def get_metrics(self, func_name: str) -> ExecutionMetrics | None:
        """Get execution metrics for a function."""
        return self._function_metrics.get(func_name)


class EnhancedAgentOrchestrator:
    """Enhanced orchestrator with better error handling and monitoring."""

    def __init__(self, max_agents: int = 100, enable_metrics: bool = True):
        self.agents: dict[str, Agent] = {}
        self.function_registry = EnhancedFunctionRegistry()
        self.switch_triggers: dict[AgentSwitchTrigger, tp.Callable] = {}
        self.current_agent_id: str | None = None
        self.execution_history: list[dict] = []
        self.max_agents = max_agents
        self.enable_metrics = enable_metrics
        self._lock = asyncio.Lock()

    async def register_agent(self, agent: Agent):
        """Register an agent with validation."""
        async with self._lock:
            if len(self.agents) >= self.max_agents:
                raise AgentError("system", f"Maximum number of agents ({self.max_agents}) reached")

            agent_id = agent.id or f"agent_{len(self.agents)}"
            agent.id = agent_id

            if agent_id in self.agents:
                raise AgentError(agent_id, "Agent already registered")

            self.agents[agent_id] = agent

            # Register functions with validation
            for func in agent.functions:
                try:
                    self.function_registry.register(func, agent_id)
                except Exception as e:
                    logger.error(f"Failed to register function {func.__name__}: {e}")
                    raise AgentError(agent_id, f"Function registration failed: {e}") from e

            if self.current_agent_id is None:
                self.current_agent_id = agent_id

            logger.info(f"Registered agent {agent_id}")

    async def switch_agent(self, target_agent_id: str, reason: str | None = None):
        """Switch to a different agent with validation."""
        async with self._lock:
            if target_agent_id not in self.agents:
                raise AgentError(target_agent_id, "Agent not found")

            old_agent = self.current_agent_id
            self.current_agent_id = target_agent_id

            self.execution_history.append(
                {
                    "action": "agent_switch",
                    "from": old_agent,
                    "to": target_agent_id,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info(f"Switched from agent {old_agent} to {target_agent_id}")
    
    def register_switch_trigger(self, trigger: AgentSwitchTrigger, handler: tp.Callable):
        """Register a custom switch trigger handler."""
        self.switch_triggers[trigger] = handler
        logger.info(f"Registered switch trigger: {trigger}")
    
    def should_switch_agent(self, context: dict) -> str | None:
        """Determine if agent switching is needed."""
        for trigger, handler in self.switch_triggers.items():
            try:
                target_agent = handler(context, self.agents, self.current_agent_id)
                if target_agent and target_agent != self.current_agent_id:
                    logger.info(f"Switch trigger {trigger} suggests switching to {target_agent}")
                    return target_agent
            except Exception as e:
                logger.error(f"Error in switch trigger {trigger}: {e}")
        return None


class EnhancedFunctionExecutor:
    """Enhanced function executor with timeout, retry, and error handling."""

    def __init__(
        self,
        orchestrator: EnhancedAgentOrchestrator,
        default_timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        max_concurrent_executions: int = 10,
    ):
        self.orchestrator = orchestrator
        self.default_timeout = default_timeout
        self.retry_policy = retry_policy or RetryPolicy()
        self.max_concurrent = max_concurrent_executions
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_executions)

    async def execute_with_timeout(
        self,
        func: tp.Callable,
        arguments: dict,
        timeout: float | None = None,
    ) -> tp.Any:
        """Execute function with timeout."""
        timeout = timeout or self.default_timeout

        try:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(func(**arguments), timeout=timeout)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(self.thread_pool, functools.partial(func, **arguments))
                return await asyncio.wait_for(future, timeout=timeout)

        except asyncio.TimeoutError:
            raise CaluteTimeoutError(func.__name__, timeout) from None
        except Exception as e:
            raise FunctionExecutionError(func.__name__, str(e), original_error=e) from e

    async def execute_with_retry(
        self,
        func: tp.Callable,
        arguments: dict,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> tp.Any:
        """Execute function with retry logic."""
        policy = retry_policy or self.retry_policy
        last_error = None

        for attempt in range(policy.max_retries + 1):
            try:
                return await self.execute_with_timeout(func, arguments, timeout)

            except CaluteTimeoutError:
                # Don't retry timeouts
                raise

            except FunctionExecutionError as e:
                last_error = e
                if attempt < policy.max_retries:
                    delay = policy.get_delay(attempt)
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}), retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Function {func.__name__} failed after {policy.max_retries + 1} attempts")

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
                # Get function and validate
                func = self.orchestrator.function_registry._functions.get(func_name)

                if not func:
                    raise FunctionExecutionError(func_name, "Function not found")

                # Validate arguments
                self.orchestrator.function_registry.validate_arguments(func_name, call.arguments)

                # Add context variables if needed
                if __CTX_VARS_NAME__ in inspect.signature(func).parameters:
                    call.arguments[__CTX_VARS_NAME__] = context_variables or {}

                # Get timeout from agent or use default
                timeout = agent.function_timeout if agent and hasattr(agent, 'function_timeout') else self.default_timeout

                # Execute with retry
                result = await self.execute_with_retry(func, call.arguments, timeout)

                call.result = result
                # Add status if not present
                if not hasattr(call, 'status'):
                    call.status = ExecutionStatus.SUCCESS
                else:
                    call.status = ExecutionStatus.SUCCESS
                if not hasattr(call, 'execution_time'):
                    call.execution_time = time.time() - start_time
                else:
                    call.execution_time = time.time() - start_time

                logger.info(f"Successfully executed {func_name} in {call.execution_time:.2f}s")

            except CaluteTimeoutError as e:
                call.result = f"Function timed out: {e}"
                if hasattr(call, 'status'):
                    call.status = ExecutionStatus.FAILURE  # Use FAILURE for timeout
                if hasattr(call, 'error'):
                    call.error = str(e)
                if hasattr(call, 'execution_time'):
                    call.execution_time = time.time() - start_time
                logger.error(f"Function {func_name} timed out: {e}")

            except (FunctionExecutionError, ValidationError) as e:
                call.result = f"Function execution error: {e}"
                if hasattr(call, 'status'):
                    call.status = ExecutionStatus.FAILURE  
                if hasattr(call, 'error'):
                    call.error = str(e)
                if hasattr(call, 'execution_time'):
                    call.execution_time = time.time() - start_time
                logger.error(f"Function {func_name} failed: {e}")

            except Exception as e:
                call.result = f"Unexpected error: {e}"
                if hasattr(call, 'status'):
                    call.status = ExecutionStatus.FAILURE  
                if hasattr(call, 'error'):
                    call.error = f"Unexpected error: {e!s}"
                if hasattr(call, 'execution_time'):
                    call.execution_time = time.time() - start_time
                logger.error(f"Unexpected error in {func_name}: {e}", exc_info=True)

            finally:
                # Record metrics
                if self.orchestrator.enable_metrics:
                    metrics = self.orchestrator.function_registry.get_metrics(func_name)
                    if metrics:
                        exec_time = getattr(call, 'execution_time', 0)
                        status = getattr(call, 'status', ExecutionStatus.SUCCESS)
                        metrics.record_execution(exec_time, status)

            return call

    async def execute_function_calls(
        self,
        calls: list[RequestFunctionCall],
        strategy: FunctionCallStrategy = FunctionCallStrategy.SEQUENTIAL,
        context_variables: dict | None = None,
        agent: Agent | None = None,
    ) -> list[RequestFunctionCall]:
        """Execute multiple function calls with specified strategy."""
        context_variables = context_variables or {}

        if strategy == FunctionCallStrategy.SEQUENTIAL:
            results = []
            for call in calls:
                result = await self.execute_single_call(call, context_variables, agent)
                results.append(result)
                # Update context with result for next call
                if result.status == ExecutionStatus.SUCCESS:
                    context_variables[f"{call.name}_result"] = result.result

        elif strategy == FunctionCallStrategy.PARALLEL:
            tasks = [self.execute_single_call(call, context_variables.copy(), agent) for call in calls]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        return results

    @asynccontextmanager
    async def batch_execution(self):
        """Context manager for batch execution with cleanup."""
        try:
            yield self
        finally:
            # Cleanup resources
            await asyncio.sleep(0)  # Allow pending tasks to complete

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=False)
