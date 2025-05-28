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

import typing as tp
from dataclasses import dataclass, field

from .types.function_execution_types import ExecutionStatus, FunctionCall


@dataclass
class FunctionChain:
    """A chain of functions to be executed in sequence with result forwarding"""

    functions: list[str | tp.Callable]
    name: str = "function_chain"
    input_key: str | None = None
    output_key: str | None = None
    context_mapping: dict[int, dict[str, str]] = field(default_factory=dict)


class ChainExecutor:
    """Executes chains of functions with result forwarding"""

    def __init__(self, calute):
        self.calute = calute

    async def execute_chain(
        self,
        chain: FunctionChain,
        initial_input: tp.Any = None,
        context: dict[str, tp.Any] | None = None,
    ) -> tp.Any:
        """Execute a chain of functions"""
        context = context or {}
        current_input = initial_input

        if chain.input_key and current_input is None:
            current_input = context.get(chain.input_key)

        results = []

        for i, func in enumerate(chain.functions):
            func_name = func if isinstance(func, str) else func.__name__

            mapping = chain.context_mapping.get(i, {})
            input_arg_name = mapping.get("input", "data")
            output_key = mapping.get("output")

            args = {input_arg_name: current_input} if current_input is not None else {}
            args["context_variables"] = context

            function_call = FunctionCall(
                name=func_name,
                arguments=args,
                timeout=30.0,
                max_retries=3,
            )
            executor = self.calute.executor
            result = await executor._execute_single_call(function_call, context)
            results.append(result)
            if result.status != ExecutionStatus.SUCCESS:
                raise RuntimeError(f"Function {func_name} failed: {result.error}")
            current_input = result.result
            if output_key:
                context[output_key] = current_input

        if chain.output_key:
            context[chain.output_key] = current_input

        return {
            "final_result": current_input,
            "intermediate_results": [r.result for r in results if r.status == ExecutionStatus.SUCCESS],
            "context": context,
        }
