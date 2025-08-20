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
import inspect
import json
import statistics
import time
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Generic, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


class ToolExecutionContext:
    """Context for tool execution with metadata"""

    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.execution_id: str = ""
        self.parent_task_id: str = ""
        self.agent_id: str = ""
        self.retry_count: int = 0
        self.error_history: list[Exception] = []

    @property
    def execution_time(self) -> float:
        return self.end_time - self.start_time if self.end_time else time.time() - self.start_time


class ToolResult(Generic[T]):
    """Typed result wrapper for tool execution"""

    def __init__(
        self,
        value: T = None,
        success: bool = True,
        error: str | None = None,
        metadata: dict | None = None,
        execution_context: ToolExecutionContext = None,
    ):
        self.value = value
        self.success = success
        self.error = error
        self.metadata = metadata or {}
        self.execution_context = execution_context or ToolExecutionContext()
        self.timestamp = datetime.now()


def enhanced_function_to_json(func: Callable) -> dict:
    """
    Enhanced version that extracts more detailed type information
    and generates OpenAI-compatible function schemas.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
        tuple: "array",
        set: "array",
        bytes: "string",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {func.__name__}: {e!s}") from e

    parameters = {}
    for param_name, param in signature.parameters.items():
        # Skip self parameter for methods
        if param_name == "self":
            continue

        param_info = {"type": "string"}  # Default type

        # Extract description from docstring if available
        param_description = _extract_param_description(func.__doc__, param_name)
        if param_description:
            param_info["description"] = param_description

        # Handle type annotations
        if param.annotation != inspect.Parameter.empty:
            param_info.update(_process_type_annotation(param.annotation, type_map))

        # Handle default values
        if param.default != inspect.Parameter.empty:
            if param.default is not None:
                param_info["default"] = param.default
                # Add enum if default is from a list of choices
                if hasattr(func, "__annotations__") and param_name in func.__annotations__:
                    if hasattr(func.__annotations__[param_name], "__args__"):
                        # Handle Literal types
                        args = get_args(func.__annotations__[param_name])
                        if args and all(isinstance(arg, str | int | float) for arg in args):
                            param_info["enum"] = list(args)

        parameters[param_name] = param_info

    # Extract required parameters (those without defaults)
    required = [
        param_name
        for param_name, param in signature.parameters.items()
        if param.default == inspect.Parameter.empty and param_name != "self"
    ]

    # Extract return type if available
    return_info = None
    if signature.return_annotation != inspect.Parameter.empty:
        return_info = _process_type_annotation(signature.return_annotation, type_map)

    result = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": _clean_docstring(func.__doc__) or f"Function {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

    if return_info:
        result["function"]["returns"] = return_info

    return result


def _process_type_annotation(annotation: Any, type_map: dict) -> dict[str, Any]:
    """Process complex type annotations"""
    result = {}

    # Get origin and args for generic types
    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation in type_map:
        result["type"] = type_map[annotation]
    elif origin is Union:
        # Handle Union types (including Optional)
        types = []
        for arg in args:
            if arg in type_map:
                types.append(type_map[arg])
            else:
                types.append(str(arg))

        if type(None) in args and len(args) == 2:
            # This is Optional[T]
            other_type = next(arg for arg in args if arg is not type(None))
            if other_type in type_map:
                result["type"] = type_map[other_type]
            else:
                result["type"] = str(other_type)
            result["nullable"] = True
        else:
            result["type"] = "union"
            result["types"] = types
    elif origin in (list, list):
        result["type"] = "array"
        if args:
            item_type = _process_type_annotation(args[0], type_map)
            result["items"] = item_type
    elif origin in (dict, dict):
        result["type"] = "object"
        if args and len(args) >= 2:
            result["additionalProperties"] = _process_type_annotation(args[1], type_map)
    elif hasattr(annotation, "__name__"):
        result["type"] = annotation.__name__
    else:
        result["type"] = str(annotation)

    return result


def _extract_param_description(docstring: str | None, param_name: str) -> str | None:
    """Extract parameter description from docstring"""
    if not docstring:
        return None

    lines = docstring.split("\n")
    for _, line in enumerate(lines):
        if param_name + ":" in line or param_name + " :" in line:
            desc_parts = line.split(":", 1)
            if len(desc_parts) > 1:
                return desc_parts[1].strip()

    return None


def _clean_docstring(docstring: str | None) -> str | None:
    """Clean and format docstring"""
    if not docstring:
        return None

    lines = docstring.strip().split("\n")
    # Take the first paragraph as description
    description_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            break
        description_lines.append(line)

    return " ".join(description_lines)


@dataclass
class ToolParameter:
    """Detailed parameter information"""

    name: str
    type: str
    description: str | None = None
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None


@dataclass
class ToolSignature:
    """Complete tool signature"""

    name: str
    description: str
    parameters: list[ToolParameter]
    returns: dict[str, Any] | None = None
    examples: list[dict[str, Any]] | None = None


class CortexTool(ABC):
    """tool with full signature support"""

    source_func: typing.ClassVar = None

    def __init__(
        self,
        name: str,
        description: str,
        max_retries: int = 3,
        timeout: float | None = None,
        cache_results: bool = False,
        requires_confirmation: bool = False,
    ):
        self.name = name
        self.description = description
        self._signature = None
        self._function_schema = None
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_results = cache_results
        self.requires_confirmation = requires_confirmation
        self._cache = {}
        self._execution_history = []

    @property
    def _direct_call(self):
        return self._run if self.source_func is None else self.source_func

    @abstractmethod
    def _run(self, **kwargs) -> Any:
        """Execute the tool"""

    def run(self, **kwargs) -> Any:
        """Public interface with validation"""
        # Validate parameters against signature
        if self._signature:
            self._validate_parameters(kwargs)

        return self._direct_call(**kwargs)

    def _validate_parameters(self, kwargs: dict[str, Any]):  # type:ignore
        """Validate parameters against signature"""
        for param in self._signature.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Required parameter '{param.name}' missing")

            if param.enum and param.name in kwargs:
                if kwargs[param.name] not in param.enum:
                    raise ValueError(f"Parameter '{param.name}' must be one of {param.enum}, got {kwargs[param.name]}")

    def get_signature(self) -> ToolSignature:
        """Get tool signature"""
        if not self._signature:
            self._signature = self._extract_signature()
        return self._signature

    def _extract_signature(self) -> ToolSignature:
        """Extract signature from _run method"""

        fn = self._direct_call

        func_schema = enhanced_function_to_json(fn)

        parameters = []
        for param_name, param_info in func_schema["function"]["parameters"]["properties"].items():
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_info.get("type", "string"),
                    description=param_info.get("description"),
                    required=param_name in func_schema["function"]["parameters"]["required"],
                    default=param_info.get("default"),
                    enum=param_info.get("enum"),
                )
            )

        return ToolSignature(
            name=self.name,
            description=self.description,
            parameters=parameters,
            returns=func_schema["function"].get("returns"),
        )

    def as_function(self) -> Callable | None:
        """Convert to callable with proper signature"""
        source_func = self._direct_call

        original_sig = inspect.signature(source_func)

        def make_wrapper():
            params = list(original_sig.parameters.values())
            param_strs = []
            call_strs = []

            for param in params:
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    param_strs.append(f"*{param.name}")
                    call_strs.append(f"*{param.name}")
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    param_strs.append(f"**{param.name}")
                    call_strs.append(f"**{param.name}")
                elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                    if param.default is inspect.Parameter.empty:
                        param_strs.append(f"*, {param.name}")
                    else:
                        param_strs.append(f"*, {param.name}={param.name}_default")
                    call_strs.append(f"{param.name}={param.name}")
                else:
                    if param.default is inspect.Parameter.empty:
                        param_strs.append(param.name)
                    else:
                        param_strs.append(f"{param.name}={param.name}_default")
                    call_strs.append(param.name)

            func_str = f"def tool_function({', '.join(param_strs)}):\n"
            func_str += f"    return self.run({', '.join(call_strs)})\n"

            namespace = {"self": self}
            for param in params:
                if param.default is not inspect.Parameter.empty:
                    namespace[f"{param.name}_default"] = param.default
            exec(func_str, namespace)
            return namespace["tool_function"]

        tool_function = make_wrapper()

        tool_function.__name__ = self.name
        tool_function.__doc__ = self.description
        tool_function.__signature__ = original_sig  # type:ignore

        tool_function.__annotations__ = getattr(source_func, "__annotations__", {}).copy()

        for attr in ["__module__", "__qualname__", "__defaults__", "__kwdefaults__"]:
            if hasattr(source_func, attr):
                try:
                    setattr(tool_function, attr, getattr(source_func, attr))
                except (AttributeError, TypeError):
                    pass

        for attr in dir(source_func):
            if not attr.startswith("_") and not hasattr(tool_function, attr):
                try:
                    setattr(tool_function, attr, getattr(source_func, attr))
                except (AttributeError, TypeError):
                    pass

        return tool_function  # type: ignore[return-value]

    def to_openai_schema(self) -> dict[str, Any]:  # type:ignore
        """Convert to OpenAI function calling schema"""
        if not self._function_schema:
            self._function_schema = enhanced_function_to_json(self._direct_call)
            self._function_schema["function"]["name"] = self.name
            self._function_schema["function"]["description"] = self.description

        return self._function_schema

    @staticmethod
    def from_function(
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        examples: list[dict[str, Any]] | None = None,  # type:ignore
    ) -> CortexTool:
        """Create a tool from a function with full signature preservation"""

        class FunctionTool(CortexTool):
            def __init__(self):
                tool_name = name or func.__name__
                tool_desc = description or _clean_docstring(func.__doc__) or f"Tool {tool_name}"

                super().__init__(tool_name, tool_desc)
                self.source_func = func
                self.examples = examples or []

            def _run(self, **kwargs):
                return self.source_func(**kwargs)

        return FunctionTool()

    @staticmethod
    def create_tool_set(
        functions: list[Callable],
        prefix: str = "",
        shared_description: str | None = None,
    ) -> list[CortexTool]:
        """Create a set of related tools from functions"""
        tools = []

        for func in functions:
            tool_name = f"{prefix}_{func.__name__}" if prefix else func.__name__
            tool = CortexTool.from_function(func, name=tool_name, description=shared_description)
            tools.append(tool)

        return tools

    @staticmethod
    def create_parameterized_tool(
        base_func: Callable,
        parameter_sets: list[dict[str, Any]],  # type:ignore
        name_template: str = "{base_name}_{variant}",
    ) -> list[CortexTool]:
        """Create multiple tool variants with different default parameters"""
        tools = []

        for i, params in enumerate(parameter_sets):
            variant_name = params.get("_name", f"variant_{i}")

            @wraps(base_func)
            def make_variant_func(preset_params):
                def variant_func(**kwargs):
                    merged_params = preset_params.copy()
                    merged_params.update(kwargs)
                    return base_func(**merged_params)

                return variant_func

            variant_func = make_variant_func(params)

            # Create tool
            tool_name = name_template.format(base_name=base_func.__name__, variant=variant_name)

            tool = CortexTool.from_function(
                variant_func,
                name=tool_name,
                description=f"{base_func.__doc__} (Variant: {variant_name})",
            )
            tools.append(tool)

        return tools

    async def execute_with_context(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute tool with full context and error handling"""
        context.start_time = time.time()
        context.execution_id = f"{self.name}_{int(time.time() * 1000)}"
        cache_key = None
        # Check cache if enabled
        if self.cache_results:
            cache_key = self._generate_cache_key(kwargs)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                context.end_time = time.time()
                return ToolResult(
                    value=cached_result,
                    success=True,
                    metadata={"cached": True},
                    execution_context=context,
                )

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                context.retry_count = attempt

                # Add timeout if specified
                if self.timeout:
                    result = await asyncio.wait_for(self._execute_async(**kwargs), timeout=self.timeout)
                else:
                    result = await self._execute_async(**kwargs)

                context.end_time = time.time()

                # Cache successful result
                if self.cache_results and cache_key is not None:
                    self._cache[cache_key] = result

                # Record execution
                self._execution_history.append(
                    {
                        "execution_id": context.execution_id,
                        "timestamp": datetime.now(),
                        "success": True,
                        "execution_time": context.execution_time,
                        "args": kwargs,
                    }
                )

                return ToolResult(value=result, success=True, execution_context=context)

            except Exception as e:
                last_error = e
                context.error_history.append(e)

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        # All retries failed
        context.end_time = time.time()
        self._execution_history.append(
            {
                "execution_id": context.execution_id,
                "timestamp": datetime.now(),
                "success": False,
                "error": str(last_error),
                "execution_time": context.execution_time,
                "args": kwargs,
            }
        )

        return ToolResult(success=False, error=str(last_error), execution_context=context)

    async def _execute_async(self, **kwargs):
        """Async wrapper for _run method"""
        if asyncio.iscoroutinefunction(self._direct_call):
            return await self._direct_call(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self._direct_call(**kwargs))

    def _generate_cache_key(self, kwargs: dict) -> str:
        """Generate cache key from arguments"""
        import hashlib

        # Sort keys for consistent hashing
        sorted_kwargs = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(sorted_kwargs.encode()).hexdigest()

    def get_execution_stats(self) -> dict:
        """Get execution statistics for this tool"""
        if not self._execution_history:
            return {"total_executions": 0}

        successful = [e for e in self._execution_history if e["success"]]
        failed = [e for e in self._execution_history if not e["success"]]

        exec_times = [e["execution_time"] for e in successful]

        return {
            "total_executions": len(self._execution_history),
            "successful_executions": len(successful),
            "failed_executions": len(failed),
            "success_rate": len(successful) / len(self._execution_history),
            "average_execution_time": statistics.mean(exec_times) if exec_times else 0,
            "min_execution_time": min(exec_times) if exec_times else 0,
            "max_execution_time": max(exec_times) if exec_times else 0,
            "cache_size": len(self._cache),
            "last_execution": self._execution_history[-1]["timestamp"] if self._execution_history else None,
        }

    def __str__(self):
        return f"CortexTool(signature={self.get_signature()})"

    __repr__ = __str__


class ComposedTool(CortexTool):
    """Tool that composes multiple tools together"""

    def __init__(
        self,
        name: str,
        description: str,
        tools: list[CortexTool],
        max_retries: int = 3,
        timeout: float | None = None,
        cache_results: bool = False,
        requires_confirmation: bool = False,
    ):
        super().__init__(
            name=name,
            description=description,
            max_retries=max_retries,
            timeout=timeout,
            cache_results=cache_results,
            requires_confirmation=requires_confirmation,
        )
        self.tools = tools
        self.composition_type = "sequential"

    def set_composition(self, composition_type: str, conditions: dict | None = None):
        """Set how tools are composed"""
        self.composition_type = composition_type
        self.conditions = conditions or {}

    def run(self, **kwargs):
        """Public interface with validation"""
        if self._signature:
            self._validate_parameters(kwargs)
        return asyncio.run(self._direct_call(**kwargs))

    async def _run(self, **kwargs):
        """Execute composed tools"""
        if self.composition_type == "sequential":
            result = kwargs
            for tool in self.tools:
                tool_result = await tool.execute_with_context(
                    ToolExecutionContext(), **result if isinstance(result, dict) else {"input": result}
                )
                if not tool_result.success:
                    raise Exception(f"Tool {tool.name} failed: {tool_result.error}")
                result = tool_result.value
            return result

        elif self.composition_type == "parallel":
            tasks = []
            for tool in self.tools:
                ctx = ToolExecutionContext()
                tasks.append(tool.execute_with_context(ctx, **kwargs))
            results = await asyncio.gather(*tasks)
            return [r.value for r in results if r.success]

        elif self.composition_type == "conditional":
            for tool in self.tools:
                condition = self.conditions.get(tool.name, lambda x: True)
                if condition(kwargs):
                    result = await tool.execute_with_context(ToolExecutionContext(), **kwargs)
                    if result.success:
                        return result.value
            raise Exception("No tool condition was met")
