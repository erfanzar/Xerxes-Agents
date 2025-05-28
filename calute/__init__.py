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

from .calute import Calute, PromptTemplate
from .chain_module import ChainExecutor, FunctionChain
from .executors import AgentOrchestrator
from .types import (
    Agent,
    AgentCapability,
    AgentFunction,
    AgentSwitch,
    AgentSwitchTrigger,
    Completion,
    ExecutionResult,
    ExecutionStatus,
    FunctionCall,
    FunctionCallInfo,
    FunctionCallsExtracted,
    FunctionCallStrategy,
    FunctionDetection,
    FunctionExecutionComplete,
    FunctionExecutionStart,
    Response,
    ResponseResult,
    StreamChunk,
    StreamingResponseType,
    SwitchContext,
)
from .workflow import (
    Workflow,
    WorkflowEngine,
    WorkflowStep,
    WorkflowStepType,
)

__all__ = (
    "Agent",
    "AgentCapability",
    "AgentFunction",
    "AgentOrchestrator",
    "AgentSwitch",
    "AgentSwitchTrigger",
    "Calute",
    "ChainExecutor",
    "Completion",
    "ExecutionResult",
    "ExecutionStatus",
    "FunctionCall",
    "FunctionCallInfo",
    "FunctionCallStrategy",
    "FunctionCallsExtracted",
    "FunctionChain",
    "FunctionDetection",
    "FunctionExecutionComplete",
    "FunctionExecutionStart",
    "PromptTemplate",
    "Response",
    "ResponseResult",
    "StreamChunk",
    "StreamingResponseType",
    "SwitchContext",
    "Workflow",
    "WorkflowEngine",
    "WorkflowStep",
    "WorkflowStepType",
)

__version__ = "0.0.5"
