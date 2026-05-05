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
"""Init module for Xerxes."""

from ..operators.config import OperatorRuntimeConfig
from .bootstrap import BootstrapResult, BootstrapStage, bootstrap
from .bridge import bootstrap_xerxes, build_tool_executor, create_query_engine, populate_registry
from .cost_tracker import CostEvent, CostTracker
from .execution_registry import (
    EntryKind,
    ExecutionRegistry,
    RegistryEntry,
    RouteMatch,
)
from .execution_registry import (
    ExecutionResult as RegistryExecutionResult,
)
from .features import AgentRuntimeOverrides, RuntimeFeaturesConfig, RuntimeFeaturesState
from .history import HistoryEvent, HistoryLog
from .loop_detection import LoopDetectionConfig, LoopDetector, LoopEvent, LoopSeverity, ToolLoopError
from .parity_audit import ModuleStatus, ParityAuditResult, run_parity_audit
from .profiles import PromptProfile, PromptProfileConfig, get_profile_config
from .query_engine import QueryEngine, QueryEngineConfig, TurnResult
from .session import RuntimeContext, RuntimeSession
from .tool_pool import ToolPool, assemble_tool_pool
from .transcript import TranscriptEntry, TranscriptStore

__all__ = [
    "AgentRuntimeOverrides",
    "BootstrapResult",
    "BootstrapStage",
    "CostEvent",
    "CostTracker",
    "EntryKind",
    "ExecutionRegistry",
    "HistoryEvent",
    "HistoryLog",
    "LoopDetectionConfig",
    "LoopDetector",
    "LoopEvent",
    "LoopSeverity",
    "ModuleStatus",
    "OperatorRuntimeConfig",
    "ParityAuditResult",
    "PromptProfile",
    "PromptProfileConfig",
    "QueryEngine",
    "QueryEngineConfig",
    "RegistryEntry",
    "RegistryExecutionResult",
    "RouteMatch",
    "RuntimeContext",
    "RuntimeFeaturesConfig",
    "RuntimeFeaturesState",
    "RuntimeSession",
    "ToolLoopError",
    "ToolPool",
    "TranscriptEntry",
    "TranscriptStore",
    "TurnResult",
    "assemble_tool_pool",
    "bootstrap",
    "bootstrap_xerxes",
    "build_tool_executor",
    "create_query_engine",
    "get_profile_config",
    "populate_registry",
    "run_parity_audit",
]
