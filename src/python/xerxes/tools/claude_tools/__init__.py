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
"""Claude-style coding and agent tools — split into focused modules.

This package re-exports the full public surface so that existing imports
(``from xerxes.tools.claude_tools import FileEditTool``) continue to work
unchanged after the split from the original monolithic ``claude_tools.py``.

Module layout:
    - :mod:`._common` — shared file-editing helpers.
    - :mod:`.file_ops` — FileEditTool, NotebookEditTool.
    - :mod:`.search` — GlobTool, GrepTool, LSPTool.
    - :mod:`.agent_ops` — AgentTool, SpawnAgents, Task*, AwaitAgents, etc.
    - :mod:`.workflow` — TodoWriteTool, AskUserQuestionTool, plan/worktree tools.
    - :mod:`.mcp_ops` — MCPTool, ListMcpResourcesTool, ReadMcpResourceTool.
    - :mod:`.remote` — RemoteTriggerTool, ScheduleCronTool.
"""

from .agent_ops import (
    AgentTool,
    AwaitAgents,
    CheckAgentMessages,
    HandoffTool,
    PeekAgent,
    ResetAgent,
    SendMessageTool,
    SpawnAgents,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskStopTool,
    TaskUpdateTool,
    _get_agent_manager,
    _parse_agents_payload,
)
from .file_ops import FileEditTool, NotebookEditTool
from .mcp_ops import ListMcpResourcesTool, MCPTool, ReadMcpResourceTool
from .remote import RemoteTriggerTool, ScheduleCronTool
from .search import GlobTool, GrepTool, LSPTool
from .workflow import (
    AskUserQuestionTool,
    EnterPlanModeTool,
    EnterWorktreeTool,
    ExitPlanModeTool,
    ExitWorktreeTool,
    PlanTool,
    SetInteractionModeTool,
    SkillTool,
    TodoWriteTool,
    ToolSearchTool,
    set_ask_user_question_callback,
)

__all__ = [
    "AgentTool",
    "AskUserQuestionTool",
    "AwaitAgents",
    "CheckAgentMessages",
    "EnterPlanModeTool",
    "EnterWorktreeTool",
    "ExitPlanModeTool",
    "ExitWorktreeTool",
    "FileEditTool",
    "GlobTool",
    "GrepTool",
    "HandoffTool",
    "LSPTool",
    "ListMcpResourcesTool",
    "MCPTool",
    "NotebookEditTool",
    "PeekAgent",
    "PlanTool",
    "ReadMcpResourceTool",
    "RemoteTriggerTool",
    "ResetAgent",
    "ScheduleCronTool",
    "SendMessageTool",
    "SetInteractionModeTool",
    "SkillTool",
    "SpawnAgents",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskOutputTool",
    "TaskStopTool",
    "TaskUpdateTool",
    "TodoWriteTool",
    "ToolSearchTool",
]
