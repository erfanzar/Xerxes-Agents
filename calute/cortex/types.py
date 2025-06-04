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

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import md5
from typing import Any

from pydantic import BaseModel, Field


class ProcessType(str, Enum):
    """CortexTask execution process types"""

    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    CONSENSUAL = "consensual"


@dataclass
class CortexAgent:
    """CortexAI-style Agent with role-based design"""

    role: str
    goal: str
    backstory: str
    memory: bool = True
    verbose: bool = True
    allow_delegation: bool = False
    tools: list[Callable] = field(default_factory=list)
    max_iter: int = 5
    max_rpm: int | None = None
    llm: str | None = None
    function_calling_llm: str | None = None
    system_template: str | None = None
    prompt_template: str | None = None
    response_template: str | None = None
    top_p: float = 0.95
    max_tokens: int = 2048
    temperature: float = 0.7
    stop: str | list[str] | None = None

    def __post_init__(self):
        self.id = f"{self.role.lower().replace(' ', '_')}_{id(self)}"

    def execute(self, task: CortexTask, context: dict[str, Any] | None = None) -> CortexTaskOutput:
        pass


@dataclass
class CortexTask:
    """CortexAI-style Task"""

    description: str
    expected_output: str
    agent: CortexAgent | None = None
    tools: list[Callable] = field(default_factory=list)
    async_execution: bool = False
    context: list[CortexTask] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    output_json: type[BaseModel] | None = None
    output_file: str | None = None
    callback: Callable | None = None
    human_input: bool = False

    def __post_init__(self):
        self.id = f"task_{id(self)}"
        self.status = "pending"
        self.output = None
        self.start_time = None
        self.end_time = None

    def __hash__(self):
        hash_object = md5(self.id.encode("utf-8"))
        return int.from_bytes(hash_object.digest(), byteorder="big")


@dataclass
class CortexTaskOutput:
    """Output from a task execution"""

    description: str
    summary: str | None = None
    raw: str = ""
    json_dict: dict[str, Any] | None = None
    agent: str | None = None
    task_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: str | None = None


class CortexOutput(BaseModel):
    """Final output from cortex execution"""

    raw: str
    tasks_output: list[CortexTaskOutput]
    json_dict: dict[str, Any] | None = None
    token_usage: dict[str, Any] = Field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Get summary of all task outputs"""
        summaries = [task.summary or task.description for task in self.tasks_output]
        return "\n\n".join(summaries)
