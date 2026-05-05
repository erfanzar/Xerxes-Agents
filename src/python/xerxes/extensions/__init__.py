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
"""Public exports for the Xerxes extensions subsystem.

Aggregates dependency resolution, hook runners, plugin registries, and skill
management so consumers can import from a single namespace.
"""

from .dependency import CircularDependencyError, DependencyResolver, DependencySpec, VersionConstraint
from .hooks import HOOK_POINTS, HookRunner
from .plugins import PluginConflictError, PluginMeta, PluginRegistry, PluginType, RegisteredPlugin
from .skills import Skill, SkillMetadata, SkillRegistry, parse_skill_md

__all__ = [
    "HOOK_POINTS",
    "CircularDependencyError",
    "DependencyResolver",
    "DependencySpec",
    "HookRunner",
    "PluginConflictError",
    "PluginMeta",
    "PluginRegistry",
    "PluginType",
    "RegisteredPlugin",
    "Skill",
    "SkillMetadata",
    "SkillRegistry",
    "VersionConstraint",
    "parse_skill_md",
]
