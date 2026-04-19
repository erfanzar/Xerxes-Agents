# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""Extensions subsystem for Xerxes.

Provides plugin registry, skill discovery, lifecycle hooks,
and dependency resolution.
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
