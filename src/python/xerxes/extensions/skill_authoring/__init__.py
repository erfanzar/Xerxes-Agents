# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Hermes-style autonomous skill authoring.

The skill authoring subsystem watches tool-call sequences during a turn,
detects when the agent has completed something worth canonicalising,
and (optionally) drafts a new SKILL.md capturing the procedure,
pitfalls observed, and a verification recipe.

Modules:
    - :mod:`tracker`: per-turn :class:`ToolSequenceTracker`.
    - :mod:`triggers`: heuristics for "is this skill-worthy?".
    - :mod:`drafter`: LLM-driven SKILL.md drafter.
    - :mod:`verifier`: verification step generator.
    - :mod:`improver`: feedback-driven skill rework.
    - :mod:`matcher`: semantic skill matcher for re-use.
"""

from .drafter import SkillDrafter, render_skill_template
from .improver import ImprovementResult, SkillImprover
from .lifecycle import (
    DeprecationDecision,
    SkillLifecycleManager,
    SkillVariant,
    SkillVariantPicker,
)
from .matcher import SkillMatch, SkillMatcher
from .pipeline import AuthoringResult, SkillAuthoringPipeline
from .telemetry import SkillStats, SkillTelemetry
from .tracker import (
    SkillCandidate,
    ToolCallEvent,
    ToolSequenceTracker,
)
from .triggers import (
    SkillAuthoringConfig,
    SkillAuthoringTrigger,
)
from .verifier import SkillVerifier, VerificationResult, VerificationStep

__all__ = [
    "AuthoringResult",
    "DeprecationDecision",
    "ImprovementResult",
    "SkillAuthoringConfig",
    "SkillAuthoringPipeline",
    "SkillAuthoringTrigger",
    "SkillCandidate",
    "SkillDrafter",
    "SkillImprover",
    "SkillLifecycleManager",
    "SkillMatch",
    "SkillMatcher",
    "SkillStats",
    "SkillTelemetry",
    "SkillVariant",
    "SkillVariantPicker",
    "SkillVerifier",
    "ToolCallEvent",
    "ToolSequenceTracker",
    "VerificationResult",
    "VerificationStep",
    "render_skill_template",
]
