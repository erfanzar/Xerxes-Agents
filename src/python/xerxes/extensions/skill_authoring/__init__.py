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
"""Public exports for the skill authoring subsystem.

Provides components for auto-drafting, improving, verifying, and managing the
lifecycle of agent skills synthesised from observed tool sequences.
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
