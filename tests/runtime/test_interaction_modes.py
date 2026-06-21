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
"""Interaction-mode normalization and guidance tests."""

from xerxes.runtime.interaction_modes import agent_name_for_mode, mode_switch_hint, normalize_interaction_mode


def test_objective_mode_aliases_normalize_to_objective() -> None:
    assert normalize_interaction_mode("goal") == "objective"
    assert normalize_interaction_mode("goal-runner") == "objective"
    assert normalize_interaction_mode("objective") == "objective"


def test_plan_mode_flag_still_wins_for_compatibility() -> None:
    assert normalize_interaction_mode("objective", plan_mode=True) == "plan"


def test_objective_mode_maps_to_objective_agent() -> None:
    assert agent_name_for_mode("goal") == "objective"


def test_objective_hint_forbids_unverified_completion() -> None:
    hint = mode_switch_hint("objective")

    assert "acceptance criteria" in hint
    assert "Do not final-answer" in hint
    assert 'SetInteractionModeTool(mode="code")' in hint
