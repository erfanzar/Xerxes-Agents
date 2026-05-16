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
"""agentskills.io open standard source — compatible with the wider ecosystem."""

from __future__ import annotations

import httpx

from .official import OfficialSkillSource


class AgentskillsIOSource(OfficialSkillSource):
    """Skill source backed by the public ``agentskills.io`` registry."""

    name = "agentskills.io"

    def __init__(self, *, client: httpx.Client | None = None) -> None:
        """Bind the source to ``https://agentskills.io/api/v1``.

        Args:
            client: Optional pre-configured ``httpx.Client`` for testing.
        """
        super().__init__(base_url="https://agentskills.io/api/v1", client=client)


__all__ = ["AgentskillsIOSource"]
