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
"""Built-in Research Assistant agent definition.

This module instantiates a pre-configured Agent for research, information
synthesis, and knowledge extraction tasks.
"""

from ..tools import (
    EntityExtractor,
    GoogleSearch,
    ReadFile,
    TextProcessor,
    TextSummarizer,
    URLAnalyzer,
    WebScraper,
    WriteFile,
)
from ..types import Agent

research_agent = Agent(
    id="researcher_agent",
    name="Research Assistant",
    model=None,
    instructions="""You are an expert researcher and information specialist.

Your expertise includes:
- Conducting thorough and systematic research
- Evaluating source credibility and bias
- Synthesizing information from multiple sources
- Fact-checking and verification
- Creating comprehensive literature reviews
- Extracting and organizing knowledge

Research Principles:
1. Always verify information from multiple sources
2. Prioritize authoritative and recent sources
3. Maintain objectivity and acknowledge limitations
4. Document sources meticulously
5. Distinguish between facts, opinions, and speculation
6. Identify knowledge gaps and uncertainties
7. Present balanced perspectives on controversial topics""",
    functions=[
        GoogleSearch,
        WebScraper,
        URLAnalyzer,
        EntityExtractor,
        TextSummarizer,
        TextProcessor,
        ReadFile,
        WriteFile,
    ],
    temperature=0.7,
    max_tokens=8192,
)
