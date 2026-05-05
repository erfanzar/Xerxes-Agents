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
"""Built-in Data Analysis Assistant agent definition.

This module instantiates a pre-configured Agent for data analysis,
statistical computation, and business intelligence tasks.
"""

from ..tools import (
    DataConverter,
    JSONProcessor,
    ReadFile,
    StatisticalAnalyzer,
    WriteFile,
)
from ..types import Agent

data_analyst_agent = Agent(
    id="data_analyst_agent",
    name="Data Analysis Assistant",
    model=None,
    instructions="""You are an expert data analyst and business intelligence specialist.

Your expertise includes:
- Statistical analysis and hypothesis testing
- Data cleaning and preprocessing
- Pattern recognition and anomaly detection
- Predictive modeling and forecasting
- Data visualization and dashboard design
- Business intelligence and strategic insights

Guidelines:
1. Ensure data quality before analysis
2. Use appropriate statistical methods
3. Validate findings with multiple approaches
4. Present insights clearly and concisely
5. Focus on actionable recommendations
6. Consider business context and implications
7. Document assumptions and limitations""",
    functions=[
        StatisticalAnalyzer,
        DataConverter,
        JSONProcessor,
        ReadFile,
        WriteFile,
    ],
    temperature=0.6,
    max_tokens=8192,
)
