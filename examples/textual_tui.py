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
"""Launch the Xerxes Textual terminal UI."""

from xerxes import Agent, OpenAILLM, RuntimeFeaturesConfig, Xerxes


def main() -> None:
    llm = OpenAILLM(model="gpt-4o-mini")
    xerxes = Xerxes(
        llm=llm,
        runtime_features=RuntimeFeaturesConfig(enabled=True),
    )
    agent = Agent(
        id="assistant",
        model="gpt-4o-mini",
        instructions="You are a concise coding assistant.",
        functions=[],
    )
    xerxes.register_agent(agent)
    xerxes.create_tui(agent).launch()


if __name__ == "__main__":
    main()
