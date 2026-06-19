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
"""Context compaction agent for summarizing conversation history.

Provides :class:`CompactionAgent` and a factory function to create one. The
agent uses an LLM to summarize long contexts or message histories.
"""

from typing import Any


class CompactionAgent:
    """Summarizes conversation context to reduce token usage.

    Supports multiple summary lengths (brief, concise, detailed) and can
    preserve specific topics during summarization.
    """

    def __init__(self, llm_client: Any, target_length: str = "concise"):
        """Build the agent.

        Args:
            llm_client: LLM client exposing ``generate_completion``. Falls
                through as an error when unavailable.
            target_length: One of ``"brief"``, ``"concise"``, ``"detailed"``;
                selects the instruction prompt sent to the LLM.
        """
        self.llm_client = llm_client
        self.target_length = target_length

        self.length_instructions = {
            "brief": (
                "Create an extremely brief summary in 2-3 sentences focusing only on the most critical information."
            ),
            "concise": (
                "Create a concise summary that captures the key points and important details in a few paragraphs."
            ),
            "detailed": (
                "Create a detailed summary that preserves important context, key decisions, and relevant details."
            ),
        }

    def summarize_context(self, context: str, preserve_topics: list[str] | None = None) -> str:
        """Summarise ``context`` via the configured LLM.

        Contexts under 200 characters are returned unchanged. The client must
        expose ``generate_completion``; callers that do not have a compaction
        LLM should skip compaction instead of using a local excerpt.
        """
        if not context or len(context) < 200:
            return context
        if not hasattr(self.llm_client, "generate_completion"):
            raise RuntimeError("CompactionAgent requires an llm_client with generate_completion.")

        length_instruction = self.length_instructions.get(self.target_length, self.length_instructions["concise"])

        prompt = f"""You are a context compaction specialist. Your job is to summarize conversation context while preserving the most important information.

{length_instruction}

IMPORTANT GUIDELINES:
- Preserve key facts, decisions, and outcomes
- Maintain chronological order where relevant
- Keep technical details that are likely to be referenced later
- Remove redundant information and verbose explanations
- Use clear, direct language
"""

        if preserve_topics:
            prompt += f"\n- Ensure these topics are covered: {', '.join(preserve_topics)}"

        prompt += f"""

CONTEXT TO SUMMARIZE:
{context}

COMPACTED SUMMARY:"""

        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(
            self.llm_client.generate_completion(prompt=prompt, temperature=0.3, max_tokens=2048, stream=False)
        )

        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content
        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "text"):
            return response.text
        if isinstance(response, str):
            return response
        return str(response)

    def summarize_messages(
        self,
        messages: list[dict[str, Any]],
        **_legacy_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Replace compactable history with a provisioner-selected summary."""
        from ..context.compaction_provisioner import CompactionProvisioner, render_messages_for_summary

        def summary_agent(compactable_messages: list[dict[str, Any]], _previous_summary: str | None) -> str:
            return self.summarize_context(render_messages_for_summary(compactable_messages))

        current_tokens = max(1, len(render_messages_for_summary(messages)) // 4)
        provisioner = CompactionProvisioner(
            model="",
            max_context_tokens=current_tokens,
            target_tokens=max(1, current_tokens // 2),
            summary_agent=summary_agent,
        )
        result = provisioner.compact(messages, force=True)
        return result.messages if result.compacted else messages


def create_compaction_agent(llm_client: Any, target_length: str = "concise") -> CompactionAgent:
    """Construct a :class:`CompactionAgent` (thin convenience factory)."""
    return CompactionAgent(llm_client=llm_client, target_length=target_length)
