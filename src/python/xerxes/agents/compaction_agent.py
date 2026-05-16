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
agent uses an LLM to summarize long contexts or message histories, falling
back to truncation if the LLM is unavailable.
"""

from typing import Any

FALLBACK_TRUNCATE_CHARS = 2000


class CompactionAgent:
    """Summarizes conversation context to reduce token usage.

    Supports multiple summary lengths (brief, concise, detailed) and can
    preserve specific topics during summarization.
    """

    def __init__(self, llm_client: Any, target_length: str = "concise"):
        """Build the agent.

        Args:
            llm_client: LLM client exposing ``generate_completion``. Falls
                back to :meth:`_fallback_truncate` when the client lacks it
                or raises.
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
        """Summarise ``context`` via the configured LLM, with a truncate fallback.

        Contexts under 200 characters are returned unchanged. If the LLM
        client lacks ``generate_completion`` or any call raises, the helper
        falls back to head+tail truncation. ``preserve_topics`` is woven
        into the prompt so the LLM is told what must not be dropped.
        """
        if not context or len(context) < 200:
            return context

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

        try:
            if hasattr(self.llm_client, "generate_completion"):
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
                elif hasattr(response, "content"):
                    return response.content
                elif hasattr(response, "text"):
                    return response.text
                elif isinstance(response, str):
                    return response
                return str(response)
            else:
                return self._fallback_truncate(context, FALLBACK_TRUNCATE_CHARS)

        except Exception as e:
            print(f"Error during summarization: {e}")
            import traceback

            traceback.print_exc()
            return self._fallback_truncate(context, FALLBACK_TRUNCATE_CHARS)

    def summarize_messages(
        self,
        messages: list[dict[str, str]],
        preserve_recent: int = 3,
    ) -> list[dict[str, str]]:
        """Replace older messages with a single summary, keeping system + recent.

        System messages always pass through verbatim. The most recent
        ``preserve_recent`` non-system messages are kept; everything older
        is concatenated, summarised via :meth:`summarize_context`, and
        inserted as one synthetic ``user`` message.
        """
        if len(messages) <= preserve_recent + 1:
            return messages

        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]

        recent_messages = other_messages[-preserve_recent:] if preserve_recent > 0 else []
        older_messages = other_messages[:-preserve_recent] if preserve_recent > 0 else other_messages

        if not older_messages:
            return messages

        context_parts = []
        for msg in older_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            context_parts.append(f"[{role.upper()}]: {content}")

        full_context = "\n\n".join(context_parts)

        summary = self.summarize_context(full_context)

        summary_message = {
            "role": "user",
            "content": f"[PREVIOUS CONVERSATION SUMMARY - {len(older_messages)} messages]:\n{summary}",
        }

        compacted = [*system_messages, summary_message, *recent_messages]

        return compacted

    def _fallback_truncate(self, context: str, max_chars: int) -> str:
        """Keep ``max_chars/2`` from the head and tail with a ``[TRUNCATED]`` marker."""
        if len(context) <= max_chars:
            return context

        half = max_chars // 2
        return context[:half] + f"\n\n... [TRUNCATED {len(context) - max_chars} characters] ...\n\n" + context[-half:]


def create_compaction_agent(llm_client: Any, target_length: str = "concise") -> CompactionAgent:
    """Construct a :class:`CompactionAgent` (thin convenience factory)."""
    return CompactionAgent(llm_client=llm_client, target_length=target_length)
