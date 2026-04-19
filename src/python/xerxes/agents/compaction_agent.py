# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Dedicated agent for intelligent context compaction through summarization.

This module provides the CompactionAgent class, which specializes in compacting
conversation context and message histories through intelligent summarization.
It helps manage context length in long-running conversations by creating concise
summaries while preserving critical information.

The agent supports features like:
- Multiple summary length modes (brief, concise, detailed)
- Topic preservation during summarization
- Message history compaction with recent message preservation
- Fallback truncation when LLM-based summarization fails
- Asynchronous LLM integration for summary generation

Typical usage example:
    from xerxes.agents.compaction_agent import CompactionAgent

    agent = CompactionAgent(
        llm_client=my_llm_client,
        target_length="concise"
    )


    summary = agent.summarize_context(long_context)


    compacted = agent.summarize_messages(messages, preserve_recent=3)
"""

from typing import Any


class CompactionAgent:
    """Agent specialized in compacting context through intelligent summarization.

    CompactionAgent provides intelligent context compaction capabilities using
    LLM-based summarization. It can summarize raw text context or entire message
    histories while preserving important information and recent interactions.

    The agent uses configurable length instructions to control output verbosity
    and supports topic preservation to ensure critical subjects are covered
    in summaries.

    Attributes:
        llm_client: LLM client instance used for generating summaries.
            Must support `generate_completion` method for async completion.
        target_length: Target summary verbosity level. One of:
            - 'brief': Extremely concise, 2-3 sentences
            - 'concise': Balanced, captures key points in paragraphs
            - 'detailed': Comprehensive, preserves context and decisions
        length_instructions: Dictionary mapping length modes to instruction prompts
            used to guide the LLM's summarization behavior.

    Example:
        >>> agent = CompactionAgent(llm_client=client, target_length="brief")
        >>> summary = agent.summarize_context("Long conversation text...")
        >>> print(summary)
    """

    def __init__(self, llm_client: Any, target_length: str = "concise"):
        """Initialize the compaction agent.

        Sets up the compaction agent with an LLM client and configures the
        target summary length. Initializes the length instruction templates
        used for guiding summarization.

        Args:
            llm_client: LLM client instance for generating summaries. Should
                support `generate_completion` method with prompt, temperature,
                max_tokens, and stream parameters.
            target_length: Target summary length mode. Valid values are:
                - 'brief': Extremely brief, 2-3 sentences only
                - 'concise': Balanced summary with key points (default)
                - 'detailed': Detailed summary preserving context

        Note:
            If an unsupported target_length is provided, the agent will
            fall back to 'concise' mode during summarization.
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
        """Summarize context intelligently using LLM-based summarization.

        Uses the configured LLM client to generate an intelligent summary of
        the provided context. The summary respects the configured target_length
        setting and can preserve specific topics during compaction.

        Args:
            context: The raw text context to summarize. If empty or under
                200 characters, returns the original context unchanged.
            preserve_topics: Optional list of topic keywords that must be
                covered in the summary. These topics are explicitly mentioned
                in the summarization prompt to ensure coverage.

        Returns:
            str: Summarized context text. Returns original context if it's
                too short (< 200 chars) or if summarization fails.

        Raises:
            No exceptions are raised; errors fall back to truncation.

        Note:
            - Uses asyncio to run the async LLM completion synchronously
            - Falls back to _fallback_truncate if LLM call fails
            - Temperature is set to 0.3 for consistent, focused summaries
            - Max tokens is limited to 2048 for the summary response

        Example:
            >>> summary = agent.summarize_context(
            ...     "Long conversation about AI and machine learning...",
            ...     preserve_topics=["neural networks", "training data"]
            ... )
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
                return self._fallback_truncate(context)

        except Exception as e:
            print(f"Error during summarization: {e}")
            import traceback

            traceback.print_exc()
            return self._fallback_truncate(context)

    def summarize_messages(
        self,
        messages: list[dict[str, str]],
        preserve_recent: int = 3,
    ) -> list[dict[str, str]]:
        """Summarize a list of messages into a compacted conversation history.

        Compacts a message history by summarizing older messages while preserving
        recent messages unchanged. System messages are always preserved separately.
        The summary is inserted as a special user message indicating it represents
        the previous conversation.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                Roles typically include 'system', 'user', and 'assistant'.
            preserve_recent: Number of most recent non-system messages to keep
                unchanged. Defaults to 3. Set to 0 to summarize all messages.

        Returns:
            list[dict[str, str]]: Compacted message list containing:
                - All original system messages (preserved as-is)
                - One summary message with role 'user' containing the summary
                - The most recent `preserve_recent` messages unchanged

        Note:
            - If total messages <= preserve_recent + 1, returns original messages
            - System messages are separated and always preserved at the beginning
            - The summary message includes a header indicating how many messages
              were summarized: "[PREVIOUS CONVERSATION SUMMARY - N messages]"

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi there!"},
            ...     {"role": "user", "content": "Tell me about AI"},
            ...     {"role": "assistant", "content": "AI is..."},
            ... ]
            >>> compacted = agent.summarize_messages(messages, preserve_recent=2)
            >>> len(compacted)
            4
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

    def _fallback_truncate(self, context: str, max_chars: int = 2000) -> str:
        """Fallback truncation when LLM-based summarization fails.

        Performs a simple truncation of the context by keeping the first and
        last portions of the text. This ensures some context is preserved
        even when the LLM client is unavailable or encounters an error.

        Args:
            context: The context string to truncate.
            max_chars: Maximum total characters to keep. The result will
                contain roughly half from the beginning and half from the end.
                Defaults to 2000 characters.

        Returns:
            str: Truncated context with a marker indicating how many characters
                were removed. Returns original context if already within limit.

        Note:
            The truncation format is:
            "[first half]... [TRUNCATED N characters] ...[last half]"
            This preserves both the beginning (often containing setup/context)
            and the end (often containing conclusions/recent info).
        """
        if len(context) <= max_chars:
            return context

        half = max_chars // 2
        return context[:half] + f"\n\n... [TRUNCATED {len(context) - max_chars} characters] ...\n\n" + context[-half:]


def create_compaction_agent(llm_client: Any, target_length: str = "concise") -> CompactionAgent:
    """Factory function to create a compaction agent.

    Convenience factory for creating CompactionAgent instances with the
    specified configuration. Provides a simple interface for agent creation
    without needing to import and instantiate the class directly.

    Args:
        llm_client: LLM client instance for generating summaries. Should
            support the `generate_completion` method for async completion.
        target_length: Target summary length mode. Valid values are:
            - 'brief': Extremely brief summaries (2-3 sentences)
            - 'concise': Balanced summaries with key points (default)
            - 'detailed': Comprehensive summaries preserving context

    Returns:
        CompactionAgent: Configured compaction agent instance ready for use.

    Example:
        >>> agent = create_compaction_agent(my_llm_client, "brief")
        >>> summary = agent.summarize_context("Long text...")
    """
    return CompactionAgent(llm_client=llm_client, target_length=target_length)
