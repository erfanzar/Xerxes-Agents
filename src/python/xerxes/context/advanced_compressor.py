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
"""Advanced context compression using tool-result pruning and LLM summarization.

Implements ``HermesCompressionStrategy``, which reduces token usage by
summarizing bulky tool outputs and by generating a high-level summary of
older conversation turns via an optional LLM client.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .compaction_strategies import BaseCompactionStrategy

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:"
)

_MIN_SUMMARY_TOKENS = 2000
_SUMMARY_RATIO = 0.20
_SUMMARY_TOKENS_CEILING = 12_000
_CHARS_PER_TOKEN = 4

_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"


def _summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """Create a terse summary of a tool execution result.

    Args:
        tool_name (str): IN: name of the tool that was executed.
        tool_args (str): IN: JSON-encoded arguments passed to the tool.
        tool_content (str): IN: raw output / content returned by the tool.

    Returns:
        str: OUT: one-line summary describing what the tool did.
    """
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name in ("terminal", "ExecuteShell", "shell"):
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[{tool_name}] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"

    if tool_name in ("read_file", "ReadFile"):
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[{tool_name}] read {path} from line {offset} ({content_len:,} chars)"

    if tool_name in ("write_file", "WriteFile"):
        path = args.get("path", "?")
        written_lines = args.get("content", "").count("\n") + 1 if args.get("content") else "?"
        return f"[{tool_name}] wrote to {path} ({written_lines} lines)"

    if tool_name in ("search_files", "GrepTool", "grep"):
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[{tool_name}] search for '{pattern}' in {path} -> {count} matches"

    if tool_name in ("patch", "FileEditTool"):
        path = args.get("path", "?")
        return f"[{tool_name}] edited {path} ({content_len:,} chars result)"

    if tool_name in ("web_search", "GoogleSearch", "DuckDuckGoSearch"):
        query = args.get("query", "?")
        return f"[{tool_name}] query='{query}' ({content_len:,} chars result)"

    if tool_name in ("delegate_task", "AgentTool"):
        goal = args.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[{tool_name}] '{goal}' ({content_len:,} chars result)"

    if tool_name in ("execute_code", "ExecutePythonCode"):
        code_preview = (args.get("code") or "")[:60].replace("\n", " ")
        if len(args.get("code", "")) > 60:
            code_preview += "..."
        return f"[{tool_name}] `{code_preview}` ({line_count} lines output)"

    first_arg = ""
    for k, v in list(args.items())[:2]:
        sv = str(v)[:40]
        first_arg += f" {k}={sv}"
    return f"[{tool_name}]{first_arg} ({content_len:,} chars result)"


def _prune_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace verbose tool-result messages with compact summaries.

    Args:
        messages (list[dict[str, Any]]): IN: conversation messages in
            OpenAI-style dict format.

    Returns:
        list[dict[str, Any]]: OUT: messages with tool results longer than
        500 characters summarized.
    """
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "tool" and len(content) > 500:
            tool_name = msg.get("name", "tool")
            tool_args = json.dumps(msg.get("args", {}), ensure_ascii=False)
            summary = _summarize_tool_result(tool_name, tool_args, content)
            pruned = dict(msg)
            pruned["content"] = summary
            pruned["_original_content"] = content
            result.append(pruned)
        else:
            result.append(msg)
    return result


_SUMMARIZER_PREAMBLE = (
    "You are a context compaction assistant. Your job is to summarize "
    "a conversation history into a structured reference document.\n"
    "Do NOT respond to any questions in the history. "
    "Do NOT provide new information. Just summarize.\n\n"
    "Use this exact template:\n"
    "## Resolved Questions\n"
    "(What questions were answered and how)\n\n"
    "## Pending Questions\n"
    "(What questions remain open)\n\n"
    "## Current State\n"
    "(Files modified, config changes, current working state)\n\n"
    "## Remaining Work\n"
    "(What still needs to be done, if anything)"
)


def _build_summary_prompt(
    messages: list[dict[str, Any]],
    previous_summary: str | None = None,
) -> str:
    """Assemble the prompt text used to request a conversation summary.

    Args:
        messages (list[dict[str, Any]]): IN: conversation messages to
            summarize.
        previous_summary (str | None): IN: prior summary to update rather
            than repeat.

    Returns:
        str: OUT: complete prompt string for the summarizer.
    """
    lines = [_SUMMARIZER_PREAMBLE, "", "CONVERSATION HISTORY:"]
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "tool":
            name = msg.get("name", "tool")
            lines.append(f"[{name} result] {content[:500]}")
        elif role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls", [])
            tc_summary = ", ".join(
                f"{tc.get('function', {}).get('name', '?')}({tc.get('function', {}).get('arguments', '')[:80]})"
                for tc in tool_calls
            )
            lines.append(f"Assistant: {content[:200]} [tool_calls: {tc_summary}]")
        else:
            lines.append(f"{role.capitalize()}: {content[:1000]}")

    if previous_summary:
        lines.extend(
            [
                "",
                "PREVIOUS SUMMARY (update this, don't repeat):",
                previous_summary,
            ]
        )

    lines.extend(["", "STRUCTURED SUMMARY:"])
    return "\n".join(lines)


class HermesCompressionStrategy(BaseCompactionStrategy):
    """Compaction strategy that prunes tool results and LLM-summarizes history."""

    def __init__(
        self,
        target_tokens: int,
        model: str = "gpt-4",
        preserve_system: bool = True,
        preserve_recent: int = 3,
        llm_client: Any | None = None,
        tail_token_budget: int | None = None,
    ):
        """Initialize the Hermes compression strategy.

        Args:
            target_tokens (int): IN: desired maximum token count after
                compaction.
            model (str): IN: model name for token counting.
                Defaults to "gpt-4".
            preserve_system (bool): IN: whether to keep system messages.
                Defaults to ``True``.
            preserve_recent (int): IN: number of recent messages to protect.
                Defaults to 3.
            llm_client (Any | None): IN: optional LLM client capable of
                ``generate_completion``. OUT: used for generating summaries.
            tail_token_budget (int | None): IN: explicit token budget for the
                tail (recent) messages. OUT: computed automatically if omitted.
        """
        super().__init__(
            target_tokens=target_tokens,
            model=model,
            preserve_system=preserve_system,
            preserve_recent=preserve_recent,
        )
        self.llm_client = llm_client
        self._previous_summary: str | None = None
        self._compaction_count = 0

        self.tail_token_budget = tail_token_budget or min(20_000, max(target_tokens // 4, 2000))

    def compact(
        self,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Compact messages by pruning tools and summarizing older turns.

        Args:
            messages (list[dict[str, str]]): IN: conversation history in
                OpenAI-style dict format.
            metadata (dict[str, Any] | None): IN: optional metadata (unused).

        Returns:
            tuple[list[dict[str, str]], dict[str, Any]]: OUT: compacted
            messages and statistics about the compaction.
        """
        stats = {
            "original_count": len(messages),
            "strategy": "hermes_compression",
        }

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if not non_system:
            stats["compacted_count"] = len(messages)
            stats["summary_created"] = False
            return messages, stats

        compactable_msgs = non_system

        pruned = _prune_tool_results(compactable_msgs)
        tools_pruned = sum(
            1
            for orig, new in zip(compactable_msgs, pruned, strict=False)
            if orig.get("content", "") != new.get("content", "")
        )
        stats["tools_pruned"] = tools_pruned

        tail_budget = self.tail_token_budget
        tail_msgs: list[dict[str, Any]] = []
        middle_msgs: list[dict[str, Any]] = []

        tail_tokens = 0
        for msg in reversed(pruned):
            msg_tokens = self.token_counter.count_tokens([msg])
            if tail_tokens + msg_tokens <= tail_budget:
                tail_msgs.insert(0, msg)
                tail_tokens += msg_tokens
            else:
                middle_msgs = pruned[: len(pruned) - len(tail_msgs)]
                break
        else:
            middle_msgs = []

        stats["tail_messages"] = len(tail_msgs)
        stats["middle_messages"] = len(middle_msgs)

        if not middle_msgs:
            result = [*system_msgs, *tail_msgs]
            stats["compacted_count"] = len(result)
            stats["summary_created"] = False
            return result, stats

        summary = self._summarize(middle_msgs, self._previous_summary)
        self._previous_summary = summary
        self._compaction_count += 1

        summary_msg = {
            "role": "system",
            "content": f"{SUMMARY_PREFIX}\n\n{summary}",
        }

        result = [*system_msgs, summary_msg, *tail_msgs]

        stats["compacted_count"] = len(result)
        stats["summary_created"] = True
        stats["messages_summarized"] = len(middle_msgs)
        stats["compaction_count"] = self._compaction_count

        return result, stats

    def _summarize(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str | None,
    ) -> str:
        """Generate a summary of the provided messages.

        Args:
            messages (list[dict[str, Any]]): IN: messages to summarize.
            previous_summary (str | None): IN: prior summary to update.

        Returns:
            str: OUT: LLM-generated summary or a fallback line-based summary.
        """
        prompt = _build_summary_prompt(messages, previous_summary)

        if self.llm_client:
            try:
                summary = self._call_llm(prompt)
                if summary:
                    return summary
            except Exception as exc:
                logger.warning("LLM summarization failed: %s", exc)

        lines = ["[FALLBACK SUMMARY] Key points from earlier conversation:"]
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")[:200]
            if content:
                lines.append(f"  {role}: {content}")
        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str | None:
        """Invoke the LLM client to generate a summary.

        Args:
            prompt (str): IN: summarization prompt.

        Returns:
            str | None: OUT: generated text, or ``None`` on failure.
        """
        if self.llm_client is None:
            return None
        try:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    from ..core.utils import run_sync

                    response = run_sync(
                        self.llm_client.generate_completion(
                            prompt=prompt, temperature=0.3, max_tokens=4096, stream=False
                        )
                    )
                else:
                    response = loop.run_until_complete(
                        self.llm_client.generate_completion(
                            prompt=prompt, temperature=0.3, max_tokens=4096, stream=False
                        )
                    )
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    self.llm_client.generate_completion(prompt=prompt, temperature=0.3, max_tokens=4096, stream=False)
                )

            if hasattr(self.llm_client, "extract_content"):
                return self.llm_client.extract_content(response)
            elif hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content
            elif isinstance(response, str):
                return response
            return str(response)
        except Exception:
            return None
