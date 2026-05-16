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
"""Token-budget-aware context compressor.

Algorithm, in order:

1. Pre-prune oversized tool outputs heuristically (cheap, no LLM).
2. Protect the first ``protect_first`` and last ``protect_last`` messages.
3. Summarize the middle slice via an injected callable.
4. Insert the summary between head and tail as a user message prefixed
   ``[CONTEXT COMPACTION — REFERENCE ONLY]`` so the model knows it's
   background, not instruction.
5. Support iterative merging — if a prior summary exists at the
   head/middle boundary, fold the new summary into it.

The summarizer is a ``Callable[[list[message], int], str]`` so tests
can substitute trivial implementations. :func:`naive_summarizer`
ships as a fallback when no LLM is wired.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .token_counter import SmartTokenCounter
from .tool_result_pruner import prune_messages

COMPACTION_REFERENCE_PREFIX = "[CONTEXT COMPACTION — REFERENCE ONLY]"

Summarizer = Callable[[list[dict[str, Any]], int], str]


def naive_summarizer(messages: list[dict[str, Any]], budget_tokens: int) -> str:
    """Fallback summarizer that emits one ``role: first line`` per message.

    Not a real summary — just enough text for tests and to keep the
    pipeline producing valid output when no auxiliary LLM is wired.
    ``budget_tokens`` is accepted for interface compatibility but
    ignored.
    """

    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        if not isinstance(content, str):
            content = str(content)
        first = content.strip().splitlines()[0] if content.strip() else ""
        if len(first) > 200:
            first = first[:200] + "…"
        if first:
            lines.append(f"- {role}: {first}")
    return "\n".join(lines)


@dataclass
class CompressionResult:
    """Outcome of one :meth:`ContextCompressor.compress` pass.

    Attributes:
        messages: the new (possibly compressed) message list.
        compressed: ``True`` when any reduction happened.
        tokens_before: token count of the original input.
        tokens_after: token count of ``messages``.
        protected_first: messages protected at the head.
        protected_last: messages protected at the tail.
        compressed_count: middle messages folded into the summary.
        pruned_tool_results: tool messages shrunk by pre-pruning.
        summary_tokens: token count of the inserted summary placeholder.
        metadata: strategy label and ad-hoc diagnostic fields.
    """

    messages: list[dict[str, Any]]
    compressed: bool = False
    tokens_before: int = 0
    tokens_after: int = 0
    protected_first: int = 0
    protected_last: int = 0
    compressed_count: int = 0
    pruned_tool_results: int = 0
    summary_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextCompressor:
    """Decide when to compact a conversation and orchestrate the steps.

    Once the conversation token count reaches
    ``threshold * context_window`` :meth:`compress` runs pre-pruning
    plus an LLM-summarised middle slice; otherwise it returns the
    input unchanged.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.75,
        context_window: int = 200_000,
        protect_first: int = 3,
        protect_last: int = 6,
        summary_min_tokens: int = 2_000,
        summary_max_tokens: int = 12_000,
        summary_budget_ratio: float = 0.20,
        model: str = "gpt-4",
        summarizer: Summarizer | None = None,
        token_counter: SmartTokenCounter | None = None,
    ) -> None:
        """Configure compaction thresholds and inject a summarizer/counter."""
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0.0, 1.0]")
        if protect_first < 0 or protect_last < 0:
            raise ValueError("protect_first and protect_last must be >= 0")
        self.threshold = float(threshold)
        self.context_window = int(context_window)
        self.protect_first = int(protect_first)
        self.protect_last = int(protect_last)
        self.summary_min_tokens = int(summary_min_tokens)
        self.summary_max_tokens = int(summary_max_tokens)
        self.summary_budget_ratio = float(summary_budget_ratio)
        self._summarizer: Summarizer = summarizer or naive_summarizer
        self._token_counter = token_counter or SmartTokenCounter(model=model)

    # ---------------------------- token counting

    def _count(self, messages: list[dict[str, Any]]) -> int:
        """Token count for a list of messages via the bound counter."""
        return int(self._token_counter.count_tokens(messages))

    def _count_text(self, text: str) -> int:
        """Token count for a plain string via the bound counter."""
        return int(self._token_counter.count_tokens(text))

    def threshold_tokens(self) -> int:
        """Return the token threshold above which compaction triggers."""
        return int(self.context_window * self.threshold)

    def should_compact(self, messages: list[dict[str, Any]]) -> bool:
        """Return ``True`` when ``messages`` exceed the compaction threshold."""
        return self._count(messages) >= self.threshold_tokens()

    # ---------------------------- summary helpers

    def _summary_budget(self, compressed_token_count: int) -> int:
        """Return the per-summary token budget, clamped to min/max bounds."""
        approx = int(compressed_token_count * self.summary_budget_ratio)
        budget = max(self.summary_min_tokens, approx)
        return min(self.summary_max_tokens, budget)

    @staticmethod
    def _looks_like_prior_summary(content: Any) -> bool:
        """Detect a previously inserted compaction reference summary."""
        return isinstance(content, str) and content.startswith(COMPACTION_REFERENCE_PREFIX)

    def _wrap_summary(self, prior_summary: str | None, new_summary: str) -> str:
        """Combine an old summary (if any) with the new one under the prefix."""
        body = new_summary.strip()
        merged = body if not prior_summary else f"{prior_summary.strip()}\n\n---\n\n{body}"
        return f"{COMPACTION_REFERENCE_PREFIX}\n\n{merged}"

    # ---------------------------- the algorithm

    def compress(self, messages: list[dict[str, Any]]) -> CompressionResult:
        """Run pre-prune + middle-summarize and return a :class:`CompressionResult`.

        If pre-pruning alone shrinks the conversation below threshold,
        the LLM summarization step is skipped. Returns the input
        unchanged (``compressed=False``) when there is no middle to
        compress.
        """

        tokens_before = self._count(messages)
        if not messages:
            return CompressionResult(messages=messages, tokens_before=0, tokens_after=0)

        # Step 1: heuristic pre-prune oversized tool outputs (always cheap).
        pruned, pruned_count = prune_messages(
            messages,
            protect_last=self.protect_last,
        )

        # Step 2: check whether we still need an LLM pass.
        tokens_after_prune = self._count(pruned)
        below_threshold = tokens_after_prune < self.threshold_tokens()
        # If pre-pruning alone suffices, ship it.
        if below_threshold and pruned_count > 0:
            return CompressionResult(
                messages=pruned,
                compressed=True,
                tokens_before=tokens_before,
                tokens_after=tokens_after_prune,
                protected_first=min(self.protect_first, len(pruned)),
                protected_last=min(self.protect_last, len(pruned)),
                compressed_count=0,
                pruned_tool_results=pruned_count,
                summary_tokens=0,
                metadata={"strategy": "prune-only"},
            )

        # Step 3: split into protected head, middle slice, and protected tail.
        n = len(pruned)
        head_n = min(self.protect_first, n)
        tail_n = min(self.protect_last, max(0, n - head_n))
        head = pruned[:head_n]
        tail = pruned[n - tail_n :] if tail_n else []
        middle = pruned[head_n : n - tail_n] if tail_n else pruned[head_n:]

        if not middle:
            # Nothing to compress — return prune output.
            return CompressionResult(
                messages=pruned,
                compressed=pruned_count > 0,
                tokens_before=tokens_before,
                tokens_after=tokens_after_prune,
                protected_first=head_n,
                protected_last=tail_n,
                compressed_count=0,
                pruned_tool_results=pruned_count,
                summary_tokens=0,
                metadata={"strategy": "no-middle"},
            )

        # Step 4: detect a prior summary; if present, merge with it.
        # Summaries usually sit right after the protected head (because
        # that's where the previous pass inserted one). Check both places.
        prior_summary: str | None = None
        if head and self._looks_like_prior_summary(head[-1].get("content")):
            prior_summary = head[-1].get("content")
            head = head[:-1]
        elif middle and self._looks_like_prior_summary(middle[0].get("content")):
            prior_summary = middle[0].get("content")
            middle = middle[1:]

        # Step 5: summarize the middle slice with the injected summarizer.
        middle_tokens = self._count(middle)
        budget = self._summary_budget(middle_tokens)
        new_summary = self._summarizer(middle, budget)
        merged_summary = self._wrap_summary(prior_summary, new_summary)
        summary_msg = {"role": "user", "content": merged_summary}

        out = [*head, summary_msg, *tail]
        tokens_after = self._count(out)
        summary_tokens = self._count_text(merged_summary)

        return CompressionResult(
            messages=out,
            compressed=True,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            protected_first=len(head) + (1 if prior_summary else 0),
            protected_last=tail_n,
            compressed_count=len(middle),
            pruned_tool_results=pruned_count,
            summary_tokens=summary_tokens,
            metadata={"strategy": "iterative" if prior_summary else "first-pass"},
        )


__all__ = [
    "COMPACTION_REFERENCE_PREFIX",
    "CompressionResult",
    "ContextCompressor",
    "Summarizer",
    "naive_summarizer",
]
