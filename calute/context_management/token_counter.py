# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Token counting using provider APIs."""

from typing import Any

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import anthropic  # type:ignore #noqa

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ProviderTokenCounter:
    """Token counter that uses actual provider APIs."""

    @classmethod
    def count_tokens_for_provider(
        cls,
        text: str | list[dict[str, str]],
        provider: str | None = None,
        model: str | None = None,
        llm_client: Any | None = None,
    ) -> int:
        """Count tokens using the appropriate provider's API.

        Args:
            text: Text string or messages list to count
            provider: Provider name (openai, anthropic, etc.)
            model: Model name
            llm_client: Optional LLM client instance

        Returns:
            Token count
        """

        if isinstance(text, list):
            text = cls._messages_to_text(text)

        if not provider and model:
            provider = cls._detect_provider(model)

        if provider == "openai":
            return cls._count_openai_tokens(text, model)
        elif provider == "anthropic":
            return cls._count_anthropic_tokens(text, model, llm_client)
        elif provider == "google":
            return cls._count_google_tokens(text, model, llm_client)
        else:
            return cls._count_fallback_tokens(text, model)

    @classmethod
    def _detect_provider(cls, model: str) -> str | None:
        """Detect provider from model name."""
        if not model:
            return None

        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "palm" in model_lower:
            return "google"
        elif "llama" in model_lower:
            return "meta"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        return None

    @classmethod
    def _messages_to_text(cls, messages: list[dict[str, str]]) -> str:
        """Convert messages to text."""
        text_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            text_parts.append(f"{role}: {content}")
        return "\n".join(text_parts)

    @classmethod
    def _count_openai_tokens(cls, text: str, model: str | None = None) -> int:
        """Count tokens for OpenAI models using tiktoken."""
        if not TIKTOKEN_AVAILABLE:
            return len(text) // 4

        try:
            if model:
                if "gpt-4o" in model or "o1" in model:
                    encoding = tiktoken.get_encoding("o200k_base")
                elif "gpt-4" in model:
                    encoding = tiktoken.get_encoding("cl100k_base")
                elif "gpt-3.5" in model:
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:
                    encoding = tiktoken.get_encoding("cl100k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except Exception as e:
            print(f"Error counting OpenAI tokens: {e}")
            return len(text) // 4

    @classmethod
    def _count_anthropic_tokens(cls, text: str, model: str | None = None, client: Any | None = None) -> int:
        """Count tokens for Anthropic models."""
        if ANTHROPIC_AVAILABLE and client:
            try:
                if hasattr(client, "count_tokens"):
                    return client.count_tokens(text)
            except Exception:
                pass

        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                pass

        return len(text) // 4

    @classmethod
    def _count_google_tokens(cls, text: str, model: str | None = None, client: Any | None = None) -> int:
        """Count tokens for Google models."""

        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")

                token_count = len(encoding.encode(text))
                return int(token_count * 1.1)
            except Exception:
                pass

        return len(text) // 4

    @classmethod
    def _count_fallback_tokens(cls, text: str, model: str | None = None) -> int:
        """Fallback token counting when provider-specific methods aren't available."""
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                pass

        return len(text) // 4


class SmartTokenCounter:
    """Smart token counter that automatically selects the best counting method."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        llm_client: Any | None = None,
    ):
        """Initialize the smart token counter.

        Args:
            provider: Provider name
            model: Model name
            llm_client: Optional LLM client with token counting capabilities
        """
        self.provider = provider
        self.model = model
        self.llm_client = llm_client

        if not self.provider and self.model:
            self.provider = ProviderTokenCounter._detect_provider(self.model)

    def count_tokens(self, text: str | list[dict[str, str]]) -> int:
        """Count tokens using the best available method."""
        return ProviderTokenCounter.count_tokens_for_provider(
            text=text,
            provider=self.provider,
            model=self.model,
            llm_client=self.llm_client,
        )

    def count_remaining_capacity(self, text: str | list[dict[str, str]], max_tokens: int) -> int:
        """Calculate remaining token capacity."""
        used_tokens = self.count_tokens(text)
        return max(0, max_tokens - used_tokens)

    def estimate_compression_ratio(self, original_text: str, compressed_text: str) -> float:
        """Estimate the compression ratio achieved."""
        original_tokens = self.count_tokens(original_text)
        compressed_tokens = self.count_tokens(compressed_text)

        if original_tokens == 0:
            return 0.0

        return 1.0 - (compressed_tokens / original_tokens)
