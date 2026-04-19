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

"""Embedder protocol and built-in providers for semantic memory.

Defines a single :class:`Embedder` protocol that all embedding backends
implement: ``embed(text)`` returns a dense vector. Providers shipped:

- :class:`HashEmbedder` — zero-dependency 256-dim hashed bag-of-words.
- :class:`SentenceTransformerEmbedder` — local model via the
  ``sentence-transformers`` package.
- :class:`OpenAIEmbedder` — remote OpenAI ``/v1/embeddings``.
- :class:`OllamaEmbedder` — local Ollama ``/api/embeddings``.

Use :func:`get_default_embedder` to pick the best available backend
without specifying a model.

The :class:`Embedder` interface is intentionally minimal: a single
:meth:`Embedder.embed` plus a :meth:`Embedder.embed_batch` for callers
that need throughput. Providers may override the batch method when the
underlying API supports it natively.
"""

from __future__ import annotations

import logging
import os
import typing as tp
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
Vector = list[float]


class Embedder(ABC):
    """Abstract embedder that turns text into a dense vector.

    Implementations must produce vectors of consistent dimension within
    a single instance. Different instances (e.g. different models) may
    use different dimensions; downstream stores must not mix them.

    Attributes:
        name: Stable identifier used for logging and storage tagging.
        dim: Vector dimension produced by :meth:`embed`. Subclasses set
            this in ``__init__`` (or after the first embed call when the
            underlying provider is dynamically sized).
    """

    name: str = ""
    dim: int = 0

    @abstractmethod
    def embed(self, text: str) -> Vector:
        """Encode a single text string into a dense vector.

        Args:
            text: The text to encode. Empty strings should yield a
                zero vector of length :attr:`dim` rather than raising.

        Returns:
            A list of floats of length :attr:`dim`.
        """

    def embed_batch(self, texts: tp.Sequence[str]) -> list[Vector]:
        """Encode multiple texts. Default implementation calls ``embed`` per item.

        Subclasses should override when the underlying provider supports
        a batched API (most remote embeddings APIs do).

        Args:
            texts: The texts to encode.

        Returns:
            One vector per input text, in the same order.
        """
        return [self.embed(t) for t in texts]


class HashEmbedder(Embedder):
    """Zero-dependency hashed bag-of-words embedder (256-dim, L2-normalised).

    Tokenises on whitespace, lowercases, hashes each token to a slot,
    accumulates term frequency, and L2-normalises the result. Cheap and
    deterministic, but semantically weak: prefer a real embedder for
    Hermes-grade recall.

    Attributes:
        name: Always ``"hash"``.
        dim: Always ``256``.
    """

    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        """Initialise with the target vector dimension.

        Args:
            dim: Number of slots in the hashed vector. Higher reduces
                collisions at the cost of memory.
        """
        self.dim = dim

    def embed(self, text: str) -> Vector:
        """Hash tokens into a fixed-dim TF vector and L2-normalise."""
        tokens = text.lower().split()
        if not tokens:
            return [0.0] * self.dim
        vec = [0.0] * self.dim
        total = float(len(tokens))
        for tok in tokens:
            idx = (hash(tok) & 0x7FFFFFFF) % self.dim
            vec[idx] += 1.0 / total
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a local ``sentence-transformers`` model.

    Lazily loads the model on first call. The default model is
    ``all-MiniLM-L6-v2`` (384 dim, ~80 MB) which gives a good
    quality/size tradeoff for cross-session recall. Raises
    :class:`ImportError` only if the package is missing **and** the
    embedder is actually used.

    Attributes:
        name: ``"sentence-transformers"``.
        model_name: The model identifier passed to ``SentenceTransformer``.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Defer model loading until first :meth:`embed` call.

        Args:
            model_name: ``sentence-transformers`` model identifier.
        """
        self.model_name = model_name
        self._model: tp.Any = None

    def _ensure_loaded(self) -> None:
        """Load the ``sentence-transformers`` model on first use and cache ``dim``."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder; "
                "install with `pip install sentence-transformers`"
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> Vector:
        """Encode text to a dense vector via the loaded model."""
        self._ensure_loaded()
        if not text:
            return [0.0] * self.dim
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: tp.Sequence[str]) -> list[Vector]:
        """Batch-encode using the model's native batching."""
        self._ensure_loaded()
        if not texts:
            return []
        arr = self._model.encode(list(texts), convert_to_numpy=True)
        return [v.tolist() for v in arr]


class OpenAIEmbedder(Embedder):
    """Embedder backed by the OpenAI ``/v1/embeddings`` API.

    Reads the API key from the constructor or the ``OPENAI_API_KEY``
    env var. Defaults to ``text-embedding-3-small`` (1536 dim) for the
    best price/quality tradeoff in 2026.

    Attributes:
        name: ``"openai"``.
        model_name: The OpenAI embeddings model identifier.
    """

    name = "openai"

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialise the OpenAI embeddings client.

        Args:
            model_name: OpenAI embeddings model. ``text-embedding-3-small``
                is 1536-dim; ``text-embedding-3-large`` is 3072-dim.
            api_key: Explicit API key. Falls back to ``OPENAI_API_KEY``.
            base_url: Optional custom base URL (e.g. for an OpenAI-compatible
                gateway).
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self._client: tp.Any = None
        if "large" in model_name:
            self.dim = 3072
        elif "ada" in model_name:
            self.dim = 1536
        else:
            self.dim = 1536

    def _ensure_client(self) -> None:
        """Build the OpenAI client on first use, requiring a configured API key."""
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required for OpenAIEmbedder; install with `pip install openai`") from exc
        if not self.api_key:
            raise RuntimeError("OpenAIEmbedder requires OPENAI_API_KEY (env or constructor arg)")
        kwargs: dict[str, tp.Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def embed(self, text: str) -> Vector:
        """Encode a single text via the OpenAI embeddings endpoint."""
        self._ensure_client()
        if not text:
            return [0.0] * self.dim
        resp = self._client.embeddings.create(input=text, model=self.model_name)
        vec = list(resp.data[0].embedding)
        self.dim = len(vec)
        return vec

    def embed_batch(self, texts: tp.Sequence[str]) -> list[Vector]:
        """Batch-encode in a single API request."""
        self._ensure_client()
        if not texts:
            return []
        resp = self._client.embeddings.create(input=list(texts), model=self.model_name)
        out = [list(d.embedding) for d in resp.data]
        if out:
            self.dim = len(out[0])
        return out


class OllamaEmbedder(Embedder):
    """Embedder backed by a local Ollama server.

    Calls ``POST {base_url}/api/embeddings``. The base URL defaults to
    ``http://localhost:11434`` and can be overridden via the
    ``OLLAMA_HOST`` env var or constructor arg. Default model is
    ``nomic-embed-text`` (768 dim).

    Attributes:
        name: ``"ollama"``.
        model_name: The Ollama model identifier.
        base_url: Ollama server URL.
    """

    name = "ollama"

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str | None = None,
    ) -> None:
        """Initialise the Ollama embedder.

        Args:
            model_name: Ollama model name. ``nomic-embed-text`` is a
                strong open default; ``mxbai-embed-large`` is also good.
            base_url: Override Ollama URL. Defaults to ``$OLLAMA_HOST``
                or ``http://localhost:11434``.
        """
        self.model_name = model_name
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.dim = 768

    def embed(self, text: str) -> Vector:
        """Encode via ``POST /api/embeddings``."""
        if not text:
            return [0.0] * self.dim
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx is required for OllamaEmbedder") from exc
        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        vec = list(resp.json().get("embedding", []))
        if vec:
            self.dim = len(vec)
        return vec


_DEFAULT_CACHE: Embedder | None = None


def get_default_embedder() -> Embedder:
    """Return a process-wide default embedder, picking the best available.

    Resolution order:

    1. ``$XERXES_EMBEDDER`` env var: ``"hash"``, ``"openai"``,
       ``"sentence-transformers"``, or ``"ollama"``.
    2. If ``OPENAI_API_KEY`` is set, use :class:`OpenAIEmbedder`.
    3. If ``sentence-transformers`` importable, use
       :class:`SentenceTransformerEmbedder`.
    4. Otherwise fall back to :class:`HashEmbedder`.

    Cached after the first call so subsequent invocations are cheap.

    Returns:
        A ready-to-use :class:`Embedder` instance.
    """
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is not None:
        return _DEFAULT_CACHE
    forced = os.environ.get("XERXES_EMBEDDER", "").strip().lower()
    if forced == "hash":
        _DEFAULT_CACHE = HashEmbedder()
        return _DEFAULT_CACHE
    if forced == "openai":
        _DEFAULT_CACHE = OpenAIEmbedder()
        return _DEFAULT_CACHE
    if forced in ("sentence-transformers", "st"):
        _DEFAULT_CACHE = SentenceTransformerEmbedder()
        return _DEFAULT_CACHE
    if forced == "ollama":
        _DEFAULT_CACHE = OllamaEmbedder()
        return _DEFAULT_CACHE
    if os.environ.get("OPENAI_API_KEY"):
        _DEFAULT_CACHE = OpenAIEmbedder()
        return _DEFAULT_CACHE
    try:
        import sentence_transformers  # noqa: F401

        _DEFAULT_CACHE = SentenceTransformerEmbedder()
        return _DEFAULT_CACHE
    except ImportError:
        pass
    _DEFAULT_CACHE = HashEmbedder()
    return _DEFAULT_CACHE


def reset_default_embedder() -> None:
    """Clear the cached default embedder. Mainly for tests."""
    global _DEFAULT_CACHE
    _DEFAULT_CACHE = None


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Return the cosine similarity in ``[-1.0, 1.0]`` between two vectors.

    Args:
        a: First vector.
        b: Second vector. Must have the same length as ``a``.

    Returns:
        Cosine similarity. Returns ``0.0`` when either vector has zero
        norm (avoids divide-by-zero).
    """
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))
