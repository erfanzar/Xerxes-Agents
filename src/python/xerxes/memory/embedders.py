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
"""Pluggable text embedding backends used by retrieval and vector storage.

Provides a small ``Embedder`` ABC with four concrete implementations:
a dependency-free ``HashEmbedder`` (always available),
``SentenceTransformerEmbedder`` (local CPU/GPU models),
``OpenAIEmbedder`` (cloud), and ``OllamaEmbedder`` (local server).
``get_default_embedder`` resolves one of these based on the
``XERXES_EMBEDDER`` env var, then ``OPENAI_API_KEY``, then the
presence of ``sentence_transformers``, falling back to hashing."""

from __future__ import annotations

import logging
import os
import typing as tp
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
Vector = list[float]


class Embedder(ABC):
    """Abstract text-to-vector encoder.

    Attributes:
        name: Short identifier used for diagnostics and dispatch.
        dim: Embedding dimensionality; may be lazily set when the
            underlying model is first loaded."""

    name: str = ""
    dim: int = 0

    @abstractmethod
    def embed(self, text: str) -> Vector:
        """Encode ``text`` into a dense vector."""
        ...

    def embed_batch(self, texts: tp.Sequence[str]) -> list[Vector]:
        """Encode a batch of texts; default implementation calls ``embed`` per item."""

        return [self.embed(t) for t in texts]


class HashEmbedder(Embedder):
    """Dependency-free fallback embedder using hashed bag-of-words counts.

    Produces a normalised ``dim``-wide vector by hashing each token to
    a bucket and accumulating uniform mass. Useful when no model is
    installed and for deterministic tests."""

    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        """Configure the output dimensionality."""

        self.dim = dim

    def embed(self, text: str) -> Vector:
        """Return an L2-normalised hashed bag-of-words vector for ``text``."""

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
    """Wrapper around ``sentence-transformers`` models.

    The model is loaded lazily on the first call to ``embed`` or
    ``embed_batch``; ``ImportError`` is raised with installation guidance
    if the dependency is missing."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Record the model name; defer loading until first use."""

        self.model_name = model_name
        self._model: tp.Any = None

    def _ensure_loaded(self) -> None:
        """Lazily import sentence-transformers and load the configured model."""

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
        """Encode ``text`` via the loaded SentenceTransformer model."""

        self._ensure_loaded()
        if not text:
            return [0.0] * self.dim
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: tp.Sequence[str]) -> list[Vector]:
        """Encode ``texts`` in one model call for efficiency."""

        self._ensure_loaded()
        if not texts:
            return []
        arr = self._model.encode(list(texts), convert_to_numpy=True)
        return [v.tolist() for v in arr]


class OpenAIEmbedder(Embedder):
    """Embedder backed by the OpenAI embeddings API.

    Reads ``OPENAI_API_KEY`` from the environment by default. ``dim``
    is initially estimated from ``model_name`` and is updated to the
    actual response width after the first successful call."""

    name = "openai"

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Capture model name, API key (or env), and optional base URL override."""

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
        """Lazily import the OpenAI SDK and construct the client.

        Raises:
            ImportError: ``openai`` is not installed.
            RuntimeError: No API key is available."""

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
        """Request an embedding for ``text`` from the OpenAI API."""

        self._ensure_client()
        if not text:
            return [0.0] * self.dim
        resp = self._client.embeddings.create(input=text, model=self.model_name)
        vec = list(resp.data[0].embedding)
        self.dim = len(vec)
        return vec

    def embed_batch(self, texts: tp.Sequence[str]) -> list[Vector]:
        """Request embeddings for ``texts`` in a single API call."""

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

    Targets ``OLLAMA_HOST`` (default ``http://localhost:11434``) and
    requests embeddings from the ``/api/embeddings`` endpoint."""

    name = "ollama"

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str | None = None,
    ) -> None:
        """Capture model name and Ollama base URL (defaults to env or localhost)."""

        self.model_name = model_name
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.dim = 768

    def embed(self, text: str) -> Vector:
        """POST to ``/api/embeddings`` and return the embedding list.

        Raises:
            ImportError: ``httpx`` is not installed."""

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
    """Return a process-wide default embedder, constructing it on first use.

    Resolution order: ``XERXES_EMBEDDER`` env var (``hash``, ``openai``,
    ``sentence-transformers``/``st``, ``ollama``), then ``OPENAI_API_KEY``,
    then a locally installed ``sentence_transformers``, finally
    ``HashEmbedder``."""

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
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is not None:
        _DEFAULT_CACHE = SentenceTransformerEmbedder()
        return _DEFAULT_CACHE
    _DEFAULT_CACHE = HashEmbedder()
    return _DEFAULT_CACHE


def reset_default_embedder() -> None:
    """Clear the cached default embedder so the next call re-resolves it.

    Primarily used by tests that mutate ``XERXES_EMBEDDER`` between cases."""

    global _DEFAULT_CACHE
    _DEFAULT_CACHE = None


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Returns ``0.0`` when the vectors have mismatched lengths or either
    has zero norm."""

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
