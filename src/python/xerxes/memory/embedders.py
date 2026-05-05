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
"""Embedders module for Xerxes.

Exports:
    - logger
    - Vector
    - Embedder
    - HashEmbedder
    - SentenceTransformerEmbedder
    - OpenAIEmbedder
    - OllamaEmbedder
    - get_default_embedder
    - reset_default_embedder
    - cosine_similarity"""

from __future__ import annotations

import logging
import os
import typing as tp
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
Vector = list[float]


class Embedder(ABC):
    """Embedder.

    Inherits from: ABC

    Attributes:
        name (str): name.
        dim (int): dim."""

    name: str = ""
    dim: int = 0

    @abstractmethod
    def embed(self, text: str) -> Vector:
        """Embed.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            text (str): IN: text. OUT: Consumed during execution.
        Returns:
            Vector: OUT: Result of the operation."""
        ...

    def embed_batch(self, texts: tp.Sequence[str]) -> list[Vector]:
        """Embed batch.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            texts (tp.Sequence[str]): IN: texts. OUT: Consumed during execution.
        Returns:
            list[Vector]: OUT: Result of the operation."""

        return [self.embed(t) for t in texts]


class HashEmbedder(Embedder):
    """Hash embedder.

    Inherits from: Embedder
    """

    name = "hash"

    def __init__(self, dim: int = 256) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            dim (int, optional): IN: dim. Defaults to 256. OUT: Consumed during execution."""

        self.dim = dim

    def embed(self, text: str) -> Vector:
        """Embed.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            text (str): IN: text. OUT: Consumed during execution.
        Returns:
            Vector: OUT: Result of the operation."""

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
    """Sentence transformer embedder.

    Inherits from: Embedder
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            model_name (str, optional): IN: model name. Defaults to 'all-MiniLM-L6-v2'. OUT: Consumed during execution."""

        self.model_name = model_name
        self._model: tp.Any = None

    def _ensure_loaded(self) -> None:
        """Internal helper to ensure loaded.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Embed.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            text (str): IN: text. OUT: Consumed during execution.
        Returns:
            Vector: OUT: Result of the operation."""

        self._ensure_loaded()
        if not text:
            return [0.0] * self.dim
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: tp.Sequence[str]) -> list[Vector]:
        """Embed batch.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            texts (tp.Sequence[str]): IN: texts. OUT: Consumed during execution.
        Returns:
            list[Vector]: OUT: Result of the operation."""

        self._ensure_loaded()
        if not texts:
            return []
        arr = self._model.encode(list(texts), convert_to_numpy=True)
        return [v.tolist() for v in arr]


class OpenAIEmbedder(Embedder):
    """Open aiembedder.

    Inherits from: Embedder
    """

    name = "openai"

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            model_name (str, optional): IN: model name. Defaults to 'text-embedding-3-small'. OUT: Consumed during execution.
            api_key (str | None, optional): IN: api key. Defaults to None. OUT: Consumed during execution.
            base_url (str | None, optional): IN: base url. Defaults to None. OUT: Consumed during execution."""

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
        """Internal helper to ensure client.

        Args:
            self: IN: The instance. OUT: Used for attribute access."""

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
        """Embed.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            text (str): IN: text. OUT: Consumed during execution.
        Returns:
            Vector: OUT: Result of the operation."""

        self._ensure_client()
        if not text:
            return [0.0] * self.dim
        resp = self._client.embeddings.create(input=text, model=self.model_name)
        vec = list(resp.data[0].embedding)
        self.dim = len(vec)
        return vec

    def embed_batch(self, texts: tp.Sequence[str]) -> list[Vector]:
        """Embed batch.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            texts (tp.Sequence[str]): IN: texts. OUT: Consumed during execution.
        Returns:
            list[Vector]: OUT: Result of the operation."""

        self._ensure_client()
        if not texts:
            return []
        resp = self._client.embeddings.create(input=list(texts), model=self.model_name)
        out = [list(d.embedding) for d in resp.data]
        if out:
            self.dim = len(out[0])
        return out


class OllamaEmbedder(Embedder):
    """Ollama embedder.

    Inherits from: Embedder
    """

    name = "ollama"

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            model_name (str, optional): IN: model name. Defaults to 'nomic-embed-text'. OUT: Consumed during execution.
            base_url (str | None, optional): IN: base url. Defaults to None. OUT: Consumed during execution."""

        self.model_name = model_name
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.dim = 768

    def embed(self, text: str) -> Vector:
        """Embed.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            text (str): IN: text. OUT: Consumed during execution.
        Returns:
            Vector: OUT: Result of the operation."""

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
    """Retrieve the default embedder.

    Returns:
        Embedder: OUT: Result of the operation."""

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
    """Reset default embedder."""

    global _DEFAULT_CACHE
    _DEFAULT_CACHE = None


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity.

    Args:
        a (Vector): IN: a. OUT: Consumed during execution.
        b (Vector): IN: b. OUT: Consumed during execution.
    Returns:
        float: OUT: Result of the operation."""

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
