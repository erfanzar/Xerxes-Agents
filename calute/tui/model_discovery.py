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

"""Helpers for discovering available models from configured TUI endpoints.

Provides functions to query remote LLM providers and local Ollama instances
for their available model catalogues, used by the TUI during startup and on
explicit ``/models`` commands.
"""

from __future__ import annotations

import typing as tp

import httpx

from ..llms import create_llm


def _extract_model_ids(data: tp.Any) -> list[str]:
    """Extract model identifiers from OpenAI-compatible list responses.

    Handles both raw dictionaries (e.g. from ``httpx`` JSON) and SDK
    response objects that expose a ``.data`` attribute containing model
    entries.  Each entry may be a dict with an ``"id"`` key or an object
    with an ``id`` attribute.

    Args:
        data: The response payload from a ``models.list()`` call or an
            equivalent JSON dict.  May also be ``None``.

    Returns:
        A sorted, deduplicated list of non-empty model ID strings found
        in the payload.
    """
    items = getattr(data, "data", data)
    results: list[str] = []
    if isinstance(items, dict):
        items = items.get("data", [])
    for item in items or []:
        if isinstance(item, dict):
            model_id = item.get("id")
        else:
            model_id = getattr(item, "id", None)
        if isinstance(model_id, str) and model_id:
            results.append(model_id)
    return sorted(set(results))


def discover_available_models(
    provider: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    """Discover available models for a provider or endpoint.

    For cloud providers (``openai``, ``anthropic``, ``gemini``), an LLM
    client is instantiated and its SDK ``models.list()`` endpoint is
    called.  For ``ollama`` / ``local``, the Ollama REST API at
    ``/api/tags`` is queried directly via ``httpx``.

    Args:
        provider: Canonical or aliased provider name (e.g. ``"openai"``,
            ``"ollama"``).
        model: Optional model hint used to seed the LLM client or as a
            fallback return value when the provider does not expose a
            model listing endpoint.
        api_key: Optional API key forwarded to the LLM client constructor.
        base_url: Optional base URL for the provider; Ollama defaults to
            ``http://localhost:11434`` when not supplied.

    Returns:
        A sorted, deduplicated list of model name strings.  May be empty if
        the provider returned no models and no *model* hint was given.

    Raises:
        httpx.HTTPStatusError: If the Ollama ``/api/tags`` request fails.
        Exception: Propagated from the SDK ``models.list()`` call for cloud
            providers.
    """
    provider = provider.strip().lower()

    if provider in {"openai", "anthropic", "claude", "gemini", "google"}:
        llm_kwargs: dict[str, tp.Any] = {}
        if model:
            llm_kwargs["model"] = model
        if api_key:
            llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        llm = create_llm(provider, **llm_kwargs)
        if hasattr(llm, "client") and hasattr(llm.client, "models"):
            return _extract_model_ids(llm.client.models.list())
        return [model] if model else []

    if provider in {"ollama", "local"}:
        endpoint = (base_url or "http://localhost:11434").rstrip("/")
        with httpx.Client(base_url=endpoint, timeout=10.0) as client:
            response = client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        names = [
            entry.get("name")
            for entry in models
            if isinstance(entry, dict) and isinstance(entry.get("name"), str) and entry.get("name")
        ]
        return sorted(set(names))

    return [model] if model else []
