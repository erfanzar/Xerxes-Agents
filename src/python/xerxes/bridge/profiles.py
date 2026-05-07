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
"""Provider profile management for the Xerxes bridge.

This module handles saving, loading, updating, and deleting LLM provider
profiles (base URL, API key, model, sampling parameters). It also provides
model list fetching from provider APIs.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..core.paths import xerxes_home

PROFILES_DIR = xerxes_home()
PROFILES_FILE = PROFILES_DIR / "profiles.json"


def _load_store() -> dict[str, Any]:
    """Load the profiles JSON store from disk.

    Returns:
        dict[str, Any]: OUT: Parsed store with ``"active"`` and ``"profiles"`` keys,
            or a default empty store if the file is missing or unreadable.
    """
    if PROFILES_FILE.exists():
        try:
            return json.loads(PROFILES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"active": None, "profiles": {}}


def _save_store(store: dict[str, Any]) -> None:
    """Persist the profiles JSON store to disk.

    Args:
        store (dict[str, Any]): IN: Profile store to persist. OUT: Serialized
            and written to ``PROFILES_FILE``.
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_FILE.write_text(json.dumps(store, indent=2, ensure_ascii=False))


def list_profiles() -> list[dict[str, Any]]:
    """List all stored profiles, marking the active one.

    Returns:
        list[dict[str, Any]]: OUT: Profile dictionaries with an ``"active"`` flag.
    """
    store = _load_store()
    active = store.get("active")
    result = []
    for name, profile in store.get("profiles", {}).items():
        result.append(
            {
                **profile,
                "active": name == active,
            }
        )
    return result


def get_active_profile() -> dict[str, Any] | None:
    """Return the currently active profile, if any.

    Returns:
        dict[str, Any] | None: OUT: The active profile dictionary, or ``None``.
    """
    store = _load_store()
    active = store.get("active")
    if active and active in store.get("profiles", {}):
        return store["profiles"][active]
    return None


SAMPLING_PARAMS = {
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "min_p",
}


def save_profile(
    name: str,
    base_url: str,
    api_key: str,
    model: str,
    provider: str = "",
    sampling: dict[str, Any] | None = None,
    set_active: bool = True,
) -> dict[str, Any]:
    """Save or update a provider profile.

    Args:
        name (str): IN: Profile name. OUT: Used as the dictionary key.
        base_url (str): IN: API base URL. OUT: Stored after stripping trailing slashes.
        api_key (str): IN: API authentication key. OUT: Stored.
        model (str): IN: Default model name. OUT: Stored.
        provider (str): IN: Provider identifier. OUT: Auto-detected from base_url
            if not provided.
        sampling (dict[str, Any] | None): IN: Sampling parameters. OUT: Defaults
            to existing values when updating.
        set_active (bool): IN: Whether to make this profile active. OUT: Updates
            the ``"active"`` key in the store.

    Returns:
        dict[str, Any]: OUT: The saved profile dictionary.
    """
    store = _load_store()
    existing = store.get("profiles", {}).get(name, {})
    profile = {
        "name": name,
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model": model,
        "provider": provider or _guess_provider(base_url),
        "sampling": sampling if sampling is not None else existing.get("sampling", {}),
    }
    store.setdefault("profiles", {})[name] = profile
    if set_active:
        store["active"] = name
    _save_store(store)
    return profile


def update_sampling(name: str, sampling: dict[str, Any]) -> dict[str, Any] | None:
    """Update sampling parameters for an existing profile.

    Args:
        name (str): IN: Profile name. OUT: Looked up in the store.
        sampling (dict[str, Any]): IN: Sampling parameter updates. OUT: Merged
            with existing values; ``None`` values cause deletion.

    Returns:
        dict[str, Any] | None: OUT: Updated profile, or ``None`` if not found.
    """
    store = _load_store()
    if name not in store.get("profiles", {}):
        return None
    existing = store["profiles"][name].get("sampling", {})
    for k, v in sampling.items():
        if k in SAMPLING_PARAMS:
            if v is None:
                existing.pop(k, None)
            else:
                existing[k] = v
    store["profiles"][name]["sampling"] = existing
    _save_store(store)
    return store["profiles"][name]


def update_active_model(model: str) -> dict[str, Any] | None:
    """Update the active profile's default model.

    Args:
        model (str): IN: Model identifier to persist. OUT: Stored on the active profile.

    Returns:
        dict[str, Any] | None: OUT: Updated active profile, or ``None`` when no
            active profile exists.
    """
    store = _load_store()
    active = store.get("active")
    if not active or active not in store.get("profiles", {}):
        return None
    store["profiles"][active]["model"] = model
    _save_store(store)
    return store["profiles"][active]


def delete_profile(name: str) -> bool:
    """Delete a profile by name.

    Args:
        name (str): IN: Profile name to delete. OUT: Removed from the store.

    Returns:
        bool: OUT: ``True`` if the profile existed and was deleted.
    """
    store = _load_store()
    if name in store.get("profiles", {}):
        del store["profiles"][name]
        if store.get("active") == name:
            store["active"] = None
        _save_store(store)
        return True
    return False


def set_active(name: str) -> bool:
    """Set the active profile.

    Args:
        name (str): IN: Profile name. OUT: Set as active if it exists.

    Returns:
        bool: OUT: ``True`` if the profile exists and was activated.
    """
    store = _load_store()
    if name in store.get("profiles", {}):
        store["active"] = name
        _save_store(store)
        return True
    return False


_MINIMAX_MODELS = [
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.7-flashspeed",
    "MiniMax-Text-01",
    "MiniMax-Text-01-MiniApp",
    "abab6.5s-chat",
    "abab6.5-chat",
    "abab6-chat",
    "abab5.5s-chat",
    "abab5.5-chat",
    "abab5-chat",
]
_PROVIDERS_WITHOUT_MODELS = {"minimax", "minimaxi"}


def fetch_models(base_url: str, api_key: str) -> list[str]:
    """Fetch the list of available models from a provider's ``/models`` endpoint.

    Args:
        base_url (str): IN: Provider base URL. OUT: Used to construct the request URL.
        api_key (str): IN: API key for authentication. OUT: Sent in the
            ``Authorization`` header.

    Returns:
        list[str]: OUT: Sorted list of model identifiers.

    Raises:
        httpx.HTTPStatusError: On HTTP errors (with a fallback for MiniMax).
        httpx.RequestError: On network-level request failures.
    """
    url = f"{base_url.rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = httpx.get(url, headers=headers, timeout=3.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404 and _guess_provider(base_url) in _PROVIDERS_WITHOUT_MODELS:
            return sorted(_MINIMAX_MODELS)
        raise
    except httpx.RequestError:
        raise

    data = resp.json()
    models = []
    for item in data.get("data", []):
        model_id = item.get("id", "")
        if model_id:
            models.append(model_id)
    return sorted(models)


def _guess_provider(base_url: str) -> str:
    """Guess the provider name from a base URL.

    Args:
        base_url (str): IN: API base URL. OUT: Matched against known provider
            substrings.

    Returns:
        str: OUT: Provider name, or ``"custom"`` if no match.
    """
    url = base_url.lower()
    if "openai" in url:
        return "openai"
    if "anthropic" in url:
        return "anthropic"
    if "localhost" in url or "127.0.0.1" in url:
        if "11434" in url:
            return "ollama"
        return "local"
    if "deepseek" in url:
        return "deepseek"
    if "together" in url:
        return "together"
    if "groq" in url:
        return "groq"
    if "kimi" in url or "moonshot" in url:
        return "kimi"
    if "minimax" in url or "minimaxi" in url:
        return "minimax"
    return "custom"
