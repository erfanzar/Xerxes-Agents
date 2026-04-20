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


"""Provider profile management.

Profiles are stored in ``~/.xerxes/profiles.json`` as a JSON object::

    {
        "active": "my-ollama",
        "profiles": {
            "my-ollama": {
                "name": "my-ollama",
                "base_url": "http://localhost:11434/v1",
                "api_key": "sk-...",
                "model": "llama3",
                "provider": "ollama"
            },
            "openai-prod": { ... }
        }
    }
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..core.paths import xerxes_home

PROFILES_DIR = xerxes_home()
PROFILES_FILE = PROFILES_DIR / "profiles.json"


def _load_store() -> dict[str, Any]:
    """Load the profiles store from disk."""
    if PROFILES_FILE.exists():
        try:
            return json.loads(PROFILES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"active": None, "profiles": {}}


def _save_store(store: dict[str, Any]) -> None:
    """Persist the profiles store to disk."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_FILE.write_text(json.dumps(store, indent=2, ensure_ascii=False))


def list_profiles() -> list[dict[str, Any]]:
    """Return all saved profiles with an ``active`` flag."""
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
    """Return the active profile, or None if none is set."""
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
    """Save a provider profile and optionally set it as active."""
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
    """Update sampling params for an existing profile. Returns the profile or None."""
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


def delete_profile(name: str) -> bool:
    """Delete a profile by name. Returns True if it existed."""
    store = _load_store()
    if name in store.get("profiles", {}):
        del store["profiles"][name]
        if store.get("active") == name:
            store["active"] = None
        _save_store(store)
        return True
    return False


def set_active(name: str) -> bool:
    """Set a profile as active. Returns True if it exists."""
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
    """Fetch available models from an OpenAI-compatible /models endpoint.

    Raises on network or HTTP errors so callers can surface the issue to the user.
    Returns known models for providers that don't expose a /models endpoint.
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
    """Guess provider name from the base URL."""
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
