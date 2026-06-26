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
"""Persistent LLM provider profile store used by the bridge and daemon.

Each profile bundles a base URL, API key, default model, provider name, and
optional sampling overrides, persisted as JSON under
``$XERXES_HOME/profiles.json``. Exactly one profile is marked ``"active"`` at
a time and is loaded by the daemon's :class:`RuntimeManager` on every
reload. ``fetch_models`` queries the provider's ``GET /models`` endpoint,
falling back to the built-in MiniMax catalogue when the provider doesn't
implement it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any

import httpx

from ..core.paths import xerxes_home

PROFILES_DIR = xerxes_home()
PROFILES_FILE = PROFILES_DIR / "profiles.json"
CLAUDE_CODE_PROFILE_NAME = "cc"


def _load_store() -> dict[str, Any]:
    """Read the profiles JSON file or return an empty store on miss/parse error."""
    if PROFILES_FILE.exists():
        try:
            return json.loads(PROFILES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"active": None, "profiles": {}}


def _builtin_profiles() -> dict[str, dict[str, Any]]:
    """Return profiles that ship with Xerxes and cannot disappear."""
    return {
        CLAUDE_CODE_PROFILE_NAME: {
            "name": CLAUDE_CODE_PROFILE_NAME,
            "base_url": "claude-code://local",
            "api_key": "",
            "model": _CLAUDE_CODE_DEFAULT_MODEL,
            "provider": "claude-code",
            "sampling": {},
        }
    }


def _merged_profiles(store: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return built-in profiles plus user profiles, letting user profiles override."""
    merged = _builtin_profiles()
    for name, profile in store.get("profiles", {}).items():
        if isinstance(profile, dict):
            merged[str(name)] = profile
    return merged


def _active_profile_name(store: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> str:
    """Resolve the active profile, falling back to the built-in Claude Code profile."""
    active = str(store.get("active") or "")
    if active in profiles:
        return active
    return CLAUDE_CODE_PROFILE_NAME


def _ensure_writable_profile(store: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Return a mutable stored profile, copying a built-in profile when needed."""
    profiles = store.setdefault("profiles", {})
    if name in profiles:
        return profiles[name]
    builtin = _builtin_profiles().get(name)
    if builtin is None:
        return None
    profiles[name] = dict(builtin)
    profiles[name]["sampling"] = dict(builtin.get("sampling", {}))
    return profiles[name]


def _save_store(store: dict[str, Any]) -> None:
    """Write the profiles store to disk, creating parent directories as needed."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_FILE.write_text(json.dumps(store, indent=2, ensure_ascii=False))


def list_profiles() -> list[dict[str, Any]]:
    """Return every stored profile with an extra ``"active"`` boolean per record."""
    store = _load_store()
    profiles = _merged_profiles(store)
    active = _active_profile_name(store, profiles)
    result = []
    for name, profile in profiles.items():
        result.append(
            {
                **profile,
                "active": name == active,
            }
        )
    return result


def get_active_profile() -> dict[str, Any] | None:
    """Return the profile marked active, or ``None`` if no active profile exists."""
    store = _load_store()
    profiles = _merged_profiles(store)
    return profiles.get(_active_profile_name(store, profiles))


SAMPLING_PARAMS = {
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "min_p",
    "thinking",
    "reasoning_effort",
    "thinking_budget",
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
    """Create or replace a profile and optionally mark it active.

    Args:
        name: Profile key.
        base_url: API base URL; trailing slashes are stripped.
        api_key: Provider API key (stored verbatim).
        model: Default model id.
        provider: Provider tag; auto-detected from ``base_url`` when empty.
        sampling: Sampling overrides; preserved from the prior profile if ``None``.
        set_active: When true, mark the profile active after saving.

    Returns:
        The saved profile dict.
    """
    store = _load_store()
    existing = store.get("profiles", {}).get(name, {})
    provider = provider.strip().lower()
    provider = {"claude_code": "claude-code"}.get(provider, provider)
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
    """Merge sampling overrides into an existing profile.

    Only keys in :data:`SAMPLING_PARAMS` are accepted; ``None`` deletes a key.
    Returns the updated profile, or ``None`` if no such profile exists.
    """
    store = _load_store()
    profile = _ensure_writable_profile(store, name)
    if profile is None:
        return None
    existing = profile.get("sampling", {})
    for k, v in sampling.items():
        if k in SAMPLING_PARAMS:
            if v is None:
                existing.pop(k, None)
            else:
                existing[k] = v
    profile["sampling"] = existing
    _save_store(store)
    return profile


def update_active_model(model: str) -> dict[str, Any] | None:
    """Persist ``model`` as the active profile's default; return it, or ``None``."""
    store = _load_store()
    profiles = _merged_profiles(store)
    active = _active_profile_name(store, profiles)
    profile = _ensure_writable_profile(store, active)
    if profile is None:
        return None
    profile["model"] = model
    _save_store(store)
    return profile


def delete_profile(name: str) -> bool:
    """Remove a profile (clearing ``"active"`` if it was the active one)."""
    store = _load_store()
    if name in store.get("profiles", {}):
        del store["profiles"][name]
        if store.get("active") == name:
            store["active"] = None
        _save_store(store)
        return True
    return False


def set_active(name: str) -> bool:
    """Mark ``name`` as the active profile; return ``False`` if it doesn't exist."""
    store = _load_store()
    if name in _merged_profiles(store):
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
_CLAUDE_CODE_MODELS_ENV = "CLAUDE_CODE_MODELS"
_CLAUDE_CODE_CLI_ENV = "CLAUDE_CODE_CLI"
_CLAUDE_CODE_DEFAULT_MODEL = "claude-code/default"


def _split_declared_models(raw: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\s,]+", raw) if part.strip()]


def _claude_code_model_option(model: str) -> str | None:
    clean = model.strip().strip("'\"")
    if not clean:
        return None
    if clean in {"default", "auto", _CLAUDE_CODE_DEFAULT_MODEL}:
        return _CLAUDE_CODE_DEFAULT_MODEL
    if clean.startswith("claude-code/"):
        return clean
    if "/" in clean:
        return None
    if re.fullmatch(r"[A-Za-z0-9_.:-]+", clean) is None:
        return None
    return f"claude-code/{clean}"


def _with_claude_code_default(models: list[str]) -> list[str]:
    return [_CLAUDE_CODE_DEFAULT_MODEL] + [model for model in models if model != _CLAUDE_CODE_DEFAULT_MODEL]


def _dedupe_claude_code_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        option = _claude_code_model_option(model)
        if option and option not in seen:
            seen.add(option)
            result.append(option)
    return result


def _claude_code_models_from_help(help_text: str) -> list[str]:
    section: list[str] = []
    capture = False
    for line in help_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--model "):
            capture = True
        elif capture and stripped.startswith("--"):
            break
        if capture:
            section.append(stripped)
    text = " ".join(section)
    if not text:
        return []
    quoted = re.findall(r"(?<![A-Za-z0-9])['\"]([A-Za-z0-9_.:-]+)['\"]", text)
    return _dedupe_claude_code_models(quoted)


def fetch_claude_code_models() -> list[str]:
    """Discover Claude Code model aliases from local configuration/CLI help."""
    env_models = os.environ.get(_CLAUDE_CODE_MODELS_ENV, "")
    if env_models:
        return _with_claude_code_default(_dedupe_claude_code_models(_split_declared_models(env_models)))

    command = os.environ.get(_CLAUDE_CODE_CLI_ENV, "claude")
    executable = command if os.path.sep in command else shutil.which(command)
    if not executable:
        return []
    try:
        proc = subprocess.run(
            [executable, "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return _with_claude_code_default(_claude_code_models_from_help(f"{proc.stdout}\n{proc.stderr}"))


def fetch_models(base_url: str, api_key: str) -> list[str]:
    """Return the provider's available model ids, sorted.

    MiniMax-flavoured providers don't implement ``GET /models``; a 404 there
    falls back to a hard-coded MiniMax catalogue rather than raising.

    Raises:
        httpx.HTTPStatusError: For non-404 HTTP errors.
        httpx.RequestError: On network-level failures.
    """
    if _guess_provider(base_url) == "claude-code":
        return fetch_claude_code_models()

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
    """Best-effort substring match of ``base_url`` to a known provider tag.

    Order matters — more specific paths are checked first so we don't
    collapse ``api.kimi.com/coding/v1`` (the Kimi Code endpoint) into the
    generic Kimi chat provider.
    """
    url = base_url.lower()
    if url.startswith("claude-code://"):
        return "claude-code"
    if "openrouter.ai" in url:
        return "openrouter"
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
    # Kimi Code — coding-specialised, distinct API host + model.
    if "kimi.com/coding" in url:
        return "kimi-code"
    if "kimi" in url or "moonshot" in url:
        return "kimi"
    if "minimax" in url or "minimaxi" in url:
        return "minimax"
    if "z.ai" in url or "zhipu" in url or "bigmodel" in url:
        return "zhipu"
    return "custom"
