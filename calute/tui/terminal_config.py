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

"""Persistent terminal configuration for Calute TUI sessions.

Provides the data-classes and file-backed store that persist user
preferences (provider, model, API keys, sampling parameters, prompt
profiles, etc.) across TUI sessions.  Configuration is stored as JSON
at the XDG-compliant path
``$XDG_CONFIG_HOME/calute/terminal_profiles.json`` (defaulting to
``~/.config/calute/terminal_profiles.json``).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SAMPLING_PARAM_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)


def default_terminal_config_path() -> Path:
    """Return the default on-disk config path for the terminal client.

    Respects the ``XDG_CONFIG_HOME`` environment variable when set;
    otherwise falls back to ``~/.config/calute/terminal_profiles.json``.

    Returns:
        A ``pathlib.Path`` pointing to the terminal profiles JSON file.
    """
    xdg_dir = os.environ.get("XDG_CONFIG_HOME")
    base_dir = Path(xdg_dir).expanduser() if xdg_dir else Path.home() / ".config"
    return base_dir / "calute" / "terminal_profiles.json"


@dataclass
class TerminalProfile:
    """Saved connection profile for the Calute terminal app.

    Captures every user-configurable setting needed to establish an LLM
    session in the TUI: provider, model, credentials, prompt profile,
    tool toggles, and sampling parameters.

    Attributes:
        name: Human-readable profile name, also used as the dictionary
            key inside ``TerminalSettings.profiles``.
        provider: Canonical LLM provider identifier (e.g. ``"openai"``).
        model: Model name string, or ``None`` to accept the provider
            default.
        api_key: API key/secret for cloud providers, or ``None``.
        base_url: Custom base URL for the provider endpoint, or ``None``
            to use the provider default.
        prompt_profile: Active prompt profile mode (``"full"``,
            ``"compact"``, ``"minimal"``, or ``"none"``).
        agent_id: Default agent identifier used at startup.
        instructions: System-level instructions for the default agent.
        include_tools: Whether built-in tools are attached to the agent.
        power_tools_enabled: Whether high-power operator tools are active.
            Defaults to ``True`` for newly created profiles.
        available_models: Cached list of model names discovered from the
            endpoint.
        sampling_params: Overrides for sampling parameters such as
            ``temperature`` and ``top_p``.
        updated_at: ISO 8601 timestamp of the last profile update.
    """

    name: str = "default"
    provider: str = "ollama"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    prompt_profile: str = "full"
    agent_id: str = "assistant"
    instructions: str = ""
    include_tools: bool = True
    power_tools_enabled: bool = True
    available_models: list[str] = field(default_factory=list)
    sampling_params: dict[str, float | int] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def from_dict(cls, data: dict) -> TerminalProfile:
        """Build a profile from a JSON-compatible dictionary.

        Unknown keys are silently discarded.  ``sampling_params`` are
        filtered to only include recognised parameter names with numeric
        values.  ``prompt_profile`` is normalised and validated against the
        allowed set; invalid values fall back to ``"full"``.

        Args:
            data: Dictionary typically produced by ``json.loads`` of a
                stored profile entry.

        Returns:
            A new ``TerminalProfile`` instance populated from *data*.
        """
        allowed = {
            "name",
            "provider",
            "model",
            "api_key",
            "base_url",
            "prompt_profile",
            "agent_id",
            "instructions",
            "include_tools",
            "power_tools_enabled",
            "available_models",
            "sampling_params",
            "updated_at",
        }
        payload = {key: value for key, value in data.items() if key in allowed}
        sampling_params = payload.get("sampling_params")
        if not isinstance(sampling_params, dict):
            payload["sampling_params"] = {}
        else:
            payload["sampling_params"] = {
                key: value
                for key, value in sampling_params.items()
                if key in SAMPLING_PARAM_KEYS and isinstance(value, int | float)
            }
        payload["power_tools_enabled"] = bool(payload.get("power_tools_enabled", True))
        prompt_profile = payload.get("prompt_profile")
        if not isinstance(prompt_profile, str) or prompt_profile.strip().lower() not in {
            "full",
            "compact",
            "minimal",
            "none",
        }:
            payload["prompt_profile"] = "full"
        else:
            payload["prompt_profile"] = prompt_profile.strip().lower()
        return cls(**payload)

    def to_dict(self) -> dict:
        """Serialize the profile to a JSON-compatible dictionary.

        Returns:
            A plain ``dict`` containing all profile fields, suitable for
            ``json.dumps``.
        """
        return asdict(self)


@dataclass
class TerminalSettings:
    """Root document persisted for terminal profiles.

    Acts as the top-level container that is serialised to and deserialised
    from the JSON configuration file.

    Attributes:
        last_profile: Name of the most recently used profile, used as
            the default on the next launch.
        profiles: Mapping of profile names to ``TerminalProfile`` instances.
    """

    last_profile: str = "default"
    profiles: dict[str, TerminalProfile] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> TerminalSettings:
        """Build settings from a JSON-compatible dictionary.

        Each value under the ``"profiles"`` key is parsed via
        ``TerminalProfile.from_dict``; non-dict entries are skipped.

        Args:
            data: Dictionary produced by ``json.loads`` of the stored
                settings file.

        Returns:
            A new ``TerminalSettings`` instance.
        """
        raw_profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
        profiles = {
            name: TerminalProfile.from_dict(profile_data)
            for name, profile_data in raw_profiles.items()
            if isinstance(profile_data, dict)
        }
        return cls(last_profile=data.get("last_profile", "default"), profiles=profiles)

    def to_dict(self) -> dict:
        """Serialize the settings to a JSON-compatible dictionary.

        Returns:
            A plain ``dict`` containing ``last_profile`` and a nested
            ``profiles`` mapping, ready for ``json.dumps``.
        """
        return {
            "last_profile": self.last_profile,
            "profiles": {name: profile.to_dict() for name, profile in self.profiles.items()},
        }


class TerminalConfigStore:
    """File-backed store for terminal endpoint and model selections.

    Reads and writes the JSON configuration file atomically, using a
    temporary file with restricted permissions (``0o600``) to prevent
    accidental credential exposure.

    Attributes:
        path: Resolved ``pathlib.Path`` to the configuration file.
    """

    def __init__(self, path: str | Path | None = None):
        """Initialise the config store.

        Args:
            path: Explicit path to the JSON file.  When ``None``, the
                default XDG-compliant location returned by
                ``default_terminal_config_path`` is used.
        """
        self.path = Path(path).expanduser() if path is not None else default_terminal_config_path()

    def load(self) -> TerminalSettings:
        """Load terminal settings from disk.

        Returns a default ``TerminalSettings`` instance when the file does
        not exist or cannot be parsed.

        Returns:
            The loaded ``TerminalSettings``, or a fresh default.
        """
        if not self.path.exists():
            return TerminalSettings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return TerminalSettings()
        return TerminalSettings.from_dict(data if isinstance(data, dict) else {})

    def save(self, settings: TerminalSettings) -> None:
        """Persist terminal settings atomically with private file permissions.

        Writes to a temporary ``.tmp`` sibling, sets ``0o600`` permissions,
        then atomically renames over the target file.  Parent directories
        are created as needed.

        Args:
            settings: The ``TerminalSettings`` instance to serialise.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(settings.to_dict(), indent=2, ensure_ascii=False)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(payload, encoding="utf-8")
        os.chmod(temp_path, 0o600)
        temp_path.replace(self.path)
        os.chmod(self.path, 0o600)

    def get_profile(self, name: str | None = None) -> TerminalProfile | None:
        """Load a named profile, defaulting to the last-used profile.

        Args:
            name: Profile name to look up.  When ``None``, the
                ``last_profile`` value from the settings is used.

        Returns:
            The matching ``TerminalProfile``, or ``None`` if no profile
            with the resolved name exists.
        """
        settings = self.load()
        resolved_name = name or settings.last_profile
        return settings.profiles.get(resolved_name)

    def upsert_profile(self, profile: TerminalProfile, *, make_default: bool = True) -> None:
        """Insert or update a profile and persist the change immediately.

        The profile's ``updated_at`` timestamp is refreshed to the current
        UTC time before saving.

        Args:
            profile: The profile to upsert.  Keyed by ``profile.name``.
            make_default: When ``True`` (default), also sets
                ``last_profile`` so this profile is loaded on the next
                launch.
        """
        settings = self.load()
        profile.updated_at = datetime.now(UTC).isoformat()
        settings.profiles[profile.name] = profile
        if make_default:
            settings.last_profile = profile.name
        self.save(settings)

    def list_profiles(self) -> list[TerminalProfile]:
        """Return all saved profiles sorted alphabetically by name.

        Returns:
            A list of ``TerminalProfile`` instances, ordered by profile
            name.
        """
        settings = self.load()
        return [settings.profiles[name] for name in sorted(settings.profiles)]
