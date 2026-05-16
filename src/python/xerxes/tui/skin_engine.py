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
"""Skin engine — load and apply named TUI themes.

Skins are tiny YAML-ish files at ``$XERXES_HOME/skins/<name>.yaml``
listing roles → hex color. The engine converts hex to ANSI 24-bit
escape codes so prompt_toolkit / rich can render with the right
palette."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._compat_shims import xerxes_subdir_safe

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse a ``#rrggbb`` (or ``rrggbb``) string into an (R, G, B) triple."""
    m = _HEX_RE.match(hex_color.strip())
    if not m:
        raise ValueError(f"invalid hex color: {hex_color!r}")
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_to_ansi_fg(hex_color: str) -> str:
    """Return the ANSI 24-bit foreground escape for ``hex_color``."""
    r, g, b = hex_to_rgb(hex_color)
    return f"\033[38;2;{r};{g};{b}m"


def hex_to_ansi_bg(hex_color: str) -> str:
    """Return the ANSI 24-bit background escape for ``hex_color``."""
    r, g, b = hex_to_rgb(hex_color)
    return f"\033[48;2;{r};{g};{b}m"


_DEFAULT_ROLES: dict[str, str] = {
    "primary": "#f7c948",
    "accent": "#3ddc97",
    "warn": "#ffb86c",
    "error": "#ff6b6b",
    "tool_name": "#6bb1ff",
    "system": "#a695e7",
    "muted": "#999999",
}

_DEFAULT_BRANDING: dict[str, str] = {
    "agent_name": "Xerxes",
    "welcome": "Welcome to Xerxes",
    "goodbye": "see you next session",
    "response_label": "xerxes",
    "prompt_symbol": "›",  # noqa: RUF001 — designed prompt glyph (U+203A), not a typo
    "help_header": "Slash commands",
    "spinner_verbs": "thinking,planning,working,searching,reading,assembling",
}

_BUILTIN_SKINS: dict[str, dict[str, str]] = {
    "default": dict(_DEFAULT_ROLES),
    "high-contrast": {**_DEFAULT_ROLES, "primary": "#ffffff", "accent": "#00ffff", "muted": "#cccccc"},
    "dim": {**_DEFAULT_ROLES, "primary": "#bcbcbc", "accent": "#808080", "muted": "#444444"},
    "ares": {**_DEFAULT_ROLES, "primary": "#ff5e57", "accent": "#ff9f1a", "warn": "#feca57", "tool_name": "#ff7675"},
    "mono": {
        **_DEFAULT_ROLES,
        "primary": "#eeeeee",
        "accent": "#bbbbbb",
        "warn": "#bbbbbb",
        "error": "#bbbbbb",
        "tool_name": "#bbbbbb",
        "system": "#bbbbbb",
        "muted": "#666666",
    },
    "slate": {
        **_DEFAULT_ROLES,
        "primary": "#90a4ae",
        "accent": "#80cbc4",
        "warn": "#ffcc80",
        "tool_name": "#82b1ff",
        "system": "#b39ddb",
        "muted": "#546e7a",
    },
    "daylight": {
        **_DEFAULT_ROLES,
        "primary": "#222831",
        "accent": "#0f4c75",
        "warn": "#fb7e21",
        "error": "#c0392b",
        "tool_name": "#0073e6",
        "system": "#3742fa",
        "muted": "#999999",
    },
}

_BUILTIN_BRANDING: dict[str, dict[str, str]] = {
    "ares": {
        **_DEFAULT_BRANDING,
        "agent_name": "Ares",
        "response_label": "ares",
        "spinner_verbs": "calculating,striking,charging,advancing",
    },
    "mono": {**_DEFAULT_BRANDING, "prompt_symbol": ">"},
}


@dataclass
class Skin:
    """One named theme: role-to-hex mapping plus branding strings.

    ``roles`` covers colors (primary/accent/warn/error/tool_name/system/muted);
    ``branding`` covers text (agent_name, welcome, goodbye, response_label,
    prompt_symbol, help_header, spinner_verbs)."""

    name: str
    roles: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_ROLES))
    branding: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_BRANDING))

    def color(self, role: str) -> str:
        """Return the hex color for ``role``, falling back to the global default."""
        return self.roles.get(role, _DEFAULT_ROLES.get(role, "#ffffff"))

    def fg(self, role: str) -> str:
        """Return the ANSI 24-bit foreground escape for ``role``'s color."""
        return hex_to_ansi_fg(self.color(role))

    def label(self, key: str) -> str:
        """Return a branding string by key (falls back to the default)."""
        return self.branding.get(key, _DEFAULT_BRANDING.get(key, ""))

    def spinner_verbs(self) -> list[str]:
        """Return the comma-separated spinner verbs as a list (always non-empty)."""
        raw = self.label("spinner_verbs")
        return [s.strip() for s in raw.split(",") if s.strip()] or ["working"]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict snapshot (name + roles + branding)."""
        return {"name": self.name, "roles": dict(self.roles), "branding": dict(self.branding)}


class SkinEngine:
    """Discover available skins and persist the active one.

    Looks up skin files in ``$XERXES_HOME/skins/<name>.{yaml,skin}``
    while also exposing the seven built-in skins (``default``,
    ``high-contrast``, ``dim``, ``ares``, ``mono``, ``slate``,
    ``daylight``)."""

    def __init__(self, base_dir: Path | None = None) -> None:
        """Anchor the engine to ``base_dir`` (default ``$XERXES_HOME/skins``)."""
        self._base_dir = base_dir or xerxes_subdir_safe("skins")
        self._active = "default"

    @property
    def base_dir(self) -> Path:
        """Return the on-disk directory scanned for user skins."""
        return self._base_dir

    def available(self) -> list[str]:
        """Return sorted skin names from built-ins union user files."""
        names = set(_BUILTIN_SKINS.keys())
        if self._base_dir.is_dir():
            for path in self._base_dir.glob("*.yaml"):
                names.add(path.stem)
            for path in self._base_dir.glob("*.skin"):
                names.add(path.stem)
        return sorted(names)

    def load(self, name: str) -> Skin:
        """Materialize a :class:`Skin` by name; raises ``KeyError`` if missing."""
        # Built-ins win first; falls through to disk lookups.
        if name in _BUILTIN_SKINS:
            return Skin(
                name=name,
                roles=dict(_BUILTIN_SKINS[name]),
                branding=dict(_BUILTIN_BRANDING.get(name, _DEFAULT_BRANDING)),
            )
        for suffix in (".yaml", ".skin"):
            path = self._base_dir / f"{name}{suffix}"
            if path.is_file():
                return self._load_file(name, path)
        raise KeyError(f"skin not found: {name}")

    def _load_file(self, name: str, path: Path) -> Skin:
        """Parse a one-key-per-line YAML-ish skin file into a :class:`Skin`."""
        roles: dict[str, str] = dict(_DEFAULT_ROLES)
        branding: dict[str, str] = dict(_DEFAULT_BRANDING)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not value:
                continue
            if key in _DEFAULT_BRANDING:
                branding[key] = value
            else:
                roles[key] = value
        return Skin(name=name, roles=roles, branding=branding)

    def save(self, skin: Skin) -> Path:
        """Persist ``skin`` to ``<base_dir>/<name>.yaml`` and return its path."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        path = self._base_dir / f"{skin.name}.yaml"
        lines = [f"{k}: {v}" for k, v in skin.roles.items()]
        lines += [f"{k}: {v}" for k, v in skin.branding.items() if v]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def set_active(self, name: str) -> Skin:
        """Mark ``name`` as the active skin and return its loaded :class:`Skin`."""
        skin = self.load(name)
        self._active = skin.name
        return skin

    def active(self) -> Skin:
        """Return a fresh :class:`Skin` for the currently active skin name."""
        return self.load(self._active)


__all__ = ["Skin", "SkinEngine", "hex_to_ansi_bg", "hex_to_ansi_fg", "hex_to_rgb"]
