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
"""Agent specification parser and resolver.

This module provides YAML-based agent specification loading, inheritance
(via ``extend``), and deep merging. It supports variable substitution in
system prompts and validates spec versions.

Main exports:
    - load_agent_spec: Load and resolve an agent.yaml file into a ResolvedAgentSpec.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from ..core.errors import AgentSpecError

logger = logging.getLogger(__name__)

DEFAULT_AGENT_SPEC_VERSION = "1"
SUPPORTED_AGENT_SPEC_VERSIONS = (DEFAULT_AGENT_SPEC_VERSION,)


class Inherit(NamedTuple):
    """Sentinel value indicating a field should be inherited from a base spec."""

    pass


INHERIT = Inherit()


def _is_inherit(value: Any) -> bool:
    """Return whether a value is the inheritance sentinel."""
    return value is INHERIT or isinstance(value, Inherit)


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge ``override`` into ``base``.

    Dicts are merged key-by-key, lists are concatenated, and scalars
    are replaced. :data:`INHERIT` on the override side keeps ``base``.
    """
    if _is_inherit(override):
        return base
    if isinstance(base, dict) and isinstance(override, dict):
        result = dict(base)
        for key, val in override.items():
            result[key] = _deep_merge(result.get(key), val) if key in result else val
        return result
    if isinstance(base, list) and isinstance(override, list):
        return list(base) + list(override)
    return override


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedAgentSpec:
    """Fully resolved agent specification after inheritance and substitution.

    Attributes:
        name (str): Resolved agent name.
        system_prompt (str): Resolved system prompt text.
        model (str | None): Model identifier, if specified.
        when_to_use (str): Description of when to use this agent.
        tools (list[str]): List of tool names.
        allowed_tools (list[str] | None): Allowed tool whitelist, if any.
        exclude_tools (list[str]): Tools to exclude.
        subagents (dict[str, SubagentSpec]): Named sub-agent specs.
        max_depth (int): Maximum delegation depth.
        isolation (str): Isolation strategy (e.g., ``"worktree"``).
        source (str): Origin of the spec (default ``"yaml"``).
    """

    name: str
    system_prompt: str
    model: str | None
    when_to_use: str
    tools: list[str]
    allowed_tools: list[str] | None
    exclude_tools: list[str]
    subagents: dict[str, SubagentSpec]
    max_depth: int
    isolation: str
    source: str = "yaml"


@dataclass(frozen=True, slots=True)
class SubagentSpec:
    """Specification for a sub-agent referenced by path and description.

    Attributes:
        path (Path): Absolute filesystem path to the sub-agent spec.
        description (str): Human-readable description.
    """

    path: Path
    description: str


@dataclass
class AgentSpec:
    """Raw agent specification before resolution.

    Fields may contain :data:`INHERIT` to indicate they should be taken from a
    parent spec during resolution.

    Attributes:
        version (str): Spec version string.
        extend (str | None): Path or ``"default"`` to a base spec to extend.
        name (str | Inherit): Agent name or inherit sentinel.
        system_prompt (str | Inherit): System prompt text or inherit sentinel.
        system_prompt_path (str | Inherit): Path to an external prompt file.
        system_prompt_args (dict[str, str]): Variable substitutions for prompts.
        model (str | None | Inherit): Model identifier or inherit sentinel.
        when_to_use (str | None | Inherit): Usage description or inherit sentinel.
        tools (list[str] | None | Inherit): Tool list or inherit sentinel.
        allowed_tools (list[str] | None | Inherit): Allowed tools or inherit sentinel.
        exclude_tools (list[str] | None | Inherit): Excluded tools or inherit sentinel.
        subagents (dict[str, SubagentSpec] | None | Inherit): Sub-agents or inherit sentinel.
        max_depth (int | Inherit): Max depth or inherit sentinel.
        isolation (str | Inherit): Isolation mode or inherit sentinel.
    """

    version: str = DEFAULT_AGENT_SPEC_VERSION
    extend: str | None = None
    name: str | Inherit = INHERIT
    system_prompt: str | Inherit = INHERIT
    system_prompt_path: str | Inherit = INHERIT
    system_prompt_args: dict[str, str] = field(default_factory=dict)
    model: str | None | Inherit = INHERIT
    when_to_use: str | None | Inherit = INHERIT
    tools: list[str] | None | Inherit = INHERIT
    allowed_tools: list[str] | None | Inherit = INHERIT
    exclude_tools: list[str] | None | Inherit = INHERIT
    subagents: dict[str, SubagentSpec] | None | Inherit = INHERIT
    max_depth: int | Inherit = INHERIT
    isolation: str | Inherit = INHERIT


@dataclass
class _RawSubagentSpec:
    """Intermediate representation used during YAML parsing."""

    path: str = ""
    description: str = ""


def _parse_agent_yaml(path: Path) -> AgentSpec:
    """Parse a single ``agent.yaml`` file into an unresolved :class:`AgentSpec`.

    Fields absent from the YAML stay as :data:`INHERIT` so the merger in
    :func:`_load_agent_spec_recursive` can fill them from a parent spec.

    Raises:
        AgentSpecError: If the file is missing, YAML is invalid, or the
            ``version`` field is not in :data:`SUPPORTED_AGENT_SPEC_VERSIONS`.
    """
    if not path.exists():
        raise AgentSpecError(f"Agent spec file not found: {path}")

    text = path.read_text(encoding="utf-8")

    if not _HAS_YAML:
        raise AgentSpecError("PyYAML is required to load agent.yaml files")

    try:
        data: dict[str, Any] = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise AgentSpecError(f"Invalid YAML in agent spec file {path}: {exc}") from exc

    version = str(data.get("version", DEFAULT_AGENT_SPEC_VERSION))
    if version not in SUPPORTED_AGENT_SPEC_VERSIONS:
        raise AgentSpecError(f"Unsupported agent spec version: {version}")

    raw: dict[str, Any] = data.get("agent", {})

    spec = AgentSpec(version=version)

    if "extend" in raw:
        spec.extend = str(raw["extend"])

    for field_name in (
        "name",
        "system_prompt",
        "system_prompt_path",
        "model",
        "when_to_use",
        "isolation",
    ):
        if field_name in raw:
            val = raw[field_name]
            if val is None and field_name == "model":
                setattr(spec, field_name, None)
            else:
                setattr(spec, field_name, str(val))

    if "system_prompt_args" in raw:
        spec.system_prompt_args = dict(raw["system_prompt_args"])

    for list_field in ("tools", "allowed_tools", "exclude_tools"):
        if list_field in raw:
            val = raw[list_field]
            if val is None:
                setattr(spec, list_field, None)
            elif isinstance(val, list):
                setattr(spec, list_field, [str(v) for v in val])
            else:
                setattr(spec, list_field, [str(val)])

    if "max_depth" in raw:
        spec.max_depth = int(raw["max_depth"])

    if "subagents" in raw and raw["subagents"] is not None:
        subs: dict[str, SubagentSpec] = {}
        for name, entry in dict(raw["subagents"]).items():
            if isinstance(entry, dict):
                rel_path = str(entry.get("path", ""))
                desc = str(entry.get("description", ""))
            else:
                rel_path = str(entry)
                desc = ""
            if rel_path:
                subs[name] = SubagentSpec(
                    path=(path.parent / rel_path).absolute(),
                    description=desc,
                )
        spec.subagents = subs

    return spec


def _resolve_system_prompt(path: Path | None, args: dict[str, str]) -> str:
    """Load and template a system-prompt file with ``${var}`` substitution.

    Returns an empty string when ``path`` is missing. Supports the shell-like
    ``${var:-default}`` fallback syntax for variables not provided in
    ``args``.
    """
    if path is None or not path.exists():
        return ""

    text = path.read_text(encoding="utf-8")

    def _replacer(match: Any) -> str:
        """Substitute one ``${var}`` or ``${var:-default}`` match."""
        full = match.group(1)
        if ":-" in full:
            key, default = full.split(":-", 1)
            return args.get(key, default)
        return args.get(full, match.group(0))

    import re

    return re.sub(r"\$\{([^}]+)\}", _replacer, text)


def load_agent_spec(path: Path) -> ResolvedAgentSpec:
    """Load ``path``, resolve every ``extend``, and return a finalised spec.

    Walks the inheritance chain via :func:`_load_agent_spec_recursive`, then
    materialises the system prompt (either inline or from
    ``system_prompt_path``) and freezes every field into a
    :class:`ResolvedAgentSpec`.

    Raises:
        AgentSpecError: If ``name`` is missing or neither ``system_prompt``
            nor ``system_prompt_path`` is provided.
    """
    raw_spec = _load_agent_spec_recursive(path)

    if _is_inherit(raw_spec.name):
        raise AgentSpecError(f"Agent name is required: {path}")
    if _is_inherit(raw_spec.system_prompt) and _is_inherit(raw_spec.system_prompt_path):
        raise AgentSpecError(f"system_prompt or system_prompt_path is required: {path}")

    if not _is_inherit(raw_spec.system_prompt):
        system_prompt = str(raw_spec.system_prompt)
    else:
        sp_path: Path | None = None
        if isinstance(raw_spec.system_prompt_path, str):
            sp_path = (path.parent / raw_spec.system_prompt_path).absolute()
        system_prompt = _resolve_system_prompt(sp_path, raw_spec.system_prompt_args)

    tools: list[str] = []
    if not _is_inherit(raw_spec.tools) and raw_spec.tools is not None:
        tools = list(raw_spec.tools)

    allowed_tools: list[str] | None = None
    if not _is_inherit(raw_spec.allowed_tools):
        allowed_tools = list(raw_spec.allowed_tools) if raw_spec.allowed_tools is not None else None

    exclude_tools: list[str] = []
    if not _is_inherit(raw_spec.exclude_tools) and raw_spec.exclude_tools is not None:
        exclude_tools = list(raw_spec.exclude_tools)

    subagents: dict[str, SubagentSpec] = {}
    if not _is_inherit(raw_spec.subagents) and raw_spec.subagents is not None:
        subagents = dict(raw_spec.subagents)

    def _resolve(val: Any, default: Any) -> Any:
        """Return ``default`` when ``val`` is the :data:`INHERIT` sentinel."""
        if _is_inherit(val):
            return default
        return val

    return ResolvedAgentSpec(
        name=str(raw_spec.name),
        system_prompt=system_prompt,
        model=_resolve(raw_spec.model, None),
        when_to_use=str(_resolve(raw_spec.when_to_use, "")),
        tools=tools,
        allowed_tools=allowed_tools,
        exclude_tools=exclude_tools,
        subagents=subagents,
        max_depth=int(_resolve(raw_spec.max_depth, 5)),
        isolation=str(_resolve(raw_spec.isolation, "")),
        source="yaml",
    )


def _load_agent_spec_recursive(path: Path) -> AgentSpec:
    """Load ``path`` and merge it onto its ``extend`` parent, recursively.

    ``extend: default`` resolves against the built-in default spec; any
    other value is treated as a path relative to ``path``'s directory. The
    returned spec still uses :data:`INHERIT` for any field neither layer
    set — :func:`load_agent_spec` is responsible for picking defaults.
    """
    spec = _parse_agent_yaml(path)

    if spec.extend is None:
        return spec

    extend = spec.extend
    if extend == "default":
        from .definitions import BUILTIN_AGENTS_DIR

        base_path = BUILTIN_AGENTS_DIR / "default" / "agent.yaml"
    else:
        base_path = (path.parent / extend).absolute()

    base_spec = _load_agent_spec_recursive(base_path)

    merged = copy.deepcopy(base_spec)

    for field_name in (
        "name",
        "system_prompt",
        "system_prompt_path",
        "model",
        "when_to_use",
        "isolation",
    ):
        child_val = getattr(spec, field_name)
        if not _is_inherit(child_val):
            setattr(merged, field_name, child_val)

    if spec.system_prompt_args:
        merged.system_prompt_args = {**merged.system_prompt_args, **spec.system_prompt_args}

    for list_field in ("tools", "allowed_tools", "exclude_tools"):
        child_val = getattr(spec, list_field)
        if not _is_inherit(child_val):
            setattr(merged, list_field, child_val)

    if not _is_inherit(spec.max_depth):
        merged.max_depth = spec.max_depth

    if not _is_inherit(spec.subagents):
        if spec.subagents is not None:
            merged.subagents = {**(merged.subagents or {}), **spec.subagents}
        else:
            merged.subagents = None

    merged.extend = None
    return merged
