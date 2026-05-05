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


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge ``override`` into ``base``.

    Args:
        base (Any): IN: The base value to merge into. OUT: Used as the fallback
            when ``override`` is :data:`INHERIT`.
        override (Any): IN: The overriding value. OUT: Replaces or extends
            ``base`` according to type (dict, list, or scalar).

    Returns:
        Any: OUT: The merged result.
    """
    if override is INHERIT:
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
    """Parse a single ``agent.yaml`` file into an :class:`AgentSpec`.

    Args:
        path (Path): IN: Filesystem path to the YAML spec. OUT: Read and parsed.

    Returns:
        AgentSpec: OUT: The parsed (but not yet resolved) agent specification.

    Raises:
        AgentSpecError: If the file is missing, YAML is invalid, or the version
            is unsupported.
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
    """Resolve a system prompt template with ``${var}`` substitution.

    Supports ``${var:-default}`` syntax for fallback values.

    Args:
        path (Path | None): IN: Path to the prompt template file. OUT: Read
            as UTF-8 text. If missing or ``None``, returns an empty string.
        args (dict[str, str]): IN: Mapping of variable names to replacement
            values. OUT: Substituted into the template.

    Returns:
        str: OUT: The resolved prompt text.
    """
    if path is None or not path.exists():
        return ""

    text = path.read_text(encoding="utf-8")

    def _replacer(match: Any) -> str:
        """Internal helper to replacer.

        Args:
            match (Any): IN: match. OUT: Consumed during execution.
        Returns:
            str: OUT: Result of the operation."""
        full = match.group(1)
        if ":-" in full:
            key, default = full.split(":-", 1)
            return args.get(key, default)
        return args.get(full, match.group(0))

    import re

    return re.sub(r"\$\{([^}]+)\}", _replacer, text)


def load_agent_spec(path: Path) -> ResolvedAgentSpec:
    """Load and fully resolve an agent spec from a YAML file.

    Args:
        path (Path): IN: Path to the ``agent.yaml`` file. OUT: Passed to
            recursive loading and resolution.

    Returns:
        ResolvedAgentSpec: OUT: The fully resolved agent specification.

    Raises:
        AgentSpecError: If required fields (name, system_prompt) are missing.
    """
    raw_spec = _load_agent_spec_recursive(path)

    if raw_spec.name is INHERIT:
        raise AgentSpecError(f"Agent name is required: {path}")
    if raw_spec.system_prompt is INHERIT and raw_spec.system_prompt_path is INHERIT:
        raise AgentSpecError(f"system_prompt or system_prompt_path is required: {path}")

    if raw_spec.system_prompt is not INHERIT:
        system_prompt = str(raw_spec.system_prompt)
    else:
        sp_path: Path | None = None
        if isinstance(raw_spec.system_prompt_path, str):
            sp_path = (path.parent / raw_spec.system_prompt_path).absolute()
        system_prompt = _resolve_system_prompt(sp_path, raw_spec.system_prompt_args)

    tools: list[str] = []
    if raw_spec.tools is not INHERIT and raw_spec.tools is not None:
        tools = list(raw_spec.tools)

    allowed_tools: list[str] | None = None
    if raw_spec.allowed_tools is not INHERIT:
        allowed_tools = list(raw_spec.allowed_tools) if raw_spec.allowed_tools is not None else None

    exclude_tools: list[str] = []
    if raw_spec.exclude_tools is not INHERIT and raw_spec.exclude_tools is not None:
        exclude_tools = list(raw_spec.exclude_tools)

    subagents: dict[str, SubagentSpec] = {}
    if raw_spec.subagents is not INHERIT and raw_spec.subagents is not None:
        subagents = dict(raw_spec.subagents)

    def _resolve(val: Any, default: Any) -> Any:
        """Internal helper to resolve.

        Args:
            val (Any): IN: val. OUT: Consumed during execution.
            default (Any): IN: default. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""
        if val is INHERIT or isinstance(val, Inherit):
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
    """Recursively load an agent spec, merging parent specs via ``extend``.

    Args:
        path (Path): IN: Path to the agent YAML file. OUT: Read and parsed;
            if ``extend`` is set, the parent is loaded recursively.

    Returns:
        AgentSpec: OUT: The merged (but not yet fully resolved) spec.
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
        if child_val is not INHERIT:
            setattr(merged, field_name, child_val)

    if spec.system_prompt_args:
        merged.system_prompt_args = {**merged.system_prompt_args, **spec.system_prompt_args}

    for list_field in ("tools", "allowed_tools", "exclude_tools"):
        child_val = getattr(spec, list_field)
        if child_val is not INHERIT:
            setattr(merged, list_field, child_val)

    if spec.max_depth is not INHERIT:
        merged.max_depth = spec.max_depth

    if spec.subagents is not INHERIT:
        if spec.subagents is not None:
            merged.subagents = {**(merged.subagents or {}), **spec.subagents}
        else:
            merged.subagents = None

    merged.extend = None
    return merged
