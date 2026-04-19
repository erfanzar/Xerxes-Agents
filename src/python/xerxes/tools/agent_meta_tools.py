# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.
"""Agent meta-tools — multi-LLM routing, skill ops, and session search.

Tools:

- :class:`mixture_of_agents`   — multi-LLM routing
- :class:`session_search`      — search past sessions
- :class:`skill_view`          — read a SKILL.md by name
- :class:`skills_list`         — list registered skills
- :class:`skill_manage`        — create / update / delete skills
"""

from __future__ import annotations

import logging
import threading
import typing as tp
from dataclasses import dataclass

from ..extensions.skills import SkillRegistry
from ..extensions.skill_authoring import SkillMatcher
from ..memory.embedders import get_default_embedder
from ..types import AgentBaseFn

logger = logging.getLogger(__name__)
LLMCallable = tp.Callable[[str], str]


@dataclass
class _MoAState:
    """Process-wide MoA configuration."""

    members: dict[str, LLMCallable]
    synthesizer: LLMCallable | None
    voting: bool


_state = _MoAState(members={}, synthesizer=None, voting=False)
_state_lock = threading.Lock()


def configure_mixture_of_agents(
    members: dict[str, LLMCallable] | None = None,
    *,
    synthesizer: LLMCallable | None = None,
    voting: bool = False,
) -> None:
    """Install the LLM panel + (optional) synthesizer used by ``mixture_of_agents``.

    Args:
        members: Mapping of friendly name → callable that takes a
            prompt and returns a string answer.
        synthesizer: Optional callable that receives a combined
            prompt summarising the panel's answers and returns the
            final reply. When omitted, the tool returns each panellist
            answer verbatim.
        voting: When ``True``, return the most common answer (after
            normalisation) instead of all answers.
    """
    global _state
    with _state_lock:
        _state = _MoAState(
            members=dict(members or {}),
            synthesizer=synthesizer,
            voting=voting,
        )


def get_moa_config() -> _MoAState:
    """Return the currently configured Mixture-of-Agents state."""
    with _state_lock:
        return _state


class mixture_of_agents(AgentBaseFn):
    """Route a problem through multiple LLMs and (optionally) synthesise."""

    @staticmethod
    def static_call(
        prompt: str,
        members: list[str] | None = None,
        synthesise: bool = True,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Run ``prompt`` against several LLM callables in parallel.

        Use this for hard reasoning tasks where you want to compare
        independent answers from different models, or where the
        synthesizer can pick the best parts of each response. The
        panel itself is configured at process startup via
        :func:`configure_mixture_of_agents`.

        Args:
            prompt: The question or task to evaluate.
            members: Optional subset of configured panellist names to
                consult. ``None`` uses every registered member.
            synthesise: When ``True`` and a synthesizer is configured,
                returns ``{"final": <synth answer>}`` plus the
                individual answers. When ``False`` (or no synthesizer),
                returns just the per-member answers.

        Returns:
            ``{"members": [...], "answers": {name: text}, "final": <opt>,
            "voted": <opt>}``.
        """
        cfg = get_moa_config()
        if not cfg.members:
            return {"error": "no MoA members configured", "members": [], "answers": {}}
        names = members or list(cfg.members.keys())
        answers: dict[str, str] = {}
        for name in names:
            fn = cfg.members.get(name)
            if fn is None:
                answers[name] = f"[unknown member {name!r}]"
                continue
            try:
                answers[name] = str(fn(prompt))
            except Exception as exc:
                answers[name] = f"[error: {exc}]"
        result: dict[str, tp.Any] = {"members": names, "answers": answers}
        if cfg.voting:
            counts: dict[str, int] = {}
            for ans in answers.values():
                key = " ".join(ans.split()).strip().lower()
                counts[key] = counts.get(key, 0) + 1
            if counts:
                result["voted"] = max(counts.items(), key=lambda kv: kv[1])[0]
        if synthesise and cfg.synthesizer is not None and answers:
            try:
                joined = "\n".join(f"[{n}] {a}" for n, a in answers.items())
                result["final"] = str(cfg.synthesizer(f"Combine these answers:\n{joined}"))
            except Exception as exc:
                result["final_error"] = str(exc)
        return result


class session_search(AgentBaseFn):
    """Search past sessions."""

    @staticmethod
    def static_call(
        query: str,
        limit: int = 5,
        agent_id: str | None = None,
        session_id: str | None = None,
        **context_variables: tp.Any,
    ):
        """Search the cross-session index for matching turns.

        The runtime must have bound a :class:`SearchHistoryTool`
        instance via :func:`set_session_searcher` for this tool to
        do anything.
        """
        searcher = get_session_searcher()
        if searcher is None:
            return {"error": "no session searcher configured", "hits": []}
        return searcher(
            query=query,
            limit=limit,
            agent_id=agent_id,
            session_id=session_id,
        )


_session_searcher_lock = threading.Lock()
_session_searcher: tp.Any | None = None


def set_session_searcher(searcher: tp.Any | None) -> None:
    """Install the callable used by :class:`session_search`.

    Pass a :class:`~xerxes.tools.history_tool.SearchHistoryTool`
    instance (or any callable matching its ``__call__`` signature).
    """
    global _session_searcher
    with _session_searcher_lock:
        _session_searcher = searcher


def get_session_searcher() -> tp.Any | None:
    """Return the currently installed session searcher callable, if any."""
    with _session_searcher_lock:
        return _session_searcher


_skill_registry_lock = threading.Lock()
_skill_registry: SkillRegistry | None = None
_skill_matcher: SkillMatcher | None = None
_matcher_lock = threading.Lock()


def set_skill_registry(registry: SkillRegistry | None) -> None:
    """Install the registry consulted by skill_* tools."""
    global _skill_registry
    with _skill_registry_lock:
        _skill_registry = registry


def get_skill_registry() -> SkillRegistry | None:
    """Return the currently installed skill registry, or ``None`` if unset."""
    with _skill_registry_lock:
        return _skill_registry


def _get_skill_matcher() -> SkillMatcher:
    """Return a lazily-initialised global SkillMatcher instance."""
    global _skill_matcher
    if _skill_matcher is None:
        with _matcher_lock:
            if _skill_matcher is None:
                registry = get_skill_registry()
                embedder = get_default_embedder()
                _skill_matcher = SkillMatcher(registry, embedder=embedder)
    return _skill_matcher


class skills_list(AgentBaseFn):
    """List every loaded skill."""

    @staticmethod
    def static_call(search: str | None = None, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return the list of skill names, descriptions, and tags.

        Args:
            search: Optional free-text query for semantic skill matching.
                When provided, returns the top 20 semantically matching skills
                instead of all skills.
        """
        reg = get_skill_registry()
        if reg is None:
            return {"error": "no skill registry configured", "skills": []}
        if search:
            matcher = _get_skill_matcher()
            matches = matcher.match(search, k=20)
            out = []
            for hit in matches:
                meta = hit.skill.metadata
                out.append(
                    {
                        "name": meta.name,
                        "version": meta.version,
                        "description": meta.description,
                        "tags": list(meta.tags or []),
                        "score": round(hit.score, 3),
                    }
                )
            return {"count": len(out), "skills": out, "query": search}
        out = []
        for skill in reg.get_all():
            meta = skill.metadata
            out.append(
                {
                    "name": meta.name,
                    "version": meta.version,
                    "description": meta.description,
                    "tags": list(meta.tags or []),
                }
            )
        return {"count": len(out), "skills": out}


class skill_view(AgentBaseFn):
    """Read a specific skill's metadata + body."""

    @staticmethod
    def static_call(name: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return the SKILL.md content for ``name``.

        Args:
            name: Skill name as listed by :class:`skills_list`, or a
                free-text description to resolve via semantic matching.
        """
        reg = get_skill_registry()
        if reg is None:
            return {"error": "no skill registry configured", "name": name}
        skill = reg.get(name)
        if skill is None:

            matcher = _get_skill_matcher()
            hits = matcher.match(name, k=1)
            if hits:
                skill = hits[0].skill
                matched = True
            else:
                matched = False
        else:
            matched = False
        if skill is None:
            return {"error": "not_found", "name": name}
        meta = skill.metadata
        result = {
            "name": meta.name,
            "version": meta.version,
            "description": meta.description,
            "tags": list(meta.tags or []),
            "instructions": skill.instructions,
            "source_path": str(skill.source_path),
        }
        if matched:
            result["_matched_query"] = name
        return result


class skill_manage(AgentBaseFn):
    """Create, update, or delete a skill on disk."""

    @staticmethod
    def static_call(
        action: str,
        name: str,
        instructions: str = "",
        description: str = "",
        version: str = "0.1.0",
        tags: list[str] | None = None,
        skills_dir: str | None = None,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Modify the skill registry by writing/removing SKILL.md files.

        Args:
            action: ``create`` / ``update`` / ``delete``.
            name: Skill identifier.
            instructions: Markdown body for ``create`` / ``update``.
            description: One-line summary.
            version: Semver string.
            tags: Optional tag list.
            skills_dir: Override the directory; defaults to
                ``$XERXES_HOME/skills``.

        Returns:
            ``{"ok": bool, "name": str, "path": str|None, ...}``.
        """
        from pathlib import Path

        try:
            from ..core.paths import xerxes_subdir
        except Exception:
            xerxes_subdir = None
        target_dir = (
            Path(skills_dir).expanduser()
            if skills_dir
            else (xerxes_subdir("skills") if xerxes_subdir else Path.home() / ".xerxes" / "skills")
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        skill_path = target_dir / name / "SKILL.md"
        if action == "delete":
            if skill_path.exists():
                skill_path.unlink()
                reg = get_skill_registry()
                if reg is not None:
                    reg._skills.pop(name, None)
                return {"ok": True, "name": name, "deleted": str(skill_path)}
            return {"ok": False, "name": name, "error": "not_found"}
        if action not in ("create", "update"):
            return {"ok": False, "error": f"unknown action {action!r}"}
        if not instructions:
            return {"ok": False, "error": "instructions required for create/update"}
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        front = [
            "---",
            f"name: {name}",
            f'description: "{description}"',
            f"version: {version}",
            f"tags: [{', '.join(tags or [])}]",
            "---",
        ]
        skill_path.write_text("\n".join(front) + "\n\n" + instructions, encoding="utf-8")
        reg = get_skill_registry()
        if reg is not None:
            try:
                reg.discover(target_dir)
            except Exception:
                pass
        return {"ok": True, "name": name, "path": str(skill_path), "action": action}


__all__ = [
    "configure_mixture_of_agents",
    "get_moa_config",
    "get_session_searcher",
    "get_skill_registry",
    "mixture_of_agents",
    "session_search",
    "set_session_searcher",
    "set_skill_registry",
    "skill_manage",
    "skill_view",
    "skills_list",
]
