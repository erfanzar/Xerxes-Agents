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
"""Agent meta tools module for Xerxes.

Exports:
    - logger
    - LLMCallable
    - configure_mixture_of_agents
    - get_moa_config
    - mixture_of_agents
    - session_search
    - set_session_searcher
    - get_session_searcher
    - set_skill_registry
    - get_skill_registry
    - ... and 3 more."""

from __future__ import annotations

import logging
import threading
import typing as tp
from dataclasses import dataclass

from ..extensions.skill_authoring import SkillMatcher
from ..extensions.skills import SkillRegistry
from ..memory.embedders import get_default_embedder
from ..types import AgentBaseFn

logger = logging.getLogger(__name__)
LLMCallable = tp.Callable[[str], str]


@dataclass
class _MoAState:
    """Mo astate.

    Attributes:
        members (dict[str, LLMCallable]): members.
        synthesizer (LLMCallable | None): synthesizer.
        voting (bool): voting."""

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
    """Configure mixture of agents.

    Args:
        members (dict[str, LLMCallable] | None, optional): IN: members. Defaults to None. OUT: Consumed during execution.
        synthesizer (LLMCallable | None, optional): IN: synthesizer. Defaults to None. OUT: Consumed during execution.
        voting (bool, optional): IN: voting. Defaults to False. OUT: Consumed during execution."""

    global _state
    with _state_lock:
        _state = _MoAState(
            members=dict(members or {}),
            synthesizer=synthesizer,
            voting=voting,
        )


def get_moa_config() -> _MoAState:
    """Retrieve the moa config.

    Returns:
        _MoAState: OUT: Result of the operation."""

    with _state_lock:
        return _state


class mixture_of_agents(AgentBaseFn):
    """Mixture of agents.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        prompt: str,
        members: list[str] | None = None,
        synthesise: bool = True,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Static call.

        Args:
            prompt (str): IN: prompt. OUT: Consumed during execution.
            members (list[str] | None, optional): IN: members. Defaults to None. OUT: Consumed during execution.
            synthesise (bool, optional): IN: synthesise. Defaults to True. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
    """Session search.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        query: str,
        limit: int = 5,
        agent_id: str | None = None,
        session_id: str | None = None,
        **context_variables: tp.Any,
    ):
        """Static call.

        Args:
            query (str): IN: query. OUT: Consumed during execution.
            limit (int, optional): IN: limit. Defaults to 5. OUT: Consumed during execution.
            agent_id (str | None, optional): IN: agent id. Defaults to None. OUT: Consumed during execution.
            session_id (str | None, optional): IN: session id. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            Any: OUT: Result of the operation."""

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
    """Set the session searcher.

    Args:
        searcher (tp.Any | None): IN: searcher. OUT: Consumed during execution."""

    global _session_searcher
    with _session_searcher_lock:
        _session_searcher = searcher


def get_session_searcher() -> tp.Any | None:
    """Retrieve the session searcher.

    Returns:
        tp.Any | None: OUT: Result of the operation."""

    with _session_searcher_lock:
        return _session_searcher


_skill_registry_lock = threading.Lock()
_skill_registry: SkillRegistry | None = None
_skill_matcher: SkillMatcher | None = None
_matcher_lock = threading.Lock()


def set_skill_registry(registry: SkillRegistry | None) -> None:
    """Set the skill registry.

    Args:
        registry (SkillRegistry | None): IN: registry. OUT: Consumed during execution."""

    global _skill_registry
    with _skill_registry_lock:
        _skill_registry = registry


def get_skill_registry() -> SkillRegistry | None:
    """Retrieve the skill registry.

    Returns:
        SkillRegistry | None: OUT: Result of the operation."""

    with _skill_registry_lock:
        return _skill_registry


def _get_skill_matcher() -> SkillMatcher:
    """Internal helper to get skill matcher.

    Returns:
        SkillMatcher: OUT: Result of the operation."""

    global _skill_matcher
    if _skill_matcher is None:
        with _matcher_lock:
            if _skill_matcher is None:
                registry = get_skill_registry()
                embedder = get_default_embedder()
                _skill_matcher = SkillMatcher(registry, embedder=embedder)
    return _skill_matcher


class skills_list(AgentBaseFn):
    """Skills list.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(search: str | None = None, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            search (str | None, optional): IN: search. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
    """Skill view.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(name: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Static call.

        Args:
            name (str): IN: name. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

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
    """Skill manage.

    Inherits from: AgentBaseFn
    """

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
        """Static call.

        Args:
            action (str): IN: action. OUT: Consumed during execution.
            name (str): IN: name. OUT: Consumed during execution.
            instructions (str, optional): IN: instructions. Defaults to ''. OUT: Consumed during execution.
            description (str, optional): IN: description. Defaults to ''. OUT: Consumed during execution.
            version (str, optional): IN: version. Defaults to '0.1.0'. OUT: Consumed during execution.
            tags (list[str] | None, optional): IN: tags. Defaults to None. OUT: Consumed during execution.
            skills_dir (str | None, optional): IN: skills dir. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, tp.Any]: OUT: Result of the operation."""

        from pathlib import Path

        xerxes_subdir: tp.Callable[..., tp.Any] | None = None
        try:
            from ..core.paths import xerxes_subdir
        except Exception:
            pass
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
