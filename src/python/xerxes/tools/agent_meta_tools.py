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
"""Meta-tools for orchestrating other agents and the skill subsystem.

Provides three families of agent-facing tools:

* :class:`mixture_of_agents` — fan a prompt out to several configured LLM
  callables, collect every answer, optionally majority-vote, and optionally
  ask a synthesizer to combine them. Members are registered via
  :func:`configure_mixture_of_agents`.
* :class:`session_search` — query historical session transcripts through a
  pluggable searcher installed with :func:`set_session_searcher`.
* :class:`skills_list` / :class:`skill_view` / :class:`skill_manage` — list,
  read, and CRUD :class:`SkillRegistry` entries on disk. Listing falls back to
  semantic matching via :class:`SkillMatcher` when a free-text query is given.

All shared state is guarded by module-level locks so the tools are safe to
invoke from concurrent agent turns.
"""

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
    """Configuration snapshot for :class:`mixture_of_agents`.

    Attributes:
        members: Mapping of member name to LLM callable taking the prompt
            string and returning a textual answer.
        synthesizer: Optional combiner that receives the joined per-member
            answers and produces a final summary.
        voting: When True the tool tallies normalized answers and reports
            the most common one alongside the raw responses.
    """

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
    """Replace the mixture-of-agents configuration atomically.

    Args:
        members: Mapping of human-readable member name to a callable that
            takes the prompt and returns its textual answer. ``None`` clears
            the roster.
        synthesizer: Optional callable used to merge member answers; when
            absent the tool returns the raw answers only.
        voting: Enable majority voting on normalized member outputs.
    """

    global _state
    with _state_lock:
        _state = _MoAState(
            members=dict(members or {}),
            synthesizer=synthesizer,
            voting=voting,
        )


def get_moa_config() -> _MoAState:
    """Return the current :class:`_MoAState` snapshot under the state lock."""

    with _state_lock:
        return _state


class mixture_of_agents(AgentBaseFn):
    """Fan a prompt out to multiple LLM members and aggregate the answers."""

    @staticmethod
    def static_call(
        prompt: str,
        members: list[str] | None = None,
        synthesise: bool = True,
        **context_variables: tp.Any,
    ) -> dict[str, tp.Any]:
        """Run ``prompt`` through configured members and combine the responses.

        Args:
            prompt: User-facing question forwarded to every selected member.
            members: Subset of configured member names to invoke. ``None``
                runs every registered member.
            synthesise: When True and a synthesizer is configured, ask it to
                merge member answers into a single ``final`` response.

        Returns:
            Dictionary with ``members`` (the names invoked), ``answers``
            (member-name to text), and optionally ``voted`` (when voting is
            enabled) and ``final`` (when synthesis succeeds) or
            ``final_error`` if synthesis raised. When no members are
            configured the dictionary contains an ``error`` field.
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
    """Search prior session transcripts via the registered searcher."""

    @staticmethod
    def static_call(
        query: str,
        limit: int = 5,
        agent_id: str | None = None,
        session_id: str | None = None,
        **context_variables: tp.Any,
    ):
        """Query historical sessions for matching transcript fragments.

        Args:
            query: Free-text search expression handed verbatim to the
                searcher (typically full-text or hybrid vector search).
            limit: Maximum number of hits to return.
            agent_id: Optional filter restricting results to a single agent
                identifier.
            session_id: Optional filter restricting results to a single
                session identifier.

        Returns:
            Whatever the installed searcher returns (typically a dict with a
            ``hits`` list). When no searcher is configured an ``error`` dict
            with an empty ``hits`` list is returned instead.
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
    """Install (or clear) the callable used by :class:`session_search`.

    Args:
        searcher: Callable accepting ``query``, ``limit``, ``agent_id`` and
            ``session_id`` keyword arguments. Pass ``None`` to detach.
    """

    global _session_searcher
    with _session_searcher_lock:
        _session_searcher = searcher


def get_session_searcher() -> tp.Any | None:
    """Return the currently installed session searcher, or ``None``."""

    with _session_searcher_lock:
        return _session_searcher


_skill_registry_lock = threading.Lock()
_skill_registry: SkillRegistry | None = None
_skill_matcher: SkillMatcher | None = None
_matcher_lock = threading.Lock()


def set_skill_registry(registry: SkillRegistry | None) -> None:
    """Install (or clear) the :class:`SkillRegistry` used by the skill tools.

    The cached :class:`SkillMatcher` is bound to the registry at first use;
    callers that need the matcher rebuilt should drop the module-level
    ``_skill_matcher`` themselves.
    """

    global _skill_registry
    with _skill_registry_lock:
        _skill_registry = registry


def get_skill_registry() -> SkillRegistry | None:
    """Return the currently installed skill registry, or ``None``."""

    with _skill_registry_lock:
        return _skill_registry


def _get_skill_matcher() -> SkillMatcher:
    """Return the lazily-constructed :class:`SkillMatcher` bound to the registry."""

    global _skill_matcher
    if _skill_matcher is None:
        with _matcher_lock:
            if _skill_matcher is None:
                registry = get_skill_registry()
                embedder = get_default_embedder()
                _skill_matcher = SkillMatcher(registry, embedder=embedder)
    return _skill_matcher


class skills_list(AgentBaseFn):
    """List registered skills, optionally filtered by a semantic query."""

    @staticmethod
    def static_call(search: str | None = None, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return skill metadata for every registered skill.

        Args:
            search: Optional free-text query. When provided, results are
                ranked by :class:`SkillMatcher` (top 20) and each entry
                carries a ``score`` field; otherwise every registered skill
                is returned in registry order.

        Returns:
            Dict with ``count`` and ``skills`` (each entry has ``name``,
            ``version``, ``description``, ``tags``, and ``score`` when a
            search query was used). When no registry is configured an
            ``error`` field is returned and ``skills`` is empty.
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
    """Return a single skill's metadata and instructions."""

    @staticmethod
    def static_call(name: str, **context_variables: tp.Any) -> dict[str, tp.Any]:
        """Fetch a skill by name, falling back to semantic match on miss.

        Args:
            name: Exact skill name. If no exact match exists the registry
                is queried via :class:`SkillMatcher`; the top hit is
                returned with ``_matched_query`` set to the original input.

        Returns:
            Dict with ``name``, ``version``, ``description``, ``tags``,
            ``instructions``, ``source_path``, and (when matched fuzzily)
            ``_matched_query``. When no registry is configured or the skill
            cannot be located, an ``error`` key is returned.
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
    """Create, update, or delete a skill on disk and refresh the registry."""

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
        """Write or remove a SKILL.md bundle and refresh the active registry.

        Args:
            action: One of ``create``, ``update``, or ``delete``. The first
                two write a ``SKILL.md`` with YAML front-matter; ``delete``
                removes the file and evicts the entry from the registry.
            name: Skill identifier; used as both the front-matter ``name``
                and the parent directory.
            instructions: Markdown body written verbatim after the
                front-matter; required for ``create``/``update``.
            description: Short human-readable description for the
                front-matter.
            version: Semantic version string written to the front-matter
                (defaults to ``0.1.0``).
            tags: Optional list of tags written into the front-matter array.
            skills_dir: Directory used to host the skill bundle. Defaults
                to ``$XERXES_DIR/skills`` (via :func:`xerxes_subdir`) or
                ``~/.xerxes/skills`` when paths utilities are unavailable.

        Returns:
            Dict reporting ``ok`` plus ``name``, ``path``/``deleted``,
            ``action``, or an ``error`` describing why the operation failed.
        """

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
