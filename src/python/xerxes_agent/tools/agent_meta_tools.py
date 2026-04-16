# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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
"""Agent meta-tools — multi-LLM routing, skill ops, and snake_case aliases.

Most of the workflow / lifecycle tools that Xerxes ships under
PascalCase names are re-exported here under shorter snake_case
identifiers (``cronjob``, ``clarify``, ``todo``, ``send_message``,
``memory``, ``session_search``, …) so prompts can reference whichever
naming convention they prefer.

The one genuinely new tool in this module is :class:`mixture_of_agents`,
which routes a single problem through N LLM client callables and
returns either the raw answers or a synthesizer's aggregated reply.

Tools:

- :class:`mixture_of_agents`   — multi-LLM routing
- :class:`clarify`             — alias of ``AskUserQuestionTool``
- :class:`cronjob`             — alias of ``ScheduleCronTool``
- :class:`delegate_task`       — wraps ``HandoffTool``
- :class:`todo`                — alias of ``TodoWriteTool``
- :class:`send_message`        — alias of ``SendMessageTool``
- :class:`session_search`      — alias of ``SearchHistoryTool`` semantics
- :class:`skill_view`          — read a SKILL.md by name
- :class:`skills_list`         — list registered skills
- :class:`skill_manage`        — create / update / delete skills
- :class:`memory`              — save/search shorthand over the memory tools
"""

from __future__ import annotations

import logging
import threading
import typing as tp
from dataclasses import dataclass

from ..extensions.skills import SkillRegistry
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


def _import_aliases() -> dict[str, tp.Any]:
    """Lazily resolve aliases (avoids circular import at module load)."""
    out: dict[str, tp.Any] = {}
    try:
        from .claude_tools import (
            AskUserQuestionTool,
            HandoffTool,
            ScheduleCronTool,
            SendMessageTool,
            TodoWriteTool,
        )

        out.update(
            askquestion=AskUserQuestionTool,
            cron=ScheduleCronTool,
            handoff=HandoffTool,
            send=SendMessageTool,
            todo=TodoWriteTool,
        )
    except Exception:
        logger.debug("claude_tools alias import failed", exc_info=True)
    return out


_aliases = _import_aliases()


class clarify(AgentBaseFn):
    """Ask the user a clarifying question (alias of AskUserQuestionTool)."""

    @staticmethod
    def static_call(question: str, options: list[str] | None = None, **context_variables: tp.Any):
        """Pause and ask the user *question*; ``options`` shows multiple-choice items.

        Use this when the agent does not have enough information to
        proceed and a follow-up answer would unblock the task.
        """
        target = _aliases.get("askquestion")
        if target is None:
            return {"error": "AskUserQuestionTool unavailable", "question": question, "options": list(options or [])}
        return target.static_call(question=question, options=list(options or []), **context_variables)


class cronjob(AgentBaseFn):
    """Schedule a recurring task (alias of ScheduleCronTool)."""

    @staticmethod
    def static_call(
        action: str,
        schedule: str | None = None,
        prompt: str | None = None,
        cron_id: str | None = None,
        **context_variables: tp.Any,
    ):
        """Manage a cron entry. ``action`` is one of ``create``, ``delete``, ``list``.

        Args:
            action: ``create``, ``delete``, or ``list``.
            schedule: Cron expression for ``create`` (e.g. ``"0 9 * * *"``).
            prompt: The task/prompt to fire on schedule for ``create``.
            cron_id: Required for ``delete``; returned by ``create``.
        """
        target = _aliases.get("cron")
        if target is None:
            return {"error": "ScheduleCronTool unavailable", "action": action}
        return target.static_call(
            action=action,
            schedule=schedule,
            prompt=prompt,
            cron_id=cron_id,
            **context_variables,
        )


class delegate_task(AgentBaseFn):
    """Hand a sub-task off to another agent (wraps HandoffTool)."""

    @staticmethod
    def static_call(
        target_agent: str,
        task: str,
        context: dict[str, tp.Any] | None = None,
        **context_variables: tp.Any,
    ):
        """Delegate ``task`` to ``target_agent`` and return its reply.

        Args:
            target_agent: Registered agent name to hand off to.
            task: Plain-language description of what to do.
            context: Optional structured context the receiver may need.
        """
        target = _aliases.get("handoff")
        if target is None:
            return {"error": "HandoffTool unavailable", "target_agent": target_agent}
        return target.static_call(
            target_agent=target_agent,
            task=task,
            context=dict(context or {}),
            **context_variables,
        )


class todo(AgentBaseFn):
    """Manage the agent's todo list (alias of TodoWriteTool)."""

    @staticmethod
    def static_call(
        action: str = "set",
        todos: tp.Any = None,
        **context_variables: tp.Any,
    ):
        """Set / add / clear todos for the active session.

        Args:
            action: ``set`` / ``add`` / ``clear`` / ``list``.
            todos: List of todo dicts, e.g. ``[{"content": "...", "status": "pending"}]``.
                A JSON-encoded string and a single dict are also accepted —
                some LLMs serialise list args that way in tool calls.
        """
        import json as _json

        if isinstance(todos, str):
            try:
                todos = _json.loads(todos)
            except Exception:
                todos = []
        if isinstance(todos, dict):
            todos = [todos]
        if not isinstance(todos, list):
            todos = []
        normalised: list[dict[str, tp.Any]] = []
        for item in todos:
            if isinstance(item, dict):
                if "content" not in item:
                    continue
                item.setdefault("status", "pending")
                item.setdefault("activeForm", item["content"])
                normalised.append(item)
            elif isinstance(item, str):
                normalised.append({"content": item, "status": "pending", "activeForm": item})
        target = _aliases.get("todo")
        if target is None:
            return {"error": "TodoWriteTool unavailable"}
        return target.static_call(action=action, todos=normalised, **context_variables)


class send_message(AgentBaseFn):
    """Send a message via a registered channel (alias of SendMessageTool)."""

    @staticmethod
    def static_call(
        channel: str,
        text: str,
        room_id: str | None = None,
        user_id: str | None = None,
        **context_variables: tp.Any,
    ):
        """Deliver ``text`` to ``channel``.

        Args:
            channel: Channel name, e.g. ``"telegram"``, ``"slack"``.
            text: Message body.
            room_id: Destination room/chat id (channel-specific).
            user_id: Destination user id (channel-specific).
        """
        target = _aliases.get("send")
        if target is None:
            return {"error": "SendMessageTool unavailable", "channel": channel}
        return target.static_call(channel=channel, text=text, room_id=room_id, user_id=user_id, **context_variables)


class memory(AgentBaseFn):
    """Save or recall a memory item (shorthand over save/search_memory)."""

    @staticmethod
    def static_call(
        action: str,
        content: str | None = None,
        query: str | None = None,
        memory_type: str = "long_term",
        **context_variables: tp.Any,
    ):
        """One-shot memory wrapper: ``action="save"`` or ``"search"``.

        A single entry-point covering both write and read so the
        agent doesn't have to remember two distinct tool names.

        Args:
            action: ``save`` or ``search``.
            content: Required for ``save`` — the text to store.
            query: Required for ``search`` — the search query.
            memory_type: ``long_term`` (default) / ``short_term`` / ``contextual``.
        """
        try:
            from .memory_tool import save_memory, search_memory
        except Exception as exc:
            return {"error": f"memory tools unavailable: {exc}"}
        if action == "save":
            if not content:
                return {"error": "content is required for save"}
            return save_memory(content=content, memory_type=memory_type, **context_variables)
        if action == "search":
            if not query:
                return {"error": "query is required for search"}
            return search_memory(query=query, memory_type=memory_type, **context_variables)
        return {"error": f"unknown action {action!r}; use 'save' or 'search'"}


class session_search(AgentBaseFn):
    """Search past sessions (snake_case alias of SearchHistoryTool)."""

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

    Pass a :class:`~xerxes_agent.tools.history_tool.SearchHistoryTool`
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


def set_skill_registry(registry: SkillRegistry | None) -> None:
    """Install the registry consulted by skill_* tools."""
    global _skill_registry
    with _skill_registry_lock:
        _skill_registry = registry


def get_skill_registry() -> SkillRegistry | None:
    """Return the currently installed skill registry, or ``None`` if unset."""
    with _skill_registry_lock:
        return _skill_registry


class skills_list(AgentBaseFn):
    """List every loaded skill (alias of registry.get_all)."""

    @staticmethod
    def static_call(**context_variables: tp.Any) -> dict[str, tp.Any]:
        """Return the list of skill names, descriptions, and tags."""
        reg = get_skill_registry()
        if reg is None:
            return {"error": "no skill registry configured", "skills": []}
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
            name: Skill name as listed by :class:`skills_list`.
        """
        reg = get_skill_registry()
        if reg is None:
            return {"error": "no skill registry configured", "name": name}
        skill = reg.get(name)
        if skill is None:
            return {"error": "not_found", "name": name}
        meta = skill.metadata
        return {
            "name": meta.name,
            "version": meta.version,
            "description": meta.description,
            "tags": list(meta.tags or []),
            "instructions": skill.instructions,
            "source_path": str(skill.source_path),
        }


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
            xerxes_subdir = None  # type: ignore[assignment]
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
    "clarify",
    "configure_mixture_of_agents",
    "cronjob",
    "delegate_task",
    "get_moa_config",
    "get_session_searcher",
    "get_skill_registry",
    "memory",
    "mixture_of_agents",
    "send_message",
    "session_search",
    "set_session_searcher",
    "set_skill_registry",
    "skill_manage",
    "skill_view",
    "skills_list",
    "todo",
]
