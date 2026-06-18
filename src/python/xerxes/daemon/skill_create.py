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
"""Skill authoring interview flow — guided skill creation via slash command.

Extracted from daemon/server.py as a mixin.
"""

from __future__ import annotations

import asyncio
import sys

from ..extensions.skills import activate_skill, inject_skill_config, skill_matches_platform
from .gateway import EmitFn

_SKILL_CREATE_STEPS: tuple[tuple[str, str, bool], ...] = (
    (
        "what",
        "What should this skill do? One or two sentences. "
        "Type `auto` to let me infer everything from this session, or `/cancel` to abort.",
        True,
    ),
    (
        "when",
        "When should a future session activate this skill? Describe the trigger. Type `auto` to let me decide.",
        True,
    ),
    (
        "tools",
        "Which tools or commands does the procedure use? List them, comma-separated. Type `auto` to let me decide.",
        True,
    ),
    (
        "pitfalls",
        "Any pitfalls or things that went wrong worth recording? Press Enter to skip, or type `auto` to let me decide.",
        False,
    ),
)
_SKILL_CREATE_AUTO = "<<auto>>"


class SkillCreateMixin:
    """Guided skill-creation interview methods.

    Handles the ``/skill create`` slash command flow: asks the user
    for a skill name and description, then launches a drafting session.
    Mixed into :class:`DaemonServer`.
    """

    async def _slash_skill(self, args: str, emit: EmitFn, *, run_now: bool = True) -> None:
        """Activate (and optionally invoke) a registered skill by name.

        Handles ``name``, ``name:sub``, and ``name:sub args`` forms. When
        ``run_now`` is true the skill's prompt section is injected into a new
        turn; otherwise the skill is just activated for future turns.
        """
        name = args.strip()
        if not name:
            await self._emit_slash(emit, "Usage: `/skill <name>` — use `/skills` to list available skills.")
            return
        skill_args = ""
        if ":" in name:
            name, skill_args = name.split(":", 1)
            name = name.strip()
            skill_args = skill_args.strip()

        self.runtime.discover_skills()
        skill = self.runtime.skill_registry.get(name)
        if skill is None:
            matches = self.runtime.skill_registry.search(name)
            if matches:
                await self._emit_slash(
                    emit, f"Skill '{name}' not found. Did you mean: {', '.join(s.name for s in matches[:5])}"
                )
            else:
                await self._emit_slash(emit, f"Skill '{name}' not found. Use /skills to list available skills.")
            return
        if not skill_matches_platform(skill):
            await self._emit_slash(emit, f"Skill '{name}' is not compatible with this platform ({sys.platform}).")
            return

        activate_skill(name)
        await emit("init_done", {"skills": self.runtime.discover_skills()})
        if not run_now:
            suffix = f"\nArguments: {skill_args}" if skill_args else ""
            await self._emit_slash(
                emit,
                f"Skill '{name}' enabled for future turns.{suffix}\n"
                "Its instructions will be included in the daemon session prompt.",
            )
            return

        await self._emit_slash(emit, f"Running skill '{name}'...")
        prompt_section = skill.to_prompt_section()
        config_block = inject_skill_config(skill)
        # Disambiguate sub-commands. The original bug: ``/autoresearch:learn``
        # arrived here as ``skill_args="learn"`` and got pasted into the prompt
        # as ``User request: learn`` — so the model saw an ambiguous one-word
        # "learn" instead of recognising the canonical sub-command. We now
        # reconstruct the full slash form when the first token of
        # skill_args matches one of the skill's declared sub-commands.
        declared_subs = skill.metadata.subcommands or []
        subcommand = ""
        free_form = skill_args
        if skill_args:
            first, _, rest = skill_args.partition(" ")
            if first in declared_subs:
                subcommand = first
                free_form = rest.strip()
        if subcommand and free_form:
            trigger = f"/{name}:{subcommand} {free_form}"
        elif subcommand:
            trigger = f"/{name}:{subcommand}"
        elif skill_args:
            trigger = skill_args
        else:
            trigger = f"Execute the '{name}' skill now."
        skill_message = f"[Skill '{name}' activated]{config_block}\n\n{prompt_section}\n\nUser request: {trigger}"
        await self._submit_turn(
            {
                "session_key": self._current_session_key,
                "text": skill_message,
                "mode": self._current_mode,
                "plan_mode": self._current_plan_mode,
            },
            emit,
        )

    async def _slash_skill_create(self, args: str, emit: EmitFn) -> None:
        """Start the multi-step ``/skill-create`` interview.

        If a slug is supplied inline (``/skill-create my-thing``) we skip the
        name prompt and go straight to the scope interview; otherwise we ask
        for the slug first. Each subsequent user message is intercepted in
        :meth:`_submit_turn` and routed to :meth:`_advance_skill_create` until
        every step in :data:`_SKILL_CREATE_STEPS` is filled.
        """
        raw = args.strip().split()[0] if args.strip() else ""
        safe_name = "".join(ch for ch in raw.lower() if ch.isalnum() or ch in {"-", "_"}).strip("-_")

        if not safe_name:
            # No slug yet — park for the name prompt; we'll start the scope
            # interview after the user answers that.
            self._pending_slash_arg = ("skill-create", self._current_session_key)
            await self._emit_slash(
                emit,
                "What should this skill be called? Type a short kebab-case slug "
                "(e.g. `commit-helper`). `/cancel` to abort.",
            )
            return

        # Slug present — pre-create the directory and start the scope interview.
        await self._start_skill_create_interview(safe_name, emit)

    async def _launch_skill_draft(self, emit: EmitFn) -> None:
        """Synthesize the draft prompt from the collected answers and submit it.

        Each answer is either a literal user reply, an empty string (which
        only ``pitfalls`` allows), or :data:`_SKILL_CREATE_AUTO` to delegate
        the field to the model. The synthesized prompt renders each kind
        differently so the model knows which fields it must infer.
        """
        state = self._pending_skill_create
        if state is None:
            return
        self._pending_skill_create = None  # consume — even on failure the loop ends

        safe_name = state["name"]
        target_path = state["target_path"]
        answers = state["answers"]

        def render(label: str, key: str, infer_hint: str) -> str:
            value = answers.get(key, "").strip()
            if value == _SKILL_CREATE_AUTO:
                return f"**{label}:** _auto — {infer_hint}_\n\n"
            if not value:
                return ""
            return f"**{label}:** {value}\n\n"

        what_block = render(
            "What the skill should do",
            "what",
            "infer from what we worked on in this session.",
        )
        when_block = render(
            "Activation trigger",
            "when",
            "pick a sensible trigger (e.g. when the user says the skill name, "
            "or when the task description matches the work we just did).",
        )
        tools_block = render(
            "Tools / commands it uses",
            "tools",
            "list the tools we actually invoked this session.",
        )
        pitfalls_value = answers.get("pitfalls", "").strip()
        if pitfalls_value == _SKILL_CREATE_AUTO:
            pitfalls_block = (
                "**Pitfalls:** _auto — list any real issues we hit this session; "
                "omit the `# Pitfalls` section if none occurred._\n\n"
            )
        elif pitfalls_value:
            pitfalls_block = f"**User-reported pitfalls:** {pitfalls_value}\n\n"
        else:
            pitfalls_block = (
                "User reported no pitfalls — omit the `# Pitfalls` section unless "
                "something in this session genuinely went wrong.\n\n"
            )

        synthetic_prompt = (
            f"Write a reusable agent skill called **`{safe_name}`**. "
            f"Do not ask follow-up questions — write the SKILL.md directly. "
            "Any field marked _auto_ below is yours to fill in based on what we "
            "did in this session so far.\n\n"
            "## Inputs\n\n"
            f"{what_block}{when_block}{tools_block}{pitfalls_block}"
            "## Output\n\n"
            f"Write the file to **`{target_path}`** using the Write tool. The file must be "
            "valid Markdown with this exact structure:\n\n"
            "1. YAML frontmatter delimited by `---` lines, containing:\n"
            f"   - `name: {safe_name}` (use this exact slug)\n"
            '   - `description:` (one short line — derived from "what the skill should do")\n'
            "   - `version: 0.1.0`\n"
            "   - `tags: [...]` (short list of topics / domain hints)\n"
            "   - `required_tools: [...]` (tool names from the tools field)\n"
            "2. `# When to use` — based on the activation trigger.\n"
            "3. `# Procedure` — numbered steps grounded in the tool list.\n"
            "4. `# Pitfalls` — only if there were real pitfalls.\n"
            "5. `# Verification` — concrete signals the procedure succeeded.\n\n"
            "After writing, confirm the final path in one short sentence. Do not output the "
            "SKILL.md body in chat; the Write tool is the only delivery channel."
        )

        auto_keys = [k for k in answers if answers[k] == _SKILL_CREATE_AUTO]
        if auto_keys:
            announcement = (
                f"Drafting skill `{safe_name}` — inferring "
                f"{', '.join(auto_keys)} from session context, saving to `{target_path}`…"
            )
        else:
            announcement = f"Drafting skill `{safe_name}` from your answers — saving to `{target_path}`…"
        await self._emit_slash(emit, announcement)
        result = await self._submit_turn({"text": synthetic_prompt, "_internal_slash": True}, emit)

        # After the agent finishes writing the SKILL.md, re-discover skills
        # so the TUI's autocomplete cache picks up the new entry. Without
        # this, ``/<new-skill>`` still won't show in the completer until the
        # user restarts xerxes or runs ``/reload``.
        turn_task = result.get("turn_task") if isinstance(result, dict) else None
        if isinstance(turn_task, asyncio.Task):

            async def _refresh_skills_after_turn() -> None:
                try:
                    await turn_task
                except Exception:
                    # Even if the agent's turn errors mid-way it may still
                    # have written the file before failing — try to refresh
                    # regardless.
                    pass
                try:
                    skills = self.runtime.discover_skills()
                except Exception:
                    return
                try:
                    await emit("init_done", {"skills": skills})
                except Exception:
                    pass

            self._track_task(_refresh_skills_after_turn())

    async def _ask_next_skill_create_question(self, emit: EmitFn) -> None:
        """Emit the next unanswered question from ``_SKILL_CREATE_STEPS``."""
        state = self._pending_skill_create
        if state is None:
            return
        for key, question, _required in _SKILL_CREATE_STEPS:
            if key not in state["answers"]:
                await self._emit_slash(emit, question)
                return
        # All four answered — kick off the actual draft.
        await self._launch_skill_draft(emit)

    async def _start_skill_create_interview(self, safe_name: str, emit: EmitFn) -> None:
        """Open the scope interview after a slug has been resolved."""
        target_dir = self.runtime.skills_dir / safe_name
        target_dir.mkdir(parents=True, exist_ok=True)
        self._pending_skill_create = {
            "session_key": self._current_session_key,
            "name": safe_name,
            "target_path": str(target_dir / "SKILL.md"),
            "answers": {},
        }
        await self._ask_next_skill_create_question(emit)

    async def _advance_skill_create(self, text: str, emit: EmitFn) -> None:
        """Record one interview answer; either ask the next or launch the draft.

        Special inputs:

        * ``auto`` (case-insensitive) — fill *this* and *every remaining*
          field with the :data:`_SKILL_CREATE_AUTO` sentinel and immediately
          launch the draft. The synthesised prompt tells the model to fill
          those fields from session context.
        * empty input on a required step — re-prompt (the field is required).
        * empty input on an optional step (pitfalls) — accept as "skip".
        """
        state = self._pending_skill_create
        if state is None:
            return

        if text.strip().lower() in {"auto", "/auto"}:
            # Hand off every still-unanswered field to the model.
            for key, _question, _required in _SKILL_CREATE_STEPS:
                state["answers"].setdefault(key, _SKILL_CREATE_AUTO)
            await self._ask_next_skill_create_question(emit)
            return

        for key, _question, required in _SKILL_CREATE_STEPS:
            if key in state["answers"]:
                continue
            if required and not text.strip():
                await self._emit_slash(
                    emit,
                    "That field is required. Type an answer, `auto` to let me decide, or `/cancel` to abort.",
                )
                return
            state["answers"][key] = text.strip()
            break
        await self._ask_next_skill_create_question(emit)
