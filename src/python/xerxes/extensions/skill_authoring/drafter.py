# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Skill drafter — turns a :class:`SkillCandidate` into a SKILL.md file.

Hermes' authored skills follow a procedure / pitfalls / verification
template. The drafter has two paths:

1. **LLM-driven**: when an LLM client is provided, it generates a rich
   draft including a natural-language procedure summary.
2. **Template fallback**: deterministic, dependency-free formatter that
   works in tests, CI, and offline. The trigger heuristic provides
   enough structured info that the template alone is useful.
"""

from __future__ import annotations

import re
import typing as tp
from datetime import datetime
from pathlib import Path

from .tracker import SkillCandidate

DEFAULT_VERSION = "0.1.0"


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert *text* into a ``kebab-case`` skill identifier."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if not text:
        text = f"skill-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return text[:max_len].rstrip("-") or "skill"


def _summarise_args(args: dict[str, tp.Any], max_chars: int = 120) -> str:
    """Compact one-line preview of an argument dict."""
    if not args:
        return "(no args)"
    parts = []
    for k, v in args.items():
        sval = str(v).replace("\n", " ")
        if len(sval) > 30:
            sval = sval[:27] + "..."
        parts.append(f"{k}={sval}")
    out = ", ".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars - 3] + "..."
    return out


def render_skill_template(
    candidate: SkillCandidate,
    *,
    name: str | None = None,
    description: str | None = None,
    version: str = DEFAULT_VERSION,
    tags: list[str] | None = None,
) -> str:
    """Format a :class:`SkillCandidate` into Hermes-style SKILL.md text.

    Sections produced (in order):

    1. YAML frontmatter (name / description / version / tags / required_tools).
    2. ``
    3. ``
    4. ``
    5. ``

    The output is deterministic — embedding the same candidate twice
    yields the same text — which keeps the drafter idempotent and
    makes diffs reviewable when skills are auto-improved later.

    Args:
        candidate: The finished tool sequence to canonicalise.
        name: Override the auto-derived skill name.
        description: Override the auto-derived description.
        version: Initial semver. Defaults to ``"0.1.0"``.
        tags: Tag list. Defaults to the unique tools list.

    Returns:
        SKILL.md-formatted text ready to be written to disk.
    """
    auto_name = name or _slugify(candidate.user_prompt or candidate.signature() or "skill")
    auto_desc = description or (candidate.user_prompt.strip() or "Auto-authored skill from tool sequence.")
    if len(auto_desc) > 200:
        auto_desc = auto_desc[:197] + "..."
    auto_tags = tags or candidate.unique_tools
    required_tools = candidate.unique_tools
    fm_lines = [
        "---",
        f"name: {auto_name}",
        f'description: "{auto_desc}"',
        f"version: {version}",
        f"tags: [{', '.join(auto_tags)}]",
        f"required_tools: [{', '.join(required_tools)}]",
        "author: hermes-skill-authoring",
        "---",
    ]
    when_lines = [
        "# When to use",
        "",
    ]
    if candidate.user_prompt:
        when_lines.append(f"Apply this skill for tasks similar to: *{candidate.user_prompt.strip()[:240]}*")
    else:
        when_lines.append("Apply this skill when the tool sequence below matches the current task.")
    proc_lines = ["", "# Procedure", ""]
    for i, ev in enumerate(candidate.successful_events, start=1):
        proc_lines.append(f"{i}. **{ev.tool_name}** — {_summarise_args(ev.arguments)}")
    pitfall_lines: list[str] = []
    failures = [e for e in candidate.events if e.status != "success"]
    retries = [e for e in candidate.events if e.retry_of is not None]
    if failures or retries:
        pitfall_lines.extend(["", "# Pitfalls", ""])
        for ev in failures:
            msg = ev.error_message or ev.error_type or ev.status
            pitfall_lines.append(f"- `{ev.tool_name}` may fail with `{msg}` — retry with adjusted args.")
        for ev in retries:
            pitfall_lines.append(f"- `{ev.tool_name}` was retried in this run; expect transient failures.")
    verify_lines = ["", "# Verification", ""]
    verify_lines.append(
        f"After running the procedure, the agent should have invoked these tools in order: `{candidate.signature()}`."
    )
    verify_lines.append(f"Total successful calls expected: **{len(candidate.successful_events)}**.")
    if candidate.final_response:
        snippet = candidate.final_response.strip()[:160].replace("\n", " ")
        verify_lines.append(f"Reference final response (truncated): *{snippet}*")
    parts = ["\n".join(fm_lines), "\n".join(when_lines)]
    parts.append("\n".join(proc_lines))
    if pitfall_lines:
        parts.append("\n".join(pitfall_lines))
    parts.append("\n".join(verify_lines))
    return "\n".join(parts).rstrip() + "\n"


class SkillDrafter:
    """Drafts a SKILL.md from a :class:`SkillCandidate`.

    Optional ``llm_client`` parameter lets callers supply an LLM that
    can rewrite the auto-generated draft into more natural language.
    The fallback (template-only) path is always available and is the
    default in tests / offline.

    Attributes:
        skills_dir: Directory where new SKILL.md files are written.
        llm_client: Optional LLM client (any object with a ``complete``
            or ``__call__`` method that accepts a string prompt).
    """

    def __init__(
        self,
        skills_dir: str | Path,
        llm_client: tp.Any | None = None,
    ) -> None:
        """Initialise the drafter.

        Args:
            skills_dir: Directory where new SKILL.md files are saved.
                Created if it does not exist.
            llm_client: Optional LLM client. When provided, its
                ``complete(prompt) -> str`` method (or ``__call__``) is
                invoked to refine the draft.
        """
        self.skills_dir = Path(skills_dir).expanduser()
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client

    def draft(
        self,
        candidate: SkillCandidate,
        *,
        name: str | None = None,
        description: str | None = None,
        write: bool = True,
    ) -> tuple[str, Path | None]:
        """Render a SKILL.md (and optionally save it).

        Args:
            candidate: The candidate to draft.
            name: Override the auto-derived skill name.
            description: Override the auto-derived description.
            write: When ``True``, write the SKILL.md to disk under
                ``skills_dir/<name>/SKILL.md``. When ``False``, return
                ``(text, None)``.

        Returns:
            ``(skill_md_text, path_or_None)``.
        """
        text = render_skill_template(candidate, name=name, description=description)
        if self.llm_client is not None:
            text = self._refine_with_llm(text, candidate)
        slug = name or self._extract_name_from_text(text) or _slugify(candidate.user_prompt or "skill")
        path: Path | None = None
        if write:
            target_dir = self.skills_dir / slug
            target_dir.mkdir(parents=True, exist_ok=True)
            path = target_dir / "SKILL.md"
            path.write_text(text, encoding="utf-8")
        return text, path

    def _refine_with_llm(self, draft: str, candidate: SkillCandidate) -> str:
        """Best-effort LLM refinement; falls back to the raw draft on error."""
        if self.llm_client is None:
            return draft
        prompt = (
            "Rewrite the following auto-generated SKILL.md to be a clear, concise, "
            "Hermes-style agent skill. Preserve the YAML frontmatter, the section "
            "headings (# When to use, # Procedure, # Pitfalls, # Verification), "
            "and the original tool sequence. Do not invent steps or remove the "
            "verification block.\n\n---\n\n" + draft
        )
        try:
            if hasattr(self.llm_client, "complete"):
                out = self.llm_client.complete(prompt)
            elif callable(self.llm_client):
                out = self.llm_client(prompt)
            else:
                return draft
            if isinstance(out, str) and out.strip().startswith("---"):
                return out.strip() + "\n"
        except Exception:
            return draft
        return draft

    @staticmethod
    def _extract_name_from_text(text: str) -> str | None:
        """Pull ``name:`` value from the YAML frontmatter."""
        m = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
        if m:
            return _slugify(m.group(1))
        return None
