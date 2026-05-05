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
"""Skill drafting from observed tool sequences.

``SkillDrafter`` turns a ``SkillCandidate`` into a ``SKILL.md`` document,
optionally refining it with an LLM.
"""

from __future__ import annotations

import re
import typing as tp
from datetime import datetime
from pathlib import Path

from .tracker import SkillCandidate

DEFAULT_VERSION = "0.1.0"


def _slugify(text: str, max_len: int = 40) -> str:
    """Normalise arbitrary text into a URL-safe slug.

    Args:
        text (str): IN: Raw text. OUT: Lowercased, non-alphanumerics replaced
            with hyphens, and truncated.
        max_len (int): IN: Maximum slug length. OUT: Enforced on the result.

    Returns:
        str: OUT: Clean slug string.
    """

    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if not text:
        text = f"skill-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return text[:max_len].rstrip("-") or "skill"


def _summarise_args(args: dict[str, tp.Any], max_chars: int = 120) -> str:
    """Format a dict of arguments into a compact human-readable string.

    Args:
        args (dict[str, tp.Any]): IN: Tool arguments. OUT: Stringified and
            truncated.
        max_chars (int): IN: Maximum output length. OUT: Enforced.

    Returns:
        str: OUT: Compact summary like ``"key=value, ..."``.
    """

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
    """Generate a ``SKILL.md`` document from a tool sequence candidate.

    Args:
        candidate (SkillCandidate): IN: Observed tool calls and final
            response. OUT: Used to build frontmatter, procedure, pitfalls, and
            verification sections.
        name (str | None): IN: Override skill name. OUT: Auto-generated from
            the prompt or signature if omitted.
        description (str | None): IN: Override description. OUT: Defaults to
            the user prompt.
        version (str): IN: Semver string. OUT: Placed in frontmatter.
        tags (list[str] | None): IN: Override tag list. OUT: Defaults to
            ``candidate.unique_tools``.

    Returns:
        str: OUT: Complete markdown document with YAML frontmatter.
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
    """Produces ``SKILL.md`` files from ``SkillCandidate`` observations.

    Args:
        skills_dir (str | Path): IN: Destination directory for drafted skills.
            OUT: Created if missing.
        llm_client (tp.Any | None): IN: Optional LLM client for refinement.
            OUT: Used by ``_refine_with_llm``.
    """

    def __init__(
        self,
        skills_dir: str | Path,
        llm_client: tp.Any | None = None,
    ) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            skills_dir (str | Path): IN: skills dir. OUT: Consumed during execution.
            llm_client (tp.Any | None, optional): IN: llm client. Defaults to None. OUT: Consumed during execution."""
        self.skills_dir = Path(skills_dir).expanduser()
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            skills_dir (str | Path): IN: skills dir. OUT: Consumed during execution.
            llm_client (tp.Any | None, optional): IN: llm client. Defaults to None. OUT: Consumed during execution."""
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            skills_dir (str | Path): IN: skills dir. OUT: Consumed during execution.
            llm_client (tp.Any | None, optional): IN: llm client. Defaults to None. OUT: Consumed during execution."""
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
        """Generate a skill document and optionally write it to disk.

        Args:
            candidate (SkillCandidate): IN: Observed tool sequence. OUT: Passed
                to ``render_skill_template``.
            name (str | None): IN: Override skill name. OUT: Passed to the
                template.
            description (str | None): IN: Override description. OUT: Passed to
                the template.
            write (bool): IN: Whether to persist the file. OUT: Determines
                whether a path is returned.

        Returns:
            tuple[str, Path | None]: OUT: Generated markdown text and optional
            written path.
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
        """Ask the configured LLM to polish the generated markdown.

        Args:
            draft (str): IN: Raw generated SKILL.md. OUT: Sent as part of the
                prompt.
            candidate (SkillCandidate): IN: Original candidate. OUT: Unused
                directly but available for context.

        Returns:
            str: OUT: Refined text, or ``draft`` on failure.
        """

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
        """Parse the ``name`` field from YAML frontmatter.

        Args:
            text (str): IN: SKILL.md content. OUT: Searched for ``name: ...``.

        Returns:
            str | None: OUT: Slugified name or ``None``.
        """

        m = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
        if m:
            return _slugify(m.group(1))
        return None
