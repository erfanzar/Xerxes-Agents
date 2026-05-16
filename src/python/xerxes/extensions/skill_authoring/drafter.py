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
    """Lowercase ``text``, replace non-alphanumerics with hyphens, and truncate to ``max_len``."""

    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if not text:
        text = f"skill-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return text[:max_len].rstrip("-") or "skill"


def _summarise_args(args: dict[str, tp.Any], max_chars: int | None = None) -> str:
    """Render ``args`` as a compact ``"key=value, ..."`` summary."""

    if not args:
        return "(no args)"
    parts = []
    for k, v in args.items():
        sval = str(v).replace("\n", " ")
        if len(sval) > 30:
            sval = sval[:27] + "..."
        parts.append(f"{k}={sval}")
    out = ", ".join(parts)
    if max_chars is not None and len(out) > max_chars:
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
    """Render a ``SKILL.md`` document from a tool-sequence candidate.

    Args:
        candidate: Observed tool calls plus the final response.
        name: Override skill name; defaults to a slug of the prompt or signature.
        description: Override description; defaults to the user prompt.
        version: Semver string placed in the frontmatter.
        tags: Override tag list; defaults to ``candidate.unique_tools``.

    Returns:
        Complete markdown document including YAML frontmatter.
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
    """Produce ``SKILL.md`` files from observed ``SkillCandidate`` tool sequences."""

    def __init__(
        self,
        skills_dir: str | Path,
        llm_client: tp.Any | None = None,
    ) -> None:
        """Initialize the drafter.

        Args:
            skills_dir: Destination directory for drafted skills; created if absent.
            llm_client: Optional LLM client used by ``_refine_with_llm``.
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
        """Render a draft skill from ``candidate`` and optionally persist it.

        Returns:
            ``(markdown, path)``: the rendered text and the file path when
            ``write`` is True, else ``(markdown, None)``.
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
        """Ask the LLM to polish ``draft``; return the raw draft on failure."""

        if self.llm_client is None:
            return draft
        prompt = (
            "Rewrite the following auto-generated SKILL.md to be a clear, concise "
            "agent skill. Preserve the YAML frontmatter, the section headings "
            "(# When to use, # Procedure, # Pitfalls, # Verification), and the "
            "original tool sequence. Do not invent steps or remove the "
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
        """Return a slugified ``name:`` value from the frontmatter, or ``None``."""

        m = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
        if m:
            return _slugify(m.group(1))
        return None
