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
"""Slash command handling for the bridge server.

Extracted from bridge/server.py as a mixin.
"""

from __future__ import annotations

import os
import sys

from ..bridge import profiles
from ..extensions.skills import activate_skill, default_skill_discovery_dirs
from ..llms.registry import calc_cost, resolve_provider
from ..runtime.bridge import populate_registry
from ..runtime.config_context import set_config as set_global_config


class SlashHandlerMixin:
    """Slash command dispatch and individual command handlers."""

    def _run_slash(self, cmd: str, args: str) -> str:
        """Dispatch a leading-slash command to the matching built-in handler.

        Returns the rendered output text, or an unknown-command message when
        ``cmd`` is neither a built-in nor a registered skill.
        """
        if cmd in ("help", "h"):
            return (
                "Commands:\n"
                "  /provider          Setup or switch provider profile\n"
                "  /sampling          View or set sampling parameters\n"
                "  /compact           Summarize conversation to free context\n"
                "  /plan OBJECTIVE    Plan and execute a multi-step task\n"
                "  /agents            List agent types and running agents\n"
                "  /skills            List available skills\n"
                "  /skill NAME        Invoke a skill by name\n"
                "  /skill-create      Create a new skill\n"
                "  /model NAME        Switch model\n"
                "  /cost              Show cost summary\n"
                "  /context           Show context info\n"
                "  /clear             Clear conversation\n"
                "  /tools             List available tools\n"
                "  /thinking          Toggle thinking display\n"
                "  /verbose           Toggle verbose mode\n"
                "  /debug             Toggle debug mode\n"
                "  /permissions       Cycle permission mode\n"
                "  /yolo              Toggle accept-all permission mode\n"
                "  /config            Show config\n"
                "  /history           Show message count\n"
                "  /exit              Exit"
            )

        if cmd == "model":
            if args:
                self._switch_model(args)
                return f"Model set to: {args}"

            current = self.config.get("model", "(none)")
            base_url = self.config.get("base_url", "")
            api_key = self.config.get("api_key", "")
            lines = [f"Current model: {current}"]
            if base_url:
                try:
                    available = profiles.fetch_models(base_url, api_key)
                except Exception as exc:
                    lines.append(f"Could not fetch models from {base_url}/models: {exc}")
                    return "\n".join(lines)
                if available:
                    switched = self._auto_switch_stale_model(available)
                    if switched:
                        previous = current
                        current = switched
                        lines[0] = f"Current model: {current}"
                        lines.append(f"Switched from unavailable model '{previous}' to '{current}'.")
                    lines.append(f"\nAvailable models ({len(available)}):")
                    for m in available:
                        marker = " (active)" if m == current else ""
                        lines.append(f"  {m}{marker}")
                    lines.append("\nUse /model <name> to switch")
                else:
                    lines.append(f"No models returned from {base_url}/models")
            return "\n".join(lines)

        if cmd == "cost":
            return self.cost_tracker.summary()

        if cmd == "history":
            return f"{len(self.state.messages)} messages, {self.state.turn_count} turns"

        if cmd == "verbose":
            self.config["verbose"] = not self.config.get("verbose", False)
            return f"Verbose: {self.config['verbose']}"

        if cmd == "thinking":
            target = args.strip().lower()
            if not target:
                effort = self.config.get("reasoning_effort", "off")
                return f"Thinking: {effort}  (levels: off | low | medium | high)"
            if target not in {"off", "low", "medium", "high"}:
                return f"Unknown thinking level: {target}. Use off|low|medium|high."
            self.config["reasoning_effort"] = target
            self.config["thinking"] = target != "off"
            return f"Thinking effort set to: {target}"

        if cmd == "sampling":
            return self._handle_sampling(args)

        if cmd == "compact":
            return self._handle_compact()

        if cmd == "skills":
            return self._handle_skills_list()

        if cmd == "skill-create":
            return self._handle_skill_create(args)

        if cmd == "debug":
            self.config["debug"] = not self.config.get("debug", False)
            return f"Debug: {self.config['debug']}"

        if cmd == "clear":
            self.state.messages.clear()
            self.state.thinking_content.clear()
            self.state.tool_executions.clear()
            self.state.turn_count = 0
            return "Conversation cleared."

        if cmd == "context":
            model = self.config.get("model", "")
            provider = resolve_provider(model, self.config)
            cost = calc_cost(model, self.state.total_input_tokens, self.state.total_output_tokens)
            return (
                f"CWD: {os.getcwd()}\n"
                f"Model: {model}\n"
                f"Provider: {provider}\n"
                f"Turns: {self.state.turn_count}\n"
                f"Messages: {len(self.state.messages)}\n"
                f"Tokens: {self.state.total_input_tokens} in / {self.state.total_output_tokens} out\n"
                f"Cost: ${cost:.4f}"
            )

        if cmd == "config":
            lines = [f"  {k}: {v}" for k, v in sorted(self.config.items()) if not k.startswith("_")]
            return "\n".join(lines) if lines else "(empty config)"

        if cmd == "permissions":
            modes = ["auto", "accept-all", "manual"]
            current = self.config.get("permission_mode", "accept-all")
            idx = modes.index(current) if current in modes else 0
            new_mode = modes[(idx + 1) % len(modes)]
            self.config["permission_mode"] = new_mode
            set_global_config(self.config)
            return f"Permission mode: {new_mode}"

        if cmd == "yolo":
            current = self.config.get("permission_mode", "accept-all")
            if current == "accept-all":
                self.config["permission_mode"] = "auto"
            else:
                self.config["permission_mode"] = "accept-all"
            set_global_config(self.config)
            return f"YOLO mode {'OFF (auto)' if current == 'accept-all' else 'ON (accept-all)'}"

        if cmd == "tools":
            registry = populate_registry()
            lines = []
            for entry in registry.list_tools():
                safe = " [safe]" if entry.safe else ""
                lines.append(f"  {entry.name}{safe} -- {entry.description[:60]}")
            lines.append(f"  ({registry.tool_count} total)")
            return "\n".join(lines)

        if cmd == "plan":
            return self._handle_plan(args)

        if cmd == "agents":
            return self._handle_agents_list()

        if cmd == "provider":
            plist = profiles.list_profiles()
            active = profiles.get_active_profile()
            active_name = active.get("name") if active else ""
            if not plist:
                return (
                    "No provider profiles configured.\n"
                    "Add one with the JSON-RPC `provider_save` method, or set\n"
                    "the env vars: XERXES_BASE_URL, XERXES_API_KEY, XERXES_MODEL."
                )

            if args:
                self.handle_provider_select({"name": args.strip()})
                return ""

            if self._wire_mode:
                NEW = "+ Create new profile…"
                options = [
                    {
                        "label": p.get("name", ""),
                        "description": (
                            f"{p.get('model', '?')} @ {p.get('base_url', '')}"
                            + ("  (active)" if p.get("name") == active_name else "")
                        ),
                    }
                    for p in plist
                ]
                options.append({"label": NEW, "description": "Add a new provider profile"})
                self._drain_queue(self._question_queue)
                self._active_question_id = self._emit_wire_question_request(
                    [
                        {
                            "id": "provider",
                            "question": "Pick a provider profile",
                            "options": options,
                            "allow_free_form": False,
                        }
                    ]
                )
                answer = self._wait_for_question_response()
                if not answer or answer == "[cancelled]":
                    return "Cancelled."
                if answer == NEW:
                    return self._provider_create_interactive()
                self.handle_provider_select({"name": answer})
                profile = profiles.get_active_profile()
                if profile and profile.get("name") == answer:
                    return f"Switched to '{answer}'  (model: {profile.get('model', '?')})"
                return f"Could not switch to '{answer}'."

            lines = ["Provider profiles:"]
            for p in plist:
                marker = "*" if p.get("name") == active_name else " "
                lines.append(f"  {marker} {p.get('name'):20s}  {p.get('model', '?')}  ({p.get('base_url', '')})")
            lines.append("\n* = active. Pass a profile name to switch: /provider NAME")
            return "\n".join(lines)

        if cmd in ("exit", "quit", "q"):
            self._emit("exit", {})
            sys.exit(0)

        skill = self._skill_registry.get(cmd)
        if skill:
            full_args = f"{cmd}:{args}" if args else cmd
            self._handle_skill_invoke(full_args)
            return ""
        return f"Unknown command: /{cmd} (type /help)"

    def _run_wire_slash(self, cmd: str, args: str) -> None:
        """Execute one of the wire-mode-only slash commands (``btw``/``plan``/``steer``)."""
        if cmd == "btw":
            self._emit_wire_event("btw_begin", {})
            self._emit_wire_event("steer_input", {"content": args})

            self._pending_skill_name = ""
            self.handle_query({"text": args})
            self._emit_wire_event("btw_end", {})
        elif cmd == "plan":
            self._emit_wire_event("plan_display", {"content": "", "file_path": None})

            result = self._handle_plan(args)
            self._emit_wire_event(
                "plan_display",
                {"content": result, "file_path": None},
            )
        elif cmd == "steer":
            self._emit_wire_event("steer_input", {"content": args})

            self.state.messages.append({"role": "user", "content": args})

    def _handle_sampling(self, args: str) -> str:
        """Run ``/sampling``: view / set ``<param> <value>`` / ``reset`` / ``save`` to the active profile."""
        valid = profiles.SAMPLING_PARAMS

        if not args.strip():
            lines = ["Sampling parameters (current session):"]
            for k in sorted(valid):
                current_val = self.config.get(k, None)
                if current_val is not None:
                    lines.append(f"  {k}: {current_val}")
                else:
                    lines.append(f"  {k}: (default)")
            lines.append("")
            lines.append("Usage: /sampling <param> <value>")
            lines.append("       /sampling reset")
            lines.append("       /sampling save  (persist to active profile)")
            return "\n".join(lines)

        parts = args.strip().split(None, 1)
        subcmd = parts[0].lower()

        if subcmd == "reset":
            for k in valid:
                self.config.pop(k, None)
            set_global_config(self.config)
            return "Sampling parameters reset to defaults."

        if subcmd == "save":
            profile = profiles.get_active_profile()
            if not profile:
                return "No active profile. Run /provider first."
            sampling = {}
            for k in valid:
                if k in self.config:
                    sampling[k] = self.config[k]
            profiles.update_sampling(profile["name"], sampling)
            return f"Sampling parameters saved to profile '{profile['name']}'."

        if len(parts) != 2:
            return f"Usage: /sampling <param> <value>\nValid params: {', '.join(sorted(valid))}"

        param = subcmd
        val_str = parts[1]

        if param not in valid:
            return f"Unknown param: {param}\nValid: {', '.join(sorted(valid))}"

        try:
            if param in ("max_tokens", "top_k"):
                val: int | float = int(val_str)
            else:
                val = float(val_str)
        except ValueError:
            return f"Invalid value: {val_str}"

        self.config[param] = val
        set_global_config(self.config)
        return f"{param} = {val}"

    def _generate_skill(self, name: str, description: str) -> str:
        """Ask the active model to author ``SKILL.md`` for ``name``; fall back to the static template."""
        model = self.config.get("model", "")
        if not model:
            return self._create_skill_template(name, description)

        try:
            from openai import OpenAI

            from ..llms.registry import PROVIDERS, bare_model, get_api_key, provider_default_headers

            provider_name = resolve_provider(model, self.config)
            api_key = self.config.get("api_key") or get_api_key(provider_name, self.config)
            prov = PROVIDERS.get(provider_name, PROVIDERS.get("openai"))
            base_url = (
                self.config.get("base_url")
                or self.config.get("custom_base_url")
                or (prov.base_url if prov else None)
                or "https://api.openai.com/v1"
            )
            client = OpenAI(
                api_key=api_key or "dummy",
                base_url=base_url,
                default_headers=provider_default_headers(provider_name) or None,
            )

            response = client.chat.completions.create(
                model=bare_model(model),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate SKILL.md files for the Xerxes agent framework. "
                            "A skill is a reusable set of instructions that an agent follows "
                            "when the skill is invoked via `/skill <name>`.\n\n"
                            "Output format (YAML frontmatter + markdown body):\n"
                            "```\n"
                            "---\n"
                            "name: skill-name\n"
                            "description: One-line description\n"
                            'version: "1.0"\n'
                            "tags: [tag1, tag2]\n"
                            "---\n\n"
                            "# Skill Title\n\n"
                            "Detailed step-by-step instructions for the agent...\n"
                            "```\n\n"
                            "Write clear, actionable instructions. Be specific about what "
                            "tools to use, what to check, and what format to output. "
                            "Output ONLY the SKILL.md content, nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Create a skill named '{name}' that does the following:\n\n{description}",
                    },
                ],
                max_tokens=2048,
                temperature=0.3,
            )

            content = response.choices[0].message.content or ""

            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            if not content.strip():
                return self._create_skill_template(name, description)

        except Exception as exc:
            return self._create_skill_template(name, description, error=str(exc))

        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(content, encoding="utf-8")

        self._skill_registry.discover(str(self._skills_dir))
        self._emit("skills_updated", {"skills": sorted(self._skill_registry.skill_names)})

        return f"Skill '{name}' generated and saved to {skill_dir}/SKILL.md\nUse /skill {name} to invoke it."

    def _handle_compact(self) -> str:
        """Run ``/compact`` through the shared agent-backed provisioner."""
        messages = self.state.messages
        if len(messages) < 4:
            return "Nothing to compact (fewer than 4 messages)."

        model = self.config.get("model", "")
        if not model:
            return "No model configured. Run /provider first."

        original_count = len(self.state.messages)
        result = self._run_context_compaction(force=True)
        if result is None:
            return "Compaction failed: no compaction provisioner is available."
        if not result.compacted:
            detail = f" ({result.error})" if result.error else ""
            return f"Compaction skipped: {result.reason or 'nothing_to_compact'}{detail}."

        new_count = len(self.state.messages)

        return (
            f"Compacted {original_count} messages -> {new_count} messages.\n"
            f"Summarized {result.summarized_count} older messages, kept {result.kept_count} live messages."
        )

    def _handle_skill_invoke(self, args: str) -> None:
        """Activate and run a skill from ``/skill <name>[:<args>]``.

        Validates the skill exists, matches the current platform, then injects
        the skill's prompt section into the conversation and dispatches a
        query against a filtered toolset that excludes ``SkillTool``.
        """
        name = args.strip()
        if not name:
            self._emit("slash_result", {"output": "Usage: /skill <name>\nUse /skills to list available skills."})
            return

        skill_args = ""
        if ":" in name:
            name, skill_args = name.split(":", 1)
            name = name.strip()
            skill_args = skill_args.strip()

        activate_skill(name)

        skill = self._skill_registry.get(name)
        if not skill:
            matches = self._skill_registry.search(name)
            if matches:
                suggestions = ", ".join(s.name for s in matches[:5])
                self._emit("slash_result", {"output": f"Skill '{name}' not found. Did you mean: {suggestions}"})
            else:
                self._emit(
                    "slash_result", {"output": f"Skill '{name}' not found. Use /skills to list available skills."}
                )
            return

        from xerxes.extensions.skills import skill_matches_platform

        if not skill_matches_platform(skill):
            self._emit(
                "slash_result",
                {"output": f"Skill '{name}' is not compatible with this platform ({__import__('sys').platform})."},
            )
            return

        from xerxes.extensions.skills import inject_skill_config

        prompt_section = skill.to_prompt_section()
        config_block = inject_skill_config(skill)
        skill_message = f"[Skill '{name}' activated]{config_block}\n\n{prompt_section}"

        self.state.messages.append(
            {
                "role": "user",
                "content": skill_message,
            }
        )

        self._emit("slash_result", {"output": f"Running skill '{name}'..."})

        trigger = skill_args if skill_args else f"Execute the '{name}' skill now."
        filtered_schemas = [s for s in self.tool_schemas if s.get("name") != "SkillTool"]
        self.handle_query({"text": trigger}, override_tool_schemas=filtered_schemas)

    def _handle_agents_list(self) -> str:
        """Render the ``/agents`` output: registered definitions, load errors, and running sub-agents."""
        from ...agents.definitions import list_agent_definition_load_errors, list_agent_definitions
        from ...tools.claude_tools import _get_agent_manager

        defs = list_agent_definitions()
        lines = [f"Agent types ({len(defs)}):"]
        for d in defs:
            source_tag = f" [{d.source}]" if d.source != "built-in" else ""
            lines.append(f"  {d.name}{source_tag} — {d.description}")
        errors = list_agent_definition_load_errors()
        if errors:
            lines.append("\nAgent spec errors:")
            for error in errors:
                lines.append(f"  {error}")

        mgr = _get_agent_manager()
        tasks = mgr.list_tasks()
        if tasks:
            lines.append(f"\nRunning agents ({len(tasks)}):")
            for t in tasks:
                agent_type = f" ({t.agent_def_name})" if t.agent_def_name else ""
                lines.append(f"  {t.name}{agent_type} [{t.status}] — {t.prompt[:60]}")
        else:
            lines.append("\nNo running agents.")

        return "\n".join(lines)

    def _handle_skill_create(self, args: str) -> str:
        """Stage ``/skill-create <name>``: the next user message becomes the description."""
        name = args.strip()
        if not name:
            return (
                "Usage: /skill-create <name>\n"
                "  Example: /skill-create code-review\n\n"
                "After entering the name, describe what the skill should do\n"
                "and the SKILL.md will be auto-generated."
            )

        if not all(c.isalnum() or c in "-_" for c in name):
            return f"Invalid skill name '{name}'. Use only letters, numbers, hyphens, and underscores."

        skill_dir = self._skills_dir / name
        if skill_dir.exists():
            return f"Skill '{name}' already exists at {skill_dir}"

        self._pending_skill_name = name
        return f"Creating skill '{name}'. Describe what this skill should do:"

    def _handle_skills_list(self) -> str:
        """Re-scan the skill directories and render the ``/skills`` output."""
        self._skill_registry.discover(*default_skill_discovery_dirs(user_skills_dir=self._skills_dir))
        skills = self._skill_registry.get_all()
        if not skills:
            return f"No skills found.\n  Skills directory: {self._skills_dir}\n  Create one with /skill-create"
        lines = [f"Skills ({len(skills)}):"]
        for s in skills:
            tags = f" [{', '.join(s.metadata.tags)}]" if s.metadata.tags else ""
            lines.append(f"  {s.name}{tags} — {s.metadata.description or 'No description'}")
        lines.append("\nUse /skill <name> to invoke a skill")
        return "\n".join(lines)

    def _create_skill_template(self, name: str, description: str, error: str = "") -> str:
        """Write a minimal ``SKILL.md`` scaffold when LLM generation is unavailable or fails."""
        title = name.replace("-", " ").replace("_", " ").title()
        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            f"---\n"
            f"name: {name}\n"
            f"description: {description[:80]}\n"
            f'version: "1.0"\n'
            f"tags: []\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"{description}\n",
            encoding="utf-8",
        )
        self._skill_registry.discover(str(self._skills_dir))
        self._emit("skills_updated", {"skills": sorted(self._skill_registry.skill_names)})
        err_note = f"\n(LLM generation failed: {error}. Created template instead.)" if error else ""
        return f"Skill '{name}' created at {skill_dir}/SKILL.md{err_note}\nUse /skill {name} to invoke it."

    def _handle_plan(self, args: str) -> str:
        """Run ``/plan`` by invoking :class:`PlanTool` with ``args`` as the objective."""
        objective = args.strip()
        if not objective:
            return "Usage: /plan <objective>\n\nExample: /plan refactor the auth module into separate files"

        from ...tools.claude_tools import PlanTool

        return PlanTool.static_call(objective=objective, execute=True)
