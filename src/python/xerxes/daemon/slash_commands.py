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
"""Slash command handlers — all interactive /command implementations.

Extracted from daemon/server.py as a mixin. Each ``_slash_*`` method handles
one command, dispatched from :meth:`SlashCommandsMixin._handle_slash`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from ..bridge import profiles
from ..context.compaction_provisioner import CompactionProvisioner, compaction_summary_agent_from_config
from ..context.window_usage import estimate_context_tokens
from ..core.paths import xerxes_subdir
from ..llms.registry import get_context_limit
from ..runtime.project_workspace import load_project_agent_workspace
from .gateway import EmitFn
from .runtime import DaemonSession, RuntimeManager, SessionManager, WorkspaceManager, render_session_system_prompt

logger = logging.getLogger(__name__)


class SlashCommandsMixin:
    """All ``/``-prefixed slash command handlers for :class:`DaemonServer`.

    The main router is :meth:`_handle_slash` which dispatches to individual
    ``_slash_*`` methods based on the command name.
    """

    config: Any
    runtime: RuntimeManager
    sessions: SessionManager
    workspaces: WorkspaceManager
    channels: Any
    _current_session_key: str
    _current_mode: str
    _current_plan_mode: bool
    _pending_slash_arg: tuple[str, str] | None
    _pending_skill_create: dict[str, Any] | None
    _background_tasks: set[Any]
    _connection_session_key: Callable[[EmitFn], str]
    _emit_init_done: Callable[[EmitFn], Awaitable[None]]
    _emit_slash: Callable[[EmitFn, str], Awaitable[None]]
    _emit_status: Callable[[EmitFn], Awaitable[None]]
    _git_branch: Callable[[Path | None], str]
    _replay_session_history: Callable[[DaemonSession, EmitFn], Awaitable[None]]
    _resolve_context_limit: Callable[[], int]
    _submit_turn: Callable[[dict[str, Any], EmitFn], Awaitable[dict[str, Any]]]
    _sync_runtime_to_connection_session: Callable[[EmitFn], None]
    _track_task: Callable[[Awaitable[Any]], Any]
    shutdown: Callable[[], Awaitable[None]]

    def _session_key_for_emit(self, emit: EmitFn) -> str:
        """Return the session key attached to this slash-command connection."""
        try:
            return self._connection_session_key(emit)
        except Exception:
            return self._current_session_key

    def _session_for_emit(self, emit: EmitFn) -> DaemonSession | None:
        """Return the session attached to this slash-command connection."""
        return self.sessions.get(self._session_key_for_emit(emit))

    async def _slash_soul(self, args: str, emit: EmitFn) -> None:
        """Show the path to the workspace's SOUL.md (or memory soul file)."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        path = session.workspace.path / "SOUL.md"
        if not path.exists():
            await self._emit_slash(emit, f"`SOUL.md` not found at `{path}`.")
            return
        await self._emit_slash(emit, f"Soul / values file: `{path}`\nEdit then `/reload` to pick up changes.")

    async def _slash_feedback(self, args: str, emit: EmitFn) -> None:
        """Show where to file feedback / report bugs."""
        await self._emit_slash(
            emit,
            "Feedback / issues:\n"
            "  • GitHub: https://github.com/erfanzar/Xerxes/issues\n"
            "  • Anything urgent? Mention it in the daemon log first (`~/.xerxes/daemon.log`).",
        )

    async def _slash_cost(self, args: str, emit: EmitFn) -> None:
        """Show the running USD cost for the active session."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        try:
            cost = session.state.cost
        except Exception:
            cost = 0.0
        model = (
            str((getattr(session, "runtime_config", {}) or self.runtime.runtime_config).get("model", ""))
            or self.runtime.model
        )
        await self._emit_slash(emit, f"Estimated cost: `${cost:.4f}` (model: `{model}`).")

    async def _slash_fast(self, args: str, emit: EmitFn) -> None:
        """Toggle fast-mode (cheaper auxiliary model for summaries/titles/etc.)."""
        cur = bool(self.runtime.runtime_config.get("fast_mode", False))
        action = args.strip().lower()
        if action == "on":
            new_value = True
        elif action == "off":
            new_value = False
        else:
            new_value = not cur
        self.runtime.runtime_config["fast_mode"] = new_value
        await self._emit_slash(emit, f"Fast mode: {'ON' if new_value else 'OFF'}.")

    async def _slash_restart(self, args: str, emit: EmitFn) -> None:
        """Schedule a daemon shutdown so the TUI respawns a fresh process on next launch."""
        await self._emit_slash(emit, "Restarting daemon — re-run `xerxes` after this shuts down.")
        self._track_task(self.shutdown())

    async def _slash_save(self, args: str, emit: EmitFn) -> None:
        """Force-persist the active session to disk right now."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session to save.")
            return
        if hasattr(self.sessions, "_session_has_history") and not self.sessions._session_has_history(session):
            await self._emit_slash(emit, "Nothing to save yet — this session has no messages.")
            return
        self.sessions.save(session)
        path = self.sessions._session_path(session.id)
        await self._emit_slash(emit, f"Saved session `{session.id}` to `{path}`.")

    async def _slash_plugins(self, args: str, emit: EmitFn) -> None:
        """List loaded plugins and their slash registrations."""
        plugins: list[Any] = []
        try:
            from ..extensions.slash_plugins import registered_slashes
        except Exception:
            slashes = []
        else:
            slashes = registered_slashes()

        lines = ["Plugins:"]
        if plugins:
            for p in plugins:
                meta = getattr(p, "meta", None)
                name = getattr(meta, "name", str(p))
                lines.append(f"  `{name}`")
        else:
            lines.append("  (no plugins loaded)")
        if slashes:
            lines.append("")
            lines.append("Plugin slash commands:")
            for s in slashes:
                lines.append(f"  `/{s.command.name}` — {s.command.description}")
        await self._emit_slash(emit, "\n".join(lines))

    def _help_text(self) -> str:
        """Render the ``/help`` output grouped by command category."""
        from ..bridge.commands import CATEGORIES, COMMAND_REGISTRY

        # Group by category for readable help output.
        by_cat: dict[str, list[Any]] = {c: [] for c in CATEGORIES}
        for cmd in COMMAND_REGISTRY:
            by_cat.setdefault(cmd.category, []).append(cmd)
        lines = ["Slash commands (run `/commands` for the flat list):"]
        for cat in CATEGORIES:
            cmds = by_cat.get(cat, [])
            if not cmds:
                continue
            lines.append(f"\n  [{cat}]")
            for c in cmds:
                # Wrap each ``/cmd <hint>`` in a backtick code span so the
                # Markdown renderer the TUI applies to slash notifications
                # doesn't silently strip angle-bracketed placeholders as
                # HTML tags.
                hint = f" {c.args_hint}" if c.args_hint else ""
                lines.append(f"    `/{c.name}{hint}`  — {c.description}")
        return "\n".join(lines)

    async def _slash_config(self, args: str, emit: EmitFn) -> None:
        """Dump non-underscore runtime config keys, with secrets redacted."""
        cfg = self.runtime.runtime_config
        redact_keys = {"api_key", "auth_token", "token", "secret"}
        lines = ["Runtime config:"]
        for key in sorted(cfg):
            if key.startswith("_"):
                continue
            value = cfg[key]
            if any(s in key.lower() for s in redact_keys) and value:
                value = "***redacted***"
            lines.append(f"  `{key}` = `{value}`")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_budget(self, args: str, emit: EmitFn) -> None:
        """Show context-window usage and remaining headroom."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        from ..llms.registry import get_context_limit

        runtime_config = getattr(session, "runtime_config", {}) or self.runtime.runtime_config
        model = str(runtime_config.get("model", "")) or self.runtime.model or ""
        limit = get_context_limit(model) if model else 0
        system_prompt = render_session_system_prompt(
            self.runtime,
            session,
            mode=str(getattr(session, "interaction_mode", runtime_config.get("mode", "code")) or "code"),
            tolerate_errors=True,
        )
        used = estimate_context_tokens(
            session.state.messages,
            model=model,
            system_prompt=system_prompt,
            tool_schemas=self.runtime.tool_schemas,
        )
        remaining = max(0, limit - used)
        pct = (used / limit * 100) if limit else 0.0
        await self._emit_slash(
            emit,
            f"Context window: `{limit or '?'}` tokens for `{model or '(unknown)'}`\n"
            f"Used: {used} ({pct:.1f}%) · Remaining: {remaining}",
        )

    async def _slash_statusbar(self, args: str, emit: EmitFn) -> None:
        """Statusbar is TUI-side; ack with a hint."""
        await self._emit_slash(emit, "Statusbar visibility is a TUI-side setting.")

    async def _slash_compact(self, args: str, emit: EmitFn) -> None:
        """Force agent-backed compaction of the active session transcript."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session to compact.")
            return
        if len(session.state.messages) < 2:
            await self._emit_slash(emit, "Nothing to compact.")
            return

        runtime_config = dict(getattr(session, "runtime_config", {}) or self.runtime.runtime_config)
        model = str(runtime_config.get("model", "")) or self.runtime.model
        if not model:
            await self._emit_slash(emit, "No model configured. Run `/provider` first.")
            return

        context_limit = int(
            runtime_config.get("max_context_tokens")
            or runtime_config.get("context_limit")
            or runtime_config.get("max_context")
            or get_context_limit(model)
            or 128_000
        )
        threshold_tokens = runtime_config.get("compaction_threshold_tokens")
        target_tokens = runtime_config.get("compaction_target_tokens")
        provisioner = CompactionProvisioner(
            model=model,
            max_context_tokens=context_limit,
            threshold_tokens=int(threshold_tokens) if threshold_tokens is not None else None,
            target_tokens=int(target_tokens) if target_tokens is not None else None,
            threshold_ratio=float(runtime_config.get("compaction_threshold", 0.75)),
            target_ratio=float(runtime_config.get("compaction_target", 0.5)),
            summary_agent=compaction_summary_agent_from_config(model, runtime_config),
        )

        original_count = len(session.state.messages)
        result = await asyncio.to_thread(provisioner.compact, session.state.messages, force=True)
        if not result.compacted:
            detail = f" ({result.error})" if result.error else ""
            await self._emit_slash(emit, f"Compaction skipped: {result.reason or 'nothing_to_compact'}{detail}.")
            return

        session.state.messages = result.messages
        session.state.metadata["last_compaction"] = {
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "summarized_count": result.summarized_count,
            "kept_count": result.kept_count,
            "max_context_tokens": context_limit,
        }
        try:
            self.sessions.save(session)
        except Exception as exc:
            logger.warning("Failed to persist compacted session %s: %s", session.id, exc)

        await self._emit_slash(
            emit,
            f"Compacted {original_count} messages -> {len(session.state.messages)} messages. "
            f"Tokens: {result.tokens_before:,} -> {result.tokens_after:,}.",
        )
        await self._emit_status(emit)

    async def _slash_update(self, args: str, emit: EmitFn) -> None:
        """Show the installed Xerxes version and release/git update status."""
        from ..runtime.update import (
            check_for_update,
            format_git_update_status,
            git_update_status,
            installed_version,
        )

        ver = installed_version()
        package_update = check_for_update()
        package_line = "Package: current or PyPI unavailable"
        if package_update is not None:
            package_line = (
                f"Package: `{package_update.latest_version}` available (installed `{package_update.installed_version}`)"
            )
        git_line = format_git_update_status(git_update_status(fetch=True, timeout=2.0))
        await self._emit_slash(
            emit,
            f"Xerxes `{ver}`\n{package_line}\nGit: {git_line}\nRun: `xerxes update`.",
        )

    async def _slash_resume(self, args: str, emit: EmitFn) -> None:
        """List or switch this TUI connection to a saved session."""
        target = args.strip()
        if target:
            matches = self._find_saved_sessions(target)
            if not matches:
                await self._emit_slash(emit, f"No saved session matches `{target}`. Run `/resume` to list sessions.")
                return
            if len(matches) > 1:
                await self._emit_resume_choices(
                    matches[:20],
                    emit,
                    body=f"Multiple sessions match `{target}`. Choose one, or type the full session id.",
                )
                return

            current_key = self._connection_session_key(emit)
            current = self.sessions.get(current_key)
            if current is not None:
                self.sessions.save(current)

            session_id = str(matches[0].get("session_id") or matches[0].get("id") or "")
            if not session_id:
                await self._emit_slash(emit, f"Saved session `{target}` is missing a session id.")
                return

            self._current_session_key = session_id
            conn_sessions = getattr(self, "_connection_sessions", None)
            if conn_sessions is not None:
                conn_sessions[emit] = session_id

            session = self.sessions.open(session_id, self.workspaces.default_agent_id)
            await emit(
                "notification",
                {
                    "id": uuid.uuid4().hex[:12],
                    "category": "history",
                    "type": "resume_begin",
                    "severity": "info",
                    "title": "",
                    "body": f"── switching to session {session.id} ──",
                    "payload": {"session_id": session.id},
                },
            )
            await emit(
                "init_done",
                {
                    "model": self.runtime.model,
                    "session_id": session.id,
                    "cwd": str(session.project_dir),
                    "git_branch": self._git_branch(session.project_dir),
                    "context_limit": self._resolve_context_limit(),
                    "agent_name": session.agent_id,
                    "skills": self.runtime.discover_skills(),
                },
            )
            await self._emit_status(emit)
            await self._replay_session_history(session, emit)
            return

        records = self._saved_session_records()
        if not records:
            await self._emit_slash(emit, "No saved sessions found.")
            return
        await self._emit_resume_choices(records[:20], emit, body="Choose a saved session, or type `/resume <id>`.")

    async def _emit_resume_choices(self, records: list[dict[str, Any]], emit: EmitFn, *, body: str) -> None:
        """Emit saved-session choices as structured data for the TUI picker."""
        choices = [self._session_choice_payload(r) for r in records]
        await emit(
            "notification",
            {
                "id": uuid.uuid4().hex[:12],
                "category": "history",
                "type": "resume_choices",
                "severity": "info",
                "title": "Resume session",
                "body": body,
                "payload": {"sessions": choices},
            },
        )

    @staticmethod
    def _session_choice_payload(record: dict[str, Any]) -> dict[str, Any]:
        """Return a compact record payload for TUI session picking."""
        sid = str(record.get("session_id") or record.get("id") or "")
        messages = record.get("messages", 0)
        if isinstance(messages, list):
            message_count = len(messages)
        else:
            try:
                message_count = int(messages or 0)
            except (TypeError, ValueError):
                message_count = 0
        return {
            "id": sid,
            "session_id": sid,
            "title": str(record.get("title") or "").strip() or sid,
            "updated_at": str(record.get("updated_at") or ""),
            "turn_count": int(record.get("turn_count", 0) or 0),
            "messages": message_count,
            "agent_id": str(record.get("agent_id") or ""),
        }

    def _saved_session_records(self) -> list[dict[str, Any]]:
        """Return saved sessions, falling back for lightweight test doubles."""
        if hasattr(self.sessions, "list_saved"):
            return self.sessions.list_saved()
        return self.sessions.list()

    def _find_saved_sessions(self, query: str) -> list[dict[str, Any]]:
        """Find saved sessions, falling back for lightweight test doubles."""
        if hasattr(self.sessions, "find_saved"):
            return self.sessions.find_saved(query)
        needle = query.strip().lower()
        return [
            record
            for record in self._saved_session_records()
            if needle
            and (
                str(record.get("session_id") or record.get("id") or "").lower().startswith(needle)
                or str(record.get("key") or "").lower() == needle
                or str(record.get("title") or "").lower() == needle
            )
        ]

    async def _slash_memory(self, args: str, emit: EmitFn) -> None:
        """Show where the agent's memory files live + their sizes."""
        from ..runtime.agent_memory import default_global_memory_dir, project_memory_dir_for

        global_dir = default_global_memory_dir()
        try:
            project_dir = project_memory_dir_for(self.config.project_dir or os.getcwd())
        except Exception:
            project_dir = None

        def _sz(path):
            try:
                return path.stat().st_size if path.exists() else 0
            except OSError:
                return 0

        lines = ["Memory:"]
        lines.append(f"  Global scope: `{global_dir}`")
        for name in ("SOUL.md", "IDENTITY.md", "USER.md", "MEMORY.md", "KNOWLEDGE.md", "INSIGHTS.md", "EXPERIENCES.md"):
            path = global_dir / name
            lines.append(f"    `{name}` — {_sz(path)} bytes" + ("" if path.exists() else "  (missing)"))
        if project_dir is not None:
            lines.append(f"  Project scope: `{project_dir}`")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_insights(self, args: str, emit: EmitFn) -> None:
        """Show top tools by call count from the session's execution log."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        counts: dict[str, int] = {}
        for ex in session.state.tool_executions:
            name = ex.get("name", "(unknown)")
            counts[name] = counts.get(name, 0) + 1
        if not counts:
            await self._emit_slash(emit, "No tools invoked in this session yet.")
            return
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        lines = ["Top tools this session:"]
        for name, n in top:
            lines.append(f"  `{name}` — {n} call{'s' if n != 1 else ''}")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_branch(self, args: str, emit: EmitFn) -> None:
        """Branch / fork the current session — saves a copy under a new id."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session to branch.")
            return
        import copy

        new_id = uuid.uuid4().hex[:12]
        clone = DaemonSession(
            id=new_id,
            key=new_id,
            agent_id=session.agent_id,
            workspace=session.workspace,
            project_dir=session.project_dir,
        )
        clone.state.messages = copy.deepcopy(session.state.messages)
        clone.state.turn_count = session.state.turn_count
        self.sessions._sessions[new_id] = clone
        self.sessions.save(clone)
        await self._emit_slash(
            emit,
            f"Branched to new session `{new_id}` ({len(clone.state.messages)} messages).\n"
            f"Open in a new terminal with `xerxes -r {new_id}`.",
        )

    async def _slash_retry(self, args: str, emit: EmitFn) -> None:
        """Resend the most recent user message after dropping the failed reply."""
        session = self._session_for_emit(emit)
        if session is None or not session.state.messages:
            await self._emit_slash(emit, "Nothing to retry.")
            return
        msgs = session.state.messages
        last_user_text = ""
        for msg in reversed(msgs):
            if msg.get("role") == "user":
                content = msg.get("content")
                last_user_text = content if isinstance(content, str) else str(content)
                break
        if not last_user_text:
            await self._emit_slash(emit, "No prior user message to retry.")
            return
        # Drop everything after the last user message so the model retries cleanly.
        while msgs and msgs[-1].get("role") != "user":
            msgs.pop()
        if msgs:
            msgs.pop()  # remove the user message too — _submit_turn re-appends it
        metadata = getattr(session.state, "metadata", None)
        if isinstance(metadata, dict):
            metadata.pop("last_connection_failure", None)
        await self._emit_slash(emit, "Retrying the last prompt…")
        await self._submit_turn({"text": last_user_text, "_internal_slash": True}, emit)

    async def _slash_retry_connection(self, args: str, emit: EmitFn) -> None:
        """Retry the most recent provider request that exhausted connection retries."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No failed provider connection to retry.")
            return

        metadata = getattr(session.state, "metadata", None)
        failure = metadata.get("last_connection_failure") if isinstance(metadata, dict) else None
        if not isinstance(failure, dict):
            await self._emit_slash(emit, "No failed provider connection to retry.")
            return

        last_user_text = str(failure.get("user_message") or "").strip()
        if not last_user_text:
            await self._emit_slash(emit, "Last provider failure did not record a retryable prompt.")
            return

        msgs = session.state.messages
        while msgs and msgs[-1].get("role") != "user":
            msgs.pop()
        if msgs and msgs[-1].get("role") == "user":
            msgs.pop()
        if isinstance(metadata, dict):
            metadata.pop("last_connection_failure", None)
        self.sessions.save(session)
        await self._emit_slash(emit, "Retrying the last failed provider connection…")
        await self._submit_turn({"text": last_user_text, "_internal_slash": True}, emit)

    async def _slash_undo(self, args: str, emit: EmitFn) -> None:
        """Drop the last user/assistant turn pair from the active session."""
        session = self._session_for_emit(emit)
        if session is None or not session.state.messages:
            await self._emit_slash(emit, "Nothing to undo.")
            return
        msgs = session.state.messages
        # Pop assistant + matching user (and any intervening tool messages).
        dropped = 0
        while msgs and msgs[-1].get("role") != "user":
            msgs.pop()
            dropped += 1
        if msgs and msgs[-1].get("role") == "user":
            msgs.pop()
            dropped += 1
        session.state.turn_count = max(0, session.state.turn_count - 1)
        self.sessions.save(session)
        await self._emit_slash(emit, f"Undone — dropped {dropped} message(s) from the conversation.")

    async def _slash_cancel_all(self, args: str, emit: EmitFn) -> None:
        """Cancel every running turn across every session."""
        count = self.sessions.cancel_all()
        await self._emit_slash(emit, f"Cancelled {count} running turn{'s' if count != 1 else ''}.")

    async def _slash_personality(self, args: str, emit: EmitFn) -> None:
        """Show the path to the workspace's persona file."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        path = session.workspace.path / "AGENTS.md"
        if not path.exists():
            await self._emit_slash(emit, f"`AGENTS.md` not found at `{path}`. Run any session prompt to seed it.")
            return
        await self._emit_slash(emit, f"Persona / instructions file: `{path}`\nEdit with your `$EDITOR`, then `/reload`.")

    async def _handle_slash(self, command: str, emit: EmitFn) -> None:
        """Dispatch a ``/cmd`` string to plugins, built-ins, or skill shorthand.

        Resolution order: slash-plugin registry, built-in commands (``/help``,
        ``/skills``, ``/yolo``, ``/permissions``, ``/context``, ...), then the
        skill registry treating ``cmd`` (or ``cmd:sub``) as a skill name.
        """
        stripped = command.strip()
        if not stripped.startswith("/"):
            return
        raw = stripped[1:]
        cmd, _, args = raw.partition(" ")
        cmd = cmd.strip().lower()
        args = args.strip()

        # Resolve aliases against the canonical registry so /q, /quit, /reset, etc. all work.
        from ..bridge.commands import resolve_command
        from ..extensions.slash_plugins import resolve_slash

        original_cmd = cmd  # preserve typed name for user-facing responses
        resolved = resolve_command(cmd)
        if resolved is not None and resolved.name != cmd:
            cmd = resolved.name

        # Plugin-registered slash commands.
        plugin = resolve_slash(cmd)
        if plugin is not None:
            try:
                out = plugin.handler(args) if args else plugin.handler()
            except Exception as exc:
                await self._emit_slash(emit, f"Plugin /{cmd} failed: {exc}")
                return
            await self._emit_slash(emit, str(out) if out is not None else f"Plugin /{cmd} ran.")
            return

        if cmd == "skills":
            await self._emit_slash(emit, self.runtime.skills_list_text())
            await emit("init_done", {"skills": self.runtime.discover_skills()})
            return

        if cmd == "skill":
            await cast(Any, self)._slash_skill(args, emit)
            return

        if cmd == "skill-create":
            await cast(Any, self)._slash_skill_create(args, emit)
            return

        if cmd == "yolo":
            new_mode = self.runtime.toggle_yolo()
            label = "ON (accept-all)" if new_mode == "accept-all" else "OFF (auto)"
            await self._emit_slash(emit, f"YOLO mode {label}")
            return

        if cmd == "permissions":
            if args:
                new_mode = self.runtime.set_permission_mode(args)
                await self._emit_slash(emit, f"Permission mode: {new_mode}")
            else:
                # Cycle through valid modes (matches bridge/server behavior).
                modes = ["auto", "accept-all", "manual"]
                current = self.runtime.permission_mode
                idx = modes.index(current) if current in modes else 0
                new_mode = self.runtime.set_permission_mode(modes[(idx + 1) % len(modes)])
                await self._emit_slash(emit, f"Permission mode: {new_mode}")
            return

        # ``/thinking`` (alias ``/reasoning``) shows/sets the reasoning effort level.
        if cmd in {"thinking", "reasoning"}:
            await self._slash_thinking(args, emit)
            return

        if cmd in {"verbose", "debug"}:
            new_value = self.runtime.toggle_flag(cmd)
            await self._emit_slash(emit, f"{original_cmd.title()}: {new_value}")
            return

        if cmd == "help":
            await self._emit_slash(emit, self._help_text())
            return

        if cmd == "commands":
            from ..bridge.commands import COMMAND_REGISTRY

            lines = [f"  /{c.name:<14} {c.description}" for c in COMMAND_REGISTRY]
            await self._emit_slash(emit, "Commands:\n" + "\n".join(lines))
            return

        if cmd == "context":
            status = self.runtime.status()
            await self._emit_slash(
                emit,
                "\n".join(
                    [
                        f"Model:           {status['model']}",
                        f"Permission mode: {status['permission_mode']}",
                        f"Tools:           {status['tools']}",
                        f"Skills:          {status['skills']}",
                        f"Workspace:       {self.config.project_dir or os.getcwd()}",
                    ]
                ),
            )
            return

        if cmd == "status":
            await self._emit_slash(emit, json.dumps(self.runtime.status(), indent=2))
            return

        if cmd == "provider":
            await cast(Any, self)._slash_provider(args, emit)
            return

        if cmd in {"exit", "quit", "q"}:
            await self._emit_slash(emit, "(use Ctrl+D or close the terminal to exit)")
            return

        if cmd == "clear":
            # Clear is a per-session concern; the TUI handles its own scrollback,
            # so we just acknowledge.
            await self._emit_slash(emit, "Cleared.")
            return

        # Dispatched via ``_BULK_SLASH_HANDLERS`` so adding a command means one
        # line in the registry + one line in the dispatch table below.
        handler = _BULK_SLASH_HANDLERS.get(cmd)
        if handler is not None:
            await handler(self, args, emit)
            return

        # ``/autoresearch:fix`` should resolve to the autoresearch skill with
        # the ``fix`` subcommand. We split before looking up the skill so a
        # skill named "autoresearch" handles "fix" as a sub-command, not the
        # whole token.
        skill_name = cmd
        subcommand = ""
        if ":" in cmd:
            skill_name, subcommand = cmd.split(":", 1)

        skill = self.runtime.skill_registry.get(skill_name)
        if skill is not None:
            # Validate subcommand against the skill's declared sub-commands.
            declared_subs = skill.metadata.subcommands
            if subcommand and declared_subs and subcommand not in declared_subs:
                await self._emit_slash(
                    emit,
                    f"Skill /{skill_name} has no sub-command '{subcommand}'. "
                    f"Available: {', '.join('/' + skill_name + ':' + s for s in declared_subs)}",
                )
                return
            # Build the args string the existing dispatcher understands:
            #   "<name>"            — no subcommand, no args
            #   "<name>:<sub>"      — subcommand only
            #   "<name>:<sub> <a>"  — subcommand + free-form args
            if subcommand and args:
                composed = f"{skill_name}:{subcommand} {args}"
            elif subcommand:
                composed = f"{skill_name}:{subcommand}"
            elif args:
                composed = f"{skill_name}:{args}"
            else:
                composed = skill_name
            await cast(Any, self)._slash_skill(composed, emit, run_now=True)
            return

        # If the command is in the registry but slipped through every branch
        # above, fail honestly so the gap can be caught by a grep rather than
        # disguised as "unknown command".
        if resolved is not None:
            await self._emit_slash(
                emit,
                f"`/{cmd}` is registered but not yet wired in the daemon. "
                "Open an issue with the command name and what you expected.",
            )
            return
        await self._emit_slash(emit, f"Unknown command: /{cmd} (type /help)")

    async def _slash_sampling(self, args: str, emit: EmitFn) -> None:
        """Show or update sampling params (temperature, top_p, max_tokens)."""
        if not args.strip():
            cfg = self.runtime.runtime_config
            lines = ["Sampling:"]
            for key in ("temperature", "top_p", "top_k", "max_tokens", "presence_penalty", "frequency_penalty"):
                if key in cfg:
                    lines.append(f"  `{key}` = `{cfg[key]}`")
            if len(lines) == 1:
                lines.append("  (provider defaults — no overrides set)")
            lines.append("\nUpdate with `/sampling <key> <value>` (e.g. `/sampling temperature 0.7`).")
            await self._emit_slash(emit, "\n".join(lines))
            return
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            await self._emit_slash(emit, "Usage: `/sampling <key> <value>`")
            return
        key, raw = parts[0].strip(), parts[1].strip()
        # Coerce numerics; non-numeric values are stored as strings.
        value: Any = raw
        try:
            value = float(raw) if "." in raw else int(raw)
        except ValueError:
            pass
        self.runtime.runtime_config[key] = value
        await self._emit_slash(emit, f"Set sampling `{key}` = `{value}`.")

    async def _slash_toolsets(self, args: str, emit: EmitFn) -> None:
        """List subagent / toolset definitions discovered on disk."""
        try:
            from ..agents.definitions import list_agent_definitions
        except Exception:
            await self._emit_slash(emit, "Toolset listing unavailable in this build.")
            return
        defs = list_agent_definitions()
        if not defs:
            await self._emit_slash(emit, "No agent toolsets configured.")
            return
        lines = [f"Agent toolsets ({len(defs)}):"]
        for d in defs:
            name = getattr(d, "name", str(d))
            descr = getattr(d, "description", "") or ""
            lines.append(f"  `{name}` — {descr[:80]}")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_model(self, args: str, emit: EmitFn) -> None:
        """Show the active model or switch to a new one.

        ``/model`` lists the active model + provider. ``/model <id>`` rebinds
        the runtime to that id (full path or short alias). Profiles are
        managed separately via ``/provider``.
        """
        target = args.strip()
        session = self._session_for_emit(emit)
        cfg = (getattr(session, "runtime_config", {}) if session is not None else None) or self.runtime.runtime_config
        if not target:
            base_url = str(cfg.get("base_url", ""))
            api_key = str(cfg.get("api_key", ""))
            active = str(cfg.get("model", "")) or self.runtime.model or "(none)"
            lines = [
                f"Active model: `{active}`",
                f"Base URL:     `{base_url or '(provider default)'}`",
            ]
            models: list[str] = []
            if base_url:
                try:
                    models = await asyncio.to_thread(profiles.fetch_models, base_url, api_key)
                except Exception as exc:
                    lines.append(f"\n(could not fetch model list from provider: {exc})")
            if models:
                lines.append("\nAvailable models from provider:")
                lines.extend(f"  {'● ' if m == active else '  '}{m}" for m in models)
            lines.append("\nSwitch with `/model <id>` or pick a profile with `/provider`.")
            await self._emit_slash(emit, "\n".join(lines))
            return
        # Validate against the provider's catalogue (best-effort): reject an
        # unknown id rather than silently binding a model that 404s at call time.
        # If the catalogue can't be fetched, we skip validation and allow it.
        base_url = str(cfg.get("base_url", ""))
        if base_url:
            from ..llms.registry import bare_model

            try:
                available = await asyncio.to_thread(profiles.fetch_models, base_url, str(cfg.get("api_key", "")))
            except Exception:
                available = []
            if available and not ({target, bare_model(target)} & set(available)):
                preview = ", ".join(available[:8]) + (" …" if len(available) > 8 else "")
                await self._emit_slash(
                    emit,
                    f"Unknown model `{target}` for this provider.\n"
                    f"Available: {preview}\n"
                    f"Run `/model` for the full list, or `/provider` to switch profiles.",
                )
                return
        try:
            self.runtime.reload({"model": target})
            self._sync_runtime_to_connection_session(emit)
        except Exception as exc:
            await self._emit_slash(emit, f"Failed to switch model: `{exc}`")
            return
        await self._emit_slash(emit, f"Model set to `{self.runtime.model}`.")
        await self._emit_init_done(emit)

    async def _slash_reload(self, args: str, emit: EmitFn) -> None:
        """Reload tools + skills from disk without restarting the daemon."""
        try:
            self.runtime.reload({})
            self._sync_runtime_to_connection_session(emit)
        except Exception as exc:
            await self._emit_slash(emit, f"Reload failed: `{exc}`")
            return
        skills = self.runtime.discover_skills()
        await emit("init_done", {"skills": skills})
        await self._emit_slash(
            emit,
            f"Reloaded. Tools: {len(self.runtime.tool_schemas)} · Skills: {len(skills)}.",
        )

    async def _slash_usage(self, args: str, emit: EmitFn) -> None:
        """Show token usage for the active session (alias-ish for /context)."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        st = session.state
        total = st.total_input_tokens + st.total_output_tokens
        lines = [
            f"Input tokens:        {st.total_input_tokens}",
            f"Output tokens:       {st.total_output_tokens}",
            f"Total tokens:        {total}",
            f"Cache read tokens:   {st.total_cache_read_tokens}",
            f"Cache create tokens: {st.total_cache_creation_tokens}",
        ]
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_nudge(self, args: str, emit: EmitFn) -> None:
        """Toggle Honcho-style background nudges."""
        cur = bool(self.runtime.runtime_config.get("nudge", True))
        action = args.strip().lower()
        if action == "on":
            new_value = True
        elif action == "off":
            new_value = False
        else:
            new_value = not cur
        self.runtime.runtime_config["nudge"] = new_value
        await self._emit_slash(emit, f"Nudge: {'ON' if new_value else 'OFF'}.")

    async def _slash_reload_mcp(self, args: str, emit: EmitFn) -> None:
        """Reload MCP server connections."""
        try:
            from ..mcp import MCPManager
        except Exception:
            await self._emit_slash(emit, "MCP support not available in this build.")
            return
        if not issubclass(MCPManager, object):
            await self._emit_slash(emit, "MCP support not available in this build.")
            return
        await self._emit_slash(
            emit,
            "MCP reload is not wired to a daemon-owned MCP registry yet. Restart Xerxes after changing MCP config.",
        )

    async def _slash_history(self, args: str, emit: EmitFn) -> None:
        """Show message and turn counts for the active session."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session yet.")
            return
        st = session.state
        await self._emit_slash(
            emit,
            f"Messages: {len(st.messages)}\nTurns: {st.turn_count}\n"
            f"Input tokens: {st.total_input_tokens}\nOutput tokens: {st.total_output_tokens}",
        )

    async def _slash_stop(self, args: str, emit: EmitFn) -> None:
        """Cancel the currently in-flight tool / turn in this session."""
        cancelled = self.sessions.cancel(self._session_key_for_emit(emit))
        await self._emit_slash(emit, "Cancelled." if cancelled else "Nothing running to cancel.")

    async def _slash_new(self, args: str, emit: EmitFn) -> None:
        """Drop the cached session and start fresh — like a clean re-launch."""
        session_key = self._session_key_for_emit(emit)
        self.sessions.evict(session_key)
        session = self.sessions.open(session_key, self.workspaces.default_agent_id)
        await self._emit_slash(emit, f"New session `{session.id}` started. Scrollback cleared on TUI side.")

    async def _steer_session(self, session_key: str, args: str, emit: EmitFn) -> None:
        """Inject a steering hint into the active turn (``/btw`` and ``/steer``).

        Mid-turn the content is queued on the active session; the streaming
        loop drains it between tool iterations and prepends it as a synthetic
        user message before the next LLM request. With no active turn the
        steer lands directly in ``state.messages`` so the next turn sees it.
        ``steer_input`` is still emitted so the TUI can render the injection.
        """
        content = args.strip()
        if not content:
            await self._emit_slash(emit, "Usage: `/steer <hint>` (also `/btw`). Anything after the command is injected.")
            return
        session = self.sessions.get(session_key)
        if session is None:
            await self._emit_slash(emit, "No active session to steer.")
            return
        await emit("steer_input", {"content": content})
        if session.active_turn_id:
            session.pending_steers.put(content)
            await self._emit_slash(emit, "Steer queued — will land before the next provider request.")
        else:
            session.state.messages.append({"role": "user", "content": f"[steer from user]\n{content}"})
            await self._emit_slash(emit, "Steer injected — will land on the next turn.")

    async def _slash_steer(self, args: str, emit: EmitFn) -> None:
        """Inject a steering hint into the current slash-command session."""
        await self._steer_session(self._session_key_for_emit(emit), args, emit)

    async def _slash_paste(self, args: str, emit: EmitFn) -> None:
        """Paste is TUI-side; ack with a hint."""
        await self._emit_slash(emit, "Paste is handled in the TUI via `Ctrl+V` / `Alt+V`.")

    async def _slash_doctor(self, args: str, emit: EmitFn) -> None:
        """Run a quick self-check: provider configured, tools loaded, skills found."""
        lines = ["Diagnostics:"]
        ok = True
        if not self.runtime.model:
            lines.append("  ✗ No model configured — run `/provider` or set `XERXES_MODEL`.")
            ok = False
        else:
            lines.append(f"  ✓ Model: `{self.runtime.model}`")
        tools = len(self.runtime.tool_schemas)
        lines.append(f"  {'✓' if tools else '✗'} Tools loaded: {tools}")
        skills = len(self.runtime.discover_skills())
        lines.append(f"  ✓ Skills discovered: {skills}")
        sessions = len(self._saved_session_records())
        lines.append(f"  ✓ Saved sessions on disk: {sessions}")
        lines.append("")
        lines.append("All good." if ok else "Something needs attention — see above.")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_title(self, args: str, emit: EmitFn) -> None:
        """Set or clear the current session's title."""
        session = self._session_for_emit(emit)
        if session is None:
            await self._emit_slash(emit, "No active session yet — start chatting first.")
            return
        title = args.strip()
        if not hasattr(session.state, "metadata") or session.state.metadata is None:
            session.state.metadata = {}
        if title:
            session.state.metadata["title"] = title
            await self._emit_slash(emit, f"Session title set to `{title}`.")
        else:
            current = session.state.metadata.get("title", "")
            await self._emit_slash(emit, f"Session title: `{current or '(unset)'}`.")

    async def _slash_thinking(self, args: str, emit: EmitFn) -> None:
        """Show or set the reasoning ('thinking') effort level.

        ``/thinking`` shows the current effort and how it maps to the active
        provider; ``/thinking <off|low|medium|high>`` sets it.
        """
        from ..llms.registry import resolve_provider

        target = args.strip().lower()
        if not target:
            state = self.runtime.reasoning_state()
            provider = resolve_provider(self.runtime.model or "", self.runtime.runtime_config) or "?"
            if state["effort"] == "off":
                mapping = "reasoning disabled"
            elif provider in {"anthropic", "claude"}:
                mapping = f"{provider} → budget_tokens={state['budget_tokens']}"
            elif provider == "claude-code":
                mapping = f"{provider} → --effort {state['effort']}"
            else:
                mapping = f"{provider} → reasoning_effort={state['effort']}"
            await self._emit_slash(
                emit,
                f"Thinking: {state['effort']}  ({mapping})\n"
                f"Levels: off | low | medium | high\n"
                f"Set with `/thinking <level>`.",
            )
            return
        try:
            state = self.runtime.set_reasoning_effort(target)
        except ValueError as exc:
            await self._emit_slash(emit, str(exc))
            return
        await self._emit_slash(emit, f"Thinking effort set to: {state['effort']}.")
        if getattr(self, "sessions", None) is not None:
            sync_session = getattr(self, "_sync_runtime_to_connection_session", None)
            if callable(sync_session):
                sync_session(emit)
            await self._emit_status(emit)

    async def _slash_skin(self, args: str, emit: EmitFn) -> None:
        """Skin is a pure TUI concern; ack and point at the live TUI command."""
        await self._emit_slash(
            emit, "Skin is a TUI-side setting. Use `/skin <name>` in the TUI, or set `XERXES_SKIN=<name>`."
        )

    @staticmethod
    def _project_init_prompt(project_dir: Path, args: str = "") -> str:
        """Build the agent-facing project initialization task."""
        extra = args.strip()
        requested = f"\nUser request for this init: {extra}\n" if extra else ""
        return f"""Initialize this repository for Xerxes by running a swarm-backed project discovery.

Project root: `{project_dir}`.{requested}
You are the setup lead. Do not use a generic template and do not assume the project shape.
The output must be project-specific enough that a future agent can work here without the user rewriting context.

Mandatory swarm stage:
- Before writing `XERXES.md` or `.agents/` files, spawn parallel discovery subagents with `SpawnAgents`.
- Do not cap the swarm with an arbitrary number. Start with the obvious discovery lanes and add more subagents if the repository shape demands it.
- Cover separate lanes for repository structure, build/test/development commands, architecture and runtime flow, coding conventions and style, existing docs and user intent, repeated workflows that deserve skills, and risks or caveats future agents must know.
- Give every subagent a concrete scope, root path, expected evidence, and instruction to return concise findings plus file paths it inspected.
- Use `AwaitAgents`, `TaskGetTool`, or `TaskOutputTool` to collect results. If subagent tools are unavailable, stop and report that `/init` cannot safely produce project context instead of guessing.
- Synthesize the swarm findings yourself. Do not paste raw subagent transcripts into project files.

Workflow:
- Inspect the repository from the project root before and during swarm work. Read top-level docs, manifests, CI/config files, existing `AGENTS.md`, existing `XERXES.md`, existing `.agents/`, and any obvious project conventions.
- If the apparent root is wrong or nested, inspect the parent just enough to identify the real project boundary, then continue from the correct root.
- Write or update `XERXES.md` only after the swarm findings are synthesized. Base it on actual repository facts: project purpose, build/test commands, runtime conventions, important paths, agent workflow rules, architecture, and project-specific caveats.
- Create or update `.agents/` only where it adds real value. Design repository-specific skills under `.agents/skills/<skill-name>/SKILL.md` for repeated workflows you can justify from the repo and the swarm results, not placeholder skills.
- Preserve user-authored content. If a target file exists, read it first and patch it deliberately instead of overwriting it blindly.
- Prefer `exec_command`/`write_stdin` for shell work, chunked file reads for large files, and project files or project memory for large notes.
- Finish with a concise report naming each subagent lane, every file you created or changed, and what evidence drove each file.
"""

    async def _run_project_init_agent(self, project_dir: Path, args: str, emit: EmitFn) -> None:
        """Delegate project initialization to the active model runtime."""
        prompt = self._project_init_prompt(project_dir, args)
        await self._emit_slash(
            emit,
            f"Project initialization swarm queued for `{project_dir}`. It will inspect the repo before authoring `XERXES.md`.",
        )
        try:
            await self._submit_turn({"text": prompt, "_internal_slash": True}, emit)
        except Exception as exc:
            await self._emit_slash(emit, f"Project initialization turn failed: `{exc}`")
            return
        try:
            self.runtime.reload({"project_dir": str(project_dir)})
            self._sync_runtime_to_connection_session(emit)
        except Exception as exc:
            await self._emit_slash(emit, f"Project initialization ran, but runtime reload failed: `{exc}`")
            return

        skills = self.runtime.discover_skills()
        await emit("init_done", {"skills": skills})
        await self._emit_slash(emit, f"Project initialization turn finished. Reloaded {len(skills)} skill(s).")

    async def _slash_init(self, args: str, emit: EmitFn) -> None:
        """Queue the agent-driven project setup workflow."""
        session_key = self._connection_session_key(emit)
        session = self.sessions.get(session_key)
        if session is None and session_key != self._current_session_key:
            session = self.sessions.get(self._current_session_key)
        project_dir = Path(
            session.project_dir if session is not None else self.config.project_dir or os.getcwd()
        ).resolve()
        await self._run_project_init_agent(project_dir, args, emit)

    async def _slash_workspace(self, args: str, emit: EmitFn) -> None:
        """Show or initialize the current project-local agent workspace."""
        session_key = self._connection_session_key(emit)
        session = self.sessions.get(session_key)
        if session is None and session_key != self._current_session_key:
            session = self.sessions.get(self._current_session_key)
        ws_path = str(session.workspace.path) if session is not None else "(no session)"
        project_dir = Path(
            session.project_dir if session is not None else self.config.project_dir or os.getcwd()
        ).resolve()

        action = args.strip().lower()
        if action == "init":
            await self._run_project_init_agent(project_dir, "", emit)
            return

        if action and action != "status":
            await self._emit_slash(emit, "Usage: `/workspace [status|init]`.")
            return

        context = load_project_agent_workspace(project_dir)
        lines = [
            f"Project dir:    `{project_dir}`",
            f"Agent workspace: `{ws_path}`",
            f"Agent id:        `{(session.agent_id if session else self.workspaces.default_agent_id)}`",
            f"Project .agents: `{context.agents_dir}` ({'ready' if context.prompt else 'not initialized'})",
        ]
        if context.loaded_files:
            lines.append("Loaded project context:")
            lines.extend(f"  `{path.relative_to(project_dir)}`" for path in context.loaded_files)
        else:
            lines.append("Run `/init` to have the agent inspect the repo and author project context.")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_background(self, args: str, emit: EmitFn) -> None:
        """List in-flight daemon background tasks (turn runners + post-turn hooks)."""
        active = [t for t in self._background_tasks if not t.done()]
        if not active:
            await self._emit_slash(emit, "No background tasks running.")
            return
        lines = [f"Background tasks ({len(active)}):"]
        for t in active:
            lines.append(f"  `{t.get_name()}`")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_rollback(self, args: str, emit: EmitFn) -> None:
        """Roll the working tree back to a snapshot (use ``/snapshots`` to find ids)."""
        target = args.strip()
        if not target:
            await self._emit_slash(emit, "Usage: `/rollback <snapshot-id>` — list with `/snapshots`.")
            return
        try:
            from ..session.snapshots import SnapshotManager
        except Exception:
            await self._emit_slash(emit, "Snapshot support not available in this build.")
            return
        try:
            mgr = SnapshotManager(self.config.project_dir or os.getcwd())
            mgr.rollback(target)
        except Exception as exc:
            await self._emit_slash(emit, f"Rollback failed: `{exc}`")
            return
        await self._emit_slash(emit, f"Rolled back to snapshot `{target}`.")

    # Mapping cmd name → bound-method-style callable so a single
    # ``handler(self, args, emit)`` line in ``_handle_slash`` covers everything.
    # Aliases are duplicated here so resolving the canonical name still hits the
    # right handler (``resolve_command`` already canonicalises before lookup).

    async def _slash_snapshot(self, args: str, emit: EmitFn) -> None:
        """Take a filesystem snapshot of the project working tree."""
        try:
            from ..session.snapshots import SnapshotManager
        except Exception:
            await self._emit_slash(emit, "Snapshot support not available in this build.")
            return
        try:
            mgr = SnapshotManager(self.config.project_dir or os.getcwd())
            record = mgr.snapshot(label=args.strip() or "manual")
        except Exception as exc:
            await self._emit_slash(emit, f"Snapshot failed: `{exc}`")
            return
        await self._emit_slash(emit, f"Snapshot `{getattr(record, 'id', '?')}` saved.")

    async def _slash_agents(self, args: str, emit: EmitFn) -> None:
        """List agent definitions and any running subagent tasks."""
        try:
            from ..agents.definitions import list_agent_definitions
        except Exception:
            defs = []
        else:
            defs = list_agent_definitions()
        lines = [f"Agents ({len(defs)}):"]
        for d in defs:
            name = getattr(d, "name", str(d))
            descr = getattr(d, "description", "") or ""
            lines.append(f"  `{name}` — {descr[:80]}")
        # Running subagent tasks, if the runtime exposes them.
        try:
            mgr = cast(Any, self.runtime).subagent_manager
            tasks = [t for t in mgr.tasks.values() if t.status in {"pending", "running"}]
        except Exception:
            tasks = []
        if tasks:
            lines.append("")
            lines.append(f"Running subagent tasks ({len(tasks)}):")
            for t in tasks:
                lines.append(f"  `{t.id}` — `{t.name or t.agent_def_name}` ({t.status})")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_queue(self, args: str, emit: EmitFn) -> None:
        """The TUI manages its own input queue; ack and tell the user."""
        await self._emit_slash(
            emit, "Pending input queue is TUI-side. The footer shows the count when items are queued."
        )

    async def _slash_branches(self, args: str, emit: EmitFn) -> None:
        """List every saved session id grouped by who their last update was."""
        records = self._saved_session_records()
        if not records:
            await self._emit_slash(emit, "No branches / saved sessions.")
            return
        lines = [f"Branches / saved sessions ({len(records)}):"]
        for r in records[:20]:
            sid = r.get("session_id", "?")
            turns = r.get("turn_count", 0)
            updated = r.get("updated_at", "")
            lines.append(f"  `{sid}` — {turns} turn(s), updated {updated}")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_voice(self, args: str, emit: EmitFn) -> None:
        """Voice mode is TUI-side; ack with a hint."""
        await self._emit_slash(emit, "Voice mode is TUI-side. Use the `xerxes-voice` CLI or the TUI's voice key.")

    async def _slash_snapshots(self, args: str, emit: EmitFn) -> None:
        """List recent filesystem snapshots."""
        try:
            from ..session.snapshots import SnapshotManager
        except Exception:
            await self._emit_slash(emit, "Snapshot support not available in this build.")
            return
        try:
            mgr = SnapshotManager(self.config.project_dir or os.getcwd())
            items = mgr.list()
        except Exception as exc:
            await self._emit_slash(emit, f"Snapshot list failed: `{exc}`")
            return
        if not items:
            await self._emit_slash(emit, "No snapshots yet. Take one with `/snapshot [label]`.")
            return
        lines = [f"Snapshots ({len(items)}):"]
        for s in items[:20]:
            sid = getattr(s, "id", "?")
            label = getattr(s, "label", "") or ""
            when = getattr(s, "created_at", "") or ""
            lines.append(f"  `{sid}` — `{label}` @ {when}")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_platforms(self, args: str, emit: EmitFn) -> None:
        """List configured messaging channel platforms and their state."""
        lines = ["Channel platforms:"]
        for name, ch in self.channels.channels.items():
            enabled = "on " if ch.enabled else "off"
            ready = "ready" if ch.instance is not None else "not-started"
            lines.append(f"  [{enabled}] `{name}` — {ready}")
        if len(lines) == 1:
            lines.append("  (none configured)")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_cron(self, args: str, emit: EmitFn) -> None:
        """List cron jobs (sub-commands ``add``/``remove``/``run`` deferred to JSON-RPC)."""
        try:
            from ..cron import JobStore
        except Exception:
            await self._emit_slash(emit, "Cron support not available in this build.")
            return
        sub = args.strip().split(maxsplit=1)
        action = sub[0].lower() if sub else "list"
        if action in ("", "list"):
            try:
                items = JobStore(xerxes_subdir("cron") / "jobs.json").list_jobs()
            except Exception as exc:
                await self._emit_slash(emit, f"Cron list failed: `{exc}`")
                return
            if not items:
                await self._emit_slash(emit, "No cron jobs scheduled.")
                return
            lines = [f"Cron jobs ({len(items)}):"]
            for j in items:
                lines.append(
                    f"  `{getattr(j, 'id', '?')}` — `{getattr(j, 'schedule', '')}` "
                    f"({'paused' if getattr(j, 'paused', False) else 'active'})"
                )
            await self._emit_slash(emit, "\n".join(lines))
            return
        await self._emit_slash(
            emit,
            f"`/cron {action}` is not yet wired in the daemon. "
            "Use the corresponding JSON-RPC method, or `/cron list` to view jobs.",
        )

    async def _slash_browser(self, args: str, emit: EmitFn) -> None:
        """List browser sessions managed by the operator subsystem."""
        sessions = []
        for sess in self.sessions._sessions.values():
            ops = getattr(sess, "operator_state", None)
            if ops is None:
                continue
            mgr = getattr(ops, "browser_manager", None)
            if mgr is None:
                continue
            try:
                sessions.extend(mgr.list_sessions())
            except Exception:
                continue
        if not sessions:
            await self._emit_slash(emit, "No browser sessions open. Use the `start_browser` tool to create one.")
            return
        lines = [f"Browser sessions ({len(sessions)}):"]
        for s in sessions:
            lines.append(f"  `{getattr(s, 'id', '?')}` — `{getattr(s, 'url', '')}`")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_tools(self, args: str, emit: EmitFn) -> None:
        """List the tool names the runtime has registered."""
        names = sorted(t.get("function", {}).get("name", "") for t in self.runtime.tool_schemas)
        names = [n for n in names if n]
        if not names:
            await self._emit_slash(emit, "No tools loaded.")
            return
        lines = [f"Tools ({len(names)}):"]
        for name in names:
            lines.append(f"  `{name}`")
        await self._emit_slash(emit, "\n".join(lines))

    async def _slash_image(self, args: str, emit: EmitFn) -> None:
        """Generate an image from a prompt via the image-generation tool."""
        prompt = args.strip()
        if not prompt:
            await self._emit_slash(emit, "Usage: `/image <prompt>`")
            return
        # Defer to the agent — it has access to the image generation tool and
        # can pick a provider + model on its own.
        synthetic = (
            f"Generate an image matching this brief and report the saved path:\n\n{prompt}\n\n"
            "Use the `generate_image` tool. Do not write any code; one tool call is enough."
        )
        await self._submit_turn({"text": synthetic, "_internal_slash": True}, emit)
        await self._emit_slash(emit, "Image-generation turn queued.")


_BULK_SLASH_HANDLERS: dict[str, Any] = {
    "new": SlashCommandsMixin._slash_new,
    "reset": SlashCommandsMixin._slash_new,
    "stop": SlashCommandsMixin._slash_stop,
    "cancel": SlashCommandsMixin._slash_stop,
    "cancel-all": SlashCommandsMixin._slash_cancel_all,
    "compact": SlashCommandsMixin._slash_compact,
    "compress": SlashCommandsMixin._slash_compact,
    "btw": SlashCommandsMixin._slash_steer,
    "steer": SlashCommandsMixin._slash_steer,
    "model": SlashCommandsMixin._slash_model,
    "sampling": SlashCommandsMixin._slash_sampling,
    "config": SlashCommandsMixin._slash_config,
    "title": SlashCommandsMixin._slash_title,
    "init": SlashCommandsMixin._slash_init,
    "workspace": SlashCommandsMixin._slash_workspace,
    "save": SlashCommandsMixin._slash_save,
    "personality": SlashCommandsMixin._slash_personality,
    "soul": SlashCommandsMixin._slash_soul,
    "tools": SlashCommandsMixin._slash_tools,
    "toolsets": SlashCommandsMixin._slash_toolsets,
    "agents": SlashCommandsMixin._slash_agents,
    "reload": SlashCommandsMixin._slash_reload,
    "reload-mcp": SlashCommandsMixin._slash_reload_mcp,
    "memory": SlashCommandsMixin._slash_memory,
    "history": SlashCommandsMixin._slash_history,
    "usage": SlashCommandsMixin._slash_usage,
    "cost": SlashCommandsMixin._slash_cost,
    "insights": SlashCommandsMixin._slash_insights,
    "budget": SlashCommandsMixin._slash_budget,
    "doctor": SlashCommandsMixin._slash_doctor,
    "update": SlashCommandsMixin._slash_update,
    "nudge": SlashCommandsMixin._slash_nudge,
    "feedback": SlashCommandsMixin._slash_feedback,
    "plugins": SlashCommandsMixin._slash_plugins,
    "platforms": SlashCommandsMixin._slash_platforms,
    "browser": SlashCommandsMixin._slash_browser,
    "image": SlashCommandsMixin._slash_image,
    "cron": SlashCommandsMixin._slash_cron,
    "fast": SlashCommandsMixin._slash_fast,
    "skin": SlashCommandsMixin._slash_skin,
    "statusbar": SlashCommandsMixin._slash_statusbar,
    "paste": SlashCommandsMixin._slash_paste,
    "voice": SlashCommandsMixin._slash_voice,
    "queue": SlashCommandsMixin._slash_queue,
    "background": SlashCommandsMixin._slash_background,
    "resume": SlashCommandsMixin._slash_resume,
    "restart": SlashCommandsMixin._slash_restart,
    "undo": SlashCommandsMixin._slash_undo,
    "retry": SlashCommandsMixin._slash_retry,
    "retry-connection": SlashCommandsMixin._slash_retry_connection,
    "branch": SlashCommandsMixin._slash_branch,
    "branches": SlashCommandsMixin._slash_branches,
    "snapshot": SlashCommandsMixin._slash_snapshot,
    "snapshots": SlashCommandsMixin._slash_snapshots,
    "rollback": SlashCommandsMixin._slash_rollback,
}
