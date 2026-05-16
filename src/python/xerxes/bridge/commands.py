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
"""Canonical slash-command registry.

Every command Xerxes exposes — to the TUI completer, the daemon's
``/`` dispatcher, and the Telegram bot ``BotCommand`` registration — comes
from :data:`COMMAND_REGISTRY`. Each entry is a :class:`CommandDef` that
records the canonical name, aliases, category, args hint, and surface
filters (``cli_only``, ``gateway_only``). Adding a new command means
appending one entry here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

CATEGORIES = (
    "session",
    "config",
    "tools",
    "skills",
    "info",
    "feedback",
    "memory",
    "voice",
    "snapshots",
    "messaging",
    "exit",
)


@dataclass(frozen=True)
class CommandDef:
    """One slash-command definition.

    Attributes:
        name: Canonical name without the leading ``/``.
        description: One-line summary shown in ``/help`` and listings.
        category: Bucket from :data:`CATEGORIES` used to group ``/help`` output.
        aliases: Extra names that resolve to this command.
        args_hint: Usage hint (e.g. ``"[name]"``) shown after the name.
        cli_only: True when the command must not be exposed in messaging gateways.
        gateway_only: True when the command is gateway-only (e.g. Telegram).
        deprecated: True when the command still resolves but is being retired.
        examples: Short snippets surfaced in extended help.
    """

    name: str
    description: str
    category: str
    aliases: tuple[str, ...] = ()
    args_hint: str = ""
    cli_only: bool = False
    gateway_only: bool = False
    deprecated: bool = False
    examples: tuple[str, ...] = field(default_factory=tuple)


# fmt: off
COMMAND_REGISTRY: tuple[CommandDef, ...] = (
    # ---- session ---------------------------------------------------------
    CommandDef("new", "Start a fresh conversation", "session", aliases=("reset",)),
    CommandDef("clear", "Clear the visible scrollback", "session"),
    CommandDef("history", "Show or search conversation history", "session"),
    CommandDef("save", "Save the session by name", "session", args_hint="<name>"),
    CommandDef("retry", "Re-run the last turn", "session"),
    CommandDef("undo", "Undo the last turn", "session"),
    CommandDef("title", "Generate or set the session title", "session", args_hint="[title]"),
    CommandDef("branch", "Fork this session", "session", args_hint="[label]"),
    CommandDef("branches", "List branches of this session", "session"),
    CommandDef("compact", "Compress the conversation", "session", aliases=("compress",)),
    CommandDef("rollback", "Restore filesystem to a snapshot", "snapshots", args_hint="<id|label>"),
    CommandDef("snapshot", "Take a filesystem snapshot", "snapshots", args_hint="[label]"),
    CommandDef("snapshots", "List filesystem snapshots", "snapshots"),
    CommandDef("stop", "Cancel the in-flight tool call", "session", aliases=("cancel",)),
    CommandDef("cancel-all", "Cancel every queued action", "session"),
    CommandDef("background", "Detach the session into the daemon", "session", cli_only=True),
    CommandDef("btw", "Inject side-channel context", "session"),
    CommandDef("queue", "Show or manage queued prompts", "session", cli_only=True),
    CommandDef("status", "Show platform / session status", "info"),
    CommandDef("resume", "Resume a saved session", "session", args_hint="<id|name>"),
    CommandDef("steer", "Course-correct mid-stream", "session", cli_only=True),
    # ---- config ----------------------------------------------------------
    CommandDef("config", "Inspect or edit runtime config", "config"),
    CommandDef("model", "Switch model", "config", args_hint="[provider:model]"),
    CommandDef("provider", "Setup or switch provider profile", "config"),
    CommandDef("sampling", "Get or set sampling params", "config"),
    CommandDef("personality", "Choose a SOUL.md preset", "config", args_hint="[name]"),
    CommandDef("statusbar", "Toggle the status bar", "config", cli_only=True),
    CommandDef("verbose", "Toggle verbose event logging", "config", cli_only=True),
    CommandDef("yolo", "Toggle accept-all permissions mode", "config"),
    CommandDef("reasoning", "Toggle reasoning display", "config", aliases=("thinking",)),
    CommandDef("fast", "Use the aux model for the next turn", "config"),
    CommandDef("skin", "Change theme/skin", "config", args_hint="[name]", cli_only=True),
    CommandDef("voice", "Toggle voice mode", "voice", args_hint="on|off|play", cli_only=True),
    CommandDef("permissions", "View / set permission mode", "config"),
    CommandDef("debug", "Toggle debug output", "config", cli_only=True),
    # ---- tools / skills --------------------------------------------------
    CommandDef("tools", "List available tools", "tools"),
    CommandDef("toolsets", "List configured toolsets", "tools"),
    CommandDef("skills", "List available skills", "skills"),
    CommandDef("skill", "Invoke a skill by name", "skills", args_hint="<name>"),
    CommandDef("skill-create", "Scaffold a new skill directory", "skills", args_hint="<name>"),
    CommandDef("cron", "Manage scheduled tasks", "tools", args_hint="list|add|remove|run"),
    CommandDef("reload", "Reload skills + tools", "tools"),
    CommandDef("reload-mcp", "Reload MCP servers", "tools"),
    CommandDef("browser", "Manage browser sessions", "tools"),
    CommandDef("plugins", "List loaded plugins", "tools"),
    CommandDef("workspace", "Inspect / edit the markdown workspace", "tools"),
    CommandDef("soul", "Show or edit SOUL.md", "tools"),
    CommandDef("agents", "List or select sub-agents", "tools"),
    # ---- info ------------------------------------------------------------
    CommandDef("help", "Show help", "info", aliases=("?",)),
    CommandDef("commands", "List every available command", "info"),
    CommandDef("restart", "Restart the agent process", "info"),
    CommandDef("usage", "Show token & cost usage", "info"),
    CommandDef("insights", "Show usage analytics", "info", args_hint="[--days N]"),
    CommandDef("platforms", "List configured messaging platforms", "info"),
    CommandDef("paste", "Paste from clipboard", "info", cli_only=True),
    CommandDef("image", "Attach a clipboard image to the next message", "info", cli_only=True),
    CommandDef("update", "Update Xerxes", "info"),
    CommandDef("cost", "Show running cost", "info"),
    CommandDef("context", "Show session info", "info"),
    CommandDef("doctor", "Diagnose configuration", "info"),
    # ---- memory + feedback ----------------------------------------------
    CommandDef("memory", "Manage memory backends", "memory", args_hint="backend list|use NAME|status"),
    CommandDef("feedback", "Give compaction / behavior feedback", "feedback"),
    CommandDef("nudge", "Toggle proactive nudges", "feedback", args_hint="on|off"),
    CommandDef("budget", "Set or show per-session budget", "feedback", args_hint="set <amount>"),
    # ---- exit ------------------------------------------------------------
    CommandDef("exit", "Quit", "exit", aliases=("quit", "q")),
)
# fmt: on


_BY_NAME: dict[str, CommandDef] = {}
_BY_ALIAS: dict[str, CommandDef] = {}
for _cmd in COMMAND_REGISTRY:
    _BY_NAME[_cmd.name] = _cmd
    for _alias in _cmd.aliases:
        _BY_ALIAS[_alias] = _cmd


def resolve_command(text: str) -> CommandDef | None:
    """Look up a command by canonical name or alias; a leading ``/`` is allowed."""
    token = text.lstrip("/").split(" ", 1)[0]
    if token in _BY_NAME:
        return _BY_NAME[token]
    return _BY_ALIAS.get(token)


def list_commands(*, category: str | None = None) -> list[CommandDef]:
    """Return commands in registry order, optionally filtered by category."""
    if category is None:
        return list(COMMAND_REGISTRY)
    return [c for c in COMMAND_REGISTRY if c.category == category]


_TELEGRAM_NAME_RE = re.compile(r"^[a-z0-9_]{1,32}$")


def telegram_bot_commands(commands: Iterable[CommandDef] | None = None) -> list[dict[str, str]]:
    """Render commands as the Telegram BotCommand JSON shape.

    CLI-only commands are excluded, hyphens are mapped to underscores, names
    are lower-cased and filtered to ``[a-z0-9_]`` (max 32 chars) so the result
    can be passed straight to ``setMyCommands``.
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for cmd in commands or COMMAND_REGISTRY:
        if cmd.cli_only:
            continue
        name = cmd.name.lower().replace("-", "_")
        if not _TELEGRAM_NAME_RE.match(name) or name in seen:
            continue
        seen.add(name)
        out.append({"command": name, "description": cmd.description[:256]})
    return out


__all__ = [
    "CATEGORIES",
    "COMMAND_REGISTRY",
    "CommandDef",
    "list_commands",
    "resolve_command",
    "telegram_bot_commands",
]
