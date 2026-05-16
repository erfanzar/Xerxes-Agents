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
"""Welcome tips shown at startup.

Roughly 150 one-liners about CLI features, slash commands, keybindings,
and workflow patterns. Tips are deterministic-by-seed so test suites can
pin a tip without flakiness.

Exports:
    - TIPS
    - random_tip
    - tip_of_the_day"""

from __future__ import annotations

import datetime as _dt
import random

# fmt: off
TIPS: tuple[str, ...] = (
    "/help shows every slash command grouped by category.",
    "Press Tab to accept the gray ghost-text suggestion from history.",
    "Up / Down navigate prompt history; PageUp / PageDown scroll the rendered output.",
    "Hit Ctrl+C once to interrupt the agent; twice within two seconds to exit.",
    "Drop @file:src/path.py into a prompt and the file content is inlined for you.",
    "@diff or @staged dumps your current git changes into the prompt.",
    "@git:HEAD or @git:my-branch resolves a ref to its latest commit.",
    "/btw injects side-channel context without breaking the active turn.",
    "/steer course-corrects the agent mid-stream — no need to cancel first.",
    "/compact summarises older turns; the next turn keeps the new summary in scope.",
    "/snapshot takes a filesystem checkpoint via a shadow git repo.",
    "/rollback <id> restores a previous snapshot — file changes only.",
    "/branch forks the current session; /branches lists them.",
    "/plan toggles read-only research mode — the agent can't execute writes.",
    "/yolo flips permission mode to accept-all. Use carefully.",
    "/permissions auto | manual | accept-all switches modes on the fly.",
    "/cost shows running cost; /insights aggregates over the last N days.",
    "/usage breaks down tokens by model and tool.",
    "/cron lets the agent schedule tasks; output is archived to disk.",
    "/voice on enables push-to-talk; hold the bound key to record.",
    "/skin <name> swaps the theme. Bundled: default, high-contrast, dim.",
    "Alt+Enter or Ctrl+J inserts a newline; plain Enter submits.",
    "Ctrl+R doesn't exist (yet); use Up/Down to navigate history.",
    "Esc twice within a second exits the TUI.",
    "Tab inside @file: completes paths from your workspace root.",
    "Bracketed paste detects clipboard images automatically.",
    "/queue stacks a prompt to run after the current turn ends.",
    "/background detaches a prompt into a side agent — finishes silently.",
    "/skills lists installed skills; /skill <name> invokes one directly.",
    "/skill-create captures the last successful workflow into a new skill.",
    "/insights --days 7 shows last week's activity at a glance.",
    "/budget set <USD> caps your per-session spend; warns at 80%.",
    "/feedback compaction good|bad helps tune compression thresholds.",
    "/nudge off silences proactive memory/skill nudges.",
    "/memory list inspects durable workspace notes in MEMORY.md.",
    "/workspace init drops AGENTS / SOUL / USER / MEMORY / TOOLS templates.",
    "/soul show inspects the persona prompt; /soul edit opens it in $EDITOR.",
    "Press Ctrl+Z to suspend Xerxes to the background (Unix).",
    "Daemon survives TUI restarts — your session stays warm.",
    "Use `xerxes -r <session-id>` to resume a specific session.",
    "Each plan in plans/ has its own audit doc — read 00-master-index.md.",
    "Subagents are sandboxed in a git worktree by default; they can't see your wip.",
    "Cortex orchestrates multi-agent flows with five process types.",
    "/agents lists sub-agent specs; /agents select <name> swaps the active one.",
    "Sampling: /sampling temperature 0.7 / top_p 0.9 / max_tokens 4096.",
    "/sampling save persists the active profile to disk.",
    "Cache hits show in /usage as cache_read tokens — way cheaper than fresh input.",
    "Anthropic ephemeral cache lives ~5 min; /compact invalidates it on purpose.",
    "Token coalescing means short streams feel faster than per-byte streaming.",
    "Long tool outputs auto-overflow to ~/.xerxes/tool_results/ and show a [ref:…] handle.",
    "Run `xerxes doctor` to diagnose providers, dependencies, and PATH.",
    "`xerxes update` detects your install mode and runs the correct upgrade.",
    "`xerxes setup` walks you through provider + model + permissions first-run.",
    "Set XERXES_HOME to relocate config away from ~/.xerxes.",
    "Set XERXES_IDENTITY_SALT to share consistent channel-side IDs across hosts.",
    "Set XERXES_SSH_HOST to default the SSH sandbox backend to a build host.",
    "OAuth credentials live at ~/.xerxes/credentials/<provider>.json (0600 perms).",
    "Try `xerxes acp` to register as an Agent Client Protocol server for IDEs.",
    "`xerxes-mcp serve` exposes 10 read/write tools for Claude Desktop & friends.",
    "Memory plugins: install xerxes-agent[memory-honcho] for Honcho AI memory.",
    "Mem0, Hindsight, Holographic, RetainDB, OpenViking, Supermemory, ByteRover are pluggable.",
    "Holographic memory works offline; it's the default fallback when no plugin is configured.",
    "/insights catches token spikes early — set /budget so they actually cost less.",
    "All slash commands have aliases — /q and /quit both exit.",
    "Tab cycles completions; Shift+Tab cycles activity modes (code/researcher/agents).",
    "Activity mode auto-detects from tool-call patterns over the last few turns.",
    "Browser providers: local Playwright + Browserbase + Browser Use + Camofox + Firecrawl.",
    "Set XERXES_CAMOFOX_SCRIPT to the camofox-bridge JS entrypoint for stealth Firefox.",
    "Modal sandbox needs `modal token new` first; then it autohibernates between turns.",
    "SSH backend uses your local OpenSSH; key auth via the SSH agent only — no keys on disk.",
    "Skin engine reads ~/.xerxes/skins/<name>.yaml; one role per line.",
    "Skin roles include primary, accent, warn, error, tool_name, system, muted.",
    "Hex colors convert to ANSI 24-bit RGB escapes — every modern terminal supports them.",
    "Press up at the empty prompt to recall your last submission.",
    "PageUp scrolls the rendered area; Ctrl+Home jumps to the top.",
    "The footer line count tells you how much vertical space is reserved for status.",
    "/context shows session id, workspace, agent, token totals at a glance.",
    "Bridge process logs to ~/.xerxes/logs/daemon.log when --verbose is set.",
    "Use --permission-mode accept-all when scripting non-interactive runs.",
    "xerxes -p '<prompt>' runs one-shot mode and auto-rejects approvals.",
    "Cron jobs run inside the same daemon; deliver=telegram routes their output.",
    "/cron add --schedule '0 9 * * *' --prompt 'Summarize my PRs'.",
    "OSV.dev malware scan gates MCP server installs by default.",
    "URL safety blocks fetches to private / internal / link-local addresses by default.",
    "Path security catches `..` escapes on every file tool — no opt-out, by design.",
    "Per-session approvals: approve-once, approve-for-session, approve-always.",
    "Always-approve persists to ~/.xerxes/approvals.json with restrictive perms.",
    "PII redaction filters phone, email, JWTs, AWS keys, GitHub PATs from logs.",
    "Tinker RL stubs ship in the [rl] extra; set TINKER_API_KEY to enable runs.",
    "RL envs registered: xerxes-terminal-test, xerxes-swe-bench, agentic-opd.",
    "Trajectory compressor ships under xerxes.training; ideal for SFT datasets.",
    "Batch runner: `xerxes-batch --in jsonl --out out.jsonl --workers 8`.",
    "/restart cleanly resets the agent and bridge without dropping your scrollback.",
    "Skill manifest: ~/.xerxes/skills/manifest.yaml drives reproducible installs.",
    "Skill quarantine for untrusted sources is enabled by default.",
    "Local Honcho-style nudging is enabled out of the box; /nudge off to disable.",
    "Each subsystem plan has tests under tests/<area>/; read plan README first.",
    "Telegram-safe slash command names are auto-derived from the registry.",
    "Telegram gateway forwards channel messages → daemon → agent → reply.",
    "Identity hash format: user_<sha16> and platform:<sha16> — same across hosts with same salt.",
    "Session reset policies per platform: timeout | msg_count | manual.",
    "Sticker cache shares storage across all platform adapters.",
    "Cross-platform mirror duplicates messages between, e.g., Slack and Discord.",
    "AgentskillsIOSource pulls public skills from the agentskills.io standard.",
    "Skill bundles end in SKILL.md — one body per directory.",
    "/insights surfaces top tools by call count — great for refactoring time-sinks.",
    "Set ANTHROPIC_API_KEY and the streaming loop enables prompt caching automatically.",
    "Auxiliary client routes summary/title/extract work through a cheaper model.",
    "Rate-limit tracker parses x-ratelimit-* headers and throttles preemptively.",
    "Error classifier turns provider errors into RATE_LIMIT / CONTEXT_OVERFLOW / TIMEOUT / etc.",
    "Auto-pruning of huge tool results happens before LLM compaction — saves $$.",
    "Pre-pruner converts binary blobs into [N bytes elided] placeholders.",
    "Iterative summary merging means /compact twice in a row doesn't waste tokens.",
    "context_window is a per-model constant; pricing table tracks it for 39+ models.",
    "Cache pricing: cache_read ≈ 0.1× input; cache_write ≈ 1.25× input (Anthropic).",  # noqa: RUF001 — U+00D7 multiplication sign
    "Cache savings show up as a positive delta in /insights.",
    "Stream → markdown → ANSI happens incrementally; long answers feel fluid.",
    "12 tool-call parsers cover Llama, Mistral, Qwen, DeepSeek, GLM, Kimi, LongCat and more.",
    "vLLM/Ollama outputs auto-route through the right parser based on model name.",
    "Codex Responses API is wired but opt-in via streaming_protocol=\"responses\".",
    "SSE reconnect uses Last-Event-ID — server has to echo it, otherwise we restart clean.",
    "Permission requests timeout at 60s in CLI; auto-deny on expiry.",
    "/voice tts reads the last response aloud — handy for hands-free flows.",
    "Tool result file storage cap: ~/.xerxes/tool_results/<session>/.",
    "Snapshots prune to the most-recent 100 by default; /snapshots prune --keep N adjusts.",
    "Schema migrations run on load; reading an old session auto-bumps it in place.",
    "Branch a session before risky experiments — rollback is per-branch.",
    "Each /branch is independent; deleting one doesn't affect siblings.",
    "Workspace files persist across sessions; treat them as agent-visible memory.",
    "AGENTS.md describes the agent registry; SOUL.md is the persona.",
    "USER.md captures stable user preferences; MEMORY.md captures durable facts.",
    "TOOLS.md documents non-default toolset configurations.",
    "Background processes registered via tools survive the turn that started them.",
    "/processes lists running background processes (when ProcessRegistry is exposed).",
    "Interrupt tokens propagate per-thread — Ctrl+C reaches into running tools cleanly.",
    "Iteration budget refunds PTC scripts: 10 tool calls inside one PTC = 1 turn used.",
    "PTC (programmatic tool calling) collapses multi-step pipelines into zero-context turns.",
    "Streaming events: TextChunk / ThinkingChunk / ToolStart / ToolEnd / TurnDone.",
    "All streaming events serialize to wire format for daemon ↔ TUI transport.",
    "If a stream stalls, the bridge auto-restarts on cancel — your TUI scrollback survives.",
    "MCP server exposes 10 tools: conversations / messages / events / permissions / channels.",
    "MCP client supports OAuth 2.1 with PKCE, plus OSV malware checks pre-install.",
    "ACP server registers via ~/.config/agent-registry/xerxes/agent.json.",
    "ACP clients (Claude Code / Cursor / Cline) discover Xerxes through that registry.",
    "Termux/Android: install with xerxes-agent[termux] — non-Android wheels are excluded.",
    "Homebrew: brew install xerxes-agent — formula included; tap repo coming.",
    "Distribution: real PyPI release ships from .github/workflows/release.yml.",
    "Cron output archive: ~/.xerxes/cron/output/<job_id>/<timestamp>.md.",
    "Session FTS index lets /history search across your entire conversation history.",
    "Each tip is just a string — drop yours into src/python/xerxes/tui/tips.py.",
)
# fmt: on


def random_tip(*, seed: int | None = None) -> str:
    """Return one random tip. Pass ``seed`` for deterministic output."""
    if seed is not None:
        return TIPS[seed % len(TIPS)]
    return random.choice(TIPS)


def tip_of_the_day(*, today: _dt.date | None = None) -> str:
    """Return a tip keyed off today's date — stable for a 24h window."""
    day = today or _dt.date.today()
    return TIPS[day.toordinal() % len(TIPS)]


__all__ = ["TIPS", "random_tip", "tip_of_the_day"]
