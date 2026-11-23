<div align="center">

<img src="https://raw.githubusercontent.com/erfanzar/Xerxes-Agents/main/assets/logo-128.png" width="96" alt="Xerxes logo" />

# Xerxes

### A Bun-native multi-agent coding workspace and runtime

[![npm](https://img.shields.io/npm/v/%40xsimurgh%2Fxerxes-agents?logo=npm&color=cb3837)](https://www.npmjs.com/package/@xsimurgh/xerxes-agents)
[![npm downloads](https://img.shields.io/npm/dm/%40xsimurgh%2Fxerxes-agents)](https://www.npmjs.com/package/@xsimurgh/xerxes-agents)
[![Bun CI](https://github.com/erfanzar/Xerxes-Agents/actions/workflows/bun-ci.yml/badge.svg)](https://github.com/erfanzar/Xerxes-Agents/actions/workflows/bun-ci.yml)
[![Bun 1.3.12+](https://img.shields.io/badge/Bun-1.3.12%2B-f9f1e1?logo=bun&logoColor=111)](https://bun.sh/)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178c6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Desktop app, OpenTUI terminal, VS Code extension, and paired Mobile/Hub cockpit — backed by one persistent Bun daemon for native multi-agent swarms (Immortals), external ACP runtimes (Antigravity, Claude Code, Codex), SSH remote hosts, MCP, skills, and scheduled jobs.**

[Install](#quick-start) · [Capabilities](#capabilities) · [External ACP](#external-acp-runtimes) · [Immortals Swarms](#immortals-multi-agent-swarms) · [Providers](#configure-a-provider) · [Documentation](#documentation)

</div>

Xerxes is an open-source, Bun-native TypeScript coding workspace. A single daemon owns provider routing, tool execution, sandboxing, persistent sessions, memory, sub-agents, swarms, and external ACP bridges over a v35 NDJSON JSON-RPC protocol — so your **Desktop app**, **OpenTUI terminal**, **VS Code sidebar**, **Mobile web cockpit**, and **remote SSH hosts** all share the same live tasks, tools, and history.

---

## Quick start

### Desktop app (macOS)

Download `Xerxes-Agents-<version>-macOS-arm64.dmg` from [GitHub Releases](https://github.com/erfanzar/Xerxes-Agents/releases/latest) and drag **Xerxes Agents.app** into `/Applications`. The app bundles its own Bun runtime, auto-manages the local daemon, and connects to remote SSH workspaces with local rendering.

### CLI & Terminal (OpenTUI)

Requires [Bun 1.3.12 or newer](https://bun.sh/). Install [`@xsimurgh/xerxes-agents`](https://www.npmjs.com/package/@xsimurgh/xerxes-agents) globally:

```bash
bun add --global @xsimurgh/xerxes-agents
# or
npm install --global @xsimurgh/xerxes-agents

xerxes
```

Or run without a global install:

```bash
bunx @xsimurgh/xerxes-agents
# or
npx --yes @xsimurgh/xerxes-agents
```

> **Package-name note:** install `@xsimurgh/xerxes-agents`, not `xerxes`. The unscoped npm package named `xerxes` is unrelated to this project.

The package exposes three binaries: `xerxes` (interactive TUI & CLI), `xerxes-acp` (Agent Client Protocol server), and `xerxes-agents` (alias).

### Install from `main`

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/erfanzar/Xerxes-Agents/main/scripts/install.sh | sh
```

Re-running the same command fast-forwards and rebuilds a managed checkout under `${XERXES_INSTALL_DIRECTORY}` and writes launchers to `${XERXES_BIN_DIRECTORY:-$HOME/.local/bin}` without killing active sessions.

```powershell
# Windows (PowerShell — writes xerxes.cmd and xerxes-acp.cmd under %LOCALAPPDATA%\Xerxes\bin)
git clone https://github.com/erfanzar/Xerxes-Agents.git
cd Xerxes-Agents
./scripts/install.ps1
```

Or run directly from source:

```bash
git clone https://github.com/erfanzar/Xerxes-Agents.git
cd Xerxes-Agents
bun install --frozen-lockfile
bun run build
bun run xerxes
```

### VS Code extension

Install the platform-specific `.vsix` (`darwin-arm64`, `darwin-x64`, `linux-arm64`, `linux-x64`, or `win32-x64`) from [GitHub Releases](https://github.com/erfanzar/Xerxes-Agents/releases/latest) via **Extensions → … → Install from VSIX…** (or from the VS Code Marketplace). It bundles Bun, shares sessions and providers with Desktop and CLI, opens diffs/files directly in VS Code, and supports **Send Selection to Xerxes** (`⌘⌥X` / `Ctrl+Alt+X`).

---

## Capabilities

| Area | Highlights |
| --- | --- |
| **Desktop Workspace** | Vibes & Neo visual styles, themed palettes & backdrops, Machine + Project grouped Sidebar V2 with inbox triage (pin/snooze/settle), 3-pane Provider/Model/Thinking/Permissions picker, Side Question (`/btw`), immediate **Steer now** (`⌘⏎`) & queued follow-ups, Files/Diff/Git/Terminal/Artifacts/Outputs tabs, and an embedded Browser with SSH port forwarding |
| **External ACP Runtimes** | Switch any thread live between **Xerxes Native**, **Google Antigravity (`agy acp`)**, **Claude Code**, **OpenAI Codex**, or custom ACP commands with per-runtime model/thinking/mode controls, `ask_question` elicitation, Turbo auto-approve, credential sync, and `XERXES.md` / `.xerxes/` bridging |
| **Immortals Swarms** | Native multi-agent swarms with a shared `#main` / custom channel message board, coordinator/worker/verifier/narrator/monitor roles, isolated git worktrees (`share` / `branch`), interactive Swarm Board & Graph, and synced Desktop/TUI/VS Code/Mobile surfaces |
| **Native Terminal (OpenTUI)** | React 19 + OpenTUI renderer with streaming Markdown, thinking, compact tool trails, live Agents rail, Background Agent View (`Left` / `/background`), and live Terminals inspector (`F8` / `/terminals`) |
| **Hub, Mobile & Channels** | Pair phones and tablets via Xerxes Hub (`xerxes hub`), run **XERXES Supervisor** across workspaces, connect **Slack** with one-click Sign in with Slack, and bridge **Telegram** or **Discord** gateways |
| **SWE Tools & Generative UI** | Native `apply_patch`, `outline`, `test_discovery`, LSP symbol/diagnostic integration, working-set compaction, and sandboxed inline HTML & 3D widgets (`Xerxes3D`) |
| **Security & Sandboxing** | Default `auto` policy with locked pre-existing inputs, macOS Seatbelt and Linux bubblewrap command sandboxes, prompt-injection scanning, explicit host ports, and zero credentials in source control |
| **Persistent State & Cron** | Durable SQLite/FTS sessions with `/resume`, `/branch`, `/compact`, `/snapshot`, `/rollback`, tiered memory, and daemon-owned scheduled jobs (`/cron`) |

---

## External ACP runtimes

Click the runtime trigger in the Desktop composer (or run `/acp` in the TUI) to bind a session to an external agent runtime while keeping Xerxes's UI, session history, approvals, file previews, and remote SSH transport:

- **Google Antigravity (`agy acp`)**:
  - Defaults to **Turbo** mode (host auto-approval) with live model, thinking-level, and mode switching.
  - Bridges project and user-global `XERXES.md`, `XERXES.local.md`, `.xerxes/*.md` notes, `.xerxes/skills`, `.xerxes/agents`, and `.xerxes/commands` read-only into `agy acp` (with dynamic mid-session content-hash refresh and `$ARGUMENTS` slash expansion, deduplicated against `.agents/skills`).
  - Projects Antigravity background `run_command` jobs and `schedule` timers into **Activity**, **Background**, live `task-*.log` inspection, and `manage_task(kill)` cancellation.
  - Supports local credential sync and forwarding to remote SSH hosts.
- **Claude Code** (`claude` stream-json) & **OpenAI Codex** (`codex` ACP) & **Custom ACP**:
  - Strict runtime isolation keeps each runtime's own slash commands, skills, and capabilities clean without cross-leaking native Xerxes commands.

---

## Immortals multi-agent swarms

Run coordinated teams of agents directly inside any workspace (see [docs/swarm.md](docs/swarm.md)):

- **Roles**: coordinator, authors/workers, verifiers, narrator, and stall monitor.
- **Shared Message Board**: real-time `@mention` wakeups, channels, status snapshots, and live worker inspection.
- **Worktree Isolation**: workers can share the workspace (`inherit`), get a shared-repo worktree (`share`), or run in an isolated clone (`branch`).
- **Interactive Surfaces**: inspect the live **Swarm Board** and **Swarm Graph** in Desktop, TUI, VS Code, and Mobile.

---

## Run it

```bash
# Interactive OpenTUI session
xerxes

# One-shot request
xerxes "explain this repository"

# Resume a persistent session
xerxes --resume <session-id>

# Pipe stdin for one-shot execution
printf 'summarize the current project' | xerxes

# Project-scoped JSON-RPC daemon
xerxes daemon --project-dir .

# Agent Client Protocol server over stdio
xerxes acp --project-dir .

# Subscription quota windows (Claude, Codex, Kimi Code, Z.ai)
xerxes usage [provider]

# Export a session transcript
xerxes export [session]

# Run a skill directly without opening the TUI
xerxes skill <skill> [arguments]

# Check installation, Bun, providers, and external runtimes
xerxes doctor
```

### Everyday slash commands

```text
/help                 show commands, plugins, and shortcuts
/provider             create or switch provider profiles
/model                choose a model and reasoning effort
/acp                  connect, configure, or disconnect external ACP runtimes
/btw <question>       ephemeral read-only side question without polluting chat
/swarm                open or control Immortals multi-agent swarms
/goal                 set or manage an iterative objective loop
/plan [msg|off]       enter plan mode or steer planning
/new                  start a fresh session
/resume <id|name>     resume saved work
/background [prompt]  detach into Agent View while the turn keeps running
/agents               inspect active sub-agents and presets
/terminals            inspect live background and interactive shells (F8)
/skills               inspect discovered skills (/skill <name> to invoke)
/permissions          select auto, accept-all (/yolo), manual, or plan
/delegate             set delegation eagerness: off, auto (default), or eager
/cron                 manage scheduled jobs
/status               show runtime, context, and session status
```

In the Desktop composer while a turn is running, press **`Enter`** to queue a follow-up or **`⌘⏎` / `Ctrl+⏎`** (or click **Steer now** on the composer or any queued item) to interrupt the current step and steer immediately.

---

## Configure a provider

Run `/provider` in the TUI or open **Settings → Providers** in Desktop to configure API keys, subscription OAuth/ADC accounts, or local OpenAI-compatible servers (Ollama, LM Studio, vLLM, eSurge, etc.). Environment variables also work out of the box:

```bash
export ANTHROPIC_API_KEY='…'
# or OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, XAI_API_KEY, ...
xerxes
```

`XERXES_HOME` controls where Xerxes stores profiles, credentials, sessions, daemon sockets, and memory (defaults to `~/.xerxes`). Credentials always stay outside the repository. See [docs/configuration-guide.md](docs/configuration-guide.md).

---

## Workspace customizations & extensions

- **Instructions**: `XERXES.md`, `XERXES.local.md`, `AGENTS.md`, and `.xerxes/*.md` notes.
- **Skills**: `SKILL.md` bundles discovered from `.xerxes/skills/`, `.agents/skills/`, `~/.xerxes/skills/`, and bundled skills.
- **Agents & Commands**: specialist agent definitions in `.xerxes/agents/` & `.agents/`, and custom slash command templates in `.xerxes/commands/` (with `$ARGUMENTS` expansion).
- **MCP & Plugins**: manage MCP servers (`/reload-mcp`) and workspace plugins (`/plugins`) directly from Desktop or TUI.
- **Channels**: connect Slack from Desktop (**Settings → Hub & Slack**), or launch Telegram directly with `xerxes telegram --token "$TELEGRAM_BOT_TOKEN"`.

---

## Architecture

```text
Desktop (Electron) · Terminal (OpenTUI) · VS Code · Mobile / Hub
                             │
                             │ v35 NDJSON JSON-RPC (local socket / SSH / Hub tunnel)
                             ▼
                        Bun Daemon
     ├── Native Provider Routing (streaming, retries, caching, budgets)
     ├── External ACP Bridges (Antigravity agy acp, Claude Code, Codex, custom)
     ├── Immortals Swarm Host & Sub-agent Orchestrator
     ├── Tool Registry, SWE Tools, LSP, Permissions & OS Sandbox (Seatbelt / bwrap)
     ├── Durable SQLite/FTS Sessions, Replay, Compaction, Snapshots & Tiered Memory
     └── Skills, MCP, Plugins, Channels (Slack/Telegram/Discord) & Cron Scheduler
```

Every surface consumes the same serializable daemon event vocabulary; the daemon guarantees a terminal event for every turn and keeps work running cleanly across reconnects and remote SSH hops.

---

## Development

```bash
bun install --frozen-lockfile

# Full repository verification gate (repo checks, all 6 TS projects, runtime/TUI/mobile tests, builds)
bun run verify

# Targeted checks & builds
bun run check                       # assertBunOnly + all 6 TypeScript projects
bun run test:runtime                # Bun runtime & desktop unit/integration tests
bun run test:ui                     # OpenTUI tests
bun run --cwd xerxes test:mobile    # Mobile web tests
bun run build                       # runtime + TUI + mobile bundles
bun run --cwd xerxes build:desktop  # macOS Xerxes Agents.app
bun run --cwd xerxes build:installer # macOS .dmg installer
bun run --cwd xerxes build:vscode   # VS Code .vsix extension
bun run docs:build                  # Native Bun docs output
git diff --check
```

### Release validation

```bash
bun run verify
RELEASE_ROOT="$(mktemp -d)"
PACKAGE_DIR="$RELEASE_ROOT/package"
ARCHIVE="$RELEASE_ROOT/xerxes-agents-$(bun -p 'require("./package.json").version').tgz"
bun run release:prepare -- --output "$PACKAGE_DIR"
(cd "$PACKAGE_DIR" && bun pm pack --filename "$ARCHIVE" --ignore-scripts)
bun run release:check -- --package "$PACKAGE_DIR" --archive "$ARCHIVE"
bun run release:smoke -- "$ARCHIVE"
```

---

## Documentation

- [Desktop workspace](docs/desktop-workspace.md)
- [Immortals swarms](docs/swarm.md)
- [Hub protocol & remote access](docs/hub-protocol.md)
- [XERXES Supervisor & Slack](docs/supervisor-slack.md)
- [Configuration guide](docs/configuration-guide.md)
- [Deployment guide](docs/deployment-guide.md)
- [System architecture](docs/system-architecture.md)
- [Testing guide](docs/testing-guide.md)
- [Changelog](docs/changelog.md)
- [Contributing](docs/contributing.md) & [AGENTS.md](AGENTS.md)

## Acknowledgements

Xerxes' OpenTUI presentation was informed by [superagent-ai/grok-cli](https://github.com/superagent-ai/grok-cli) (MIT License). Xerxes keeps separate branding and is not affiliated with Grok or xAI.

## License

Licensed under the [Apache License 2.0](LICENSE).

Created by [Erfan Zare Chavoshi](https://github.com/erfanzar).
