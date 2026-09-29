# Changelog

Selected highlights from the project's git history, grouped by major theme. For the full history, run `git log --oneline --no-merges`.

The current native work is the Bun/TypeScript cutover. Historical entries below intentionally
describe the earlier implementation and are not current setup instructions.

---

## Unreleased

- Runtime: compaction now fires on the provider's real prompt size. The context
  estimate ran about 1.8x low on code-heavy Claude transcripts, so a 1M window the
  meter put near half full was rejected as full and auto-compaction never started.
  Each round's reported prompt tokens now calibrate the estimate, which is kept on
  the session for the pre-turn check and the context meter.
- Runtime: a long goal round can overflow and compact more than once. The
  compact-and-retry was spent once per turn, so the second overflow in an
  hours-long round blocked the goal.
- Runtime: Stop takes effect at once. A tool that ignored the stop (a
  `check_command` waiting up to a minute on a background build) held the turn
  open; interruptible tools are no longer awaited after a stop, and
  `check_command`'s wait ends on it.
- Desktop: Escape needs a second press within two seconds to stop a running task.
  One stray press (reaching for a screenshot shortcut) cancelled the task and every
  agent it had spawned; it now shows "Press esc again to stop". The Stop button is
  still one click.
- Desktop: connecting to an SSH workspace copies this Mac's key-based provider
  profiles (OpenRouter, Z.ai, Kimi and the like) to the host, so they work there
  without re-entering keys. The host's own profiles and selection are kept;
  Claude Code and ChatGPT sign-ins stay per machine.
- Providers: the built-in Claude Code profile is now `claude-code` instead of
  `cc`. Saved profiles and selections migrate on load, and `cc` still resolves
  for older sessions, `/provider cc` and `cc/<model>` ids.

## 0.6.3 — 2026-09-29

- Desktop: much lower energy use. Every workspace page a window has opened kept
  listing all saved and live sessions every five seconds, even when hidden; with
  several workspaces open that polling dominated the runtime's CPU. Hidden or covered
  pages now skip it and catch up when shown, an unfocused window refreshes its lists
  every thirty seconds, and background output, agent and run pollers pause while hidden.
- Runtime: an idle-only runtime update no longer restarts while a goal is armed; a
  restart silently disarmed the goal and stopped autonomous work at its next round.

## 0.6.2 — 2026-09-28

- Usage: OpenRouter keys show credit left, spend today, this week, this month and
  all time. A new "By model" section totals the last 30 days per model from the
  runtime's own record of every task and subagent round, priced at published prices
  ("price unknown" when none exists).
- Providers: the model is optional when adding a provider. Left blank, the runtime
  asks the provider which models the key can use and starts on the first; an edit
  keeps the saved model.
- Desktop: command and skill suggestions, menus and pickers stay opaque over a
  transparent or image background.
- Desktop: the rail's Touched card expands in place, and clicking a file opens
  Session edits on that file's diff.

## 0.6.1 — 2026-09-28

- Desktop: fresh macOS installs open with the transparent window, system blur, an
  opaque chat column and see-through side panels; other platforms keep solid surfaces.
- README: a real screenshot of the desktop app replaces a design render that never
  loaded on GitHub.

## 0.6.0 — 2026-09-28

Desktop application
- Claude-style layout: list sidebar, one reading column, a composer with the
  repository bar (branch, diff, Review, Create PR), model, effort and a context ring.
- Dotted agent status orbs for tools, reasoning, writing and waiting, on one shared
  low-rate timer that stops when the window is unfocused, covered or settled.
- Files open and edit in place with rendered Markdown, save conflicts and revert;
  read, write and edit tool rows show their content inline.
- Themes and backgrounds (Xerxes default, Claude colours), working blur, a transparent
  mode, copy buttons on replies, chat deletion and Claude-sized type.
- Lower energy use: covered workspace views are throttled, stream updates are
  batched, and looping animations pause while unfocused.
- Running chats reattach after an update or reconnect instead of opening a new task.

Runtime
- Leased clients' turns and armed goals keep running while the desktop is away
  (a locked laptop, a dropped SSH link) and reattach on return.
- Goal rounds lost to transient provider failures retry after 1, 3 and 10 minutes
  instead of blocking the goal.
- Session recovery never splices journal entries from older compaction generations
  onto a compacted session; a live session that loses a save conflict is preserved
  under `sessions/divergent/` before the saved copy is reloaded.
- Claude Code: streams stay alive while a large tool call is written, tag-form tool
  calls parse, the system prompt is passed by file and the CLI is kept current.
- Goal evidence can cite the latest successful tool call; reasoning effort reports
  "default" rather than a false "off"; outdated SSH runtimes update when idle.

Terminal UI
- Welcome screen, transparent-mode panels, accurate status and turn durations;
  diff panel seeks land after layout.

## 0.5.0 — 2026-09-20

- Add a selectable transparent TUI background alongside Chrome styling.
- Reuse supported local providers for SSH tasks with one scoped setup approval;
  credentials and refresh stay local, with explicit expiry and revocation.
- Keep approved local models discoverable and usable by delegated agents.
- Continue finished spawned agents through SendMessageTool with stable identity,
  saved history, and ordered follow-up input.
- Persist model, mode, effort and permission choices atomically; preserve prior
  settings on save failures and restore task policy after daemon restart.
- Improve task navigation, history, output, cancellation and recovery behavior.
- Correct empty optional provider fields in model inventory.
- Avoid Linux watcher failures caused by unrelated Unix sockets by polling scoped
  file metadata; report unsupported Windows PTY runtimes without orphaned panels.


- Reconcile restored active-agent statuses with workers owned by this daemon.
- Show provider response waits explicitly and avoid generic working verbs during silence.
- Deliver compaction and retry progress before the first model output, and retain terminal errors from silent attempts.

- Show the compaction spinner for automatic and mid-turn summaries, preserve
  lifecycle statuses through the TUI adapter, and clear the indicator on failure.

- Bound compaction input chunks independently of the model context window and
  retry timeouts with smaller chunks, including mid-turn compaction.
- Identify automatic compaction failures explicitly while retaining the original conversation.

## 0.4.5 — 2026-09-09

- Explicit model selections now take precedence over intelligence tier hints when
  spawning agents; working-tree isolation accepts an explicit HEAD reference.

- Fixed saving custom agents whose names are derived from their filenames,
  including agents created by project setup.
- The custom-agent editor now lists and edits nested Markdown specialists and
  preserves valid declared names that differ from their filenames.
- Agent drafts without a discovery description now fail validation instead of
  saving successfully and silently disappearing from the runtime catalog.
- Background task creation now exposes, validates, and forwards worktree settings.
- Cancelling SSH setup returns promptly even if SSH ignores termination.
- A tunnel drop during renderer handoff no longer launches a TUI on a dead tunnel.

## 0.4.4 — 2026-09-09

- Added bounded SSH reconnects, manual retry, connection progress, and recovery of
  the last recorded remote session.
- Redesigned the custom-agent browser and added model-generated, editable drafts.
- Enabled automatic workspace snapshots before model turns and shell commands.
- Fixed automatic compaction checks inside long-running tool loops, bounded
  oversized compaction requests, and preserved compaction state across saves.
- Kept status RPCs responsive during compaction and extended the manual
  compaction timeout.

- Interrupted commands retain partial stdout/stderr and report cancellation with
  its available reason, instead of mislabeling valid arguments as a validation error.
- Added `/activity`: a unified panel for session shells/watches and workspace
  schedules, with elapsed times, output/details, stop/pause controls and brief
  completion/failure badges. Lifecycle events replace composer status polling.
- Bound provider profiles to conversations and inherited subagents. Model picker
  selection applies the provider and model together; GPT models cannot be routed
  to Kimi subscriptions. HTML provider error pages now show concise messages.

- Added live blue shell/watcher counts beside the composer settings, including
  while idle. Click a count to inspect background work; completed work clears.

- Remote workspaces now render locally over a private SSH-forwarded daemon
  socket. Tunnel failures return to the original local workspace.
- F7 collects changes on the daemon host and includes navigable untracked-file
  previews, with empty-file and binary-file handling.

- Fixed inline swarm status lists splitting comma-containing titles into phantom
  queued agents; saved transcripts reconcile titles with actual agent records.
- Added `TaskOutputTool` character pagination (`offset`, optional `limit`, up to
  8,000 characters). Follow the returned next offset to read complete saved
  reports without rerunning agents. For new work on a completed agent, use
  `AgentTool` with `resume` and `prompt`; `SendMessageTool` targets running agents.

## 0.4.3 — 2026-09-07

- Redesigned operational TUI dialogs, forms, empty states, and capability guides.
- Added SSH config host discovery, remote project folder browsing, and automatic
  installation and updates of managed remote Xerxes builds.
- Added Plugin Creator mode with native TypeScript plugin examples and Bun tests.
- Exposed custom-agent editing and skill/plugin controls, with project specialist
  discovery and Claude-style agent workflows.
- Improved goal configuration, provider/model discovery, reasoning visibility,
  session restoration, and attachment handling.
- Fixed bang-command output and exit codes; command results now receive a model
  follow-up and remain available when restoring the conversation.

## 0.4.0 — 2026-09-03

- Published the native distribution under the scoped npm identity
  [`@xsimurgh/xerxes-agents`](https://www.npmjs.com/package/@xsimurgh/xerxes-agents).
- Added Claude Code compatibility for project instructions, `/init`, shell mode, memory capture,
  hooks, project MCP, headless output formats, model fallback, and injected skills.
- Added timeout-driven command backgrounding and persistent PTY sessions with live terminal
  inspection and control from the TUI.
- Added a live spawned-agent roster with animated activity cubes, elapsed time, tool summaries,
  completion state, and restoration after session reattachment.
- Improved Agent View attachment, session-tab switching, reasoning-model pinning, provider-profile
  capability resolution, and full tool-argument summaries after resume.
- Corrected cumulative LLM/TTFT accounting and rejected fabricated Codex throughput samples from
  tool-only terminal events.
- Added provider-stream retries, fresh instruction-file overlays, and complete daemon-capability
  mapping across the TUI and desktop renderer.

## 0.3.0 — Native Bun/TypeScript migration

- Added the native runtime, CLI, daemon, session, streaming, API, and OpenTUI client paths under the
  Bun workspace.
- Replaced the handwritten RST/Sphinx documentation branch with Markdown sources and generated
  TypeScript API pages.
- Added native examples, installer, CI/release paths, and focused contract/parity coverage.
- Completed the Bun-only cutover: retired Python source, tests, tooling, playground paths,
  virtual-environment artifacts, caches, and distributions are removed; the frozen install,
  typecheck, full native test suites, build, release-package check, and Bun-only guard pass.

Project identity has shifted over time: originally `eLLM`, renamed to `AgentX`, then `Calute`, and now **`xerxes-agent`** with the native `xerxes` command.

---

## 0.2.6 — 2026-06-23

- Added the bundled `bug-bounty-hunter` multi-iteration swarm repair skill.
- Bumped the package/runtime version to 0.2.6.

## 0.2.5 — 2026-06-22

- Added the bundled `eternal-army` swarm-orchestration skill.
- Fixed skill discovery refresh for source checkouts and resilient frontmatter parsing.
- Tightened `/steer` delivery so queued guidance reaches the next provider request or is saved for the next turn.
- Bumped the package/runtime version to 0.2.5.

## 0.2.4 — 2026-06-19

- Replaced static/clamped tool-result handling with project-memory spillover plus agent-written summaries.
- Added context-window provisioning before provider calls and after tool batches.
- Tightened DeepScan so subagents save full findings to project memory and return only compact pointers.
- Bumped the package/runtime version to 0.2.4.

## 0.2.3 — 2026-06-19

- Fixed bare `/provider` with zero saved profiles so it opens the add-profile picker instead of returning dead-end text.
- Bumped the package/runtime version to 0.2.3.

## 0.2.2 — 2026-06-19

- Added the managed `~/.xerxes-venv` installer flow and terminal alias setup.
- Updated `xerxes update --force` to target the managed venv when present.
- Bumped the package/runtime version to 0.2.2.

## 0.2.1 — 2026-06-19

- Bumped the package/runtime version to 0.2.1.

## 0.2.0 — 2026-04-16

### Project rename and restructure

- **Renamed Python module** `xerxes_agent` → `xerxes` ([543fb74](https://github.com/erfanzar/Xerxes/commit/543fb74)).
- Distribution package stays `xerxes-agent` for PyPI compatibility.
- All build config (pyproject, hatch hook, pytest.ini, CI, Dockerfile, pre-commit) repointed to the new module.
- Orphan `src/python/xerxes_agent/` shim directory removed.
- Metric `xerxes_agent_switches_total` renamed to `xerxes_switches_total`.

### Security hardening

- **Fixed sandbox pickle-escape vulnerability.** Child-to-parent IPC in both `docker_backend` and `subprocess_backend` now uses JSON instead of pickle. The old symmetric pickle design was an arbitrary-code-execution vector via `__reduce__` (a malicious sandboxed tool could craft a payload that would execute `os.system(...)` in the parent process upon deserialization).
- **Fixed `math_tools.Calculator` eval-sandbox bypass.** Replaced `eval(expr, {"__builtins__": {}})` with an AST-whitelist evaluator that accepts only numeric literals, the arithmetic operators `+ - * / // % **`, unary `+/-`, the functions `sin cos tan log sqrt abs pow exp`, and the constants `pi` and `e`. The previous pattern was bypassable via `().__class__.__bases__[0].__subclasses__()`.
- Both fixes verified with negative proof-of-concept tests.

### Feature work

- CLI feature buildout and scrollable scrollback ([543fb74](https://github.com/erfanzar/Xerxes/commit/543fb74)).
- Agent orchestration, hand-off, and planning features ([a816780](https://github.com/erfanzar/Xerxes/commit/a816780)).
- Markdown rendering and CLI enhancements ([edfcfa9](https://github.com/erfanzar/Xerxes/commit/edfcfa9)).
- `Cmd-Enter` submit in the TUI ([7e88262](https://github.com/erfanzar/Xerxes/commit/7e88262)).
- Hardened tool routing and TUI behavior ([20e143b](https://github.com/erfanzar/Xerxes/commit/20e143b)).

### Removed

- **Chainlit UI package.** The `xerxes/ui/` subpackage wrapping Chainlit was removed; UI now belongs outside the core framework. Removed `chainlit` from optional dependencies, eliminated the `ui`/`research`/`full` extras' Chainlit dependencies, and stripped `create_ui` methods from `Xerxes`, `Cortex`, `CortexAgent`, `CortexTask`, `TaskCreator`, and `DynamicCortex`.
- Outdated `docs/api_docs/calute.rst` and other stale Sphinx references.

### Test fixes

- 15 previously-failing tests restored to passing. All referenced the old `xerxes_agent.*` module via `mock.patch` / `monkeypatch.setattr` strings that static-analysis couldn't catch. Rewritten to use `xerxes.*`.
- pytest coverage target corrected from nonexistent `src/python/xerxes_agent` to the real `src/python/xerxes`.

---

## 0.2.0 prerelease — early 2026

- **Modular runtime and TUI** ([88e9ac8](https://github.com/erfanzar/Xerxes/commit/88e9ac8)) — released as "Calute 0.2.0".
- Enhanced LLM functionality: reasoning-token extraction, sampling parameter overrides, compat shims for OpenAI-wire-compatible providers ([dcdbfdf](https://github.com/erfanzar/Xerxes/commit/dcdbfdf)).
- Comprehensive docstrings added across the codebase ([d2067be](https://github.com/erfanzar/Xerxes/commit/d2067be)).

---

## 0.1.x series

### 0.1.2 ([bda33b9](https://github.com/erfanzar/Xerxes/commit/bda33b9))

- Modified `create_ui` parameter signature (since removed).

### 0.1.1 ([659cf05](https://github.com/erfanzar/Xerxes/commit/659cf05))

- Added `create_ui` method across `Xerxes` / `Cortex` / `CortexAgent` / `CortexTask` (since removed in 0.2.0).
- Gradio-based UI components added ([6194f5f](https://github.com/erfanzar/Xerxes/commit/6194f5f)) — later replaced by Chainlit, now removed.

### 0.1.0 core

- **MCP (Model Context Protocol) integration** ([474d7f7](https://github.com/erfanzar/Xerxes/commit/474d7f7), [8d1afaf](https://github.com/erfanzar/Xerxes/commit/8d1afaf)) — stdio + SSE + streamable-HTTP transports; `MCPClient`, `MCPManager`, `MCPTool`, `MCPResource` public API.

---

## 0.0.x series (Calute era)

- **Architecture overhaul with new LLM abstraction layer** ([7779931](https://github.com/erfanzar/Xerxes/commit/7779931)) — introduced `BaseLLM` as the abstract provider interface.
- **Migrated from custom chat types to OpenAI proxy types** ([9b4fbf1](https://github.com/erfanzar/Xerxes/commit/9b4fbf1)); improved API server streaming.
- **API server (FastAPI)** added in 0.0.19 ([0535858](https://github.com/erfanzar/Xerxes/commit/0535858)).
- **EnhancedMemoryStore** and general helper assistant ([cf05bec](https://github.com/erfanzar/Xerxes/commit/cf05bec)).
- Comprehensive tools suite: web search, code execution, browser tools, data tools, math tools, etc. ([45edc91](https://github.com/erfanzar/Xerxes/commit/45edc91), [061a762](https://github.com/erfanzar/Xerxes/commit/061a762)).
- Advanced delegation engine + memory management system ([02255dc](https://github.com/erfanzar/Xerxes/commit/02255dc)).
- `reinvoke_after_function` option for re-prompting LLM after tool execution ([d54b9ec](https://github.com/erfanzar/Xerxes/commit/d54b9ec)).
- Python 3.13 support ([9909837](https://github.com/erfanzar/Xerxes/commit/9909837)).
- XerxesConfig hierarchical config system; `/context`, `/cost`, `/compact` slash commands.

---

## Historical (pre-Xerxes: eLLM / AgentX era)

Deep history predating the current multi-agent framework shape. Highlights:

- eLLM → AgentX rename with project restructure ([8ecb339](https://github.com/erfanzar/Xerxes/commit/8ecb339)).
- Early Gradio GUI experiments (later deprecated).
- Initial Ollama backend support ([38654fc](https://github.com/erfanzar/Xerxes/commit/38654fc)).
- Original RAG work, LLM quantization support, WebSocket server experiments.

These commits are preserved for archaeological purposes but don't reflect current API or design.

---

## How to read this

Commit SHAs link to the GitHub commit if the repository is public. Dates are approximate (inferred from tag history and commit clustering, not author dates).

For a precise diff between any two points in history:

```bash
git log --oneline <from>..<to>
git diff --stat <from>..<to>
```

For a live changelog of the most recent work:

```bash
git log --oneline --no-merges -20
```
