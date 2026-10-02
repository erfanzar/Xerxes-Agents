# Changelog

Selected highlights from the project's git history, grouped by major theme. For the full history, run `git log --oneline --no-merges`.

The current native work is the Bun/TypeScript cutover. Historical entries below intentionally
describe the earlier implementation and are not current setup instructions.

---

## 0.6.18 — 2026-10-02

Reliability release: 46 bugs found by a whole-codebase review, each confirmed by an independent check, fixed with a test that fails without the fix.

- Messages and sessions: a turn that was running when the app quit, crashed or updated is recovered on reopen, including a brand-new session's first turn. /undo and /retry after a compaction no longer wipe the session. Opening a chat mid-turn, or reconnecting, no longer hides, duplicates or reorders messages. A save conflict no longer leaves a window on a hidden empty session. A damaged pre-compaction archive no longer makes a session unopenable.
- Goals: after a runtime update every armed goal continues, not only the session you reopen. A steer sent while a turn is saving is applied. A forced restart no longer drops the goal.
- Agents and workflows: agents are not lost or merged after a restart, and retries keep their settings and count correctly. Reconnecting an MCP server or changing LSP settings no longer kills running agents. Workflows no longer fail at a hidden 100-agent cap, and an agent's background commands end with it.
- Claude Code: tool calls are no longer cut at a literal "</function>" inside their arguments, a reply is no longer swallowed after markup mentioned in prose, and a call cut off by the output limit is not run half-written.
- App and SSH: windows can be closed, reloading a window keeps its own session, and windows restore on Windows and Linux. A dead remote runtime is restarted instead of reconnecting forever, and connecting no longer overwrites the server's own provider settings.
- Tools: large command output keeps its beginning and its final error lines. An agent's own file append no longer blocks its next edit, and agents no longer evict each other's read-before-edit records.

## 0.6.17 — 2026-10-02

- Memory guard: an agent command, with everything it starts, is stopped once it uses more than a limit (default: half the computer's memory; GPU memory counts on macOS). The agent is told why and how to run less at once. Set the limit, or turn it off, in Settings → General.
- Desktop: opening a chat shows your latest message. A chat opened on its newest 100 actions, and a goal or workflow session can run hundreds of tool actions after the last thing you said, so the conversation looked lost. It now pages back until your last message is on screen; the button reads "Show earlier conversation".

## 0.6.16 — 2026-10-01

- Claude Code: a reply rejected with "The model's tool call could not be parsed" is retried (twice at most) with a note telling the model it used the native <invoke> syntax and must call tools as <function=NAME>{json}</function>. Sonnet slips into the native form in long, tool-heavy agent runs, and each slip used to fail the whole agent. Text already shown is not repeated.

## 0.6.15 — 2026-09-30

- Agents: a reply with no text and no readable tool call no longer ends an agent as "completed without a final response". After tool work, the agent is asked again (twice at most) to call a tool in the required form or give its answer. Workflow agents on Claude Code hit this after large tool results.
- Settings → Agent intelligence: point the light, balanced and smart tiers at any provider, any model (from the provider's list or typed in) and any reasoning effort that model supports, and choose which tier new agents use.
- Updates: the app checks for a new version every five minutes. "Check for Updates…" now says when you are up to date instead of showing nothing, and "Not now" keeps a version from reappearing until you ask or relaunch.

## 0.6.14 — 2026-09-30

- Compaction: a long session's summary no longer grows until compaction cannot make room. Each pass used to keep the previous summary word for word and add the new one after it; a goal session reached a 25K-token summary against a 2K budget and its turns failed with "Automatic compaction could not make room". Once the carried summary and the new one pass twice the budget, they are folded into one summary. The full history stays in the pre-compaction archive.

## 0.6.13 — 2026-09-30

- Tools: a large file can be read and edited. ReadFile refused any file over 256 KB, even a 50-line window of it, and since an edit needs a read first, such a file could not be edited at all. Windows of any file up to 32 MB now read normally; only a whole-file read (limit=-1) keeps the 256 KB limit.
- Agents ask with clickable choices. AskUserQuestionTool takes `options` (recommended first); the desktop shows them as buttons with number keys plus a field for your own answer, and the question text renders as Markdown. The daemon used to drop options from the plain question form.
- Desktop: a question's options are single-choice; a second click moves the choice instead of sending both.

## 0.6.12 — 2026-09-30

- Desktop: a new task no longer shows the task it replaced. After switching, a window could keep asking the runtime about the previous task, so the new one showed that task's goal, background jobs and model time. It now always uses the task the runtime says it opened, ignores answers meant for the task it left, and a finished turn's totals only reach the window still showing that task.
- Desktop: the side panel's tabs no longer overlap when it is narrow. They keep their width and scroll, the open tab stays in view, and only an edge with hidden tabs fades.

## 0.6.11 — 2026-09-30

- Desktop: the side panel's edge runs unbroken around its rounded corners, and the resize divider beside it no longer draws a second line through the gaps above and below the panel (it still highlights on hover).

## 0.6.10 — 2026-09-29

- Desktop: a workflow agent's full output shows in the inspector. Agents in a workflow used to be saved with only their first 2,000 characters; they now keep up to 16,000 like any agent, and only the oldest agents of a very large run are trimmed. Trimmed output is labelled as such.
- Desktop: an agent's structured result (a workflow schema answer) shows as labelled fields instead of one line of JSON, and its output is pretty-printed.
- Desktop: workflow phase headings line up with their agents, and the inspector's "All activity" link points back.

## 0.6.9 — 2026-09-29

- Desktop: the Files panel lists dot folders and files (.github, .vscode, .env.example…); only .git and .DS_Store stay hidden.
- Desktop: a rendered Markdown file fills the preview and scrolls as one page; it was capped at 380px, which cut long files off at the first big code block.
- Desktop: with a transparent background, the side panel's corners no longer show a sliver of bare desktop beside the conversation.

## 0.6.8 — 2026-09-29

- Desktop: "Restart workspace runtime" works while the runtime is busy. The popover names what is still running and offers Restart now; an active goal picks up again on the new runtime (older runtimes are shut down and the goal is resumed).
- Desktop: a clearer icon for the Agents chip and card, and the Tasks / Agents switch is a proper segmented control instead of a bent underline.

## 0.6.7 — 2026-09-29

- Desktop: the app updates itself from GitHub releases, with your permission.
  It checks at startup and every six hours (and from Xerxes Agents › Check for
  Updates…), shows the new version and its release notes, and asks: Install and
  restart, Not now, or Skip this version. On yes it downloads the installer,
  checks it against GitHub's published SHA-256, verifies the app's signature,
  and swaps it in when the app quits, then relaunches. Development builds and
  unwritable Applications folders get the release page instead.
- Desktop: reopening the app during a long turn no longer hides what that turn
  already did. The conversation only catches up when a turn ends, so a goal
  round that ran for hours looked like it had lost its last several steps
  until it finished; the runtime now serves the running turn's messages to a
  client that attaches mid-turn.

- Agents: an **Agents** chip beside Plan in the composer (and `/delegate`):
  **Off** works alone, **Auto** (default) fans out when work splits and asks
  once before a run of more than about ten agents whether to be budget-minded or
  thorough, **Eager** delegates whenever work can split, with a verify step.
- Agents: the model gets a playbook for choosing between working alone, one
  agent, a batch and a Workflow — reviews, sweeps, migrations, research,
  claim-checking, competing debugging theories and benchmark matrices each have
  a shape, and the costs of fanning out are spelled out.
- Workflows report their cost at published prices; the run's card shows it.
- Git: **Review** runs large changes as a find-then-verify workflow (a reviewer
  per slice, a skeptic per finding) and reports how many findings survived.
  **Create PR** runs the project's checks and writes a sectioned description:
  summary, changes by area, behaviour changes, how it was tested, risks and
  rollback, follow-ups.

## 0.6.6 — 2026-09-29

- Desktop: agents and workflows stay visible in the conversation. They used to
  fold away with the tool calls around them; now only the tool calls fold.
- Desktop: the agents card follows Claude Code's shape — "Running 4 agents…"
  over a tree, one branch per agent with its model, tool uses, tokens and
  time, and under it what it is doing now or "Done (15 tool uses · 23.4K
  tokens · 1m 13s)"; "+N more tool uses" opens its latest calls in place.
  Workflows keep their phases, progress bar and dot grid for large runs.
- Desktop: workflow runs get their own card in the Activity panel, by phase;
  the Agents list keeps the agents started outside a workflow.
- Agents' live activity reads as a command line ("Running env
  JAX_PLATFORMS=cpu,mps python bench.py") instead of "cmd=env, args=…".

## 0.6.5 — 2026-09-29

- Runtime: an update no longer waits forever behind an armed goal. It holds
  the next goal round, installs when the current one ends, and the fresh
  runtime re-arms the same goal and carries on. 0.6.3 had stopped restarts
  while a goal was armed, so a long-running goal kept the old runtime (and
  new tools such as Workflow) indefinitely. The sidebar says "Update after
  this goal round" meanwhile.
- Desktop: a steer leaves "Queued messages" as soon as the model receives it.
  It used to stay there until the turn ended, which in a goal round can be
  hours after the model had already answered it.

## 0.6.4 — 2026-09-29

- Agents: a `Workflow` tool. The model writes a short script with `agent()`,
  `parallel()`, `pipeline()` and `phase()` to fan work out to as many
  subagents as the job needs (reviews, audits, migrations, find-then-verify),
  picking a model per agent and getting structured results back. The script
  runs in its own process without provider keys; stopping the task stops the
  script and every agent it started.
- Agents: the model delegates on its own initiative and matches the model to
  the job, using a new prompt section that lists the models each configured
  provider reports (with published prices) so mechanical work can go to
  cheaper, faster models.
- Desktop: the in-chat agents card is rebuilt for one agent to thousands. A
  workflow shows its name, phases, each agent's model, what it is doing right
  now and its clock; large phases draw as a dot grid with rows only for agents
  still running or failed. Spawn batches are no longer cut at 24 in the card.
- TUI: a Workflow row shows its run's phases, tallies and live or failed agents.
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
