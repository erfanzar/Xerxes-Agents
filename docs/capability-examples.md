# Capability examples

Type slash commands in the TUI. Quoted messages below are prompts to send after closing the relevant picker. External services require your own configuration.

## Plugin Creator

Run `/plugin-creator`, then send:

> Build a text-statistics plugin that counts Unicode characters, words and lines. Include Bun tests for empty and invalid input and a README with usage examples.

This opens a new session with the plugin authoring preset. Your prior chat stays saved in Agent View. Ordinary new sessions keep their configured default preset. `/creator` remains the separate agent-preset authoring mode.

A working reference is [text-stats.ts](../examples/plugins/text-stats.ts). Verify its actual installation and execution path offline, from the repository root:

```sh
bun test xerxes/test/pluginCreator.test.ts
```

To install the reference in the TUI, substitute your absolute checkout path:

```text
/plugins install "/absolute/path/to/Xerxes-Agents/examples/plugins/text-stats.ts"
/plugins inspect text-statistics
```

Ask “Call plugin_text_stats with args [\"Hello world\\n🙂\"].” Expected result: `{"characters":13,"words":3,"lines":2}`. Model calls pass positional arguments using `{"args":[...]}`.

```text
/plugins disable text-statistics
/plugins enable text-statistics
```

Disabled plugins reject new calls. Keep installed source files in place. Restart Xerxes after editing installed module code. Authoring does not automatically install a plugin. Managed installation supports native tool modules; hooks, providers and channels need an embedding host.

## Specialists and skills

Run `/custom-agents`, press **N**, enter this example, then **Ctrl+S**:

```markdown
---
name: reviewer
description: Review changes for reproducible correctness bugs without editing files.
tools: ReadFile, GlobTool, GrepTool, ListDir
---
Read relevant code and report concrete findings with file paths, triggers and impact.
Do not edit files. Say when evidence is insufficient.
```

Ask “Use the reviewer specialist on my current changes.” Inspect delegation with **F6**. Existing `.xerxes/agents` definitions load automatically. `/init` can inspect the repository and generate missing setup while preserving existing files.

For a local skill, save this as `review-checklist/SKILL.md`:

```markdown
---
name: review-checklist
description: Review input validation, error propagation and state restoration.
---
Inspect changed code. Check invalid inputs, cancellation and persistence.
Report confirmed bugs with a reproduction, or say none were found.
```

Run `/skills install "/absolute/path/to/review-checklist"`, then `/skills search review-checklist`. Ask “Use review-checklist on my changes.” Referenced assets belong inside the skill directory. Shell preprocessing requires explicit skill trust. Run `/skills lint` to see all authoring problems at once.

For a complete agent preset, run `/creator` and ask “Create a testing-focused coding preset with examples of when to use it.” `/preset list` lists presets; `/preset use <id>` selects one before the first turn. `/reload` refreshes externally edited project definitions. `/` completion and `/help` show commands actually available in your session.

## Review what Xerxes learned

These native Bun/TypeScript features retain evidence and let you review improvements before they change anything.

### Shared lessons across clones and machines

Lessons recorded in a Git project carry a path-independent repository key: normalized `origin`, or the smallest root commit when there is no origin. Clones with the same key share local recall even at different paths. Old path-only lessons still work in their original workspace.

Run `/xerxes lessons share` to opt this machine into the repository and create `.xerxes/lessons/` at the **Git toplevel**, even from a nested session folder. Commit that folder normally; the command warns if Git ignores it. **On each fresh machine, run `/xerxes lessons sync` once** to authorize file imports/exports locally. Both commands persist the repo-keyed decision in `~/.xerxes/governed-memory/lesson-sharing.json` (under `XERXES_HOME` when overridden), outside the repository. A cloned folder or PR branch cannot grant this opt-in. Later turns sync opted-in repos before recall; a busy queue is skipped and waiting is capped at about 1.5 seconds, with cancellation observed. `/xerxes lessons sync` also forces a pass and reports counts. Invalid/conflicting files are reported and never deleted; writes replace complete files atomically. Pending and confidential/restricted records are never exported. Superseded approved IDs remain represented for convergence, but are not injected into turns. A retirement tombstone is exported only for an ID already shared, never for rejected/rolled-back text that stayed private.

**Imported lessons are automatically approved**, by the owner's choice. Original evidence is provenance from another machine, not locally verified proof. Host/provenance fields are frozen persistently, so changing hostnames never changes a lesson ID's content. Corrections get new IDs; retirement tombstones travel too, so an old active copy cannot resurrect retired guidance. When two retirements meet, the earlier timestamp wins. The context walker excludes the shared folder; only bounded, matching approved lessons enter a turn.

When a Hub is configured, capable daemons also expose signed lesson sync RPCs (`session.read` for listing, agent-allowed `lessons.sync` for applying). The Hub can synchronize matching repositories without moving provider credentials. Hub list/apply do not require the file-sharing opt-in: explicitly enrolling that Hub is their trust decision. Set `XERXES_LESSON_SYNC=off` on a daemon to disable **all** lesson import/export (file and Hub). No Git repository identity means sharing is unavailable; no identity is invented.

### Evidence-backed lessons

Ask: “Use `lesson_record` to retain the reusable lesson from that successful tool call, with a short cue and confidence.” The host resolves the latest successful call, or verifies an explicit `evidence_tool_call_id`. Alternatively, `test_evidence` records a reported command, exit code and summary; it is not labelled host-verified. Missing/failed tool-call evidence is refused. `lesson_recall` matches cues within the active workspace, and `lesson_review` requests approval, rejection or rollback of an exact id with a reason through the normal permission gate; automatic turns cannot approve their own lessons. Corrections and rollback retain governed history; rendered INSIGHTS-style Markdown is only a view.

```text
/xerxes lessons
/xerxes lessons Bun fixtures
/xerxes lessons approve <id> Reviewed the evidence
/xerxes lessons reject <id> Not generally useful
/xerxes lessons rollback <id> Guidance is now outdated
```

Lessons start **pending**. Approve a lesson yourself after checking its evidence; reject and rollback require a reason. At the start of each later turn, up to five approved lessons matching your message are attached to the turn context, not the cached system prompt. The block is bounded to 2,000 characters, clips with `[clipped]`, includes cues and evidence provenance, and identifies lessons as reviewed guidance that cannot override your instructions. Pending, rejected, expired and superseded lessons are not automatically injected; no match means no added context. Recall failures are logged without failing your turn.

### Review reflection proposals before changing anything

By default, a goal transitioning to complete or blocked, or a long turn (25 tool calls or 20 minutes), can request one background reflection using the session's provider. The turn ends first. Aborted/errored turns and subagent sessions do not reflect. The model must cite exact `[ACTION]`, `[OBSERVATION]` or `[ERROR]` lines from the shared trajectory, and may return no proposal. A clipped preview is not proof of failure. Configure `runtime.reflection: all | goals | off` (default `all`); `goals` limits it to goal transitions. `XERXES_REFLECTION` overrides the startup setting.

```text
/xerxes
/xerxes help
/xerxes proposals
/xerxes show <id>
/xerxes show <id> --json
/xerxes approve <id> Reviewed evidence and full content
/xerxes reject <id> Too specific to this task
```

The desktop has a **Learned** panel with **Proposals** and **Lessons** tabs and in-chat proposal cards. In the TUI, typing `/xerxes` alone opens a review overlay; a status chip shows how many items need review. Headless clients receive readable help for the bare command. A new proposal sends a structured reflection notification: “Reflection proposal <id>: <kind> <target>. Nothing was changed. Review it with /xerxes.” Its fields include the id, kind, target, first rationale line and evidence count. `show` reads like a review document: status, trigger/objective, rationale, quoted evidence and exact proposed content, including bundle file sizes and contents. Add `--json` for the full stored record.

`reflection_list` exposes the same stored proposals to the model; `reflection_review` requires human authority to approve and cannot self-approve from an automatic goal round. A typed `/xerxes approve` is a human decision. Skills pass through the linted bundle writer in the user skills directory, preserving version/tags on update. Agent YAML must validate and match the approved name before atomic writing under the active project's `.xerxes/agents`. Invalid proposals/lint refusals never partially write. Plugin ideas only return a `/plugin-creator` hint. Run `/reload` after approved agent edits to refresh definitions.

### Preview a declared improvement budget

```text
/xerxes improve
/xerxes improve --proposals proposals.json --max-candidates 3 --max-task-runs 64
```

The daemon **always forces `--dry-run`**, never imports a transport, and prints the declared candidate/task budget plus the real CLI command. Run that command yourself only after reviewing the plan:

```sh
bun run --cwd xerxes playground:improve -- --transport /absolute/path/to/transport.ts --proposals /absolute/path/to/proposals.json --max-candidates 3 --max-task-runs 64
```

Candidates declare INV (general) or DOM (domain) full-file edits. Baseline, individual candidates and any conditional composition use the same selected hard tasks in private workspaces; budgets count all runs. `--keyword`, `--max-tokens`, `--max-wall-ms` and `--dry-run` further bound the experiment. Transports are explicit host ports; no credentials or provider are discovered. Recommendations never apply themselves to the source harness.

Inside a `Workflow` script, blind `select()` compares anonymous values before optional scoring:

```js
const chosen = await select(
  [{ value: baselineText, model: 'hidden-baseline' }, { value: candidateText, model: 'hidden-candidate' }],
  { judge: 'Prefer correctness, clear evidence and a minimal change.', baseline: 0 }
)
return { value: chosen.value, fallback: chosen.fallback, reason: chosen.reason }
```

Wrapper metadata stays hidden from the judge; letters A–Z are shuffled. An optional `score(value)` runs only after commitment. Invalid judge replies, failed selected scoring, or a score below the baseline cause an explicit baseline fallback. Judge agents count toward Workflow budgets and `expect_agents`.

### Measure long-session recall after compaction

```sh
bun run --cwd xerxes playground:recall -- --dry-run --facts 10 --seed 42 --filler-tokens 384000
bun run --cwd xerxes playground:recall -- --transport /absolute/path/to/transport.ts --facts 10 --seed 42 --filler-tokens 384000
```

Like `playground:hard` and `playground:improve`, the transport explicitly exports `createEvaluationSessionPort(context)` and owns provider configuration; no credentials are discovered. RECALL plants seeded synthetic keys, paths, thresholds, decisions and reasons, adds generated CI log-review turns, then asks exact-answer questions in the **same session**. Answers match only after case/whitespace normalization (no LLM judge). Increase `--filler-tokens` for larger context windows: this character-based estimate does **not** guarantee compaction. The report gives per-fact results, recall rate, wall time and measured pre-recall compactions if the port implements `getCompactionCount()` (a cumulative active-session counter); otherwise it says **unknown**, never inferring compaction from token counts. Dry-run imports no transport. No `session_search` tool is required; available memory/search behavior remains part of the agent being measured. SIGINT cancels execution and closes the private session.

## Models and intelligence

Open `/config`, then choose provider, model and reasoning effort for **light**, **balanced** and **smart**. Ask “List available models and reasoning levels; use a light agent to locate tests and a smart agent to review concurrency.” Choose configured models, not guessed names.

`/provider` and `/reasoning` change the current conversation. The composer displays reasoning effort. Subscription quota is only available when the provider exposes it; unknown usage is not a guaranteed remaining allowance.

## Terminals, agents and runs

Ask “Start a background shell that prints progress once per second for five seconds, then exits.” Open `/terminals` (**F8**) to inspect output and exit status. Use its controls to send input or stop a long-running command.

Ask “Use two agents: one maps source modules, the other locates tests; each should give a short summary.” `/agents` (**F6**) inspects them. `/runs` shows history and unread results. `/workspaces` inspects retained agent workspaces.

## Goals

Ask “Create a goal to audit the parser with three todos: invalid input, cancellation and persistence; verify each with tests.” `/goal` (**F10**) shows progress. New goals have no default token or duration cap. `/goal unlimited` removes caps from an existing goal while preserving progress. `/goal resume` resumes a paused or blocked goal when eligible. Completed todos alone do not establish that the goal is finished.

## Schedules, monitors and loops

- `/schedules`, **N**: prompt “Review the working tree without editing”; recurring cron `0 9 * * *`; choose your timezone. Save paused, review the next-run preview, then **P** to enable. Check results in `/runs`.
- `/monitors`, **N**: choose a file source and an existing absolute build-log path. Fill the shown fields and save. Append a line yourself to verify detection. Terminal sources observe sessions started in Xerxes; network sources need configuration.
- `/loop`: follow the setup for a bounded follow-up such as “Check the build output and report when it finishes.” Review timing and limits before starting.

The owning daemon must remain running. Pause or remove your test schedules and monitors afterward.

## Machines and browser

Open `/machine` and press **N** (or **Enter** when empty) to add a workspace using the form. **Tab** moves between name, SSH host and project folder; **Enter** saves and **Esc** returns without saving. On **SSH host**, press **F2** to choose an alias from `~/.ssh/config` and its `Include` files. On **Project folder**, press **F2** to browse that host (starting at its home directory when blank): **Enter** opens a folder, **Left/Backspace** goes up, and **Space** selects the current folder. **Esc** returns with your draft intact; manual entry also works. Browsing requires an already trusted SSH host and noninteractive authentication (such as your SSH agent). If needed, first connect using `ssh <alias>` in a terminal. Saving does not connect automatically. The equivalent command is:

```text
/machine add compute my-ssh-alias "/home/me/My Project"
/machine
```

Select `compute`, press **Enter** and authenticate through your SSH configuration. Connection automatically installs Bun if missing or too old, and builds the latest published GitHub main revision under ~/.xerxes/remote-runtime. Unchanged builds are reused; older releases and existing installations are preserved. Setup requires Git (and curl, bash, unzip if Bun needs installing). Providers still need remote authentication; local credentials are never copied. Setup errors are logged in ~/.xerxes/remote-runtime/setup.log. The TUI renders locally while an SSH tunnel carries daemon RPC. Exit the remote workspace view to return to your previous local chat. It does not migrate conversations or synchronize files. `/machine remove compute` removes the saved entry.

For browser tools, supply an already-running Chromium-compatible browser endpoint with `/browser connect <endpoint>`. Ask “Read the current page title.” Xerxes attaches to the supplied CDP endpoint; it does not launch a browser.

## Review, recovery and integrations

| Entry point | Try it |
| --- | --- |
| `/diff` or F7 | Make a small local edit; inspect its patch and file entry. |
| `/snapshots` | Browse a saved snapshot and preview a restore before applying it. |
| `/context` | Inspect context and memory for the current conversation. |
| `/config mcp` | Configure a server you control, then inspect its exposed tools using `/tools`. |
| `/config lsp` | Configure your project's language server; check diagnostics on a known syntax error. |
| `/features` | Discover capability entry points. |
| `/help` | Find commands and keyboard shortcuts. |

These examples are walkthroughs, not proof of live external connectivity. The plugin reference test uses a temporary manifest, makes no network calls and leaves real plugin configuration untouched.

### Run a shell command and discuss the result

Type `!ls` (or another `!command`) to execute it in the project directory.
Xerxes displays stdout/stderr and nonzero exit codes, then asks the model to
review the result without rerunning the command. The submitted context retains
the command and output for later turns and session restoration. `{!pwd}` inside
a normal prompt substitutes command output without creating a separate follow-up.
If the model is already working, a new `!command` waits in the queue so its
follow-up does not race the active turn.
