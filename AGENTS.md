# AGENTS.md

Xerxes (`@xsimurgh/xerxes-agents` v0.8.60) is a Bun-native (Bun 1.3.12+) strict TypeScript monorepo built around one global v35 JSON-RPC daemon serving OpenTUI, Electron Desktop, VS Code, Mobile PWA, Hub/Supervisor, Channels, and external ACP bridges.

## Universal Commandments

1. **Strict Bun-only**: Zero Python runtime, `.py`/`.pyc` files, Python tests, Python packaging metadata, or Python subprocess fallbacks (`bun run repo:check` enforces this). Never run `python`/`pytest`/`ruff`/`mypy` or Node wrappers when Bun runs directly.
2. **Strict TypeScript**: Keep `exactOptionalPropertyTypes: true` and `noUncheckedIndexedAccess: true`. Never widen `key?: T` to `key?: T | undefined`; use `...(x === undefined ? {} : { x })`. Use `.js` local ESM extensions and `import type`.
3. **Fail-closed security & git**: Never commit credentials, fake external success, or stage, commit, push, open PRs, alter external accounts, or publish unless explicitly asked.
4. **Verify before handoff**: Run the narrowest test while iterating, then `bun run verify` and `git diff --check` (plus `bun run smoke` / `bun run docs:build` when relevant), reporting only completed checks.
5. **Repo harness**: Consult `.agents/README.md`, `.agents/rules/`, `.agents/skills/`, `.agents/commands/`, `.agents/agents/`, and `.agents/references/projects-map.md`.

## Sub-Projects Matrix

| Sub-Project | Primary Path | Tsconfig | Test Command | Build Command |
| --- | --- | --- | --- | --- |
| **Core Runtime & Daemon** | `xerxes/src/{core,daemon,streaming,llms,executors,tools,security,session,context,memory,runtime,extensions,acp,swarm,agents,...}` | `xerxes/tsconfig.json` | `bun run test:runtime` (narrow: `bun test xerxes/test/<f>.test.ts`) | `bun run build:runtime` |
| **OpenTUI Terminal UI** | `xerxes/src/ui/` | `xerxes/tsconfig.ui.json` | `bun run test:ui` (`vitest.ui.config.ts`) | `bun run build:ui` |
| **Electron Desktop** | `xerxes/src/desktop/` | `xerxes/tsconfig.desktop.json` (`main/` also in `tsconfig.json`) | `bun test xerxes/test/desktop*.test.*` | `bun run --cwd xerxes build:desktop` |
| **VS Code Extension** | `xerxes/src/vscode/` | `xerxes/tsconfig.vscode.json` (host), `tsconfig.desktop.json` (`ui/`) | `bun test xerxes/test/vscode*.test.*` | `bun run build:vscode` |
| **Mobile PWA & Shared Client** | `xerxes/src/mobile/`, `xerxes/src/client/` | `xerxes/tsconfig.mobile.json` (`types: []`) | `bun run --cwd xerxes test:mobile` (`vitest.mobile.config.ts`) | `bun run --cwd xerxes build:mobile` |
| **Hub, Supervisor & Slack Broker** | `xerxes/src/hub/`, `xerxes/src/slack-broker/` | `xerxes/tsconfig.json` (`hub/protocol.ts` also in `tsconfig.mobile.json`) | `bun test xerxes/test/hub*.test.ts xerxes/test/slackBroker.test.ts` | `bun run build:runtime` + `bun run --cwd xerxes build:mobile` |
| **Examples** | `examples/` | `examples/tsconfig.json` | `bun test xerxes/test/rootExamples.test.ts` | `bunx tsc --noEmit -p examples/tsconfig.json` |
| **Packaging, Docker & Releases** | `packaging/runtime/`, `Dockerfile`, `scripts/`, `xerxes/scripts/` | `xerxes/tsconfig.json` | `bun test xerxes/test/{containerDistribution,nativeWorkflowContract,installerParity}.test.ts` | `bun run release:prepare` / `release:check` / `release:smoke` |
| **Evaluation Playground** | `xerxes/playground/`, `xerxes/scripts/realUseCheck.ts` | `xerxes/tsconfig.json` | `bun test xerxes/test/realUseCheck.test.ts` | `bun run real-use` / `playground:{warmup,hard}` |
| **Bundled Skills & Docs** | `xerxes/skills/` (119), `docs/` | `xerxes/tsconfig.json` | `bun test xerxes/test/bundledSkill*.test.ts` | `bun run docs:build` / `copyBundledSkills.ts` |

## Essential Commands

Run from the repository root unless noted otherwise:

```bash
# Install locked workspace dependencies (Bun 1.3.12+)
bun install --frozen-lockfile

# Run Xerxes locally
bun run xerxes
bun run xerxes "explain this function"
bun run xerxes --resume <session_id>
bun run xerxes daemon --project-dir .
bun run xerxes acp --project-dir .
bun run xerxes hub status

# Narrow typechecks & tests (match the owning runner!)
bun run --cwd xerxes check:runtime          # also check:ui | check:desktop | check:vscode | check:mobile
bun test xerxes/test/<file>.test.ts         # core, daemon, acp, swarm, hub, desktop*, vscode*
bunx --bun --no-install vitest run --config xerxes/vitest.ui.config.ts xerxes/src/ui/<path>.test.ts
bunx --bun --no-install vitest run --config xerxes/vitest.mobile.config.ts xerxes/src/mobile/__tests__/<file>.test.ts

# Root quality gates (Note: `bun run build` builds runtime + ui + mobile, NOT desktop/vscode)
bun run check                               # repo:check + check:image-catalog + all 5 xerxes tsconfigs
bun run test                                # test:runtime + test:ui + test:mobile
bun run build                               # build:runtime + build:ui + build:mobile
bun run verify                              # check && test && build
bun run smoke                               # live Bun v35 daemon TUI gateway + completion smoke
bun run docs:build                          # deterministic Markdown + AST-only TS API docs
git diff --check

# Desktop, VS Code & Release packaging
bun run --cwd xerxes build:desktop          # build:runtime -> buildDesktop.ts -> packageDesktopMac.ts
bun run build:vscode                        # build:runtime -> buildDesktop.ts -> buildVscode.ts (.vsix)
bun run release:prepare -- --output <dir>   # then bun pm pack -> release:check -- --package <dir> --archive <tgz> -> release:smoke -- <tgz>
bun run --cwd xerxes fix-license-headers    # normalizes `#` headers in .sh/.yml/.yaml/Dockerfile
```

## How a Turn Works & Non-Negotiable Invariants

1. `xerxes/src/cli.ts` dispatches interactive OpenTUI, one-shot, `daemon`, `acp`, `hub`, `immortals`/`swarm`, or explicit subcommands.
2. Surfaces speak NDJSON JSON-RPC 2.0 to the single global Bun daemon (`DAEMON_PROTOCOL_VERSION = 35`) over per-user Unix socket / Windows named pipe or localhost WS (`127.0.0.1:11996`), except Mobile PWA which speaks Hub v1 (`src/hub/protocol.ts`).
3. `daemon/server.ts` -> `daemon/runtime.ts` -> `WorkspaceTurnRunner` -> `daemon/turnRunner.ts` assembles cache-stable session layers (`SessionPromptSnapshots` + `assembleContextLayers`) and volatile per-turn context (`assembleTurnContext` -> `<turn-context>`), gates outbound LLM traffic (`outboundGate.ts`), and invokes `streaming/loop.ts: runTurn`.
4. `runTurn` streams provider deltas, runs fail-closed policy/data-boundary/hook/permission checks, batches consecutive `concurrencySafe` tools (max 8), normalizes outputs (`normalizeToolOutput` after `tool_result_persist`), and maps stream events to frozen v35 wire events via `daemonEventFromStream`.
5. `DaemonTranscriptStore` persists transcripts with generation CAS + crash `.jsonl` journal; post-persistence `afterTurn` runs reflection, auto-titling, and goal continuation without blocking `turn_end`.

Key non-negotiable invariants (full rules in `.agents/rules/`):

- **v35 `PROTOCOL.md` & Prompt-Cache Separation**: Every daemon RPC or wire event change MUST update `xerxes/src/ui/PROTOCOL.md` in the same patch (`daemonProtocolContract.test.ts`). Never place session-mutating RPCs in `QUEUE_BYPASS_METHODS` or `CONCURRENT_DISPATCH_METHODS`. Never put per-turn mutable state (memory diffs, live goal status, instruction diffs, approved lessons, context deltas) into `assembleContextLayers()`; always use `assembleTurnContext()` (`<turn-context>`).
- **Default Agent Tool Allow-List**: Registering a tool in `ToolRegistry` does NOT expose it to the default agent until it is added to `xerxes/src/agents/default/agent.yaml` `tools:` and audited across the 7 security/tracking sets (`permissions.ts`, `dataBoundary.ts`, `runSafety.ts`, `sweWorkingSet.ts`, `leanPrompt.ts`, `toolRegistry.ts`, `subagentManager.ts`).
- **Antigravity Thin-Client & External ACP Isolation**: Forward only verbatim human-authored turns (`userAuthored === true`) to Antigravity; inject zero Xerxes instructions, skills, commands, goals, or forged actions; keep `agyBrainBridge.ts` strictly read-only, omit `fs.writeTextFile`, never auto-approve `interaction_*`, strip outer/Xerxes env in `agyRuntimeEnv.ts`, never read/export AGY OAuth tokens in `agyCredentials.ts`, and preserve Claude Code model-only flags + Codex single-flight refresh.
- **Immortals Swarm Surface & Worktree Safety**: OpenTUI, Desktop, VS Code, and Mobile MUST NEVER import `src/swarm/` directly (v35 RPC-only). Killed worker names are permanent; `share` and `branch` workspaces are permanently deleted on `kill`; all git calls use `-c core.hooksPath=/dev/null` and strip `GIT_*`.
- **Browser Purity (`types: []`) & CSS/Token Cascade**: `src/mobile/`, `src/client/`, `src/hub/protocol.ts`, and `src/auth/usageView.ts` compile under `tsconfig.mobile.json` (`types: []`) and MUST contain zero Node or Bun globals. Never hand-edit generated `theme.css` (edit `src/ui/theme.ts` + `src/desktop/tokens.ts`); preserve stylesheet order (`theme -> xterm -> app -> atelier -> layout -> vibes -> neo -> vscode`) and WCAG AA `>= 4.5:1` contrast.
- **Fail-Closed Security, Sandbox & Offline-Default Tests**: `ToolCapabilities`, `tool_permission_check`, `DataBoundary` (project `.xerxes/data-policy.yaml` can only tighten), `PolicyRouter`, `CommandSandbox`, and `WorkspacePathResolver` (`resolve` + `recheck`) all fail closed. Native browser automation (`BrowserManager`) attaches only to an explicitly supplied CDP endpoint. All unit tests and `examples/` default to deterministic offline injected ports (credentials external and opt-in).

## Code & Git Conventions

- **License header**: Every `.ts`/`.tsx` source file starts with:
  ```ts
  // Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
  // Licensed under the Apache License, Version 2.0.
  ```
- **TypeScript style**: Strict types, discriminated unions over boolean bags, `unknown` + narrowing over `any`, top-level functions over thin classes, top-of-file imports (lazy only for startup/cycle isolation), `.js` local ESM extensions, `import type`, and `...(x === undefined ? {} : { x })` for optional properties.
- **Bun & boundaries**: Prefer `Bun.file`, `Bun.write`, `Bun.spawn`, `Bun.serve`, `Bun.sqlite`, `Bun.YAML`, `Bun.Glob`, and `Bun.CryptoHasher`; validate all external inputs at boundaries; call `timer.unref?.()` on background timers; never swallow errors.
- **Windows lockstep**: Guard named-pipe paths with `!isNamedPipePath()`, route `.cmd`/`.bat` spawns via `planSpawn()` with `windowsVerbatimArguments: true`, and keep `core/hostPlatform.ts` aligned with `ui/lib/hostPlatform.ts` (`windowsSupport.test.ts`).
- **Zero-DOM React tests & artifacts**: Neither `jsdom` nor `happy-dom` is installed; React `.test.tsx` suites use `react-dom/server` (`renderToStaticMarkup`), pure reducers, and fakes. Never manually edit generated `dist/` files.
- **Git & commits**: Use Conventional Commits (`type(scope): concise description`). Do not stage, commit, push, open a PR, alter external accounts, or publish artifacts unless explicitly requested.

## `.agents/` Harness Map

Start at `.agents/README.md` for how the repository harness is organized, and consult `.agents/references/projects-map.md` for deep tables (6 tsconfigs, subsystem map, build dependency graph, 9 version-bearing files + 4 tests, and prompt/memory budgets).

| Layer | Paths |
| --- | --- |
| **Rules** (always-on invariants) | `.agents/rules/core-runtime-daemon.md`, `.agents/rules/client-surfaces-and-hub.md`, `.agents/rules/acp-swarms-and-agents.md`, `.agents/rules/testing-packaging-and-ci.md` |
| **Skills** (procedural checklists) | `.agents/skills/{xerxes-verify,xerxes-daemon-protocol,xerxes-tools-and-providers,xerxes-surfaces-dev,xerxes-acp-and-swarms,xerxes-release-and-packaging,xerxes-playground-and-docs}/SKILL.md` |
| **Commands** (invocable routines) | `.agents/commands/{verify,add-rpc-or-slash,add-tool,audit-acp-isolation,audit-surface-parity,debug-turn,release-prep,run-evals}.md` |
| **Specialist Agents** (read-only auditors) | `.agents/agents/{protocol-auditor,acp-isolation-auditor,surface-reviewer,release-auditor}.md` |
| **References & Index** | `.agents/README.md`, `.agents/references/projects-map.md` |
