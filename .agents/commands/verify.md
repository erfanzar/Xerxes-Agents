---
name: verify
description: Run targeted or full Bun verification gates for Xerxes-Agents (core, ui, desktop, vscode, mobile, hub, acp, swarm, docs, examples, or all).
tags: [verification, testing, ci, bun]
---

# Verify Xerxes-Agents

Run the appropriate Bun-only verification gate from the repository root for the requested scope: `$ARGUMENTS`.

## 1. Parse Target Scope

Interpret `$ARGUMENTS` (defaulting to `all` when empty). Multiple space-separated scopes are allowed; deduplicate commands when combining scopes. Never use Python tooling (`python`, `python3`, `uv`, `pytest`, `ruff`, `mypy`) to validate this repository.

| Scope | Commands to Execute (from repo root) |
| --- | --- |
| `core` | `bun run repo:check && bun run --cwd xerxes check:image-catalog && bun run --cwd xerxes check:runtime && bun test xerxes/test/daemonProtocolContract.test.ts xerxes/test/windowsSupport.test.ts xerxes/test/toolDescriptionContracts.test.ts xerxes/test/streamingLoopParity.test.ts` (plus targeted tests for any edited core subsystem under `xerxes/test/`) |
| `ui` | `bun run --cwd xerxes check:ui && bun run test:ui && bun run build:ui && bun run smoke` |
| `desktop` | `bun run --cwd xerxes check:desktop && bun test xerxes/test/desktop*.test.*` (add `bun run --cwd xerxes build:desktop` when build, preload, renderer assets, tokens, or packaging changed) |
| `vscode` | `bun run --cwd xerxes check:vscode && bun run --cwd xerxes check:desktop && bun test xerxes/test/vscode*.test.* xerxes/test/swarmClientSurfacesDesktopVscode.test.ts` (add `bun run build:vscode` when manifest, sheets, assets, or bundling changed) |
| `mobile` | `bun run --cwd xerxes check:mobile && bun run --cwd xerxes test:mobile && bun run --cwd xerxes build:mobile` |
| `hub` | `bun run --cwd xerxes check:runtime && bun run --cwd xerxes check:mobile && bun test xerxes/test/hub*.test.ts xerxes/test/slackBroker.test.ts` |
| `acp` | `bun run --cwd xerxes check:runtime && bun run --cwd xerxes check:desktop && bun test xerxes/test/acp*.test.ts xerxes/test/agy*.test.ts xerxes/test/cliAcpDelegation.test.ts xerxes/test/claudeCode*.test.ts xerxes/test/codex*.test.ts xerxes/test/desktopAcp*.test.tsx xerxes/test/desktopAgyParity.test.tsx xerxes/test/desktopClaudeLanguageAndAcpDiscovery.test.tsx && bunx --bun --no-install vitest run --config xerxes/vitest.ui.config.ts xerxes/src/ui/__tests__/tuiAcpCompatibility.test.tsx` |
| `swarm` | `bun run check && bun test xerxes/test/nativeSwarm*.test.* xerxes/test/swarmClientSurfacesDesktopVscode.test.ts xerxes/test/swarmGraph.test.tsx xerxes/test/desktopSwarmInspectorFormat.test.tsx xerxes/test/swarmDaemonMemory.test.ts xerxes/test/swarmHubSurfaces.test.ts && bunx --bun --no-install vitest run --config xerxes/vitest.ui.config.ts xerxes/src/ui/__tests__/nativeSwarmUi.test.ts xerxes/src/ui/__tests__/swarmClientSurfacesUi.test.ts && bunx --bun --no-install vitest run --config xerxes/vitest.mobile.config.ts xerxes/src/mobile/__tests__/swarmClientSurfacesMobile.test.ts` |
| `docs` | `bun run docs:build && bun test xerxes/test/bundledSkillsLint.test.ts xerxes/test/bundledSkillAssets.test.ts` |
| `examples` | `bunx tsc --noEmit -p examples/tsconfig.json && bun test xerxes/test/rootExamples.test.ts` |
| `all` (default) | `bun run verify && bun run smoke && bun run docs:build && git diff --check` |

## 2. Execution Rules

1. Always run `git diff --check` at the end of any verification run to catch whitespace or conflict-marker regressions.
2. Remember the build dependency order:
   - `bun run build` covers `build:runtime`, `build:ui`, and `build:mobile` (it does **not** run `build:desktop` or `build:vscode`).
   - `buildDesktop.ts` requires a fresh `xerxes/dist/build-id` from `build:runtime`.
   - `buildVscode.ts` requires both `build:runtime` and `buildDesktop.ts` (use `bun run build:vscode`).
3. Do not manually edit generated output under `xerxes/dist/` or `docs/_bun/`; regenerate via the owning Bun build command.
4. For optional desktop visual or live-workspace checks, note `bun xerxes/scripts/previewDesktop.ts` and `bun xerxes/scripts/verifyDesktopWorkspace.ts` (do not run headless GUI flows unless requested).
5. Note test runner boundaries:
   - `bun test ./test` (`test:runtime`) runs core, daemon, hub, slack-broker, desktop, and VS Code tests.
   - `vitest.ui.config.ts` (`test:ui`) runs `xerxes/src/ui/**/*.test.{ts,tsx}`.
   - `vitest.mobile.config.ts` (`test:mobile`) runs `xerxes/src/mobile/__tests__/**/*.test.{ts,tsx}`.

## 3. Report Format

Summarize only commands actually executed in this worktree:
- **Scope**: requested scope(s) and exact commands run.
- **Results table**: step, exit code, pass/fail counts (or build artifacts verified).
- **Failures**: exact file paths, test names, and root cause snippets for any failing check.
- **Not run**: explicitly list any out-of-scope checks (for example `build:desktop`, `build:vscode`, or `bun run real-use`).
