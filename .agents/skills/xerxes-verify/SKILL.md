---
name: xerxes-verify
description: Use when selecting and running the right verification ladder, typechecks, test runners, builds, smoke checks, or docs checks after changing any part of the Xerxes-Agents repository.
version: 1.0
author: Xerxes-Agents
tags: [xerxes, verify, testing, typecheck, ci]
---

# Xerxes Verification Ladder

Pick the narrowest checks while iterating, then escalate to the CI-parity gate before handoff. See `.agents/references/projects-map.md` and `.agents/rules/testing-packaging-and-ci.md` for full sub-project and tsconfig boundaries.

## When to use

- After editing source, tests, bundled skills, examples, docs, or CI workflows.
- Before declaring a change verified or handing off cross-cutting work.

## Step 1: Map changed directories to narrow checks

Run from the repository root:

| Changed directory | Narrow check | Narrow test runner | Targeted build / smoke |
| --- | --- | --- | --- |
| `xerxes/src/{core,daemon,streaming,llms,tools,executors,security,session,context,memory,runtime,extensions,channels,cron,audit,bridge,cortex,workspace,images,local,operators}/` | `bun run --cwd xerxes check:runtime` | `bun test xerxes/test/<file>.test.ts` | `bun run build:runtime` |
| `xerxes/src/ui/` | `bun run --cwd xerxes check:ui` | `bunx --bun --no-install vitest run --config xerxes/vitest.ui.config.ts <file>` (or `bun run test:ui`) | `bun run build:ui && bun run smoke` |
| `xerxes/src/desktop/` | `bun run --cwd xerxes check:desktop` (+ `check:runtime` for `main/**`) | `bun test xerxes/test/desktop*.test.*` | `bun run --cwd xerxes build:desktop` |
| `xerxes/src/vscode/` | `bun run --cwd xerxes check:vscode` (`*.ts`) + `check:desktop` (`ui/**/*.tsx`) | `bun test xerxes/test/vscode*.test.*` | `bun run build:vscode` |
| `xerxes/src/{mobile,client,hub,slack-broker}/`, `xerxes/src/auth/usageView.ts` | `bun run --cwd xerxes check:mobile` (+ `check:runtime` for `hub`/`slack-broker`) | `bun run --cwd xerxes test:mobile` + `bun test xerxes/test/hub*.test.ts xerxes/test/slackBroker.test.ts` | `bun run --cwd xerxes build:mobile` |
| `xerxes/src/acp/` | `bun run --cwd xerxes check:runtime && bun run --cwd xerxes check:desktop` | `bun test xerxes/test/acp*.test.ts xerxes/test/agy*.test.ts xerxes/test/cliAcpDelegation.test.ts` + UI ACP Vitest | — |
| `xerxes/src/swarm/` | `bun run --cwd xerxes check` | `bun test xerxes/test/nativeSwarm*.test.* xerxes/test/swarm*.test.*` + UI & Mobile Vitest swarm tests | — |
| `xerxes/skills/`, `xerxes/src/agents/default/` | `bun run --cwd xerxes check:runtime` | `bun test xerxes/test/bundledSkillsLint.test.ts xerxes/test/bundledSkillAssets.test.ts xerxes/test/skills.test.ts xerxes/test/harnessLint.test.ts` | `bun run build:runtime` |
| `examples/` | `bunx tsc --noEmit -p examples/tsconfig.json` | `bun test xerxes/test/rootExamples.test.ts` | — |
| `docs/`, `xerxes/src/docs/` | `bun run --cwd xerxes check:runtime` | — | `bun run docs:build` |
| `.github/workflows/`, `Dockerfile`, `packaging/runtime/` | `bun run repo:check` | `bun test xerxes/test/nativeWorkflowContract.test.ts xerxes/test/containerDistribution.test.ts` | — |
| `@earendil-works/pi-ai` / image catalog | `bun run --cwd xerxes check:image-catalog` | `bun test xerxes/test/imageModels.test.ts` | `bun run --cwd xerxes generate:image-catalog` |

Add contract tests when touching core boundaries:
- Protocol / RPC / events: `bun test xerxes/test/daemonProtocolContract.test.ts xerxes/test/streamingWireEvents.test.ts`
- Tool descriptions / Windows parity: `bun test xerxes/test/toolDescriptionContracts.test.ts xerxes/test/windowsSupport.test.ts`

## Step 2: Escalate to the full repository gate

```bash
bun run check        # repo:check + check:image-catalog + all 5 tsconfigs
bun run test         # test:runtime (bun test) + test:ui (vitest) + test:mobile (vitest)
bun run build        # build:runtime + build:ui + build:mobile
bun run verify       # check && test && build (single CI-parity gate)
bun run smoke        # live v35 daemon RPC + completion smoke (no API key needed)
bun run docs:build   # HTML + TS API docs (required for docs/API edits)
bun run real-use     # optional 40-check offline harness
git diff --check     # whitespace / conflict-marker hygiene
```

## Pitfalls

- **`bun run build` excludes Desktop and VS Code**: Run `bun run --cwd xerxes build:desktop` or `bun run build:vscode` explicitly when changing those surfaces.
- **Test runner split**: Desktop and VS Code tests live in `xerxes/test/` (`bun test`), NOT Vitest. Neither `jsdom` nor `happy-dom` is installed; `.tsx` tests use `react-dom/server` (`renderToStaticMarkup`).
- **Hermetic home**: Both `bunfig.toml` files preload `xerxes/test/support/isolatedHome.ts` (temp `XERXES_HOME`, Claude Code auto-install/autoupdate off).
- **Bun-only**: Never invoke `python`/`pytest`/`ruff`/`mypy` or leave unignored `.py`/`.pyc` files (`repo:check` fails). Never hand-edit `dist/`.
