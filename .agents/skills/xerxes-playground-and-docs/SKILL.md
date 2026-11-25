---
name: xerxes-playground-and-docs
description: Use when running Xerxes evaluation playground suites or real-use checks, authoring or linting bundled skills, building native Bun docs, running maintenance scripts, or validating offline examples.
version: 1.0
author: Xerxes-Agents
tags: [xerxes, playground, evals, skills, docs, examples]
---

# Xerxes Playground Evals, Bundled Skills, Docs & Examples

Run native TypeScript evals, author/lint bundled skills (`xerxes/skills/`), build Bun docs, run maintenance scripts, or validate offline `examples/`. See `.agents/rules/testing-packaging-and-ci.md`.

## When to use

- Running any suite in `xerxes/playground/` or `bun run real-use`.
- Adding, editing, or linting bundled skills in `xerxes/skills/`.
- Building `docs/`, running `xerxes/src/maintenance/` utilities, or updating `examples/`.

## Workflow A: Evaluation Playground (6 Suites) & `real-use`

All suites are Bun-native, never read ambient credentials, and never mutate `process.env.XERXES_HOME`. Every CLI requires `--transport <module>` exporting `createEvaluationSessionPort(context)` -> `EvaluationSessionPort` and isolates runs under `--run-root <dir>` (`run-<pid>-<uuid>/{home,workspace}` + PID lock, deleted in `finally`; copies only `profiles.json` when `--profile-dir` is given).

| Suite | Command | Notes |
| --- | --- | --- |
| **Warmup** | `bun run playground:warmup -- --transport <mod> --run-root <dir>` | 8 TS fixture tasks; `-v`, `-k <filter>`, `--profile-dir <dir>` |
| **Hard** | `bun run playground:hard -- --transport <mod> --run-root <dir>` | 16 difficulty 4–5 tasks across 7 categories with watchdog graders |
| **SWE** | `bun xerxes/playground/sweCli.ts --transport <mod> --run-root <dir> [--json]` | Direct invocation (no package script); 4 TS bugfixes, hidden holdout tests, 5 SWE KPIs |
| **Improve** | `bun run --cwd xerxes playground:improve -- --transport <mod> --proposals <json> --max-candidates <n> --max-task-runs <n> [--max-tokens <n>] [--max-wall-ms <ms>] [--dry-run] [--run-root <dir>]` | Budget-capped A/B harness evaluator; isolated copies, never mutates repo |
| **Recall** | `bun run --cwd xerxes playground:recall -- --transport <mod> [--facts 10] [--filler-tokens 384000] [--seed 42] [--dry-run] [--run-root <dir>]` | Continuous-session synthetic recall under compaction |
| **Matrix** | `bun run --cwd xerxes playground:matrix -- --matrix <json> --out <dir> [--run-root <dir>]` | Local dataset matrix (`dabstep`, `spreadsheetbench`, `gdpval`, `terminal-bench`, `synthetic-private`), `constrained` vs `unrestricted` |

- **`real-use`**: `bun run real-use` (`xerxes/scripts/realUseCheck.ts`) runs 40 local checks in temp dirs with redacted output. Offline by default (pass `--online` for OSV; host probes require explicit injection). Covered by `xerxes/test/realUseCheck.test.ts`.

## Workflow B: Authoring & Linting Bundled Skills (`xerxes/skills/`)

- **Frontmatter (`skillLint.ts`)**: Start at byte 0 (`---`...`---`), use ONLY `SKILL_FRONTMATTER_KEYS` (`name`, `description`, `version`, `author`, `platforms`, `tags`, `source`, `resources`, `dependencies`, `required_tools`, `config_vars`, `trust_level`, `setup_command`, `subcommands`), keep scalar values single-line (no `|`/`>` block scalars, nested mappings, multiline `[...]`, or duplicate keys), match `name:` to the directory basename, and avoid `COMMAND_REGISTRY` or existing skill collisions.
- **Assets & security (`copyBundledSkills.ts`, `harnessLint.ts`)**: Zero `.py`/`.pyc` files, zero symlinks, zero executable (`0o111`) files, zero unshipped relative script invocations, zero `promptScanner.ts` patterns or credential shapes, and none of the blocked non-redistributable dirs (`powerpoint/*` except `SKILL.md`, `creative-ideation`, `research/research-paper-writing/templates`, `training/{axolotl,pytorch-fsdp,unsloth}`). Keep descriptions concise (`MAX_SKILL_INDEX_ENTRIES = 128`, `MAX_SKILL_INDEX_BYTES = 16 KiB`; 119 bundled skills ship).

## Workflow C: Native Bun Docs, Maintenance & Offline `examples/`

- **Docs**: `bun run docs:build` (`--dry-run` supported) reads `docs/bun-docs.json`, renders `.md` only, extracts TS API via AST without executing code, and writes `docs/_bun` with `.xerxes-bun-docs.json` ownership manifest (rejects unowned non-empty dirs and retired `legacy` key). Standalone Markdown API docs: `bun run --cwd xerxes docs:api` (`--dry-run`, `--no-clean`, `--no-atomic`; rejects `--format`).
- **Maintenance**: `bun run --cwd xerxes fix-license-headers` (`#` headers in `.sh`/`.yml`/`.yaml`/`Dockerfile`, NOT `.ts`/`.tsx`), `bun run --cwd xerxes generate-jsdoc` (idempotent AST JSDoc inserter, `--dry-run`), `bun run --cwd xerxes generate:image-catalog` (syncs `imageModels.generated.json` from `@earendil-works/pi-ai`).
- **Examples**: Every demo in `examples/` MUST default to deterministic offline ports (zero ambient env/network); live mode requires explicit `--live --model --base-url --api-key`, file output requires `--write`.

## Verification commands

```bash
bun test xerxes/test/bundledSkillsLint.test.ts xerxes/test/bundledSkillAssets.test.ts xerxes/test/skills.test.ts xerxes/test/harnessLint.test.ts
bun run docs:build && bun test xerxes/test/realUseCheck.test.ts
bunx tsc --noEmit -p examples/tsconfig.json && bun test xerxes/test/rootExamples.test.ts xerxes/test/pluginCreator.test.ts
```
