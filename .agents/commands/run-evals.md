---
name: run-evals
description: Dry-run or execute a Xerxes evaluation suite (real-use, warmup, hard, swe, recall, improve, or matrix) with explicit --transport and isolated --run-root.
tags: [evals, playground, benchmarks, testing]
---

# Run Xerxes Evaluation Suites

Parse and execute (or dry-run) the evaluation suite requested in: `$ARGUMENTS`.

## 1. Strict Isolation & Transport Rules (`xerxes/playground/`)

- **Zero Python**: Every evaluation harness is Bun-native TypeScript. Never invoke Python tools.
- **No ambient credentials or home mutation**: Playground CLIs never discover ambient provider keys and never mutate `process.env.XERXES_HOME`.
- **Explicit transport module**: Every playground suite (`warmup`, `hard`, `swe`, `recall`, `improve`) requires `--transport <module>` exporting `createEvaluationSessionPort(context)` returning an `EvaluationSessionPort` (except `real-use` and `--dry-run` modes).
- **Explicit isolated `--run-root`**: Always pass a dedicated temp directory such as `/tmp/xerxes-eval-<suite>`. `xerxes/playground/isolation.ts` allocates `run-<pid>-<uuid>/{home,workspace}` with `.evaluation-owner.json`, copies only `profiles.json` when `--profile-dir` is supplied, and removes the run directory in `finally`.

## 2. Suite Command Matrix

| Suite | Description | Exact Command |
| --- | --- | --- |
| `real-use` | 40 local end-to-end checks across plans 01-32 + TUI/LIVE/HARDWARE; offline by default (pass `--online` only if OSV network check requested) | `bun run real-use` (contract test: `bun test xerxes/test/realUseCheck.test.ts`) |
| `warmup` | 8 TS fixture tasks (`reasoning`, `file_read`, `file_edit`, `bug_fix`, `tool_search`, `shell`, `multiturn`, `memory_recall`) | `bun run playground:warmup -- --transport <module> --run-root /tmp/xerxes-eval-warmup [-v] [-k <filter>] [--profile-dir <dir>]` |
| `hard` | 16 difficulty 4-5 tasks across 7 categories (`coding`, `memory`, `tools`, `recovery`, `reason`, `instruction`, `robustness`) | `bun run playground:hard -- --transport <module> --run-root /tmp/xerxes-eval-hard [-v] [-k <filter>] [--profile-dir <dir>]` |
| `swe` | 4 multi-file TS bugfixes with hidden holdout tests and 5 Core SWE KPIs (no npm script entry — invoke directly) | `bun xerxes/playground/sweCli.ts --transport <module> --run-root /tmp/xerxes-eval-swe [--json] [--profile-dir <dir>]` |
| `recall` | Single continuous session (no resets) testing fact recall across filler tokens (`--facts` default 10, max 1000; `--filler-tokens` default 384000, max 2M; `--seed` 42) | `bun run --cwd xerxes playground:recall -- --transport <module> --run-root /tmp/xerxes-eval-recall [--facts 10] [--filler-tokens 384000] [--seed 42] [--dry-run]` |
| `improve` | Budget-capped A/B harness evaluator (`agent_yaml`, `skill`, `prompt_addendum`; never mutates repo) | `bun run --cwd xerxes playground:improve -- --proposals <json> --transport <module> --run-root /tmp/xerxes-eval-improve --max-candidates <n> --max-task-runs <n> [--max-tokens <n>] [--max-wall-ms <ms>] [--dry-run]` |
| `matrix` | Local dataset comparison (`dabstep`, `spreadsheetbench`, `gdpval`, `terminal-bench`, `synthetic-private`) split into `constrained` vs `unrestricted` privacy sections | `bun run --cwd xerxes playground:matrix -- --matrix <json> --out /tmp/xerxes-eval-matrix-out --run-root /tmp/xerxes-eval-matrix` |

## 3. Preflight Validation

1. Parse suite name (`real-use|warmup|hard|swe|recall|improve|matrix`) and any extra flags from `$ARGUMENTS`.
2. If `$ARGUMENTS` is empty or unrecognized, ask which suite to run and list the 7 options above.
3. For `warmup`, `hard`, `swe`, `recall`, or `improve`:
   - If `--transport <module>` is omitted and `--dry-run` is not requested (note only `recall` and `improve` support `--dry-run`), stop and ask the user for the `--transport` module path rather than inventing credentials or hitting ambient providers.
   - Verify the `--transport` file path (and `--proposals` / `--matrix` JSON path when applicable) exists before running.
4. For `real-use`, keep execution offline unless the user explicitly passed `--online`.

## 4. Reporting Results

Report:
- Exact suite, flags, transport module (or `offline`/`dry-run`), and `--run-root` used.
- Pass/fail/skip totals and per-task failure taxonomy (`timeout`, `cancelled`, `grader_exception`, `transport_error`, `assertion_failed`).
- For `swe`, include all 5 Core SWE KPIs: `resolvedRate`, `localizationEfficiency`, `editMismatchRate`, `holdoutRegressionRate`, and `costAndLatencyPerResolved`.
- For `recall`, include recall accuracy and `compactionCount` if reported.
- Confirm whether the temporary run directory under `--run-root` was cleaned up cleanly.
