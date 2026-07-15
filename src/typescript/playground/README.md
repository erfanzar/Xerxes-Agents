# Native Bun evaluation playground

This directory is the TypeScript/Bun replacement for the Python playground harness.
It provides the same evaluation building blocks without importing Python, changing
`XERXES_HOME`, spawning a Python interpreter, or discovering provider credentials.

## Host-owned execution

`NativeEvaluationAgent` accepts an `EvaluationSessionPort`. The embedding host is
therefore responsible for choosing and starting a real runtime/provider, then may
pass the private `homeDirectory` and `workspaceDirectory` created by
`createEvaluationIsolation()` into that port. The harness never creates a provider
client or reads profile/environment credentials on its own.

Failure diagnosis is likewise opt-in through `EvaluationJudgePort`. Without one,
the scorecard reports the grader detail and explicitly says no judge was injected.

## Run the normal warm-up CLI

The standard eight-task suite has a standalone Bun entry point:

```bash
bun run playground:warmup -- \
  --transport /absolute/path/to/evaluation-transport.ts \
  --profile-dir ~/.xerxes \
  --verbose
```

It accepts the original `-v` / `--verbose` and `-k` / `--keyword` switches,
plus `--run-root` for caller-selected private run storage. The CLI creates one
private home and workspace per invocation, copies only the explicit
`profiles.json` source when `--profile-dir` is supplied, and removes that run
after reporting its score.

The transport module must export this factory:

```ts
import {
  DaemonEvaluationSessionPort,
  type EvaluationTransportContext,
} from '/path/to/Xerxes-Agents/src/typescript/playground/index.js'

export function createEvaluationSessionPort(context: EvaluationTransportContext) {
  const runtime = createPrivateNativeRuntime({
    homeDirectory: context.homeDirectory,
    workspaceDirectory: context.workspaceDirectory,
  })
  return new DaemonEvaluationSessionPort(runtime)
}
```

`createPrivateNativeRuntime` belongs to the embedding host because provider
selection and credentials remain explicit. `DaemonEvaluationSessionPort`
adapts the real Bun daemon event vocabulary (text, tools, status, terminal
events) and creates a fresh native session for each `freshSession` task step.
No Python process or shell command is involved.

## Suites

- `runWarmupSuite()` rebuilds a TypeScript fixture workspace for the eight warm-up
  checks. The coding checks invoke exported functions through a caller-selected
  `WorkspaceModuleEvaluator`; `BunWorkspaceModuleEvaluator` is supplied for direct,
  shell-free Bun child-process execution.
- `runHardSuite()` preserves per-task workspaces, `{dir}` substitution, fresh-session
  steps, failure diagnosis, watchdog-bounded checkers, and category/difficulty
  scorecards. `createNativeHardTasks()` supplies the complete built-in typed
  TypeScript/Bun catalog, and `bun run --cwd src/typescript playground:hard -- --transport <module>`
  runs it through an explicit host transport.
