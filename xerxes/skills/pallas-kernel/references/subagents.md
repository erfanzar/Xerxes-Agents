# Pallas Kernel Subagents

This file maps the MaxKernel `auto_agent/subagents` design into Xerxes-native delegation contracts.

## AutonomousPipelineAgent

Purpose: coordinate a multi-iteration kernel improvement loop.

Delegates in this order:

1. `PlanKernelAgent`
2. `ImplementKernelAgent`
3. `ValidateKernelCompilationAgent`
4. `ValidatedTestGenerationAgent`
5. `UnifiedTestAgent`
6. `AutotuneAgent`
7. `ProfileAgentOrchestrator`
8. snapshot and best-iteration selection

State to maintain:

- source/base kernel path;
- optimized kernel path;
- plan path;
- test path;
- profiling script path;
- autotune specs path;
- autotune results path;
- tolerances;
- iteration history.

## PlanKernelAgent

Creates or revises an optimization plan.

Required behavior:

- save pasted/reference code to the base kernel path;
- read the current plan when revising;
- incorporate compilation/test/autotune/profile results;
- consult Pallas/TPU docs before proposing API-sensitive changes;
- write the plan to the exact plan path.

## ImplementKernelAgent

Writes the optimized Pallas kernel from the plan.

Required behavior:

- follow the plan exactly;
- preserve unrelated setup code;
- create module-level `kernel`, `computation`, and `main`;
- include `debug=True`;
- add shape, memory-space, and transfer comments;
- write the full optimized file and stop.

## ValidateKernelCompilationAgent

Ensures a kernel file exists, then runs a compilation-fix loop.

Subroles:

- file finder;
- compilation checker;
- compile fixer;
- debug statement inserter;
- debug cleanup;
- validation summarizer.

Rules:

- validate before tests;
- loop through distinct fixes, with a default ceiling of six attempts;
- add debug statements only after repeated failures;
- cleanup debug statements after successful compilation;
- record `FIX_SUMMARY` for every fix attempt.

## ValidatedTestGenerationAgent

Generates a pytest file and validates the test harness before full execution.

Validation checks:

- syntax parse;
- import/compile check;
- pytest collection;
- setup validation;
- baseline/mock execution when possible.

Rules:

- require both base and optimized kernels;
- do not randomly search for kernel files;
- fix only the test file when validation fails;
- respect `atol` and `rtol`.

## UnifiedTestAgent

Runs the generated pytest file with full tracebacks and captures:

- exit code;
- stdout/stderr;
- pass/fail status;
- performance metric when emitted.

If no test file exists, stop and report the missing artifact.

## AutotuneAgent

Subroles:

- planner creates `autotune_specs.json`;
- runner evaluates candidates;
- applier patches only best-config constants;
- summarizer reports status.

Rules:

- candidate grids must be small;
- use values likely to divide real shapes;
- print `CORRECTNESS` and `RESULT_TIME`;
- apply only successful results with `best_config` and `best_time_ms`;
- do not dump all candidate results into the user response.

## ProfileAgentOrchestrator

Subroles:

- profiling script generator;
- profiling script reader;
- profile executor;
- profile summarizer.

Rules:

- generate a script with JIT, warmups, three measured profile executions, and XProf trace collection when available;
- analyze DMA/memory transfer ratio and compute ratio;
- inspect XPlane/HLO when trace paths are available;
- finish with `NEEDS_IMPROVEMENT: True|False`.

## ExplanationAgent

Answers Pallas/JAX/TPU questions.

Rules:

- retrieve docs before answering conceptual/API questions;
- read referenced files first;
- use parallel retrieval for predictable subtopics and sequential retrieval for gaps;
- return concise grounded explanations with examples and caveats.

## Xerxes delegation rules

When spawning these roles in Xerxes:

- use names matching the role, e.g. `pallas-plan`, `pallas-validate`, `pallas-profile`;
- assign one phase per agent;
- require final artifact paths and summaries only;
- forbid long transcripts and raw tool logs;
- store large reports in project memory or project-local files;
- keep the main agent responsible for verifying claims before changing source.
