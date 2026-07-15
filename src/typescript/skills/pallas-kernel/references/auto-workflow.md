# Pallas Kernel Autonomous Workflow

Use this workflow when the user asks for an autonomous MaxKernel-style pass: generate or improve a Pallas kernel over multiple iterations until it compiles, passes tests, and has evidence-backed performance.

## 1. Establish the run state

Create a project-local run directory:

```text
pallas-kernel/<kernel_stem>/
```

Inside it, maintain:

- `base_kernel.py` or a pointer to the source file;
- `optimized_kernel.py`;
- `base_kernel_plan.md`;
- `test_optimized_kernel.py`;
- `profile_optimized_kernel.py`;
- `autotune_specs.json`;
- `autotune_results.json`;
- `iterations/round-###/`;
- `history.md`.

Never use `/tmp` for durable artifacts.

Default tolerances:

- `atol=1e-2`
- `rtol=1e-2`

Tighten these when dtype and algorithm allow it.

## 2. Run the phase loop

For each iteration:

1. Plan or revise the plan.
2. Implement the optimized kernel.
3. Validate compilation and JIT execution.
4. Generate and validate tests.
5. Run tests.
6. Autotune a small high-probability search space.
7. Profile and decide whether more improvement is worthwhile.
8. Snapshot artifacts and metrics.

The loop can stop when:

- compilation cannot be fixed after repeated distinct attempts;
- test generation cannot produce a valid pytest file;
- correctness cannot be made to pass without changing semantics;
- profiling says there is no significant room for improvement;
- the user-defined iteration limit is reached;
- the run is blocked by missing TPU/JAX/Pallas dependencies.

## 3. Compilation gate

Compilation is a hard gate. Run validation before test generation. If validation fails:

- preserve the strategy;
- fix only compile/JIT/API issues;
- use raw errors;
- do not repeat failed fixes;
- after repeated failures, add temporary debug statements, validate again, then remove them after success.

Do not proceed to tests with a non-compiling kernel.

## 4. Test gate

Generated tests must pass validation before full execution:

- syntax parse;
- import/compile check;
- pytest collection;
- baseline/mock execution when possible;
- full test execution after validation passes.

If a generated test is invalid, fix the test file only. Do not alter the kernel while repairing test structure.

## 5. Autotune gate

Autotune is split into:

1. planner writes `autotune_specs.json`;
2. runner evaluates candidates;
3. applier changes only tuned constants from the best config;
4. summarizer writes the result.

Apply a best config only if the runner reports success, a valid `best_config`, and a valid `best_time_ms`. Do not list every tested configuration in the user response.

## 6. Profile gate

Profile after tests and autotune. The profiler must decide:

```text
NEEDS_IMPROVEMENT: True|False
```

Base the decision on evidence: timing, XProf if available, memory-transfer ratio, compute ratio, MXU/ALU utilization, HBM/VMEM traffic, pipeline bubbles, and interaction with surrounding XLA code.

## 7. Snapshot and choose best

After each iteration, snapshot:

- optimized kernel;
- test file;
- autotune specs and results;
- compile status;
- test status;
- latency in milliseconds if available;
- profiling summary.

When multiple valid iterations exist, choose the one with the lowest latency. If latency is missing, choose by profiling summary and explain the uncertainty.

## 8. Final response

Return:

- run directory;
- best iteration;
- optimized kernel path;
- test path;
- profile report path;
- autotune report/spec paths;
- compile/test/performance status;
- whether another iteration is recommended.
