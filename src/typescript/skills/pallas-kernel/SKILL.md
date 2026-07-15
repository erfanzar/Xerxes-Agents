---
name: pallas-kernel
description: Plan, implement, validate, test, profile, and autotune JAX/Pallas TPU kernels.
version: 1.0.0
tags: [jax, pallas, tpu, kernels, performance, maxkernel]
required_tools:
  - ReadFile
  - WriteFile
  - FileEditTool
  - GrepTool
  - GlobTool
  - exec_command
  - write_stdin
subcommands:
  - auto
  - plan
  - implement
  - validate
  - test
  - profile
  - autotune
  - explain
  - gpu-to-jax
---

# Pallas Kernel

This skill teaches the agent to work like a TPU/Pallas kernel engineer: plan first, implement only from an approved plan, validate with raw compilation errors, write tests, profile bottlenecks, then autotune a small high-value search space.

The workflow is adapted from the MaxKernel agent design in `AI-Hypercomputer/accelerator-agents`, but expressed for Xerxes tools and project memory. Do not call external MaxKernel code unless the user explicitly asks for it.

# When to use

Use this skill when the user asks to:

- write, optimize, debug, benchmark, profile, or autotune a JAX/Pallas kernel;
- convert CUDA, Triton, PyTorch CUDA, or accelerator-oriented code into JAX/Pallas;
- build a TPU kernel plan before touching implementation;
- create correctness or performance tests for a kernel;
- improve TPU utilization, MXU usage, HBM/VMEM data movement, tiling, or block specs.

Do not use this skill for ordinary Python refactors, frontend work, or non-accelerator code unless the task explicitly involves JAX, Pallas, TPU, CUDA, Triton, GPU kernels, or benchmark kernels.

# How to use

## Operating law

Work in phases and stop at the natural handoff point. Do not run the whole pipeline unless the user asked for a full autonomous pass.

1. **Plan**: inspect the source kernel, create a markdown optimization plan next to it, and stop for review.
2. **Implement**: read the approved plan and source file, write an optimized kernel file next to the source, and stop.
3. **Validate**: run compilation/JIT validation and fix only compilation errors, preserving the plan.
4. **Test**: generate and run pytest correctness/performance tests against a reference implementation.
5. **Profile**: collect timing/profile evidence and explain bottlenecks.
6. **Autotune**: create a bounded parameter search, run it, apply the best config only when evidence supports it.

If a subcommand was invoked, follow the matching reference workflow exactly:

- `pallas-kernel:auto`: `references/auto-workflow.md`
- `pallas-kernel:plan`: `references/plan-workflow.md`
- `pallas-kernel:implement`: `references/implement-workflow.md`
- `pallas-kernel:validate`: `references/validate-workflow.md`
- `pallas-kernel:test`: `references/test-workflow.md`
- `pallas-kernel:profile`: `references/profile-workflow.md`
- `pallas-kernel:autotune`: `references/autotune-workflow.md`
- `pallas-kernel:explain`: `references/explain-workflow.md`
- `pallas-kernel:gpu-to-jax`: `references/gpu-to-jax-workflow.md`

Before planning or implementation, read `references/knowledge-base.md` for the compact Pallas/TPU rules distilled from MaxKernel's knowledge base. When delegating to subagents, use the topology in `references/subagents.md` and the role prompts in `references/prompt-pack.md`. Keep each subagent scoped to one phase and require artifact paths instead of long raw transcripts.

## File and artifact contract

- Keep artifacts beside the source kernel unless the user specifies another directory.
- Plan path: `<kernel_stem>_plan.md`
- Optimized kernel path: `<kernel_stem>_optimized.py`
- Test path: `test_<optimized_kernel_stem>.py`
- Profiling script path: `<kernel_stem>_profile_run.py`
- Profile report path: `<kernel_stem>_profile.md`
- Autotune spec path: `autotune_specs.json`
- Autotune results path: `autotune_results.json`
- Autotune report path: `<kernel_stem>_autotune.md`
- Autonomous run directory: `pallas-kernel/<kernel_stem>/`
- Iteration snapshots: `pallas-kernel/<kernel_stem>/iterations/round-###/`

Never use `/tmp` for durable plans, reports, generated kernels, or test files. If scratch execution needs a transient file, remove it before final response and do not treat it as project memory.

For long-running investigations, also write a concise project-memory note:

```text
agent_memory_append("project", "pallas-kernel/report.md", summary, section="<kernel path>")
```

## Reading rules

- Use `ReadFile(file_path=..., offset=0, limit=400)` by default.
- Continue with the returned next offset when needed.
- Use `limit=-1` only for intentionally small files or when the whole file is needed.
- Prefer `rg`, `rg --files`, `GlobTool`, and targeted reads over dumping entire trees.

## Pallas implementation rules

Every generated Pallas kernel file must be self-contained and include:

- imports for `jax`, `jax.numpy as jnp`, and `jax.experimental.pallas as pl`;
- module-level `kernel(...)` function containing memory-reference logic;
- module-level `computation(...)` function that invokes `pl.pallas_call`;
- optional module-level helpers/constants when the plan requires them;
- `main()` that creates sample inputs and runs the computation without hiding errors.

Do not add `try/except` around compilation, JIT, or kernel execution. Raw errors are required for diagnosis.

Use `debug=True` in `pl.pallas_call` during development and validation unless the user explicitly asks for final stripped production code.

## Documentation rules

Generated kernel code must include:

- shape comments for significant tensors and references;
- memory-space comments such as HBM, VMEM, SMEM, and registers;
- transfer comments for HBM to VMEM, VMEM to registers, registers to VMEM, and VMEM to HBM writeback;
- comments explaining grid indices, block specs, boundary masking, accumulation, and synchronization.

## TPU/Pallas design checklist

Before writing code, reason about:

- operation shape and dtype;
- target TPU generation if known;
- reference implementation and expected numerical tolerance;
- block dimensions and whether they divide the real tensor shapes;
- MXU-friendly tile sizes, usually aligned to 32, 64, or 128 where appropriate;
- HBM bandwidth versus compute bound behavior;
- VMEM footprint per block;
- whether `pl.program_id` dimensions map cleanly to output blocks;
- whether masks are required for edge tiles;
- whether accumulation precision needs `float32`;
- whether repeated benchmark runs are affected by donated buffers.

If any API detail is uncertain, do not guess. Consult the local project docs if present, `references/knowledge-base.md`, and current JAX/Pallas documentation before editing.

## Validation checklist

Validation is not optional for generated kernels unless the user explicitly says not to run it.

- Run a syntax/import check.
- Run the kernel's `main()` or a small direct call.
- Run `jax.jit(computation)` on representative inputs.
- Compare against a reference implementation with `jnp.allclose` or `numpy.testing.assert_allclose`.
- Time with warmups and `.block_until_ready()`.
- Report exact commands, exact files touched, and the current status.

## Common pitfalls

- Hiding compiler failures with `try/except`.
- Changing the algorithm instead of fixing the Pallas API issue.
- Guessing `BlockSpec` rank or index maps without checking the source shapes.
- Using huge autotune grids.
- Timing without warmup or `.block_until_ready()`.
- Benchmarking host dispatch time instead of device work.
- Reusing donated buffers in a loop.
- Failing to preserve the source kernel's public function names or expected input layout.

## Source Inspiration

This skill is inspired by MaxKernel from `AI-Hypercomputer/accelerator-agents`, especially its autonomous phase split: planning, implementation, compilation validation, test generation, test execution, autotuning, profiling, artifact snapshots, best-iteration selection, explanation, and GPU-to-JAX conversion.
