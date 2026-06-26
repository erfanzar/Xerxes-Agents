# GPU-to-JAX Workflow

Use this workflow for CUDA, Triton, PyTorch CUDA, or GPU-oriented code that should become JAX/Pallas.

## 1. Identify the source framework

Read the source and classify it as:

- CUDA C++/CUDA kernels;
- Triton;
- PyTorch CUDA extension;
- PyTorch tensor code intended for GPU;
- mixed framework.

If the source is ambiguous, ask the user.

## 2. Create a conversion plan

Write `<source_stem>_gpu_to_jax_plan.md` beside the source.

The plan must include:

- source framework and entry points;
- hardware-specific parts to remove or replace;
- equivalent JAX/Pallas computation;
- expected input/output shapes;
- validation strategy;
- risks and missing information.

Stop for review unless the user requested an autonomous conversion.

## 3. Simplify before converting

Separate:

- algorithmic computation;
- memory layout;
- thread/block scheduling;
- GPU-specific intrinsics;
- synchronization.

Preserve algorithmic semantics. Do not blindly translate CUDA thread indices into Pallas program IDs.

## 4. Convert to JAX/Pallas

Use:

- plain `jax.numpy` for non-kernel reference implementation;
- Pallas only where explicit tiling/memory control is needed;
- the required `kernel` + `computation` structure from the main skill.

## 5. Validate in stages

Run:

1. syntax/import check;
2. non-JIT execution on small inputs;
3. JIT execution;
4. correctness against a reference;
5. performance comparison if requested.

## 6. Write summary

Save `<source_stem>_gpu_to_jax_summary.md` with:

- what changed;
- what GPU-specific concepts were removed;
- new JAX/Pallas files;
- validation commands and results;
- known gaps.
