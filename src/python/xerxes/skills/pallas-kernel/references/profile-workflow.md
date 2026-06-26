# Pallas Kernel Profile Workflow

Use this workflow when the user asks for profiling, bottleneck analysis, or TPU utilization.

## 1. Establish the profiling target

Identify:

- optimized kernel path;
- reference path if available;
- representative input shapes;
- target TPU generation if known.

If shapes are unknown, infer conservative examples from the kernel or ask the user.

## 2. Collect evidence

Prefer existing project benchmark commands. Otherwise create a small profiling script beside the kernel.

Timing requirements:

- warmup before recording;
- use `.block_until_ready()`;
- record median or mean across multiple runs;
- include dtype, shape, device, and JIT state.

If XProf or Perfetto traces are available, collect the trace path and write it into the report. Do not require traces for a basic profile.

## 3. Analyze bottlenecks

Look for:

- low MXU utilization;
- excessive HBM traffic;
- DMA or memory transfer dominance;
- vector-unit bottlenecks;
- compilation/recompilation effects;
- masks or edge tiles causing branch overhead;
- layout transposes around the kernel;
- too-small tile sizes underutilizing compute;
- too-large tiles exceeding VMEM or reducing occupancy.

## 4. Write profile report

Save `<kernel_stem>_profile.md` beside the kernel.

Include:

- command and environment;
- input shapes/dtypes;
- baseline time;
- optimized time;
- speedup;
- observed bottlenecks;
- recommended next changes;
- whether autotuning is worthwhile.

## 5. Final response

Return the profile report path, headline speedup or slowdown, and 3-5 actionable recommendations.
