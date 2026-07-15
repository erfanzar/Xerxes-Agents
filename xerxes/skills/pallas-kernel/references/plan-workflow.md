# Pallas Kernel Plan Workflow

Use this workflow when the user asks to create, revise, or inspect an optimization plan.

## 1. Identify the source kernel

Determine the source in this order:

1. Code pasted in the user message.
2. File path in the user message.
3. Existing active kernel path from conversation context.

If no source is available, ask for the file path or pasted code and stop.

If the user pasted code, write it to a project-local file with a clear name such as `kernel_source.py` or a name supplied by the user. Do not write to `/tmp`.

## 2. Read the source

Read the source with chunked reads. Identify:

- public function names and signatures;
- input shapes, dtypes, and layouts;
- reference computation;
- current use of `jax`, `pallas`, `jit`, or plain `jnp`;
- likely bottleneck: memory bandwidth, compute, dispatch, layout, synchronization, or fusion.

## 3. Research the pattern

Use available docs, local references, or web search for current Pallas examples when you are uncertain about APIs. Verify:

- `pl.pallas_call` signature and required `out_shape`;
- `pl.BlockSpec` rank and `index_map`;
- `pl.program_id` usage;
- masks and boundary handling;
- TPU memory hierarchy and MXU-friendly tile sizes.

## 4. Write the plan

Save a markdown plan beside the source as `<kernel_stem>_plan.md`.

The plan must include:

```markdown
# Kernel Optimization Plan: <kernel name>

## 1. Current Kernel Analysis
- What the source computes
- Current implementation approach
- Bottlenecks or correctness risks

## 2. Optimization Strategy
- High-level approach
- Transformations to apply
- Why each transformation should help

## 3. Memory Layout and Tiling
- Proposed block sizes
- HBM/VMEM/SMEM/register usage
- VMEM footprint estimate
- Whether edge masks are needed

## 4. TPU-Specific Optimizations
- MXU/vector-unit usage
- Pipelining or prefetching plan
- Synchronization or memory fence needs
- Accumulation precision

## 5. Implementation Details
- `kernel(...)` reference arguments
- `computation(...)` signature
- Grid dimensions
- BlockSpec configuration
- Boundary behavior and edge cases

## 6. Validation Plan
- Compilation/JIT command
- Correctness comparison method
- Tolerances
- Benchmark sizes and timing method

## 7. Expected Performance Impact
- Expected speedup direction
- Risks
- Alternative fallback strategy

## 8. Documentation Requirements
- Required shape comments
- Required memory-space comments
- Required transfer comments
```

## 5. Stop for review

Final response:

- name the plan path;
- summarize the strategy in 3-6 bullets;
- ask whether to approve, revise, or show the plan.

Do not implement in the same turn unless the user explicitly requested a full autonomous workflow.
