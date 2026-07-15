# Pallas Kernel Implementation Workflow

Use this workflow after an optimization plan exists and the user approves implementation.

## 1. Read the approved plan and source

Read:

- the approved `<kernel_stem>_plan.md`;
- the source kernel file;
- any local helper modules required by the source.

If the plan path is missing or ambiguous, ask for it and stop.

## 2. Preserve the approved strategy

Follow the plan exactly:

- use the planned block sizes unless they are impossible;
- use the planned grid shape;
- keep the planned memory layout;
- preserve source initialization and setup code outside the kernel/computation functions;
- do not rewrite unrelated code.

If the plan is contradictory, stop and ask to revise the plan.

## 3. Generate the optimized file

Write the optimized file beside the source as `<kernel_stem>_optimized.py`.

The file must contain:

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def kernel(...):
    """Core Pallas kernel operating on memory references."""
    ...

def computation(...):
    """Set up pallas_call and invoke kernel."""
    return pl.pallas_call(
        kernel,
        out_shape=...,
        grid=...,
        in_specs=...,
        out_specs=...,
        debug=True,
    )(...)

def main():
    """Create sample inputs and run the kernel without hiding errors."""
    ...

if __name__ == "__main__":
    main()
```

## 4. Documentation requirements

Every important tensor/reference needs shape and memory annotations:

- `# Shape: (...)`
- `# Memory: HBM|VMEM|SMEM|Registers`
- `# Transfer: HBM -> VMEM`
- `# Load: VMEM -> Registers`
- `# Store: Registers -> VMEM`
- `# Writeback: VMEM -> HBM`

Explain grid IDs, block coordinates, masks, accumulation, and why tile sizes match the plan.

## 5. Do not hide failures

Do not add `try/except` around:

- imports;
- `pallas_call`;
- `jax.jit`;
- sample execution;
- validation.

Let errors surface so the validation workflow can fix the real issue.

## 6. Final response

Return:

- optimized file path;
- plan file path used;
- key optimizations implemented;
- whether validation was run;
- next recommended command or workflow.
