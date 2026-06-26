# Pallas Kernel Knowledge Base

This is a compact, Xerxes-native extraction of the MaxKernel `auto_agent/knowledge_base` material. It is not a copy of the source docs; it captures the durable rules the model should use before writing or fixing Pallas kernels.

## Documentation scope

The MaxKernel knowledge base was built from current JAX/Pallas documentation covering:

- Pallas quickstart;
- grids and `BlockSpec`;
- software pipelining;
- TPU details;
- TPU pipelining;
- TPU matmul;
- sparse kernels;
- profiling and XProf analysis.

When exact API signatures matter, verify against current docs before editing code.

## Pallas programming model

- A Pallas kernel operates on mutable `Ref` objects, not ordinary arrays.
- The callable returned by `pl.pallas_call` receives and returns `jax.Array` values.
- Inputs and outputs are partitioned by `grid`, `in_specs`, and `out_specs`.
- `pl.program_id(axis)` identifies the current grid invocation.
- `pl.BlockSpec(block_shape, index_map, memory_space=...)` maps grid IDs to blocks.
- On TPU, blocks must have rank at least 1.
- Every input and output normally needs a matching `BlockSpec`.
- Writes from different programs must target disjoint HBM locations unless using a valid accumulation/aliasing pattern.

## Memory model

Use these terms consistently in plans and code comments:

- HBM/DRAM: large device memory.
- VMEM/SRAM: on-chip vector memory used by Pallas block buffers.
- SMEM: TPU scalar memory for scalar values.
- Registers: closest compute storage.

On TPU, Pallas fetches HBM data into SRAM/VMEM before the kernel body runs, executes vector/MXU work, then writes output SRAM back to HBM after the kernel completes.

Good comments should describe:

- HBM to VMEM block transfer;
- VMEM to register load;
- register compute;
- register to VMEM store;
- VMEM to HBM writeback.

## Grid and BlockSpec rules

- Treat `grid` as a structured loop nest.
- Map `program_id` axes to semantic dimensions in the plan before coding.
- `index_map` must accept the right number of grid indices and return the right number of block indices.
- Block rank must match the array rank unless using supported squeezed dimensions such as `None`.
- Edge masks are required when dimensions are not cleanly divisible by block sizes.
- For reductions and accumulations, put the reduction dimension in the minor-most grid axis so the same output SRAM block can be reused until accumulation is complete.

## TPU optimization rules

- The goal is usually to become compute-bound and, for matrix-heavy kernels, MXU-bound.
- Memory-bound kernels spend too much time in HBM/VMEM transfer or synchronization waits.
- Increase block sizes for arithmetic-intense operations until VMEM footprint or pipeline bubbles become limiting.
- Prefer bf16 or lower precision when accuracy allows.
- Fuse operations while data is already in VMEM, especially around matmul.
- Arrange index maps so repeated consecutive iterations can reuse blocks.
- TPU Pallas generally relies on double buffering for HBM/VMEM pipelining.
- Larger blocks may hurt short pipelines because startup/teardown bubbles become a larger fraction of runtime.

## Mosaic TPU constraints

- Matmul-style accumulation should use 32-bit accumulation when required by TPU lowering.
- The last two dimensions of TPU block shapes often need hardware-friendly divisibility, commonly multiples aligned around 8 and 128, unless matching the full array dimension exactly.
- Rank-zero blocks are unsupported on TPU.
- Use `debug=True` in `pl.pallas_call` during development and validation.

## Testing and timing rules

- Always compare against a base/reference implementation.
- Use small correctness shapes first, then representative performance shapes.
- Warm up before timing.
- Use `.block_until_ready()` or `jax.block_until_ready(...)`.
- Report a structured metric such as `PERF_METRICS: <milliseconds>` or `RESULT_TIME: <milliseconds>`.
- If donation is used, do not reuse donated buffers across repeated timing calls.

## Profiling rules

Use roofline analysis for a theoretical read and XProf for empirical evidence.

Look for:

- DMA and memory-transfer ratio;
- compute ratio;
- Tensor Core Sync Flag / sync wait time;
- MXU utilization;
- ALU/vector bottlenecks;
- HBM traffic and VMEM footprint;
- XLA optimization barriers around Pallas calls;
- pipeline bubbles;
- HLO ops dominating duration.

The desired endpoint is usually: no dominant memory wait, high compute/MXU utilization, and no surrounding XLA regressions.

## TPU specs reference

Use these as rough planning anchors, not exact runtime guarantees:

| TPU | Peak bf16 TFLOPs | HBM GiB | HBM TB/s | Topology |
| --- | ---: | ---: | ---: | --- |
| v4 | 275 | 32 | 1.2 | 3D Torus |
| v5e | 197 | 16 | 0.8 | 2D Torus |
| v5p | 459 | 95 | 2.76 | 3D Torus |
| v6e / Trillium | 918 | 32 | 1.6 | 2D Torus |
| TPU 7x / Ironwood | 2157 | 192 | 7.4 | 3D Torus |

If the user provides the real target TPU, prefer that over this table.

## Hard no-go patterns

- Guessing Pallas API signatures.
- Continuing after compilation failure without fixing it.
- Generating tests without both base and optimized implementations.
- Running broad random file searches when state or user paths already specify the files.
- Applying autotune results without a valid best config and measured time.
- Reporting performance without warmup and device synchronization.
- Hiding kernel errors behind `try/except`.
