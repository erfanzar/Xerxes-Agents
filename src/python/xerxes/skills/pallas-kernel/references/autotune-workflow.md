# Pallas Kernel Autotune Workflow

Use this workflow when the user asks to tune block sizes or kernel parameters.

## 1. Identify tunable parameters

Read the optimized kernel and plan. Candidate parameters usually include:

- `BLOCK_M`, `BLOCK_N`, `BLOCK_K`;
- tile sizes;
- group sizes;
- num warps or pipeline stage equivalents when applicable;
- unroll factors;
- prefetch/pipeline toggles.

Do not tune semantic constants.

## 2. Keep the search small

Use high-probability values only:

- prefer multiples of 32, 64, or 128;
- prefer values that divide the relevant tensor dimensions;
- use 2-3 candidates per parameter;
- keep total combinations roughly between 10 and 100;
- avoid broad blind sweeps.

## 3. Create autotune specs

Write `autotune_specs.json` beside the kernel:

```json
{
  "kernel_name": "...",
  "code_template": "...",
  "search_space": {
    "BLOCK_M": [64, 128],
    "BLOCK_N": [64, 128]
  },
  "original_file_path": "..."
}
```

The generated benchmark template must print:

```text
RESULT_TIME: <milliseconds>
```

Use one warmup and exactly 10 measured iterations by default. Call `.block_until_ready()`.

## 4. Run and compare

For each candidate:

- write or patch the candidate;
- run the benchmark;
- capture the result;
- discard failed candidates with their error category;
- keep the best valid config.

## 5. Apply only evidence-backed changes

Apply the best config only if it improves timing beyond noise. Otherwise report that the current config remains best.

## 6. Write report

Save `<kernel_stem>_autotune.md` with:

- search space;
- results table;
- best config;
- applied changes;
- follow-up ideas.
