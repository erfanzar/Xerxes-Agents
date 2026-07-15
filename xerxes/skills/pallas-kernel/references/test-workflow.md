# Pallas Kernel Test Workflow

Use this workflow to generate or run tests for a Pallas kernel.

## 1. Require two implementations

You need:

- base/reference implementation;
- optimized Pallas implementation.

If either path is missing, ask the user to provide both paths and stop. Do not search randomly.

## 2. Read both files

Identify:

- function names;
- input signatures;
- expected shapes and dtypes;
- output structure;
- tolerances implied by dtype and accumulation precision.

## 3. Generate pytest file

Write `test_<optimized_kernel_stem>.py` beside the optimized kernel.

Test structure:

- `TestCompilation`: import, run, and JIT both implementations.
- `TestCorrectness`: compare outputs across representative shapes, zeros, ones, and random inputs.
- `TestPerformance`: warmup, run several timed iterations, call `.block_until_ready()`, report speedup.

Use small default shapes first so the test is fast. Add larger benchmark cases only if the user requested performance work or TPU access is confirmed.

## 4. Test timing rules

- Warm up before timing.
- Use `.block_until_ready()` on outputs.
- Do not time only host dispatch.
- Avoid huge iteration counts.
- If the kernel donates buffers, create fresh inputs for each timed iteration or disable donation in the benchmark.

## 5. Run tests

Run:

```bash
uv run pytest <test_file.py> -q
```

If TPU/JAX deps are unavailable, report that clearly and still leave the test file.

## 6. Final response

Return:

- test file path;
- command run;
- pass/fail summary;
- correctness tolerance;
- performance result if measured.
