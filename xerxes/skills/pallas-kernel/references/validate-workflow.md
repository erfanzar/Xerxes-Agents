# Pallas Kernel Validation Workflow

Use this workflow when the user asks to validate compilation or fix compile/JIT errors.

## 1. Identify files

Find:

- optimized kernel path;
- plan path if available;
- source/reference path if available.

If the optimized file is missing, ask for it and stop.

## 2. Run narrow validation

Use `exec_command` for commands. For long commands, poll with `write_stdin`.

Start with:

```bash
uv run python <optimized_kernel.py>
```

Then run a minimal JIT check if not already covered:

```bash
uv run python -c "import importlib.util, pathlib; p=pathlib.Path('<optimized_kernel.py>'); spec=importlib.util.spec_from_file_location('k', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(m.computation)"
```

If the project does not use `uv`, use the project's documented Python runner.

## 3. Fix only compilation errors

When validation fails:

- read the exact traceback;
- read the kernel and plan;
- identify the smallest change that fixes the current error;
- do not change the optimization strategy unless it is the direct cause;
- do not repeat a fix that already failed;
- keep `debug=True` in `pl.pallas_call`;
- keep raw errors visible.

Common fixes:

- BlockSpec rank does not match input rank.
- `index_map` returns wrong number of indices.
- `out_shape` dtype/shape does not match returned output.
- Missing masks for non-divisible dimensions.
- Reference writes use incompatible shapes.
- `program_id` dimension order mismatches grid.

## 4. Record attempt summary

After each fix, summarize:

```text
FIX_SUMMARY:
- Error: <exact error category>
- Cause: <root cause>
- Fix: <specific change made>
```

## 5. Stop condition

Stop when:

- syntax/import passes;
- non-JIT run passes;
- JIT run passes;
- a correctness test exists and passes, or you clearly state it still needs to be created.

If the same error survives multiple attempts, stop and report the blocker rather than thrashing.
