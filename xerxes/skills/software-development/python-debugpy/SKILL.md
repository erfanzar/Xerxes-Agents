---
name: python-debugpy
description: Debug Python with pdb plus debugpy remote attach.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [debugging, python, pdb, debugpy, breakpoints]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/software-development/python-debugpy/SKILL.md
---

# Python Debugger (pdb + debugpy)

## Overview

Xerxes is Bun-native, but its tools inspect user-owned projects of any
language. When a user's Python project is the debug target, pick by
situation:

| Tool | When |
|---|---|
| **`breakpoint()` + pdb** | Local, interactive, simplest. Get a REPL at that line. |
| **`python -m pdb`** | Launch a script under pdb with no source edits. |
| **`debugpy`** | Remote / headless / attach to an already-running process. |

Start with `breakpoint()`: the cheapest thing that works.

## When to Use

- A test fails and the traceback does not reveal why a value is wrong
- You need to step through a function and watch a collection mutate
- A long-running process misbehaves and cannot be restarted
- Post-mortem: inspect locals at the crash site of an exception

Do not use for problems `print()` or `pytest -vv --tb=long --showlocals`
already reveals.

## pdb Quick Reference

| Command | Action |
|---|---|
| `n` / `s` / `r` / `c` | next / step into / return / continue |
| `unt N` / `j N` | continue until line N / jump to line N |
| `l` / `ll` | source around current line / full function |
| `w` / `u` / `d` | stack trace / up / down |
| `a` | print current function args |
| `p expr` / `pp expr` | print / pretty-print expression |
| `display expr` | auto-print on every stop |
| `b file:line[, cond]` | breakpoint, optionally conditional |
| `tbreak file:line` | one-shot breakpoint |
| `!stmt` | execute arbitrary Python (assignments included) |
| `interact` | full REPL in current scope |
| `q` | quit |

## Recipes

**Local breakpoint.** Add `breakpoint()` in the source, run normally. Remove
it before committing; check with a search tool for `breakpoint\(\)` in
`*.py` files.

**Launch under pdb.**
```bash
python -m pdb path/to/script.py arg1
# then: b path/to/script.py:42 ; c
```

**Debug a pytest test.** Run pytest directly — parallel or
output-capturing runners silently swallow the pdb prompt:
```bash
python -m pytest tests/foo_test.py::test_bar --pdb     # drop on failure
python -m pytest tests/foo_test.py::test_bar --trace   # drop at test start
```

**Post-mortem on any exception.**
```bash
python -m pdb -c continue script.py
```
Or in code:
```python
import pdb, sys
try:
    run_the_thing()
except Exception:
    pdb.post_mortem(sys.exc_info()[2])
```

**Remote debug with debugpy — launch with the inspector:**
```bash
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client your_script.py
python -m debugpy --listen 127.0.0.1:5678 --wait-for-client -m your.module
```
Or in source, near the entry point:
```python
import debugpy
debugpy.listen(("127.0.0.1", 5678))
debugpy.wait_for_client()
```

**Attach to an already-running process:**
```bash
python -m debugpy --listen 127.0.0.1:5678 --pid <pid>
```
Hardened kernels may block ptrace-based injection
(`/proc/sys/kernel/yama/ptrace_scope`); launching under debugpy from the
start avoids that.

**Terminal-friendly alternative: `remote-pdb`.** Usually what you actually
want from an agent:
```bash
pip install remote-pdb
# in code:
from remote_pdb import set_trace
set_trace(host="127.0.0.1", port=4444)   # blocks until connection
# from the terminal:
nc 127.0.0.1 4444                        # gives a normal (Pdb) prompt
```
Use `debugpy` only when IDE integration is genuinely needed; from a shell
session `remote-pdb` is cleaner.

## Common Pitfalls

1. **pdb under parallel/capturing runners hangs silently.** Run pytest
   directly on a single file for interactive debugging.
2. **`breakpoint()` in CI or non-TTY contexts hangs the process.** Never
   commit it; grep before committing.
3. **`PYTHONBREAKPOINT=0` disables all `breakpoint()` calls.** Check the env
   when a breakpoint does not hit.
4. **`debugpy.listen` blocks only with `wait_for_client()`.** Otherwise
   execution continues and breakpoints may fire before you attach.
5. **Threads.** pdb debugs only the current thread; use debugpy for
   multithreaded code.
6. **asyncio.** pdb works in coroutines, but `await` inside pdb needs Python
   3.13+; on older versions use `interact` mode.
7. **Forking / multiprocessing.** pdb does not follow forks; each child needs
   its own `breakpoint()`. Debug one process at a time.

## Verification Checklist

- [ ] `python -c "import debugpy; print(debugpy.__version__)"` succeeds
- [ ] Port actually listening: `ss -tlnp` shows 5678
- [ ] First breakpoint actually hits (else check `PYTHONBREAKPOINT`, runner
      capture, or late attach)
- [ ] `w` shows the expected call stack
- [ ] Cleanup: no stray `breakpoint()` / `set_trace()` / `debugpy.listen` in
      committed code

## One-Shot Recipes

"Missing dict key": add `breakpoint()` above the KeyError site, then
`pp d`, `pp list(d.keys())`, `w`.

"Passes alone, fails in the suite": `python -m pytest tests/ -x --pdb` so it
traps after accumulated state.

"Async handler deadlocks": `remote_pdb.set_trace()` at handler entry, trigger
it, connect with `nc`, then `w` and inspect pending tasks.

---

Adapted from the `python-debugpy` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Hermes Agent.
