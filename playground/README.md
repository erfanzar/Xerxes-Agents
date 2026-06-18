# Xerxes agent eval playground

A scored, re-runnable evaluation harness for the real Xerxes agent. It spins up
the agent (daemon + whatever provider profile is **active**, e.g. MiniMax) in an
**isolated sandbox workspace**, runs a task battery through it, auto-grades each
task, and prints a scorecard.

## Run

```bash
.venv/bin/python playground/eval.py            # run the whole suite
.venv/bin/python playground/eval.py -v         # also print each prompt + reply
.venv/bin/python playground/eval.py -k memory  # run only tasks whose name matches
```

Example:

```
  Xerxes eval playground — model: MiniMax-M2.7-highspeed   sandbox: .../playground/workspace

  [PASS] reasoning      reason     4.7s  391
  [PASS] file_read      tools      6.9s  4.2.0
  [PASS] file_edit      coding    13.4s  greet() -> 'goodbye'
  [PASS] bug_fix        coding    15.9s  add(2,3) -> 5
  [PASS] tool_search    tools      6.9s  tools=['GrepTool'] ans='notes/b.txt'
  [PASS] shell          tools      2.6s  eval-ok-7
  [PASS] multiturn      context    5.8s  42
  [PASS] memory_recall  memory     5.4s  fresh session -> 'fly.io'
  ------------------------------------------------------------
  SCORE: 8/8 passed   (100%)   total 61.6s   model=MiniMax-M2.7-highspeed
```

## What it tests

| Task | Category | What it proves |
|------|----------|----------------|
| `reasoning` | reason | basic correctness (`17x23 = 391`) |
| `file_read` | tools | reads a file and extracts a value |
| `file_edit` | coding | edits code — **graded by executing** `greet()` |
| `bug_fix` | coding | fixes a real bug — **graded by executing** `add(2,3)==5` |
| `tool_search` | tools | finds content across files (uses Grep/Glob) |
| `shell` | tools | runs a shell command and reports output |
| `multiturn` | context | remembers within a session (fav number → x6) |
| `memory_recall` | memory | **always-on memory**: a fact stated in one session is recalled by a brand-new session via the journal |

Coding tasks are graded **behaviorally** (the harness executes the resulting
file), not by string-matching — so a "looks right" answer that doesn't run fails.

## How it's isolated

- Each run rebuilds `playground/workspace/` from scratch (toy files with a known
  value, a bug, a hidden marker).
- It **kills any running daemon and starts a fresh one whose working directory
  is the sandbox**, so the agent's file tools and its memory journal stay inside
  `playground/workspace/` — your Xerxes repo is never touched.
- `accept-all` permission mode so tasks don't block on approvals.

## Evaluate a different model

The harness uses the **active provider profile** (it does not change it). To
compare models, switch the profile and re-run:

```bash
# point the active profile at another model, then:
.venv/bin/python -c "import sys;sys.path.insert(0,'src/python');from xerxes.bridge import profiles;profiles.update_active_model('MiniMax-M3')"
.venv/bin/python playground/eval.py
```

…or in the TUI: `/model MiniMax-M3`, then re-run the script.

## Add or change tasks

Edit `task_suite()` in `eval.py`. A task is:

```python
dict(
    name="my_task", cat="coding",
    steps=[("prompt for step 1", True),          # (prompt, fresh_session?)
           ("prompt for step 2", False)],         # False = same session as prev
    check=lambda step_results: (bool_pass, "detail string"),
)
```

`fresh_session=True` starts a clean session (memory persists across sessions, so
that's how the cross-session memory test works). `check` receives the list of
per-step results (`{"text", "tools", "latency", "ctx", "error"}`) and returns
`(passed, detail)`. For coding tasks prefer behavioral checks (exec the file)
over string matching — see `_behaves`.
