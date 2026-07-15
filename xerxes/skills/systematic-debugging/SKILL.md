---
name: systematic-debugging
description: Reproduce, trace, test a falsifiable root-cause hypothesis, fix narrowly, and verify unexpected behavior in a Bun-native workflow.
version: "2.0.0"
tags: [debugging, troubleshooting, root-cause, verification, bun]
required_tools: [ReadFile, GrepTool, GlobTool, ListDir, exec_command, FileEditTool, WriteFile, AgentTool]
---

# Systematic Debugging

Use this workflow for bugs, failed tests, build errors, performance regressions,
and unexpected production behavior. Scale the depth to the problem, but keep the
order: **reproduce -> trace -> hypothesize -> fix -> verify**.

## Operating rules

- Preserve user changes. Inspect relevant worktree state before editing and do
  not run destructive Git commands.
- Do not change code until evidence identifies a plausible root cause.
- Change one causal variable at a time. Do not bundle cleanup or refactoring
  into a debugging patch.
- Never claim a command, test, external call, or production observation ran
  unless it completed in the current environment.
- Follow the repository's native runtime and instructions. In Xerxes, use Bun;
  do not create Python runtime dependencies, Python tests or packaging metadata,
  or a Python subprocess fallback.

## Production tool forms

Use the registered tool names and input fields exactly:

```text
ReadFile(file_path="src/module.ts", offset=0, limit=200)
GrepTool(pattern="error|failed", path="src", glob="**/*.ts", output_mode="content")
GlobTool(pattern="**/*.test.ts", path="src")
ListDir(directory_path="src", recursive=true, max_depth=2)
exec_command(cmd="bun", args=["test", "xerxes/test/example.test.ts"], workdir=".")
```

`exec_command` is direct argv execution: `cmd` contains one executable, while
every argument is a separate `args` item. Do not put pipes, redirects, `cd`, or a
whole shell command in `cmd`. Use `FileEditTool` for exact replacements,
`WriteFile` for a genuinely new file, and the smallest safe read range needed.

## 1. Reproduce

1. Record the expected behavior, observed behavior, exact error, environment,
   and shortest known trigger.
2. Read the complete error and relevant stack frames; do not stop at the first
   familiar line.
3. Run the narrowest deterministic reproduction with `exec_command`. For this
   repository, prefer a focused command such as:

   ```text
   exec_command(cmd="bun", args=["test", "xerxes/test/failing.test.ts"], workdir=".")
   ```

4. If the failure is intermittent, record frequency and vary only one condition
   at a time. If it cannot be reproduced, gather logs or state and report the
   uncertainty instead of guessing.

## 2. Trace the root cause

Build an evidence chain from the symptom back to the first invalid decision or
value.

- Use `GrepTool` for the error string, symbol, event, or configuration key;
  `GlobTool` and `ListDir` for bounded discovery; and chunked `ReadFile` calls for
  the relevant implementations and tests.
- Trace callers, inputs, mutations, serialization, and component boundaries.
  At each boundary, compare what entered, what left, and what was expected.
- Inspect relevant recent changes with direct argv Git commands, for example
  `exec_command(cmd="git", args=["diff", "--", "src/module.ts"])`.
- Find a nearby working path and list meaningful differences. Do not assume a
  small difference is irrelevant before testing it.
- Add temporary diagnostics only when existing evidence cannot distinguish the
  failing boundary; keep them bounded and remove them unless they are useful
  permanent observability.

End this phase with one sentence: **"The likely root cause is X because Y."**
If the evidence cannot support that sentence, continue tracing.

## 3. Test one hypothesis

Write a falsifiable prediction before editing: **"If X is causal, then changing
or observing Y will produce Z."** Test it with the smallest isolated experiment.
If the prediction fails, discard the hypothesis and return to tracing; do not
stack another speculative change on top.

Independent investigation may be delegated, but the subagent must not patch:

```text
AgentTool(
  title="Trace cache invalidation",
  prompt="Reproduce the failure, trace the invalid value to its source, and return evidence plus one falsifiable root-cause hypothesis. Do not edit files.",
  subagent_type="researcher",
  wait=true
)
```

Keep titles single-line, specific, and at most 48 characters. The parent remains
responsible for checking the evidence before authorizing a fix.

## 4. Fix narrowly

1. Add the smallest failing regression test first when the behavior is
   testable. Confirm it fails for the expected reason.
2. Apply one root-cause fix with `FileEditTool`, or `WriteFile` only when the fix
   genuinely requires a new file.
3. Avoid unrelated formatting, dependency changes, and compatibility shims for
   retired behavior.

If three evidence-based fix attempts fail, stop editing. Reassess the design,
shared state, and component boundaries, then discuss an architectural change
before attempt four.

## 5. Verify and report

Run, in order:

1. the regression test;
2. the nearest affected suite plus typecheck or build required by project
   instructions;
3. the original reproduction and relevant error/cancellation path;
4. broader repository gates proportional to the blast radius; and
5. `exec_command(cmd="git", args=["diff", "--check"])` when files changed.

Report the root cause, evidence, exact fix, commands that completed and their
results, and any unverified production or external behavior. A green unrelated
test is not proof that the original bug is fixed.
