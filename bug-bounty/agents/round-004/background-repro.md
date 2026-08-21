# Round 004 — BackgroundCommandManager cross-session re-attempt

**Status:** confirmed · **Severity:** high · **Class:** cross-session authorization / IDOR

## Reproduction

On 2026-08-21, I registered the production process tools with one shared `BackgroundCommandManager`. Session A started `/bin/sh -c 'echo ROUND004_A_SECRET; sleep 30'`; session B then listed, checked, and killed A's command through `ToolRegistry.execute` using its own `sessionId`.

```text
$ bun -e '<production ToolRegistry/process-tools reproducer>'
{"aProcId":"4c22749f239e","bListed":["4c22749f239e"],"bRead":"ROUND004_A_SECRET","bKilled":true}
```

Exit code was `0`; cleanup used `disposeAll()` and removed the temporary workspace.

## Cause

The daemon intentionally creates one manager outside `runnerFactory` so handles survive registry/settings rebuilds (`xerxes/src/cli.ts:782-809`). However, `BackgroundEntry` has no owner, and `require()` authorizes only by `proc_id` (`xerxes/src/tools/backgroundCommands.ts`). Although every tool handler receives trusted `ToolExecutionContext.sessionId`, `registerProcessTools` ignores the context for start/check/list/kill (`xerxes/src/tools/processTools.ts:166-217`). Thus B can discover A's ID, consume A's buffered output, and terminate A's child.

## Lowest-risk ownership fix

Preserve the single host-lifetime manager; do **not** return to per-registry managers. Add an immutable manager-private owner to each entry and make the scoped operations require it:

```ts
start(ownerSessionId, options)
check(ownerSessionId, procId, ...)
list(ownerSessionId)
kill(ownerSessionId, procId, signal)
disposeSession(ownerSessionId)
```

Bind `ownerSessionId` exclusively from handler `context.sessionId`, never tool/model input. Fail closed if absent. On owner mismatch, return the existing unknown-`proc_id` error to avoid an existence oracle; filter `list` by owner. Keep `disposeAll()` for host shutdown. This is lower risk than manager-per-session because it preserves handles across runner rebuilds and central process/terminal tracking while changing only the manager/tool boundary.

Minimum regression: A can list/check/kill A; B lists nothing and cannot check/kill A; B cannot consume A's output; `disposeSession(B)` leaves A running. Lifecycle cleanup and subagent parent-vs-child ownership should be handled separately rather than widening this authorization fix.

No production or test files were edited; only this report was written.
