# Round 003 — Concurrent shutdown cleanup

## Confirmed issue

`BackgroundSessionManager.shutdown()` and `ExecutionRegistry.shutdown()` both used a `shuttingDown` boolean guard. The first caller set the flag and waited for active runner cleanup, while every concurrent caller observed the flag and returned immediately. Thus a caller awaiting shutdown was not guaranteed that cleanup had completed.

## Fix

Both managers now memoize the first shutdown operation in `shutdownPromise` and return that same promise to every caller. The first call still atomically stops submissions before beginning the existing cancellation/wait/pruning sequence. Shutdown options are consequently owned by the initiating call, matching the prior first-call-wins guard behavior.

Regression tests in both focused test files hold an active runner behind a deferred gate and verify that:

- concurrent calls receive the same shutdown promise;
- the concurrent caller remains pending while runner cleanup is incomplete; and
- it resolves only after the runner settles and the active count reaches zero.

## Files changed

- `xerxes/src/runtime/backgroundSessions.ts`
- `xerxes/src/runtime/executionRegistry.ts`
- `xerxes/test/backgroundSessions.test.ts`
- `xerxes/test/executionRegistry.test.ts`

## Verification

- `bun test xerxes/test/backgroundSessions.test.ts xerxes/test/executionRegistry.test.ts` — **11 pass, 0 fail**.
- `git diff --check -- xerxes/src/runtime/backgroundSessions.ts xerxes/src/runtime/executionRegistry.ts xerxes/test/backgroundSessions.test.ts xerxes/test/executionRegistry.test.ts` — **passed**.
- `bun run --cwd xerxes check` — **blocked by unrelated current-worktree TypeScript errors**:
  - `src/acp/transport.ts`: missing `StdioJsonRpcServer.stopPrompts` (lines 101 and 271).
  - `src/llms/client.ts`: missing `throwIfResponsesCompletionError` (line 1133).
