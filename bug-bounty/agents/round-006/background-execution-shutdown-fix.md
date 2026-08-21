# Round 006 — Background and execution shutdown parity

## Issue

`BackgroundSessionManager` and `ExecutionRegistry` previously coupled the caller-facing timeout to the memoized shutdown operation. Whichever caller initiated shutdown therefore fixed the timeout and `cancelRunning` policy for every later caller. A zero-timeout first call could make later callers return immediately even while physical cleanup was still running, and a later caller could not escalate a graceful shutdown to cancellation. Invalid first-call options were also observed after the managers had begun mutating shutdown state.

## Fix

Both managers now memoize only the shared physical cleanup promise. Every `shutdown()` call:

1. validates `timeoutMs` and `cancelRunning` before changing manager state;
2. starts the shared pending-cancellation and active-runner cleanup once;
3. may independently escalate active work to cancellation; and
4. applies its own timeout while waiting on the shared cleanup promise.

The cleanup promise itself has no timeout, so a caller timing out does not replace or terminate cleanup and a later caller can still wait for actual settlement.

## Tests

Focused parity tests for both managers verify that:

- a zero-timeout caller returns while physical cleanup remains active;
- a concurrent caller with a longer timeout continues waiting on that same cleanup;
- a later `cancelRunning: true` caller aborts the active runner without starting duplicate work;
- late runner success cannot overwrite cancellation; and
- invalid timeout and cancellation options throw before shutdown state changes, leaving submission available.

## Files changed

- `xerxes/src/runtime/backgroundSessions.ts`
- `xerxes/src/runtime/executionRegistry.ts`
- `xerxes/test/backgroundSessions.test.ts`
- `xerxes/test/executionRegistry.test.ts`

## Verification

- `bun test xerxes/test/backgroundSessions.test.ts xerxes/test/executionRegistry.test.ts` — **13 pass, 0 fail**.
- `git diff --check -- xerxes/src/runtime/backgroundSessions.ts xerxes/src/runtime/executionRegistry.ts xerxes/test/backgroundSessions.test.ts xerxes/test/executionRegistry.test.ts` — **passed**.
- `bun run --cwd xerxes check` — **blocked by unrelated current-worktree TypeScript errors** in `src/auth/codexAuth.ts`: unresolved `refreshOAuthToken` and `OAuthError`, followed by `unknown` catch-variable accesses (lines 307–327).
