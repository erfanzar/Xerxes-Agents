# Round 006 — Channel manager lifecycle status fix

## Outcome

Fixed the channel manager status defect where calling `enable()` after a failed stop cleared the recorded stop failure even though the adapter was still marked started and no real restart occurred. A later successful stop retry now clears that stale failure.

## Changes

- `xerxes/src/channels/manager.ts`
  - Capture whether the adapter was already started before `registry.start()`.
  - Clear lifecycle failures only when `enable()` performs a genuine transition from not-started to started.
  - Preserve a failed-stop diagnostic on an idempotent enable of an adapter whose stop failed.
  - Existing successful `disable()` behavior clears the diagnostic after the teardown retry succeeds.
- `xerxes/test/channelManager.test.ts`
  - Added a focused regression covering failed stop, idempotent re-enable without another `start()`, retained stop error, and successful stop retry clearing the stale error.

## Verification

- `bun test xerxes/test/channelManager.test.ts xerxes/test/channels.test.ts`
  - Passed: 14 tests, 0 failures, 70 assertions.
- `git diff --check -- xerxes/src/channels/manager.ts xerxes/test/channelManager.test.ts`
  - Passed with no output.
- `bun run --cwd xerxes check`
  - Blocked by an unrelated pre-existing TypeScript error: `src/auth/codexAuth.ts(307,20): error TS2304: Cannot find name 'refreshOAuthToken'.`

## Remaining risk

The full type-check could not complete because of the unrelated `codexAuth.ts` error. Focused channel tests pass.
