# Round 003 — Channel stop retry fix

## Outcome

Fixed the confirmed `ChannelRegistry` lifecycle defect: a failed channel `stop()` no longer removes the channel from the registry's started-state map, allowing a later stop attempt to retry teardown.

## Changes

- `xerxes/src/channels/registry.ts`
  - Moved `started.delete(name)` from `finally` to the successful-stop path.
  - Existing failure reporting and propagation/isolation behavior remains unchanged.
- `xerxes/test/channels.test.ts`
  - Updated the existing lifecycle-failure assertion to require failed stops to remain started.
  - Added a focused regression test where the first stop fails and the second succeeds, proving retry occurs and state clears only after success.

## Verification

- RED before implementation: `bun test xerxes/test/channels.test.ts`
  - 9 passed, 1 failed; regression expected started state after the failed stop but received `false`.
- GREEN after implementation: `bun test xerxes/test/channels.test.ts`
  - 10 passed, 0 failed, 52 assertions.
- Type check: `bun run --cwd xerxes check`
  - Passed runtime and UI TypeScript checks.

## Blockers / risks

None observed. Full repository test/build gates were not requested or run.
