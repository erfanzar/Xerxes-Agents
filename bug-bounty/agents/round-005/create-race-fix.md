# Round 005: session.create active-key race

## Fix

`startNewSession` now queues `session.create` through the same `runClientSwitch` tail as `session.resume` and `session.activate`. The generation check remains outside the queued callback, so a superseded create may finish and release the queue without applying stale visible state; the newer mutation then runs without waiting on itself or deadlocking.

Added a regression test that starts a deferred create, requests activation, verifies activation waits, resolves create, and confirms activation proceeds and becomes the sole visible session.

## Verification

- `bunx --bun --no-install vitest run --config vitest.ui.config.ts src/ui/__tests__/sessionLifecycle.test.ts` — 3 passed.
- `bun run check:ui` — passed.
