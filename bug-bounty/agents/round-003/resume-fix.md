# Round 003 — Resume/activate overlap fix

## Finding

`useSessionLifecycle` used a generation check to suppress stale React/UI updates, but `GatewayClient.sessionResume` and `sessionActivate` independently commit `activeSessionKey` when their requests resolve. Overlapping calls could therefore resolve out of order: the visible state stayed on the newest selection while the older request committed the client's active key last.

## Fix

Added a hook-local promise tail for gateway session-switch mutations. Resume and activate requests now execute serially, and a queued request is skipped if its generation is stale before it starts. Existing generation checks remain responsible for visible-state commits and error reporting. No `gatewayClient.ts` change was required.

Added a focused regression test that starts resume, selects activate before resume settles, verifies activate is not sent concurrently, and confirms gateway commits occur in request order with only the newest session applied to UI/composer state.

## Files

- `xerxes/src/ui/app/useSessionLifecycle.ts`
- `xerxes/src/ui/__tests__/sessionLifecycle.test.ts`

## Verification

- `bun test xerxes/src/ui/__tests__/sessionLifecycle.test.ts` — 2 pass, 0 fail
- `bun run --cwd xerxes check:ui` — passed (`tsc --noEmit -p tsconfig.ui.json`)
- `git diff --check -- xerxes/src/ui/app/useSessionLifecycle.ts xerxes/src/ui/__tests__/sessionLifecycle.test.ts` — passed
