# ACP worker cleanup fix — Round 005

## Scope

Reviewed and updated:

- `xerxes/src/acp/transport.ts`
- `xerxes/test/acpTransport.test.ts`

## Finding and fix

`startPrompt()` previously discarded the promise returned by `worker.finally(...)`. When the prompt worker rejected (for example because the injected writer rejected a streamed update), `finally()` created a second rejected promise with no observer even though the original worker was later passed to `Promise.allSettled()`. This could emit an `unhandledRejection`.

Cleanup now uses `worker.then(onFulfilled, onRejected)`, with both branches removing the worker. The rejection branch handles the derived promise while preserving the original worker rejection for the transport's `Promise.allSettled()` shutdown path.

## Regression coverage

Added `ACP prompt worker cleanup observes writer rejection`. It installs a temporary `unhandledRejection` listener, makes the ACP writer reject a streamed update, waits for delayed rejection reporting, and asserts that no unhandled rejection escaped.

## Post-shutdown update assessment

A prompt handler may outlive transport shutdown when it ignores cancellation or never settles. A broad queue/lifecycle rewrite was not needed. The prompt worker now has a local `acceptsUpdates` guard that becomes false when its shutdown race settles, preventing a detached handler from publishing later `session/update` frames. Updates emitted while a prompt is still contractually settling remain allowed, preserving the existing EOF cancellation response behavior.

## Verification

- `bun test xerxes/test/acpTransport.test.ts` — **11 pass, 0 fail, 36 assertions**
- `bun run --cwd xerxes check` — **passed** (`check:runtime` and `check:ui`)
- `git diff --check -- xerxes/src/acp/transport.ts xerxes/test/acpTransport.test.ts` — **passed**
