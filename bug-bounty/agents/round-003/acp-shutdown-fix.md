# ACP stdio shutdown fix — Round 003

## Confirmed defect

`StdioJsonRpcServer.serve()` called `AcpServer.shutdown()` and then awaited every prompt worker. A custom `promptHandler` is not required to observe runner cancellation, so a handler returning a permanently pending promise kept the worker—and therefore stdio shutdown—pending forever.

## Fix

- Added a transport-local shutdown signal.
- Prompt workers now race handler settlement against transport shutdown.
- Shutdown still calls `AcpServer.shutdown()` first, preserving session cancellation and runner abort behavior.
- A one-task grace period lets cancellation-aware prompts settle and emit their existing JSON-RPC success/error response before only non-settling workers are detached.
- The `shutdown` request response is still written before cancellation starts; a detached non-settling prompt emits no fabricated response.

## Regression

Added `ACP shutdown does not await a non-settling prompt handler` in `xerxes/test/acpTransport.test.ts`. It starts a permanently pending handler, sends `shutdown`, and verifies:

- serving completes within 500 ms;
- shutdown request id `2` receives `{ ok: true }`;
- no response is invented for pending prompt id `1`;
- the ACP session is marked cancelled.

The existing cancellation-aware runner regression also continues to verify its cancellation error response.

## Verification

- `bun test xerxes/test/acpTransport.test.ts` — **10 pass, 0 fail, 35 assertions**.
- `bun run --cwd xerxes check:runtime` — **passed** (`tsc --noEmit`).
- `git diff --check -- xerxes/src/acp/transport.ts xerxes/test/acpTransport.test.ts` — **passed**.
