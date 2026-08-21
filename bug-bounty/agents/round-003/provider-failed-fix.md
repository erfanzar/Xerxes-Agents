# Round 003 — Responses HTTP-200 failure fix

## Fix

`ResponsesApiClient.complete()` now rejects successful HTTP envelopes whose decoded Responses payload has `status: "failed"` or `status: "error"`.

The semantic failure is checked before output, tool-call, usage, or finish-reason parsing. Its provider error uses the streaming Responses normalization:

```text
stream returned API error (server_error): Model exploded
```

This prevents failed responses from resolving as ordinary completions with `finishReason: "failed"` or `"error"`.

## Regression coverage

Added a deterministic table-style test in `xerxes/test/responsesApiClient.test.ts` covering both terminal statuses with an HTTP 200 JSON response and asserting rejection with the normalized provider code/message.

The new test failed before the implementation change because `complete()` resolved. It passes after the fix.

## Files changed

- `xerxes/src/llms/client.ts`
- `xerxes/test/responsesApiClient.test.ts`
- `bug-bounty/agents/round-003/provider-failed-fix.md`

## Verification

- `bun test xerxes/test/responsesApiClient.test.ts` — 7 pass, 0 fail, 11 assertions.
- `bun run --cwd xerxes check` — passed runtime and UI TypeScript checks.
- `git diff --check -- xerxes/src/llms/client.ts xerxes/test/responsesApiClient.test.ts` — passed.
