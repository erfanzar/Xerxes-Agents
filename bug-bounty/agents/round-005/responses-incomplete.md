# Responses non-stream incomplete finish mismatch

**Confirmed.** `ResponsesApiClient.complete()` returns `finishReason: "incomplete"` for a successful Responses payload with:

```json
{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}
```

The equivalent streaming `response.incomplete` event is normalized to `finishReason: "length"` by `incompleteFinishReason()` in `xerxes/src/streaming/responsesApi.ts`. Non-stream parsing in `parseResponsesCompletion()` (`xerxes/src/llms/client.ts`) ignores `incomplete_details.reason` and falls back to raw `status`, creating transport-dependent agent-loop behavior.

## Exact regression test

Isolated reproducer: `bug-bounty/agents/round-005/responses-incomplete.repro.test.ts`. Permanent placement should be `xerxes/test/responsesApiClient.test.ts` beside the native non-stream completion test. It stubs HTTP 200 with partial text, `status: incomplete`, reason `max_output_tokens`, and usage, then expects the completion to preserve content/usage and report `finishReason: "length"`.

## Evidence

```text
bun test bug-bounty/agents/round-005/responses-incomplete.repro.test.ts
0 pass, 1 fail
Expected finishReason "length"; received "incomplete".
```

No production files changed.
