# Round 005 — Responses API streamed tool-call correlation ID

## Finding

The streamed Responses API translator conflated two provider identifiers:

- `item.id` / streamed `item_id` identifies the output item used to route argument delta events.
- `call_id` is the correlation identifier required by the later `function_call_output` item.

When both were present, `ResponsesEventTranslator.upsertPendingCall()` replaced the pending tool call's public ID with `item.id`. The runtime therefore emitted a neutral `ToolCall.id` such as `item_1`; subsequent history serialization reused that value as `function_call_output.call_id`, losing the provider's actual `call_1` correlation ID.

## Fix

Updated `xerxes/src/streaming/responsesApi.ts` to keep the item ID as the pending-map lookup key while preserving `call_id` as `PendingFunctionCall.id`. It falls back to the item ID only when the provider supplies no `call_id`.

Updated `xerxes/test/responsesApi.test.ts` expectations for:

- normal streamed function-call assembly;
- duplicate completed-item suppression; and
- streams whose argument deltas alternate between `call_id` and `item_id` aliases.

The aliasing test now explicitly verifies that the emitted neutral tool call uses `call_9`, not `item_9`.

## Verification

A regression-first focused run failed before the translator patch with the observed mismatch:

```text
Expected id: call_1
Received id: item_1
11 pass, 1 fail
```

After the patch:

```text
bun test xerxes/test/responsesApi.test.ts xerxes/test/responsesApiClient.test.ts
19 pass, 0 fail, 31 expect() calls
```

TypeScript checks also passed:

```text
bun run --cwd xerxes check
# check:runtime: tsc --noEmit -p tsconfig.json
# check:ui: tsc --noEmit -p tsconfig.ui.json
```

A scoped `git diff --check` for the source and focused test files passed with no output.
