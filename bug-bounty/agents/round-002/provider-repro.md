# Round 002 — Provider reproduction

## Verdict

Two issues reproduced against the current `main` worktree.

### 1. Responses HTTP-200 `status: "failed"` is accepted

**Affected path:** `ResponsesApiClient.complete()` → `parseResponsesCompletion()` in `xerxes/src/llms/client.ts`.

Exact case:

```ts
test('Responses completion rejects an HTTP-200 failed response', async () => {
  const client = new ResponsesApiClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async () => Response.json({
      id: 'resp_1',
      status: 'failed',
      error: { code: 'server_error', message: 'Model exploded' },
      output: [],
    }),
  })

  await expect(client.complete({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'hi' }],
  })).rejects.toThrow('server_error')
})
```

**Observed:** resolves to

```json
{"content":"","toolCalls":[],"finishReason":"failed"}
```

The HTTP status check cannot catch Responses API semantic failures carried in a successful HTTP envelope. Streaming is not affected: `ResponsesEventTranslator` already throws for `response.failed` and `error` events.

### 2. `extraBody` can overwrite protected chat payload fields

**Affected path:** `openAiCompatiblePayload()` → `addSampling()`; `Object.assign(payload, request.extraBody)` runs after protected fields are constructed.

Exact case:

```ts
test('extraBody cannot overwrite protected OpenAI-compatible fields', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body))
      return Response.json({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] })
    },
  })

  await client.complete({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'hi' }],
    extraBody: {
      model: 'overwritten',
      messages: [],
      stream: true,
      temperature: 99,
      custom_flag: 'kept',
    },
  })

  expect(payload).toMatchObject({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'hi' }],
    stream: false,
    custom_flag: 'kept',
  })
  expect(payload?.temperature).not.toBe(99)
})
```

**Observed payload:**

```json
{"model":"overwritten","messages":[],"stream":true,"temperature":99,"custom_flag":"kept"}
```

This permits request/model substitution and makes `complete()` send a streaming request while trying to parse it as JSON.

## API compatibility assessment

- Keeping `CompletionRequest.extraBody` is source-compatible and preserves legitimate provider extensions.
- Protect at least adapter-owned routing/protocol fields (`model`, `messages`/`input`, `stream`, tools/tool choice, and adapter-managed sampling fields) while continuing to merge unknown extension keys.
- This intentionally changes behavior only for callers relying on collisions; that behavior violates the typed request fields and is unsafe to preserve.
- Responses payloads currently ignore `extraBody` entirely. Reproduction retained canonical `model`, `input`, and `stream`, but also dropped `custom_flag`. If Responses is intended to honor the documented provider-specific `extraBody` contract, add a filtered merge there too; otherwise narrow the public documentation/type contract by transport.
- Rejecting HTTP-200 failed Responses is behaviorally compatible for successful responses and aligns non-streaming behavior with the existing streaming error contract.

## Evidence

Executed a Bun in-memory reproduction on 2026-08-21. Output:

```text
CASE1_RESOLVED {"content":"","toolCalls":[],"finishReason":"failed"}
CASE2_CHAT_BODY {"model":"overwritten","messages":[],"stream":true,"temperature":99,"custom_flag":"kept"}
CASE3_RESPONSES_BODY {"model":"gpt-4o","input":[{"role":"user","content":"hi"}],"stream":false}
```

No production or test files were edited.
