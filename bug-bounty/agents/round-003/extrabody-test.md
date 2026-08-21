# Round 003 — OpenAI chat `extraBody` canonical-field protection

## Confirmed regression

`openAiCompatiblePayload()` builds canonical chat-completions fields and then `addSampling()` runs:

```ts
Object.assign(payload, request.extraBody)
```

This lets provider extensions overwrite canonical request semantics including `model`, `messages`, `stream`, sampling controls, and `stop`. Fields assigned later happen to survive (`tools`, `tool_choice`, and streaming `stream_options`), but that ordering is accidental and does not protect the request consistently.

## Test-only patch

Added `xerxes/test/openAiExtraBodyProtection.test.ts` with observable fetch-payload coverage for both `complete()` and `stream()`:

- canonical `model`, `messages`, `stream`, sampling fields, tools, and tool choice must win;
- canonical streaming `stream_options.include_usage` must win;
- legitimate provider extension keys (`chat_template_kwargs`, `service_tier`) must remain on the wire.

Current implementation evidence:

```text
bun test xerxes/test/openAiExtraBodyProtection.test.ts
0 pass
2 fail
```

The failures directly show `extraBody` replacing canonical `model`, `messages`, `stream`, `temperature`, `max_tokens`, `top_p`, penalties, and `stop`.

## Proposed production patch (not applied due to shared ownership)

In `xerxes/src/llms/client.ts`, define one protected key set for every field owned by the chat payload builder, then copy only non-protected extension entries:

```ts
const OPENAI_CHAT_PROTECTED_BODY_KEYS = new Set([
  'model',
  'messages',
  'stream',
  'stream_options',
  'temperature',
  'max_tokens',
  'top_p',
  'frequency_penalty',
  'presence_penalty',
  'stop',
  'reasoning_effort',
  'thinking_budget',
  'tools',
  'tool_choice',
  'top_k',
  'min_p',
  'repetition_penalty',
])

function addOpenAiChatExtensions(
  payload: Record<string, unknown>,
  extraBody: Readonly<Record<string, unknown>> | undefined,
): void {
  if (!extraBody) return
  for (const [key, value] of Object.entries(extraBody)) {
    if (!OPENAI_CHAT_PROTECTED_BODY_KEYS.has(key)) payload[key] = value
  }
}
```

Call this from `openAiCompatiblePayload()` after all canonical fields have been assembled, and remove the raw `Object.assign` from `addSampling()`. This keeps sampling translation focused, protects even conditionally absent canonical keys, and preserves unknown provider extension keys. If the project prefers explicit rejection over filtering, throw `ConfigurationError('extraBody', ...)` listing collisions; the authored tests currently specify the backward-compatible filtered-merge behavior.
