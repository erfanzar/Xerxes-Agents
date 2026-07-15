---
name: add-llm-provider
description: Add a Bun-native LLM provider entry with routing, pricing, headers, limits, and Bun tests.
version: 2.0.0
tags: [llm, provider, registry, typescript, bun, xerxes]
required_tools: [ReadFile, WriteFile, FileEditTool]
---

# When to use

Use this skill when adding a provider that can use one of Xerxes' native
transports: OpenAI-compatible HTTP, Anthropic, or the local Claude Code
transport. Use a separate native client implementation when a provider needs a
new wire protocol.

# How to use

## 1. Inspect the native registry and client boundary

Read:

- `src/typescript/src/llms/providerRegistry.ts`
- `src/typescript/src/llms/client.ts`
- `src/typescript/test/providerRegistry.test.ts`

`ProviderConfig` uses native camel-case fields: `apiKeyEnv`, `baseUrl`,
`contextLimit`, `models`, and `transport`. Provider selection supports explicit
`provider/model` notation before model-prefix routing.

## 2. Add the provider metadata

Add a `provider(...)` entry to `PROVIDERS` in
`src/typescript/src/llms/providerRegistry.ts`:

```ts
myprovider: provider('myprovider', 'openai', {
  apiKeyEnv: 'MYPROVIDER_API_KEY',
  baseUrl: 'https://api.myprovider.example/v1',
  contextLimit: 128_000,
  models: ['myprovider-chat-1', 'myprovider-chat-2'],
}),
```

Use an existing `ProviderTransport` only when the client can actually speak the
provider's protocol. A local endpoint that needs no key can omit `apiKeyEnv` and
use `defaultApiKey` only when the endpoint requires a non-secret placeholder.

## 3. Complete routing and accounting

Update every applicable native registry location in the same change:

1. Add model rates to `COSTS` as `[inputUsdPerMillion,
   outputUsdPerMillion]`. Use `[0, 0]` only when the model is genuinely free
   or its price is intentionally unknown.
2. Add distinctive model prefixes to `PREFIX_MAP`. It is sorted longest-first;
   avoid collisions with existing providers.
3. Add a `MODEL_CONTEXT_LIMITS` entry only when a model differs from the
   provider-wide `contextLimit`.
4. Add a `providerDefaultHeaders()` branch only when the provider explicitly
   requires a header.

Keep model spellings consistent between the provider entry, cost table, prefix
rules, context overrides, documentation, and tests.

## 4. Implement a new transport when required

Do not pretend a non-compatible endpoint is OpenAI-compatible. Add an explicit
native client in `src/typescript/src/llms/`, make the transport discriminant
and `createLlmClient()` handle it, and use a deterministic injected fetch or
transport in tests. Surface provider errors with redacted context; never log an
API key.

## 5. Add Bun tests and verify

Extend `src/typescript/test/providerRegistry.test.ts` for prefix resolution,
explicit provider notation, context limit, pricing, and required headers. Add a
client test when transport behavior changes.

Run:

```bash
bun test src/typescript/test/providerRegistry.test.ts
bun test src/typescript/test/llmTypesParity.test.ts
bun run --cwd src/typescript check
```

## Common pitfalls

- `transport` is a native protocol choice, not a marketing label.
- The explicit `provider/model` form takes precedence over prefixes; test both.
- Do not use environment discovery in tests. Pass an environment record or
  provider override explicitly.
- Prefix collisions route the wrong client. Add the narrowest distinctive
  prefix and inspect the sorted map.
- Keep costs and context limits as source-controlled metadata, not ad-hoc
  runtime configuration.
