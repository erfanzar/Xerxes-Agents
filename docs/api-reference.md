# HTTP API reference

The native HTTP surface is implemented in
[`src/typescript/src/api-server/server.ts`](../src/typescript/src/api-server/server.ts). It exposes
an OpenAI-compatible chat-completions handler that an embedding host starts with Bun's HTTP server.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness response with the configured model count. |
| `GET` | `/v1/models` | OpenAI-style list of advertised model identifiers. |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion, streamed or buffered. |

Unknown paths return a structured OpenAI-style error. Methods not accepted by an endpoint return
`405` with an `Allow` header.

## Embedding the server

Create the handler with `createOpenAiApiServer({ llm, models, ...options })`, then call
`server.listen({ hostname, port })`. The `llm` option accepts one native `LlmClient` or a resolver
for model-specific clients; `models` is the explicit allow-list exposed by `/v1/models`.

Optional protections are explicit:

- `auth` enables bearer-token validation while leaving `/health` available by default.
- `cors` emits a declared browser CORS policy.
- `rateLimit` enables an in-memory sliding-window guard.
- `maxRequestBodyBytes` bounds JSON request bodies before completion work begins.

See the generated TypeScript API pages for the full request/response contracts and option types.

## Completion behavior

The endpoint accepts standard OpenAI chat messages, tool declarations, sampling controls, and the
`stream` flag. For streaming requests it emits Server-Sent Events and ends with the expected
`[DONE]` marker. Tool calls and usage are normalized from the native streaming event model.

Provider credentials and network clients are supplied by the embedding application. The API server
does not discover secrets, start a provider, or grant tools implicitly.
