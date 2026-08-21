# Round 002 — protocol reproduction

Date: 2026-08-21

## Findings

### 1. ACP shutdown can hang on an active non-runner prompt — reproduced

**Result:** confirmed.

A deterministic in-process reproduction used `AcpServer({ promptHandler })` with a handler that never resolves, submitted `session/prompt`, then submitted `shutdown`. After 300 ms the serve promise was still pending, although shutdown request `id: 2` had received `{ "ok": true }`.

Observed output:

```json
{"result":"TIMEOUT","output":[{"jsonrpc":"2.0","id":2,"result":{"ok":true}}]}
```

Root cause:

- `xerxes/src/acp/transport.ts`: shutdown sets `running = false`, replies successfully, and calls `server.shutdown()`.
- The transport `finally` then awaits `Promise.allSettled([...this.workers])` without a bound.
- `xerxes/src/acp/server.ts`: `shutdown()` can cancel prompts only through the optional runner (`this.runner?.cancel(...)`). A caller-supplied `promptHandler` has no abort/cancel contract, so its worker can remain pending forever.

The existing test `ACP EOF aborts active runner prompts before awaiting transport workers` covers only `AcpAgentRunner`, whose LLM stream honors an `AbortSignal`; it does not cover the supported `promptHandler` construction path or an explicit `shutdown` frame.

**Expected policy:** acknowledged shutdown/EOF must not leave the stdio protocol process indefinitely waiting on an uncooperative active worker. The source comment explicitly says shutdown should abort active prompts “so transport shutdown cannot wait forever.”

**Focused regression tests:**

1. In `xerxes/test/acpTransport.test.ts`, start a never-resolving custom `promptHandler`, send `session/prompt` followed by `shutdown`, and assert `serveACPStdio` settles within a short bound and emits the shutdown response.
2. Add the equivalent EOF case for a custom handler, distinct from the existing cancellable-runner test.
3. Preserve the existing runner cancellation assertion to ensure cooperative prompts are still aborted and cleaned up.

### 2. MCP OAuth accepts plaintext remote endpoints — reproduced

**Result:** confirmed.

A deterministic injected-fetch reproduction supplied:

```ts
{
  authorizeUrl: 'http://oauth.example.test/authorize',
  tokenUrl: 'http://oauth.example.test/token',
  clientId: 'client',
}
```

Observed output:

```json
{"authorize":"http://oauth.example.test/authorize?response_type=code&client_id=client&redirect_uri=http%3A%2F%2F127.0.0.1%3A5454%2Fcallback&state=state&code_challenge=challenge&code_challenge_method=S256","requestUrl":"http://oauth.example.test/token","accessToken":"plaintext-accepted"}
```

Root cause: `xerxes/src/mcp/oauth.ts` validates only that endpoint strings are non-empty. `buildAuthorizeUrl()` checks absolute URL syntax but not scheme, and `requestToken()` sends authorization codes, PKCE verifiers, refresh tokens, and returned access tokens to the configured `tokenUrl` without requiring TLS.

The loopback default redirect URI (`http://127.0.0.1:5454/callback`) is a distinct native-app exception and should remain allowed. The remote authorization and token endpoints are the affected values.

**Existing expected security policy:** `xerxes/src/channels/oauth.ts` already normalizes both `authorizeUrl` and `tokenUrl` through `httpsUrl()` and throws `<field> must use HTTPS`. MCP HTTP transport separately permits both HTTP and HTTPS for local/operator-owned MCP servers, but that transport policy should not be applied to credential-bearing OAuth endpoints.

Current MCP OAuth coverage (`xerxes/test/mcpHardening.test.ts`) checks refresh-token scope retention using HTTPS fixtures; there is no plaintext rejection test.

**Focused regression tests:** in `xerxes/test/mcpHardening.test.ts` (or a dedicated MCP OAuth test), assert that both `buildAuthorizeUrl()` and `exchangeCode()`/`refreshToken()` reject remote `http://` authorization/token URLs before invoking fetch, while HTTPS endpoints succeed and the loopback HTTP redirect remains accepted.

## Verification

- `bun test xerxes/test/acpTransport.test.ts` — **9 pass, 0 fail**.
- `bun test xerxes/test/mcpHardening.test.ts` — **8 pass, 0 fail**.
- Both standalone reproductions exited successfully and produced the outputs quoted above.

No production or test source was edited.
