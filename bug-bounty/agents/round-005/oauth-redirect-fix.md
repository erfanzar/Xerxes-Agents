# MCP OAuth token redirect hardening — Round 005

## Finding

OAuth token requests used the Fetch default redirect mode (`follow`). HTTP 307 and 308 responses preserve the POST method and body, so an OAuth authorization code, PKCE verifier, or refresh token could be forwarded to a redirect target selected by the token endpoint.

## Fix

- Set `redirect: 'manual'` on the shared token request in `xerxes/src/mcp/oauth.ts`.
- Redirect responses are therefore returned to the OAuth client without issuing a second request.
- Existing non-success handling rejects the response as `OAuth token request failed with HTTP <status>`; response bodies and redirect locations are not exposed in that error.
- The shared request path protects both authorization-code exchange and token refresh.

## Focused coverage

Added tests in `xerxes/test/mcpHardening.test.ts` for:

- a 307 response during authorization-code exchange, asserting manual redirect mode, one fetch invocation, and HTTP 307 rejection;
- a 308 response during refresh, asserting manual redirect mode, one fetch invocation, and HTTP 308 rejection.

## Verification

- `bun test xerxes/test/mcpHardening.test.ts` — 12 pass, 0 fail, 57 assertions.
- `bun run --cwd xerxes check` — runtime and UI TypeScript checks passed.
- `git diff --check -- xerxes/src/mcp/oauth.ts xerxes/test/mcpHardening.test.ts` — passed.
