# MCP OAuth endpoint hardening — Round 003

## Fix

- MCP OAuth authorization and token endpoints now require HTTPS by default.
- Added `OAuthConfig.allowInsecureLoopback` as an explicit development opt-in.
- The opt-in permits HTTP only for exact loopback hosts: `localhost`, `127.0.0.1`, and `::1`; private/LAN and remote HTTP hosts remain rejected.
- Endpoint validation occurs during config resolution, before an authorization URL is returned or a token request is sent.

## Focused coverage

Added tests in `xerxes/test/mcpHardening.test.ts` for:

- remote HTTP authorization endpoint rejection;
- remote HTTP token endpoint rejection before fetch invocation;
- explicit localhost/IPv6-loopback HTTP acceptance;
- private-network HTTP rejection despite the loopback opt-in.

## Verification

- `bun test xerxes/test/mcpHardening.test.ts xerxes/test/authOAuth.test.ts` — 19 pass, 0 fail.
- `bun run --cwd xerxes check` — runtime and UI TypeScript checks passed.
- `git diff --check -- xerxes/src/mcp/oauth.ts xerxes/test/mcpHardening.test.ts` — passed.
