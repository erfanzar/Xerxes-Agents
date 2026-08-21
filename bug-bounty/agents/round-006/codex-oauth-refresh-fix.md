# Round 006 — Codex OAuth refresh endpoint/redirect hardening

## Status

Confirmed and fixed.

## Finding

`CodexSession.refresh()` constructed and sent its token request directly instead of passing the configured token URL through the shared OAuth endpoint validation. Consequently, `CODEX_REFRESH_TOKEN_URL_OVERRIDE=http://attacker.invalid/oauth/token` was accepted and the refresh token was posted over plaintext HTTP.

The direct fetch also omitted `redirect: 'manual'`. Fetch therefore used its default redirect-following behavior, allowing an OAuth token endpoint redirect to move the POST request outside the validated endpoint boundary.

## Reproduction evidence

Two regression tests were added to `xerxes/test/codexAuth.test.ts`. Before the fix:

- the insecure override test observed one request and failed with `OAuth token response missing required access_token`, proving the plaintext endpoint was contacted;
- the redirect test observed `init.redirect === undefined`, proving automatic redirect following remained enabled.

Focused pre-fix result: **44 passed, 2 failed**.

## Fix

In `xerxes/src/auth/codexAuth.ts`:

- resolve the Codex OAuth configuration through the shared hardened OAuth configuration path (`buildAuthorizeUrl`) before sending refresh credentials, which applies the shared absolute-URL and HTTPS endpoint checks;
- set `redirect: 'manual'` on the refresh fetch so 3xx responses are surfaced as failures instead of followed.

Existing Codex-specific response/error behavior remains intact, including quota messaging and invalid-grant CLI-session healing.

## Verification

- `bun test xerxes/test/codexAuth.test.ts`: **46 passed, 0 failed, 126 assertions**.
- `git diff --check -- xerxes/src/auth/codexAuth.ts xerxes/test/codexAuth.test.ts`: passed.
- `bun run --cwd xerxes check`: blocked by an unrelated existing TypeScript error in `src/security/fileSync.ts:166`: `TS2554: Expected 2 arguments, but got 4.` The owned files produced no reported check error before TypeScript stopped.
