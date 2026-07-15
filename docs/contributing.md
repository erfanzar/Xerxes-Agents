# Contributing to Xerxes

Xerxes is a strict TypeScript project that runs on Bun. Keep a change focused, preserve unrelated
dirty work, and do not add a second runtime or compatibility shim that masks an unavailable
integration.

## Local checks

```sh
bun install --frozen-lockfile
bun run check
bun run test
bun run build
git diff --check
```

Run the narrowest relevant test while iterating, then the root checks once concurrent work has
settled. Use `bun test xerxes/test/<file>.test.ts` for a runtime test and
`bun run test:ui` for the terminal client.

## Change rules

- Keep public daemon protocol and persisted-session changes deliberate and covered by contract
  tests.
- Put network, credential, browser, hardware, and host execution behind explicit interfaces;
  tests must inject a fake rather than discover ambient credentials.
- Do not add Python source, tests, packaging metadata, or subprocess fallbacks to the repository.
- Make unavailable host integrations fail explicitly with useful guidance; never return a fabricated
  success response.
- Browser automation attaches to an explicitly supplied, already-running Chromium CDP endpoint.
  Do not launch a browser implicitly or place credentials in configuration fixtures.
- Do not add copied third-party visual assets or product copy. Terminal visuals should be original,
  accessible, and work at narrow widths.

See [XERXES.md](../XERXES.md) and [the system architecture](system-architecture.md) for runtime
and contract guidance.
