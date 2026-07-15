# Testing guide

The active test path uses Bun's test runner and strict TypeScript checking.

```sh
bun install --frozen-lockfile
bun run check
bun run test
bun run build
```

The root commands cover the native runtime and the OpenTUI terminal client. Use the smallest
relevant test while iterating:

```sh
# Runtime test file
bun test xerxes/test/streamingLoopParity.test.ts

# Terminal client tests
bun run test:ui

# Native integration-style local checks
bun run real-use
bun run swarm
```

## Test design

- Use Bun's `test` and `expect` APIs with deterministic injected clocks, file systems, clients,
  and process ports where a subsystem touches the host.
- Exercise public contracts: daemon v35 frames, persisted session records, OpenAI-compatible
  responses, stream events, YAML shapes, and tool schemas.
- Keep credentialed, browser, hardware, network, and third-party channel tests opt-in. A skipped
  host boundary is preferable to a fabricated success.
- Add a regression before removing a behavior oracle. A native slice needs a focused test and a
  consuming entrypoint check before its old implementation can be deleted.

## Delivery check

Run this from the repository root once parallel changes have settled:

```sh
bun run check
bun run test
bun run build
git diff --check
```

`bun run xerxes --help` and `bun run xerxes doctor` are fast CLI smoke checks. `doctor` may warn
about an intentionally absent provider key; that is not a successful remote-provider test.
