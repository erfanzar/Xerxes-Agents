# Xerxes documentation

Xerxes is a Bun-native TypeScript runtime for building, serving, and operating LLM-powered
agents. The public command, daemon protocol, HTTP surface, and OpenTUI terminal client share typed
contracts from `xerxes/`.

The npm package is `xerxes-bun` (the package named `xerxes` is unrelated). With Bun 1.3.12 or
newer installed, either install it globally with `bun add --global xerxes-bun` or run it directly
with `bunx xerxes-bun`.

## Start here

```sh
bun install --frozen-lockfile
bun run check
bun run test
bun run build
bun run xerxes --help
```

Use `bun run xerxes doctor` to verify the local runtime. It reports missing optional provider
credentials without attempting a remote request.

## Guides

- [Native documentation build](BUN_BUILD.md)
- [Contributing](contributing.md)
- [Configuration](configuration-guide.md)
- [Deployment](deployment-guide.md)
- [Testing](testing-guide.md)
- [System architecture](system-architecture.md)
- [API reference](api-reference.md)
- [Release history](changelog.md)

## Source layout

- `xerxes/src/` — runtime, daemon, provider, tool, service, and OpenTUI implementation.
- `xerxes/src/ui/` — OpenTUI terminal client that speaks the native v35 gateway.
- `xerxes/test/` — Bun unit, protocol, and integration coverage.
- `examples/` — runnable Bun examples.

The generated TypeScript API reference is available in the built site under `typescript-api/`.
