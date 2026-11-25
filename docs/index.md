# Xerxes documentation

Xerxes is a Bun-native TypeScript agent runtime built around one global v35 JSON-RPC daemon. It serves the OpenTUI terminal client, Electron desktop app, VS Code extension, Mobile Web PWA, Hub, and external ACP bridges from shared typed contracts.

Install `@xsimurgh/xerxes-agents` globally with Bun 1.3.12 or newer (the npm package named `xerxes` is unrelated):

```sh
bun add --global @xsimurgh/xerxes-agents
xerxes doctor
```

Or run it once without installing globally:

```sh
bunx @xsimurgh/xerxes-agents
```

## Quick start from source

```sh
bun install --frozen-lockfile
bun run check
bun run test
bun run build
bun run xerxes --help
```

Run `bun run xerxes doctor` to inspect local configuration and provider availability. It reports missing credentials without making a remote request.

## Surfaces

| Surface | Entry point | Purpose |
| --- | --- | --- |
| Terminal UI (OpenTUI) | `xerxes` | Interactive terminal workspace, overlays, and slash commands |
| Desktop app (Electron) | `xerxes/src/desktop/` | Multi-workspace desktop client with Files, Diff, Git, Terminal, and Immortals |
| VS Code extension | `xerxes/src/vscode/` | Single-column sidebar and editor chat integrated with VS Code diff and Source Control |
| Mobile PWA & Hub | `xerxes/src/mobile/`, `xerxes/src/hub/` | Remote approvals, live transcripts, voice calls, Autopilot, and Watch from a phone |
| Daemon & ACP | `xerxes daemon`, `xerxes acp` | Shared local v35 JSON-RPC runtime and external ACP bridge |

## Guides

- [Native documentation build](BUN_BUILD.md)
- [Configuration](configuration-guide.md)
- [Goal mode](goal-mode.md)
- [Deployment](deployment-guide.md)
- [Testing](testing-guide.md)
- [Contributing](contributing.md)
- [GitHub Actions](github-action.md)
- [System architecture](system-architecture.md)
- [Release history](changelog.md)
- [TypeScript API reference](typescript-api/index.html)

## Source layout

- `xerxes/src/`: runtime, daemon, provider transports, native tools, Hub, and client surfaces.
- `xerxes/src/ui/`: OpenTUI terminal client speaking the native v35 daemon gateway.
- `xerxes/test/`: Bun unit, protocol, surface, and integration test suites.
- `examples/`: offline-default runnable Bun examples.
