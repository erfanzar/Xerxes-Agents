# Xerxes-Agents Deep Reference Map

## 1. 6-Tsconfig Matrix (`strict: true` in all)

| Tsconfig (`check:*`) | Mod/Lib/Extra Flags | `types` | `include` & Ownership |
| --- | --- | --- | --- |
| `xerxes/tsconfig.json` (`runtime`) | `ESNext`/`bundler`/`ESNext,DOM`; `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `verbatimModuleSyntax`, `isolatedModules` | `["bun"]` | `src/**/*.ts`, `test/**/*.ts` (excl `ui`, `desktop/renderer`, `vscode`; checks `desktop/main`, `hub`, `client`, `acp`, `swarm`) |
| `xerxes/tsconfig.ui.json` (`ui`) | `ES2023`/`nodenext`/`ES2023` (no DOM) | `["bun","node"]` | `src/ui/**/*.{d.ts,ts,tsx}` (excl `__tests__`; `.js` imports) |
| `xerxes/tsconfig.desktop.json` (`desktop`) | `ES2023`/`bundler`/`ES2023,DOM`; `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes` | `["bun","node"]` | `src/desktop/**`, **`src/vscode/ui/**/*.tsx`** |
| `xerxes/tsconfig.vscode.json` (`vscode`) | `ES2023`/`bundler`/`ES2023,DOM`; `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes` | **`["node","vscode"]`** | **`src/vscode/*.ts`** (no `bun`; `ui/` -> `desktop`) |
| `xerxes/tsconfig.mobile.json` (`mobile`) | `ES2023`/`bundler`/`ES2023,DOM`; `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `verbatimModuleSyntax`, `isolatedModules` | **`[]`** | `src/mobile/**`, `src/client/**/*.ts`, `src/hub/protocol.ts`, `src/auth/usageView.ts` (excl `__tests__`; browser-only) |
| `examples/tsconfig.json` | Extends `xerxes/tsconfig.json` (`noEmit`) | `["bun"]` | `./**/*.ts` (`rootExamples.test.ts`) |

## 2. Subsystem Map (`xerxes/src/` & Repo)

- **CLI/Core/Wire**: `{cli,xerxes,index}.ts` (no `cortex/*`), `core/` (`XerxesConfig`, `XERXES_VERSION`, Windows pipe/`planSpawn`), `protocol/`, `types/`.
- **Daemon (v35)**: `daemon/{server,runtime,turnRunner,workspaceResources,paths,providerReadiness,outboundGate,compactionRunner,interactions,websocketGateway,acpBridgeHost,agyBrainBridge,swarmHost,subagentHost,goalCommand,superagent*}.ts` (WS `127.0.0.1:11996`, LRU 12, `SessionPromptSnapshots`).
- **Streaming/LLMs/Auth**: `streaming/` (`runTurn`, max 8 `concurrencySafe`), `llms/` (40+ providers), `auth/` (`CredentialStorage`, `codexAuth`, `agyRuntimeEnv`, `usageView.ts`), `routing/` (`PolicyRouter`).
- **Tools/LSP/MCP/Security**: `executors/toolRegistry.ts`, `tools/`, `lsp/` (32), `mcp/` (1k), `security/` (`PolicyEngine`, `DataBoundary`, `CommandSandbox`, `SandboxRouter`, `promptScanner`, `hubIdentityVerifier`).
- **Session/Memory/Runtime/Ext**: `session/` (`DaemonTranscriptStore` CAS+`.jsonl`), `context/` (`assembleContextLayers` vs `<turn-context>`, `ContextCompressor`), `memory/` (`AgentMemory`, `LessonStore`), `runtime/` (`bootstrap`, `projectWorkspace`, `leanPrompt`, `goal*`, `subagentWorktrees`, `reflection`), `tasks/`, `extensions/` (`SkillRegistry`, `skillLint`, `harnessLint`, 12 hooks, `ProjectPlugins`).
- **Channels/ACP/Swarms/Agents**: `{channels,cron,api-server,audit,bridge,cortex,workspace,images,local,operators,docs,maintenance}/` (15+ channels, `wx` lease, `COMMAND_REGISTRY`, `GitScm`, CDP `BrowserManager`), `acp/`, `swarm/`, `agents/` (`default/*.yaml`, `SubAgentManager`).
- **Surfaces/Hub/Repo**: `ui/`, `desktop/`, `vscode/`, `client/`, `mobile/` (PWA+WASM voice), `hub/`+`slack-broker/`, `xerxes/playground/` (6), `xerxes/skills/` (119), `docs/`, `examples/` (8), `packaging/runtime/`.

## 3. Build Dependency Graph

```text
`bun run build` = build:runtime && build:ui && build:mobile (omits desktop, vscode)
build:runtime  -> dist/cli.js + dist/{skills/,default/,sandboxShim.ts} + dist/build-id
build:ui       -> dist/ui/entry.js (react & @opentui/* external; verifyTuiBuild.ts)
build:mobile   -> dist/mobile/ (hashed assets, sw.js, 3 voice workers, 4 ort-wasm)
build:desktop  -> build:runtime -> buildDesktop.ts (preload CJS, theme.css) -> packageDesktopMac.ts
build:vscode   -> build:runtime -> buildDesktop.ts -> buildVscode.ts (CJS, webview/chat, Bun -> 5x .vsix)
release/Docker -> packaging/runtime/{package.json,bun.lock} + bin/{xerxes*,default/} + ui/entry.js + skills/
```

## 4. Version Sites (`0.8.58`: 9 Files + 4 Tests)

- **Manifests (3)**: `package.json`, `xerxes/package.json` (`releasePackage.ts`, `vscode-marketplace.yml`), `packaging/runtime/package.json` (+ `@opentui/*`/`react` & `bun.lock`: `containerDistribution`, `nativeWorkflowContract`).
- **Source & Docs (6, manual)**: `src/core/version.ts`, `src/core/config.ts:547`, `src/desktop/renderer/buildInfo.ts:14`, `src/llms/{anthropic.ts:760,mistral.ts:365}`, `docs/changelog.md:10`.
- **Tests (4)**: `xerxes/test/{installerParity:70,daemonServer:3699,desktopStore:64+753,bridgePythonParity:320}.test.ts`.

## 5. Budgets & Test Runners

- **Prompts**: system `24k` tok (`12k`/`6k`/`4k`/`2k`); `AGENTS.md`/`XERXES.md`/`CLAUDE.md` `16/32 KiB`; freshness `16/32 KiB`; git `8/12 KiB`, extra `16 KiB`; lean (`<=32k`/Bonsai) **`1,500`c** instructions, `200` tool, `80` schema.
- **Workspace `.agents/` & `.xerxes/`**: scanned by `projectWorkspace.ts` (`6,000 B`/file, `96 KiB` total, `200` files, depth `8`; `.agents/skills/` excluded).
- **Memory/Skills/Compaction/Wire**: memory `4k B`/file, `8 KiB`/`48` idx, `32 KiB` total (`16 KiB` diff, <=7d journals); `5` lessons/turn; skills `16 KiB`/`128` (119 bundled), `1 MiB`/file, `32 MiB` pass; compact `0.6`, `<swe-working-set>` `1,600`c; Daemon/ACP `16 MiB`, Hub v1 `1 MiB` (`256 KiB` target).
- **Runners**: `test:runtime` (`bun test xerxes/test/<f>.test.{ts,tsx}`, `isolatedHome.ts` — Core/Daemon/ACP/Swarm/Hub/Slack/**Desktop**/**VS Code**/Examples/Skills/CI); `test:ui` (`vitest --config xerxes/vitest.ui.config.ts`); `test:mobile` (`vitest --config xerxes/vitest.mobile.config.ts`).
