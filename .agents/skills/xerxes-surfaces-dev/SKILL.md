---
name: xerxes-surfaces-dev
description: Use when building or modifying OpenTUI terminal UI, Electron Desktop, VS Code extension, Mobile Web PWA, shared src/client modules, or Hub, Supervisor, and Slack surfaces.
version: 1.0
author: Xerxes-Agents
tags: [xerxes, tui, desktop, vscode, mobile, hub, ui]
---

# Xerxes Client Surfaces & Hub Development

Workflow for OpenTUI, Electron Desktop, VS Code, Mobile PWA, shared `src/client/`, and Hub/Supervisor/Slack. See `.agents/rules/client-surfaces-and-hub.md` and `.agents/references/projects-map.md`.

## When to use

- Modifying `xerxes/src/{ui,desktop,vscode,mobile,client,hub,slack-broker}/`.

## Surface 1: OpenTUI Terminal UI (`xerxes/src/ui/`)

- **JSX & tsconfig**: Begin every OpenTUI `.tsx` file with `/** @jsxImportSource @opentui/react */`. `tsconfig.ui.json` uses `moduleResolution: "nodenext"` (`.js` extensions required) and no DOM lib.
- **Overlays (`src/ui/app/overlayStore.ts`)**: Adding a flag to `OverlayState` requires entries in `OVERLAY_FLAG_KEYS`, `OVERLAY_BLOCKS_BACKGROUND_HOTKEYS` (`agents: false` so F6 toggles it; others `true`), and `OVERLAY_CLOSED_VALUES` (for `dismissOverlayForCtrlC`). Decide whether `resetFlowOverlays()` clears it on turn end, build dialogs on `src/ui/opentui/dialogChrome.tsx`, and always restore prior screen/transcript state on cancel or error.
- **Bundle & startup invariants**: `entry.tsx` imports `@opentui/core` before `@opentui/react` sequentially (never `Promise.all`). `xerxes/scripts/buildTui.ts` externalizes `react`, `react/jsx-*`, `react-reconciler`, `@opentui/core`, `@opentui/react`, and all 8 `@opentui/core-*` platform packages. `xerxes/scripts/verifyTuiBuild.ts` asserts literals in `dist/ui/entry.js` (`'Many agents, one terminal.'`, `'START WITH'`, `'createRequire as __cr'`, `'@opentui/core'`, and no leading shebang). Keep `src/core/hostPlatform.ts` and `src/ui/lib/hostPlatform.ts` aligned (`windowsSupport.test.ts`).

## Surface 2: Electron Desktop & VS Code (`src/desktop/`, `src/vscode/`)

- **Shared bridge parity**: Both shells share `src/desktop/bridgeApi.ts`, `main/daemon.ts`, `renderer/store.ts`, core views, and stylesheets. Every new `window.xerxes` channel in `bridgeApi.ts` must be handled in **both** `src/desktop/main.ts` and `src/vscode/panelHost.ts` (`PanelHost` stays pure of `vscode` imports for headless tests) or safely optional-checked (like `cockpitApiFrom`). Branch shell-only behavior with `isVscodeHost()`.
- **IPC batching & preload**: Route streaming/telemetry events through `DaemonEventBatcher` (`src/desktop/main/ipcEventBatcher.ts`, 16ms / max-128 coalescing, FIFO flush before lifecycle events). Keep host-owned RPCs in `HOST_ONLY_DAEMON_METHODS` (`windowRoutes.ts`). `src/desktop/preload.ts` MUST build as `format: 'cjs'`; keep `NODE_ENV="production"` (`buildDesktop.ts` fails if React DevTools banner leaks).
- **Tokens & CSS cascade**: Never hand-edit `theme.css` — edit `NOCTURNE_DARK`/`NOCTURNE_LIGHT` in `src/ui/theme.ts` and `TOKEN_NAMES` in `src/desktop/tokens.ts` (`desktopTokens.test.ts`). Keep exact stylesheet order (`desktopNeoStyle.test.ts`): `theme.css` -> `xterm.css` -> `app.css` -> `atelier.css` -> `layout.css` -> `vibes.css` -> `neo.css` (last in Desktop `index.html`, `buildDesktop.ts`, `previewDesktop.ts`) -> `vscode.css` (last in VS Code `extension.ts`; keep `SHEETS` synced with `xerxes/scripts/buildVscode.ts`). In `neo.css`, prefix every selector with `:root[data-style='neo']` + `:not(#neo)`, prefix `@keyframes` with `neo-`, and keep every text/surface pair + `--x-on-accent` on `--x-accent` at **WCAG AA >= 4.5:1**.
- **Typecheck & build order**: `tsconfig.desktop.json` owns `src/desktop/**/*.{ts,tsx}` AND `src/vscode/ui/**/*.tsx`; `tsconfig.vscode.json` (`types: ["node", "vscode"]`, no `bun` types) owns `src/vscode/*.ts`. Run `build:runtime` before `buildDesktop.ts`, and both before `buildVscode.ts`. Preview UI fixtures via `bun xerxes/scripts/previewDesktop.ts --style=vibes` (or `--style=neo --theme=light`).

## Surface 3: Mobile PWA, Shared Client & Hub/Supervisor/Slack

- **`types: []` browser purity**: `tsconfig.mobile.json` enforces zero Node/Bun globals across `src/mobile/**`, `src/client/**`, `src/hub/protocol.ts`, and `src/auth/usageView.ts`. Use Web APIs (`globalThis.crypto`, `TextEncoder`, `fetch`, `WebSocket`) only.
- **Hub, Supervisor & Slack**: Hub v1 (`src/hub/protocol.ts`) uses strict `HUB_METHODS` / `HUB_COMMAND_ACTIONS` allowlists (1 MiB cap, never raw daemon passthrough, never trust `X-Forwarded-*`). Keep `assertSupervisorToolCeiling` blocking shell/workflow/generic RPC/approval tools, `fenceUntrusted` on task data, `AGENT_SCOPE_CEILING` human-only limits, and Slack (`src/hub/slack/*`, `src/slack-broker/`) restricted to approved 1:1 user-token DMs.
- **Voice WASM build checks**: `kokoro-js` and `@huggingface/transformers` must resolve to one `@huggingface/transformers` root; `buildMobile.ts` copies 4 `ort-wasm-simd-threaded*.{wasm,mjs}` files and fails if `onnxruntime-node` or `sharp` leaks (`voiceCallBuild.test.ts`). Never use WebGPU for q8 Moonshine, int8 ONNX files, or COEP headers (single-threaded WASM only).

## Pitfalls

- **Zero DOM test libs**: All React `.test.tsx` tests across Desktop, VS Code, and Mobile use `react-dom/server` (`renderToStaticMarkup`) and pure reducers — no jsdom/happy-dom.
- **Running macOS `.app`**: Set `XERXES_DESKTOP_PACKAGE_DIR` when packaging Desktop while an installed `.app` is running.

## Verification commands

```bash
# OpenTUI
bun run --cwd xerxes check:ui && bun run test:ui && bun run build:ui && bun run smoke

# Desktop & VS Code (tests run on bun test, not Vitest)
bun run --cwd xerxes check:desktop && bun run --cwd xerxes check:vscode
bun test xerxes/test/desktop*.test.* xerxes/test/vscode*.test.*
bun run --cwd xerxes build:desktop   # or: bun run build:vscode

# Mobile PWA, Shared Client & Hub/Slack
bun run --cwd xerxes check:mobile && bun run --cwd xerxes test:mobile && bun run --cwd xerxes build:mobile
bun test xerxes/test/hub*.test.ts xerxes/test/slackBroker.test.ts
```
