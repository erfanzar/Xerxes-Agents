---
name: audit-surface-parity
description: Audit a feature or diff across Daemon, OpenTUI, Electron Desktop, VS Code, and Mobile/Hub for protocol sync, bridgeApi/client parity, browser purity, CSS cascade/WCAG AA, and SSR tests.
tags: [surfaces, ui, desktop, vscode, mobile, hub, audit]
---

# Audit Cross-Surface Parity & Boundaries

Audit `$ARGUMENTS` (default: `git diff --name-only`) across **Daemon**, **OpenTUI**, **Electron Desktop**, **VS Code**, and **Mobile PWA / Hub**.

## 1. Protocol, Adapter & Isolation Parity

1. **Daemon v35 Sync**: Verify `xerxes/src/ui/PROTOCOL.md` matches `xerxes/src/daemon/server.ts` and `turnRunner.ts` (including `*(bridge only)*` tags). Check `gatewayTypes.ts`, `gatewayAdapter.ts`, `gatewayClient.ts` (`SNAPSHOT_COVERED_EVENTS`), and `desktop/main/ipcEventBatcher.ts` (FIFO flush before lifecycle/interactive events).
2. **Prompt Filters & Shared Views**: Keep `looksLikeInternalUserPrompt` (`src/ui/gatewayAdapter.ts`), `isLegacyHarnessPrompt` (`src/desktop/renderer/transcriptContent.ts`), and daemon `looksLikeInternalReplayMessage` aligned. Keep `usage.report` formatting in `xerxes/src/auth/usageView.ts`.
3. **Hard Boundaries**: Zero direct `src/swarm/` imports in TUI, Desktop, VS Code, or Mobile UI code. Host-only RPCs stay in `HOST_ONLY_DAEMON_METHODS` (`desktop/main/windowRoutes.ts`); Hub uses `HUB_DAEMON_METHODS` (`hub/runtimePool.ts`) + `HUB_METHODS`/`HUB_COMMAND_ACTIONS` (`hub/protocol.ts`), never raw daemon passthrough.

## 2. Desktop <-> VS Code Sharing Parity

1. Any channel in `xerxes/src/desktop/bridgeApi.ts` is handled in **both** Electron (`desktop/main/ipc.ts`/`main.ts`) and VS Code (`src/vscode/panelHost.ts`), or explicitly optional-checked (like `cockpitApiFrom`).
2. Host divergence uses `isVscodeHost()` (`data-host="vscode"`); `PanelHost` stays pure of the `vscode` module; `SHEETS` in `src/vscode/extension.ts` matches `xerxes/scripts/buildVscode.ts`.
3. `src/vscode/*.ts` uses zero `Bun.*` types (`tsconfig.vscode.json`: `types: ["node", "vscode"]`); `src/vscode/ui/**/*.tsx` compiles under `tsconfig.desktop.json`.

## 3. OpenTUI & Cross-Platform Contracts

1. Every OpenTUI `.tsx` file declares `/** @jsxImportSource @opentui/react */`.
2. `src/ui/opentui/entry.tsx` imports `@opentui/core` before `@opentui/react` sequentially; `buildTui.ts` keeps `react`, `react-reconciler`, and `@opentui/*` external; `verifyTuiBuild.ts` assertions remain valid.
3. Every `OverlayState` flag in `src/ui/app/overlayStore.ts` updates `OVERLAY_FLAG_KEYS`, `OVERLAY_BLOCKS_BACKGROUND_HOTKEYS`, `OVERLAY_CLOSED_VALUES`, and `resetFlowOverlays()`.
4. `src/core/hostPlatform.ts` and `src/ui/lib/hostPlatform.ts` remain aligned (`windowsSupport.test.ts`).

## 4. Browser Purity (`types: []`) & Build Hygiene

1. `src/mobile/**`, `src/client/**`, `src/hub/protocol.ts`, and `src/auth/usageView.ts` contain **zero** Node (`node:*`, `Buffer`, `process`) or Bun (`Bun.*`) imports/globals (`tsconfig.mobile.json`).
2. Desktop preload builds as CJS (`format: 'cjs'`), Desktop/Mobile builds define production `NODE_ENV`, and Mobile voice build leaks zero `onnxruntime-node` or `sharp` (`voiceCallBuild.test.ts`).

## 5. Design System, CSS Cascade & WCAG AA

1. Edit palette tokens in `src/ui/theme.ts` (`NOCTURNE_DARK`/`NOCTURNE_LIGHT`) + `src/desktop/tokens.ts` (`TOKEN_NAMES`); NEVER hand-edit generated `dist/desktop/renderer/theme.css`.
2. Preserve load order: `theme.css` -> `xterm.css` -> `app.css` -> `atelier.css` -> `layout.css` -> `vibes.css` -> `neo.css` (last in Desktop `index.html`, `buildDesktop.ts`, `previewDesktop.ts`) -> `vscode.css` (last in VS Code `extension.ts`).
3. In `neo.css`, every selector starts with `:root[data-style='neo']` and includes `:not(#neo)`, every `@keyframes` starts with `neo-`, and text/surface pairs plus `--x-on-accent` on `--x-accent` pass WCAG AA `>= 4.5:1` in dark and light.

## 6. Zero-DOM SSR Tests & Verification

Confirm `.test.tsx` suites use `react-dom/server` (`renderToStaticMarkup`) + pure reducers (no `jsdom`/`happy-dom`) in the right runner:

```bash
bun run check
bun test xerxes/test/daemonProtocolContract.test.ts xerxes/test/windowsSupport.test.ts xerxes/test/desktopTokens.test.ts xerxes/test/desktopNeoStyle.test.ts xerxes/test/vscodePanelHost.test.ts xerxes/test/vscodeChatView.test.tsx
bun run test:ui && bun run --cwd xerxes test:mobile
```
Add `build:ui`, `build:mobile`, `build:desktop`, or `build:vscode` when bundling/assets changed. Report **Critical**, **Warning**, and **Suggestion** findings with `file:line` citations and a 5-surface parity matrix.
