# Client Surfaces & Hub Rules

Applies to `xerxes/src/{ui,desktop,vscode,client,mobile,hub,slack-broker}/` and shared modules (`src/hub/protocol.ts`, `src/auth/usageView.ts`).

## 1. Typecheck & Browser-Purity Boundaries

- **Strict browser-only `types: []` (`tsconfig.mobile.json`)**: `src/mobile/**`, `src/client/**`, `src/hub/protocol.ts`, and `src/auth/usageView.ts` MUST contain zero Node (`node:*`, `Buffer`, `process`) or Bun (`Bun.*`) imports/globals; use Web APIs (`globalThis.crypto`, `TextEncoder`, `fetch`, `WebSocket`).
- **VS Code split**: `tsconfig.vscode.json` (`types: ["node", "vscode"]`, no `bun`) checks ONLY `src/vscode/*.ts`. `src/vscode/ui/**/*.tsx` is owned by `tsconfig.desktop.json`.
- **Desktop main overlap**: `src/desktop/main/**/*.ts` is checked by BOTH `tsconfig.desktop.json` and `xerxes/tsconfig.json` (which excludes `src/desktop/renderer`, not `main/`).
- **OpenTUI tsconfig**: `tsconfig.ui.json` uses `nodenext` (`.js` local imports), `lib: ["ES2023"]` (no DOM), and omits `exactOptionalPropertyTypes`/`noUncheckedIndexedAccess`/`verbatimModuleSyntax`.
- **Single usage source**: Keep `src/auth/usageView.ts` as the sole `usage.report` parser/formatter across TUI, Desktop, and Mobile.

## 2. Protocol, Bridge & Hub v1 Boundaries

- **Daemon v35 parity**: Keep `ui/PROTOCOL.md`, `ui/gatewayTypes.ts`, `ui/gatewayAdapter.ts` (`looksLikeInternalUserPrompt`), and `desktop/renderer/transcriptContent.ts` (`isLegacyHarnessPrompt`) synced with daemon events and replay filters.
- **Desktop IPC**: Keep renderer sandboxed (`nodeIntegration: false`, `contextIsolation: true`, strict CSP, validated `structuredClone` preload). Block `HOST_ONLY_DAEMON_METHODS` (`provider.remote.reply`, `connection.lease`) in `windowRoutes.ts`. Preserve `DaemonEventBatcher` FIFO ordering (flush deltas before interactive/lifecycle events).
- **Desktop ⇄ VS Code parity**: Handle every new `src/desktop/bridgeApi.ts` channel in BOTH `src/desktop/main.ts` and `src/vscode/panelHost.ts` (keep `panelHost.ts` pure of `vscode` for headless tests) OR optional-check safely. Gate VS Code behavior via `isVscodeHost()` (`data-host="vscode"`). Sync `SHEETS` between `src/vscode/extension.ts` and `scripts/buildVscode.ts`.
- **Hub v1 (`src/hub/protocol.ts`)**: Never treat Hub v1 as a daemon v35 passthrough. Enforce `HUB_METHODS`, `HUB_COMMAND_ACTIONS`, and internal `HUB_DAEMON_METHODS` / `HUB_MODEL_DAEMON_METHODS` / `HUB_LESSON_DAEMON_METHODS`; 1 MiB frame cap; actor from authenticated session; fail closed without `identity`; ignore `X-Forwarded-*`.
- **Supervisor ceiling**: `assertSupervisorToolCeiling` blocks shell, workflow, generic RPC, and approval tools; fence task payloads via `fenceUntrusted`; honor `AGENT_SCOPE_CEILING` (human-only `permission.respond`/`grants.manage`).
- **SSH & Browser Tab**: Enforce `known_hosts` fingerprint confirmation (no `ControlMaster` reuse); `DesktopProviderForwarding` revokes on transport loss and never writes credentials to remote disk; `browserTab.ts` uses an ephemeral partition (`xerxes-browser-*`), no preload/Node/downloads/popups, `http(s)` only, and routes SSH traffic including loopback (`<-loopback>`) through SOCKS5.

## 3. OpenTUI Mechanics & Build Traps

- Every OpenTUI `.tsx` file MUST declare `/** @jsxImportSource @opentui/react */`. Import `@opentui/core` before `@opentui/react` sequentially (never `Promise.all`).
- Adding a flag to `OverlayState` (`src/ui/app/overlayStore.ts`) requires entries in `OVERLAY_FLAG_KEYS`, `OVERLAY_BLOCKS_BACKGROUND_HOTKEYS`, and `OVERLAY_CLOSED_VALUES`. Preserve `resetFlowOverlays()` and never let a slash picker erase a transcript permanently.
- Keep `react`, `react/jsx-*`, `react-reconciler`, `@opentui/core`, `@opentui/react`, and all 8 `@opentui/core-*` packages external in `scripts/buildTui.ts`. Sync welcome copy edits with `scripts/verifyTuiBuild.ts` and `src/ui/__tests__/buildScript.test.ts` (`'Many agents, one terminal.'`, `'START WITH'`, `'createRequire as __cr'`).

## 4. Design System, CSS Cascade, Packaging & SSR Tests

- **Tokens**: Never hand-edit generated `dist/desktop/renderer/theme.css`; edit `NOCTURNE_DARK`/`NOCTURNE_LIGHT` in `src/ui/theme.ts` and `TOKEN_NAMES` in `src/desktop/tokens.ts` (`desktopTokens.test.ts`).
- **Stylesheet load order (`desktopNeoStyle.test.ts`)**: `theme.css` -> `xterm.css` -> `app.css` -> `atelier.css` -> `layout.css` -> `vibes.css` -> `neo.css` (last in Desktop `index.html`, `buildDesktop.ts`, `previewDesktop.ts`) -> `vscode.css` (last in VS Code `extension.ts`). `classic` and `shading` are retired.
- **Neo & VS Code CSS**: Every `neo.css` selector starts with `:root[data-style='neo']` with `:not(#neo)`; `@keyframes` start with `neo-`; every text/surface pair + `--x-on-accent`/`--x-accent` passes **WCAG AA >= 4.5:1** in dark and light. `vscode.css` uses double-ID specificity `:root[data-host='vscode']:not(#vscode-host):not(#vscode-layer)`.
- **Packaging traps**: `preload.ts` MUST build as `format: 'cjs'`. `buildDesktop.ts` & `buildMobile.ts` must define `'process.env.NODE_ENV': '"production"'` (fails if React DevTools banner leaks). macOS `.app` copying requires `{ verbatimSymlinks: true }` and inside-out signing. Mobile voice needs one shared `@huggingface/transformers` root, exact 4 `ort-wasm-simd-threaded*.{wasm,mjs}` files, zero `onnxruntime-node`/`sharp`, no WebGPU for q8 Moonshine, no int8 ONNX, and no COEP header.
- **Zero-DOM tests**: No `jsdom`/`happy-dom`; use `react-dom/server` (`renderToStaticMarkup`), pure reducers, and fakes. Run Desktop/VS Code/Hub/Slack under `bun test ./test`, TUI under `vitest.ui.config.ts`, Mobile/Client under `vitest.mobile.config.ts`.
