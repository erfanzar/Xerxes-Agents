---
name: add-rpc-or-slash
description: Guided end-to-end procedure for adding or modifying a v35 daemon JSON-RPC method, wire event, or slash command across daemon, PROTOCOL.md, surfaces, and Hub.
tags: [daemon, protocol, rpc, slash, surfaces]
---

# Add or Modify a Daemon v35 RPC, Wire Event, or Slash Command

Follow this checklist for: `$ARGUMENTS`. Preserve wire formats and persisted-session compatibility unless a migration was explicitly requested.

## 1. Classify the Change

- **RPC Method**: `DAEMON_PROTOCOL_VERSION = 35` in `xerxes/src/daemon/server.ts`
- **Wire Event**: `emit`/`broadcast` in `xerxes/src/daemon/` or `daemonEventFromStream` in `xerxes/src/daemon/turnRunner.ts`
- **Slash Command**: `xerxes/src/bridge/{commands,slashRouter}.ts`, `xerxes/src/daemon/server.ts`, `xerxes/src/ui/app/slash/registry.ts`

## 2. Daemon RPC Method Checklist

1. **Dispatch (`xerxes/src/daemon/server.ts`)**: Validate params at boundary (`xerxes/src/protocol/jsonRpc.ts`), return JSON-serializable payloads, and use `...(x === undefined ? {} : { x })` (`exactOptionalPropertyTypes`).
2. **Queue Tier**:
   - Default (mutators): serialized per connection + `SessionOperationQueue`.
   - `QUEUE_BYPASS_METHODS`: ONLY read-only/status/active-turn controls (auto-spread into `CONCURRENT_DISPATCH_METHODS`, asserted by test).
   - `CONCURRENT_DISPATCH_METHODS` only: sessionless slow I/O. **Never** add session mutators to either list.
3. **Protocol Sync (`xerxes/src/ui/PROTOCOL.md`)**: Document exact method name, params, response, and tier (`daemonProtocolContract.test.ts` blocks undocumented methods and phantom rows).
4. **Desktop & VS Code**:
   - Host-owned RPCs (`provider.remote.reply`, `connection.lease`): add to `HOST_ONLY_DAEMON_METHODS` (`xerxes/src/desktop/main/windowRoutes.ts`).
   - Renderer `window.xerxes` channels: update `xerxes/src/desktop/bridgeApi.ts` and implement in **both** `xerxes/src/desktop/main/ipc.ts`/`main.ts` and `xerxes/src/vscode/panelHost.ts` (or optional-check).
5. **Hub & Mobile** (only if Hub needs it — never expose raw daemon RPCs to Hub clients):
   - Add to `HUB_DAEMON_METHODS` (`HUB_MODEL_DAEMON_METHODS` / `HUB_LESSON_DAEMON_METHODS`) in `xerxes/src/hub/runtimePool.ts`.
   - If sessionless, update `HUB_SESSIONLESS_METHODS` + `hubMethodPolicy` (`xerxes/src/security/hubIdentityVerifier.ts`).
   - Add typed Hub v1 `HUB_METHODS` / `HUB_COMMAND_ACTIONS` (`xerxes/src/hub/protocol.ts`), update `docs/hub-protocol.md`, and wire `xerxes/src/client/` with zero Node/Bun globals (`tsconfig.mobile.json` `types: []`).

## 3. Daemon Wire Event Checklist

1. Emit in `xerxes/src/daemon/` (map `StreamEvent` in `daemonEventFromStream()` in `turnRunner.ts`); keep frozen wire names (`think_part`, not `thinking_part`) in `PRODUCTIVE_TURN_EVENTS`.
2. Update `xerxes/src/ui/PROTOCOL.md` event table; remove `*(bridge only)*` when a non-`bridge/` producer emits it.
3. Update consumers:
   - **OpenTUI**: `src/ui/gatewayTypes.ts` (`normalizeEventType`), `gatewayAdapter.ts` (`adaptDaemonEvent`), `gatewayClient.ts` (`SNAPSHOT_COVERED_EVENTS`).
   - **Desktop/VS Code**: `src/desktop/main/ipcEventBatcher.ts` (FIFO flush before lifecycle/interactive events) and `src/desktop/renderer/store.ts`.
   - **Hub/Mobile**: `src/hub/projections.ts` and `src/client/hubStore.ts` if projected.

## 4. Slash Command Checklist (3 Layers)

1. **Daemon & Bridge**: Register metadata in `COMMAND_REGISTRY` (`src/bridge/commands.ts`); add to `HANDLED_CANONICAL_COMMANDS` (or `DAEMON_EXTENSION_COMMANDS` for `mode`/`plan`/`ultra`/`search`) in `daemon/server.ts`; if bridge-routed, update `ROUTED_COMMANDS` + `switch` in `src/bridge/slashRouter.ts`. Never pass secrets in slash args (`plugins.secret.set` is password-field/CLI only). Colliding commands on external ACP forward verbatim.
2. **TUI & Surfaces**: Never duplicate daemon command logic (`commands.catalog` -> `slash.exec`); put presentation/overlay commands only in `src/ui/app/slash/registry.ts`. Any new `OverlayState` flag in `src/ui/app/overlayStore.ts` updates `OVERLAY_FLAG_KEYS`, `OVERLAY_BLOCKS_BACKGROUND_HOTKEYS`, `OVERLAY_CLOSED_VALUES`, and `resetFlowOverlays()`. Check completion in `src/ui/hooks/useCompletion.ts`.
3. **Collisions & Docs**: Confirm zero skill collisions (`builtin-command-conflict` in `src/extensions/harnessLint.ts`); update `PROTOCOL.md` and docs.

## 5. Prompt-Cache & Verification

Never place per-turn mutable state in `assembleContextLayers()` (`src/context/assembly.ts`); use `assembleTurnContext()` (`<turn-context>`).

```bash
bun run check
bun test xerxes/test/daemonProtocolContract.test.ts xerxes/test/streamingWireEvents.test.ts xerxes/test/bridgeCommands.test.ts xerxes/test/bridgeSlashRouter.test.ts xerxes/test/bundledSkillsLint.test.ts
bunx --bun --no-install vitest run --config xerxes/vitest.ui.config.ts xerxes/src/ui/__tests__/createSlashHandler.test.ts xerxes/src/ui/__tests__/slashCompletionCatalog.test.ts xerxes/src/ui/__tests__/gatewayAdapter.test.ts
```
Add `hub*.test.ts` + `test:mobile` for Hub/client changes, and `desktop*.test.*` + `vscodePanelHost.test.ts` for bridge changes.
