---
name: xerxes-daemon-protocol
description: Use when adding, renaming, or modifying a v35 daemon JSON-RPC method, wire event, or slash command across the daemon, PROTOCOL.md, bridge router, TUI, Desktop, VS Code, or Hub.
version: 1.0
author: Xerxes-Agents
tags: [xerxes, daemon, protocol, json-rpc, slash-commands]
---

# Xerxes Daemon v35 Protocol & Slash Commands

End-to-end procedure for changing the v35 NDJSON JSON-RPC surface (`DAEMON_PROTOCOL_VERSION = 35`), wire events, or slash commands. See `.agents/rules/core-runtime-daemon.md` and `.agents/rules/client-surfaces-and-hub.md`.

## When to use

- Adding or changing a JSON-RPC method in `xerxes/src/daemon/server.ts`.
- Adding or modifying a v35 wire event or `StreamEvent` mapping.
- Adding or changing a slash command across daemon, bridge, TUI, Desktop, or VS Code.

## Checklist A: Adding or Modifying a Daemon RPC Method

1. **Dispatch & validate (`xerxes/src/daemon/server.ts`)**:
   - Add handler (`method === "..."` or `[...].includes(method)`), validate params at the boundary, and keep `JsonRpcRequest.id` (`string | number`) and payloads JSON-serializable.
2. **Pick the exact queue tier (`xerxes/src/daemon/server.ts`)**:
   - `QUEUE_BYPASS_METHODS`: read-only/status/control RPCs that must answer during active turns or schedules (included in `CONCURRENT_DISPATCH_METHODS` via `...QUEUE_BYPASS_METHODS`, asserted by test).
   - `CONCURRENT_DISPATCH_METHODS` only: sessionless slow I/O outside the per-connection queue.
   - Default serialized (`SessionOperationQueue` per session): ALL session mutators. **Never** place session-mutating RPCs in bypass or concurrent sets.
3. **Update Hub auth & allow-lists (if Hub-reachable)**:
   - Add sessionless RPCs to `HUB_SESSIONLESS_METHODS` (`daemon/server.ts`) and register policy in `hubMethodPolicy` (`xerxes/src/security/hubIdentityVerifier.ts`).
   - Add to internal daemon allow-list in `xerxes/src/hub/runtimePool.ts` (`HUB_DAEMON_METHODS`, `HUB_MODEL_DAEMON_METHODS`, `HUB_LESSON_DAEMON_METHODS`).
   - Expose versioned Hub v1 methods/actions in `xerxes/src/hub/protocol.ts` (`HUB_METHODS`, `HUB_COMMAND_ACTIONS`) and `docs/hub-protocol.md` (never raw daemon v35 passthrough).
4. **Update renderer blocklist**:
   - For host-owned RPCs that renderers/webviews must never invoke, add to `HOST_ONLY_DAEMON_METHODS` in `xerxes/src/desktop/main/windowRoutes.ts`.
5. **Sync `xerxes/src/ui/PROTOCOL.md` in the same patch**:
   - Document the exact method name, params, and result. `xerxes/test/daemonProtocolContract.test.ts` enforces bidirectional sync (no undocumented handlers, no phantom rows).
6. **Wire clients**:
   - Update `xerxes/src/ui/{gatewayClient,gatewayTypes}.ts` and, if bridged to Desktop/VS Code, `xerxes/src/desktop/bridgeApi.ts` + `main.ts` + `src/vscode/panelHost.ts`.

## Checklist B: Adding or Modifying a Wire Event

1. **Emit & map**: Emit via `emit`/`broadcast` in `xerxes/src/daemon/`. If sourced from `StreamEvent`, map in `daemonEventFromStream()` (`xerxes/src/daemon/turnRunner.ts`: `text`->`text_part`, `thinking`->`think_part`, `tool_start`->`tool_call`, `tool_end`->`tool_result`, `permission_request`->`approval_request`, `provider_wait`/`compaction`/`usage_update`->`status_update`, `turn_done`->`turn_end`).
2. **`PRODUCTIVE_TURN_EVENTS`**: Use exact **wire names** (`think_part`, not `thinking_part`; enforced by contract test).
3. **Sync `xerxes/src/ui/PROTOCOL.md`**: Add the event row and drop `*(bridge only)*` if a non-`bridge/` producer emits it.
4. **Update adapters & batcher**: Update `xerxes/src/ui/gatewayTypes.ts` (`normalizeEventType`), `xerxes/src/ui/gatewayAdapter.ts` (`adaptDaemonEvent`), `SNAPSHOT_COVERED_EVENTS` (`gatewayClient.ts`), and `DaemonEventBatcher` (`src/desktop/main/ipcEventBatcher.ts`) for high-frequency events.

## Checklist C: Adding or Modifying a Slash Command

Update all three layers together:
1. **Metadata (`xerxes/src/bridge/commands.ts`)**: Add entry and aliases to `COMMAND_REGISTRY`.
2. **Daemon & bridge**: Add canonical name to `HANDLED_CANONICAL_COMMANDS` (or `DAEMON_EXTENSION_COMMANDS` for `mode`/`plan`/`ultra`/`search`-style controls) and dispatch in `daemon/server.ts`. If bridge-routed, add to `ROUTED_COMMANDS` and `switch` in `xerxes/src/bridge/slashRouter.ts`.
3. **TUI & docs**: Local presentation/overlay commands live in `xerxes/src/ui/app/slash/registry.ts` (`createSlashHandler.ts`, `overlayStore.ts`); daemon commands advertise via `commands.catalog` and run via `slash.exec`. Ensure overlays restore prior screen/transcript on cancel, completion, or error. Check zero skill-name collision (`harnessLint.ts` `builtin-command-conflict`) and update docs.

## Pitfalls

- **Prompt cache**: Never add volatile per-turn data to `assembleContextLayers()`; put per-turn diffs/lessons/goal status in `assembleTurnContext()` (`<turn-context>`).
- **No secrets over slash**: `plugins.secret.set` is password-field/CLI only.
- **Replay parity**: Keep `looksLikeInternalUserPrompt` (`ui/gatewayAdapter.ts`) and `isLegacyHarnessPrompt` (`desktop/renderer/transcriptContent.ts`) synced with daemon `looksLikeInternalReplayMessage`.

## Verification commands

```bash
bun run --cwd xerxes check:runtime && bun run --cwd xerxes check:ui && bun run --cwd xerxes check:desktop
bun test xerxes/test/daemonProtocolContract.test.ts xerxes/test/daemonServer.test.ts xerxes/test/streamingWireEvents.test.ts xerxes/test/bundledSkillsLint.test.ts
bun run test:ui && bun run smoke
```
