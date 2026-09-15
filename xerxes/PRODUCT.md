# Xerxes

<!-- impeccable:product-schema 1 -->

## Platform

web

The desktop uses a React web renderer inside Electron, with macOS window controls and desktop integrations. This platform value describes the graphical rendering surface; Xerxes also has a separate terminal interface and programmable runtime. It does not imply a browser-only product or a mobile application.

## Users

People doing coding and general agent work, with equal priority for both. Users manage long-running tasks across multiple projects and need to understand, steer, and review work performed by models, tools, and delegated agents.

## Product Purpose

Make sustained agent work usable and inspectable: start a task, supply context, follow progress, intervene, inspect results, and resume later without losing the conversation or its working state. Users should be able to work in several workspaces concurrently.

## Positioning

Xerxes is a Bun-native, provider-configurable agent runtime with desktop and terminal clients. Persistent sessions, tool execution, permissions, skills, MCP integrations, delegated agents, and scheduled work belong to the runtime. The graphical interface should expose those capabilities clearly rather than make users switch to the terminal to understand their tasks. This is the product mechanism, not a claim of unique market capabilities or completed interface parity.

## Operating Context

- Tasks include coding, reviewing changes, research, planning, and general tool-assisted work.
- Users open local projects or connect to configured SSH workspaces, select models, give instructions, and return to saved sessions.
- Long conversations, lengthy tool output, many agents, deep file paths, and simultaneous running tasks are normal states.
- A task's working directory, instructions, skills, permissions, provider selection, and saved state must remain correctly associated with that task when windows or workspaces change.
- Runtime processes can outlive the desktop window. Closing a window must not silently cancel ongoing work.

## Capabilities and Constraints

- Preserve existing sessions, runtime integrations, public protocol contracts, and persisted-session behavior during redesign.
- The runtime and repository workflows are Bun-native TypeScript. Do not introduce Python runtime dependencies or substitute npm lifecycle wrappers for Bun commands.
- Keep authentication, configuration, permission, and certificate failures observable. Never bypass security checks to make a connection appear healthy.
- Recoverable connection failures should retain the task and present one updating recovery state.
- Provide readable tool results with access to full underlying details; expose agent status, current work, goals, todos, queued instructions, and failures without requiring prolonged transcript scrolling.
- Preserve keyboard operation, viewport-contained menus, multiline drafting, and usable narrow and large windows.
- Verify actual interactions and populated rendered states. Screenshots alone do not establish functionality; fixture tests alone do not establish native-build correctness.
- Do not disrupt active daemons or overwrite unrelated changes while developing.

### Shared runtime

Desktop and TUI share a per-user daemon with workspace-scoped capabilities and concurrent sessions. Idle legacy daemons migrate automatically; busy legacy daemons remain attached; the shared daemon refuses overlapping workspace ownership during migration. Explicit custom sockets and remote runtimes remain supported.

Full GUI access to TUI capabilities is an objective, not a verified current property. Capability gaps should be tracked and reported explicitly.

## Brand Commitments

- Preserve the Xerxes name and the expressive serif XERXES wordmark.
- Retain recognizable Xerxes identity while adapting the desktop experience to the user's current reference: **Claude desktop Code view**, specifically its presentation of tools, agents, and ongoing work. The reference is not Claude's terminal interface.
- Earlier Hermes and experimental visual treatments are historical evidence, not authority over the user's latest direction. Detailed visual decisions belong in the design record, not this product record.
- Use plain, actionable language. State what is running, completed, blocked, or failed honestly.

## Evidence on Hand

- `../AGENTS.md`: repository requirements and public-runtime constraints.
- `../README.md`: runtime capabilities and entry points; existing project-daemon descriptions document current architecture, not the requested replacement.
- `../docs/desktop-workspace.md`: current desktop workflows and integration boundaries.
- `src/desktop/`: implemented Electron shell and graphical client.
- `src/daemon/`, `src/ui/PROTOCOL.md`, and `src/ui/`: runtime, wire contract, and terminal-client behavior.
- `../assets/`: existing Xerxes identity and installer assets.
- User-provided screenshots and the sibling `xerxes-desktop-verification` directory document prior defects and verification captures. Confirm build freshness and distinguish native captures from fixtures before relying on them.
- Official Claude desktop documentation is an external reference. Local Claude required renewed sign-in during inspection; the user explicitly chose to continue from official references. Populated Claude interactions have not been verified locally.

## Product Principles

1. Coding and general agent work deserve equal clarity and control.
2. Task continuity survives window changes, workspace switches, and recoverable disconnections.
3. Progress and results are understandable without reading raw transport data or searching a long transcript.
4. A simpler shared service must not weaken workspace boundaries or permissions.
5. Claim completion only with evidence from the implemented behavior.

## Open Decisions

- Global-daemon ownership, workspace-service isolation, and migration of existing running processes need an implementation design.
- The remaining GUI/TUI capability gaps need a current, behavior-based audit.
- Platform support beyond the existing desktop and terminal implementations is not expanded by this record.
