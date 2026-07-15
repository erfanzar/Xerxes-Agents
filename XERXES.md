# XERXES.md — Bun / TypeScript Project Context

> **Project root:** the current Xerxes repository checkout
> **Runtime target:** Bun + strict TypeScript
> **License:** Apache-2.0

## Current status

Xerxes is a Bun-native TypeScript multi-agent framework. The runtime and OpenTUI terminal UI live
in one package under `src/typescript/`, with the UI source in `src/typescript/src/ui/`. The CLI, daemon,
ACP server, one-shot path, session export, API surface, and TUI execute through
the Bun runtime.

Use the delivery gate in this document before a release or handoff. Report only
the verification results run against the current worktree.

## Native commands

Use Bun for all supported development and runtime commands:

~~~bash
# Install locked workspace dependencies
bun install --frozen-lockfile

# Strict TypeScript checks, all Bun/UI tests, and production bundles
bun run check
bun run test
bun run build

# CLI smoke checks
bun run xerxes --help
bun run xerxes doctor
bun run xerxes "explain this function"

# Native services
bun run xerxes daemon
bun run xerxes acp
bun run xerxes export --list

# Native tooling
bun run docs:build
bun run real-use
bun run swarm

# Focused runtime tests
bun test src/typescript/test/streamingLoopParity.test.ts
bun test src/typescript/test/playgroundEval.test.ts

# Native warm-up evaluation (caller provides the runtime transport)
bun run playground:warmup -- --transport /absolute/path/to/evaluation-transport.ts
~~~

bun run xerxes doctor is the first check for a local install: it reports the
detected Bun runtime, Xerxes home, PATH setup, and missing provider credentials.
Real model runs require an explicitly configured provider key or profile. Do
not put credentials in fixtures, source files, or test logs.

The portable installer is Bun-native:

~~~bash
bash scripts/install.sh
~~~

It requires Bun to be installed already, installs the locked workspace, and
creates a launcher in XERXES_BIN_DIRECTORY (default: ~/.local/bin).

## Repository shape

~~~text
Xerxes-Agents/
├── src/typescript/                 # Native runtime package
│   ├── src/cli.ts                  # Bun CLI and service dispatch
│   ├── src/core/                   # Configuration, errors, paths, validation
│   ├── src/llms/                   # Provider registry and wire clients
│   ├── src/streaming/              # Async turn loop and event parsers
│   ├── src/runtime/                # Bootstrap, budgets, profiles, diagnostics
│   ├── src/executors/              # Tool registry and execution contracts
│   ├── src/tools/                  # Native tool modules and host ports
│   ├── src/security/               # Policy, scanning, approvals, sandbox ports
│   ├── src/session/                # Durable sessions, search, replay, export
│   ├── src/memory/                 # Four-tier stores and retrieval adapters
│   ├── src/daemon/                 # v35 socket/WebSocket daemon
│   ├── src/api-server/             # OpenAI-compatible HTTP surface
│   ├── src/acp/                    # Agent Client Protocol server
│   ├── src/channels/               # Channel lifecycle and adapters
│   ├── src/extensions/             # Skills, plugins, hooks, authoring
│   ├── src/cortex/                 # Multi-agent topology orchestration
│   ├── src/ui/                     # React + OpenTUI terminal client
│   └── test/                       # Bun unit, contract, and integration tests
├── examples/                       # Root TypeScript examples
├── docs/                           # Native static documentation configuration
├── src/typescript/playground/      # Evaluation fixtures and local artifacts
├── scripts/                        # Bun-native installer and repository scripts
└── .agents/projects/               # Project planning and verification notes
~~~

## How a native turn works

The native core is an async event pipeline:

~~~text
CLI / OpenTUI / daemon / API
  -> runtime bootstrap and session state
  -> provider detection + normalized streaming request
  -> AsyncGenerator<StreamEvent>
  -> permission and policy gate for each tool call
  -> tool executor, sandbox or explicitly injected host port
  -> normalized tool result added to the transcript
  -> text, thinking, tool, permission, and terminal events to the caller
~~~

Important native boundaries:

- src/typescript/src/streaming/loop.ts owns cancellation repair, tool-turn
  iteration, late steer handling, and event emission.
- src/typescript/src/daemon/server.ts exposes the v35 local protocol and
  persists session changes after commands such as compaction.
- src/typescript/src/runtime/ composes prompt profiles, features, budgets,
  transcript state, diagnostics, and CLI bootstrap.
- src/typescript/src/executors/ and src/typescript/src/tools/ validate,
  authorize, execute, and audit tool calls.
- src/typescript/src/memory/, context/, and session/ own durable memory,
  retrieval, prompt injection bounds, compaction, search, replay, and export.

Provider calls, browser/computer control, media APIs, hardware training, and
remote channel transports are modeled as explicit interfaces where the host must
make the security and credential decision. Never invent a successful external
operation when no host implementation was supplied.

## Supported CLI surface

~~~text
xerxes [prompt]                    # native one-shot or TUI
xerxes daemon [--project-dir DIR]  # native daemon
xerxes acp [--project-dir DIR]     # ACP over stdio
xerxes telegram --token TOKEN      # native daemon with Telegram settings
xerxes install --cloud-code ...    # companion-install planning/execution
xerxes doctor                      # local diagnostic report
xerxes update ...                  # update planning/execution
xerxes export [session]            # saved-session export
~~~

Run bun run xerxes --help for the authoritative command list. A command that
does not have a handler must fail explicitly; it must not silently pretend to
succeed.

## Code conventions

- Use strict TypeScript and Bun APIs. Run bun run check before handing work off;
  use bun test for focused and complete coverage.
- Keep ESM imports explicit, including .js extensions for local TypeScript
  modules. Do not add CommonJS or a Node/tsx execution path.
- Prefer typed discriminated unions, immutable values, and small concrete
  helpers over compatibility wrappers or any casts.
- Keep provider, storage, browser, channel, evaluator, and accelerator access
  behind explicit ports. Tests inject fakes; production code injects a real
  adapter after a deliberate configuration choice.
- Validate untrusted input at the boundary. Keep path traversal checks,
  permission gates, policy enforcement, prompt scanning, and redaction on the
  native path.
- Use apply_patch for source edits. Do not overwrite unrelated dirty changes,
  reset the worktree, or restore files just to simplify a rewrite.
- Every new TypeScript source file starts with the Apache-2.0 copyright header
  used by the native tree. Run bun run --cwd src/typescript
  fix-license-headers to normalize supported headers.

## Native maintenance rules

1. Keep public contracts stable where they are intentionally retained: daemon
   protocol v35, OpenAI-compatible API frames, persisted session formats, YAML
   agent/skill conventions, and tool schemas.
2. Do not add a Python runtime, Python test, packaging metadata, or subprocess
   fallback. Tools may still inspect user-owned files of any language.
3. Keep browser, media, provider, channel, hardware, and storage integrations
   behind explicit native interfaces. A missing host implementation must return
   an actionable error rather than a simulated success.
4. Do not port unsafe bypass or jailbreak behavior. Security evaluation code is
   defensive, bounded, and offline unless a caller explicitly injects a model.

## Evaluation playground

The native evaluation building blocks live in src/typescript/playground/. They
create private run directories without mutating global environment state, accept
a caller-owned evaluation session port, and run warm-up or typed hard tasks with
behavioral graders. The playground does not discover credentials, start a hidden
provider client, execute a foreign-language grader, or run shell commands
through a shell.

~~~bash
bun test src/typescript/test/playgroundEval.test.ts
~~~

`createNativeHardTasks()` supplies the complete 16-task hard battery. Its
fixtures and behavioral checkers are TypeScript/Bun-native, and
`bun run --cwd src/typescript playground:hard -- --transport <module>` runs it
through an explicitly supplied host transport.

## Delivery gate

Before handing off a cross-cutting change, run the relevant focused tests and
then the root gate when concurrent work has settled:

~~~bash
bun run check && bun run test && bun run build
git diff --check
~~~

The delivery gate is required before a release or cross-cutting handoff. Do not
report it as passing until it has completed successfully in the current
worktree.
