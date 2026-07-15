# AGENTS.md

Guidance for agents working in Xerxes. Read this before changing the repository.

## Project overview

Xerxes (`xerxes-agents` v0.3.0) is a Bun-native TypeScript multi-agent runtime. It provides a
terminal UI, JSON-RPC daemon, OpenAI-compatible API, provider routing, tool execution, MCP,
subagents, channels, persistent sessions, and tiered memory.

- Runtime: Bun 1.3+
- Language: TypeScript (strict)
- Runtime source: `src/typescript/src/`
- Runtime tests: `src/typescript/test/`
- TUI source: `src/typescript/src/ui/`
- TUI tests: `src/typescript/src/ui/__tests__/`
- Bundled skill content: `src/typescript/skills/`

The repository is Bun-native. Do not add a Python runtime, Python test, Python packaging
metadata, or a Python subprocess fallback. User-requested tools may operate on Python files in a
user's workspace; that is different from making Xerxes depend on Python.

## Commands

Run these from the repository root unless a command says otherwise.

```bash
# Install locked workspace dependencies
bun install --frozen-lockfile

# Start Xerxes
bun run xerxes
bun run xerxes "explain this function"
bun run xerxes --resume <session_id>
bun run xerxes daemon --project-dir .
bun run xerxes acp --project-dir .

# Runtime validation
bun run --cwd src/typescript check
bun test src/typescript/test/<file>.test.ts
bun run --cwd src/typescript test:runtime

# TUI validation
bun run --cwd src/typescript check:ui
bun run --cwd src/typescript test:ui

# Full repository gate
bun run check && bun run test && bun run build
git diff --check

# Documentation and maintenance
bun run docs:build
bun run --cwd src/typescript fix-license-headers
```

Do not validate repository changes with `python`, `python3`, `uv`, `pytest`, `ruff`, or `mypy`.
Do not add Node/npm lifecycle wrappers when Bun can run the command directly.

## Repository layout

```text
src/typescript/
├── src/
│   ├── cli.ts              # CLI entry point
│   ├── xerxes.ts           # Embedded facade
│   ├── daemon/             # v35 JSON-RPC daemon and session bridge
│   ├── streaming/          # Async turn loop and stream events
│   ├── llms/               # Provider routing and transports
│   ├── executors/          # Tool registry and dispatch
│   ├── tools/              # Built-in and Claude-compatible tools
│   ├── agents/, cortex/    # Agent specs and orchestration
│   ├── session/, memory/   # Durable state, FTS, replay, retrieval
│   ├── security/           # Policies, scanning, sandbox routing
│   ├── mcp/, acp/          # MCP and Agent Client Protocol surfaces
│   ├── channels/           # Messaging adapters and gateways
│   ├── api-server/         # OpenAI-compatible HTTP service
│   ├── extensions/         # Skills, hooks, plugins, authoring
│   └── ui/                 # React + OpenTUI terminal client
├── test/                   # Bun contract and integration tests
└── skills/                 # Bundled SKILL.md content and safe assets

docs/                       # Markdown documentation and Bun docs output
examples/                   # Bun/TypeScript examples
scripts/install.sh           # Bun installer and launcher setup
```

## How a turn works

1. `src/typescript/src/cli.ts` selects interactive TUI, one-shot, daemon, ACP, or an explicit
   command such as `doctor`, `export`, or `skill`.
2. The TUI communicates with the Bun daemon through the v35 newline-delimited JSON-RPC protocol.
3. `streaming/loop.ts` normalizes provider deltas into serializable stream events, routes tool
   calls through permissions and the tool registry, and guarantees a terminal turn event.
4. `runtime/` enforces turn, budget, and compaction limits while `session/` persists data.
5. The TUI renders the same event vocabulary the daemon, API, MCP, and channels consume.

Preserve public wire formats and persisted-session behavior unless the user explicitly authorizes a
protocol migration. Do not invent a successful fallback for an unsupported external integration:
return a typed, actionable error or require an explicit injected host port.

## Code conventions

- Use strict TypeScript and explicit public types. Prefer discriminated unions over boolean state
  flags and `unknown` plus narrowing over unchecked `any`.
- Use native async iterators and `AbortSignal` for stream cancellation. Do not block the event loop.
- Prefer top-level functions and small concrete helpers over class hierarchies without a real
  lifecycle or shared mutable state.
- Keep imports at the top. Use lazy imports only for genuine startup cost or cycle isolation.
- Validate external input at the boundary: CLI arguments, JSON-RPC frames, provider responses,
  persisted records, YAML agent specs, and webhook payloads.
- Use `Bun.file`, `Bun.write`, `Bun.spawn`, `Bun.serve`, and `Bun.sqlite` where the implementation
  is runtime-specific. Keep a narrow interface around privileged or host-owned capabilities.
- Let errors propagate unless a catch adds context or intentionally changes control flow. Never
  swallow provider, filesystem, or tool failures.
- Keep security decisions explicit. Tool-policy and sandbox denials must remain observable.
- Every TypeScript source file begins with the repository Apache-2.0 header used by adjacent files.
- Do not extend obsolete compatibility shims just to preserve retired internals.

## Tests and verification

Add or update focused Bun tests beside the subsystem you change. A test should exercise observable
behavior, not merely an exported symbol. At minimum cover error and cancellation behavior at
boundaries that stream, persist, call a provider, or execute a tool.

Run the narrowest relevant test while iterating, then run the full root gate before handing over a
cross-cutting change. Inspect `git diff --check` after generated or bulk asset work. Do not edit
generated `dist/` output manually; use the owning Bun build command.

For live-provider, browser, channel, email, or cloud tests, keep credentials outside the repository
and make the external call opt-in. Offline tests must use a deterministic injected port or fixture.

## Skills, extensions, and assets

`src/typescript/skills/` is recursively copied into the runtime distribution by
`src/typescript/scripts/copyBundledSkills.ts`. Preserve safe references, templates, and assets when
moving a bundled skill. Do not leave a duplicate `name` frontmatter entry that can shadow the native
skill. Executable or privileged integrations belong in native TypeScript with an explicit host port;
do not revive unsafe legacy scripts.

## TUI and daemon behavior

Keep the UI protocol-independent: `src/typescript/src/ui` talks only to the documented daemon RPC
surface.
When changing a slash command, update all three layers together:

1. daemon command/handler and its contract test;
2. TUI command registry, completion, and state restoration behavior; and
3. user-facing help or docs.

Never make a slash picker erase a transcript permanently. Overlay state must restore the prior
screen after cancel, completion, or a provider error. Preserve keyboard access and narrow-terminal
fallbacks when changing layout.

Native browser automation uses Chromium DevTools Protocol (CDP). It only attaches to an explicitly
supplied, already-running Chromium-compatible browser endpoint (for example through `/browser
connect <endpoint>`); Xerxes does not launch or own a browser process on the user's behalf.

## Documentation and release work

Documentation is Markdown and native Bun-generated API content. Keep README commands, install
instructions, CLI help, examples, CI workflows, Docker files, and `AGENTS.md` aligned with the
actual Bun executable. Before a release or cross-cutting handoff, run the full Bun gate and report
only the results that completed in the current worktree.

Commit messages use Conventional Commit/Commitizen form:

```text
type(scope): concise description
```

Do not stage, commit, push, create a PR, alter external accounts, or publish artifacts unless the
user explicitly asks.
