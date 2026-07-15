# Xerxes product overview

Xerxes is a multi-agent orchestration framework for running LLM-powered workflows through a CLI,
terminal client, daemon, HTTP API, ACP, MCP, and channel adapters. The native target is Bun with
strict TypeScript, using one event vocabulary across every surface.

## Product goals

- Make a turn observable from input through provider streaming, tool authorization, execution, and
  durable session state.
- Keep capabilities composable: agents, tools, profiles, memory stores, channels, and skills have
  typed contracts rather than hidden global behavior.
- Make high-power work deliberate. External clients, credentials, browser control, training
  hardware, and remote transports are supplied by explicit host ports.
- Preserve user-facing formats that matter—daemon v35, OpenAI-compatible frames, session records,
  and YAML agent/skill files—while simplifying internal implementation.

## Native experience

```sh
bun run xerxes
bun run xerxes "plan a change"
bun run xerxes daemon --project-dir .
bun run xerxes acp --project-dir .
bun run xerxes doctor
```

The OpenTUI client provides an interactive keyboard-first view over the native daemon. One-shot
calls, the daemon, and the API use the same normalized streaming events, so cancellation, tool
lifecycle, usage, and terminal errors have consistent meaning.

## Runtime status

Xerxes is a Bun-native TypeScript runtime with one event vocabulary shared by the CLI, terminal
client, daemon, API, ACP, MCP, and channel surfaces. Keep integrations that require credentials or
privileged host access behind explicit native ports. Browser control attaches only to an
explicitly supplied, already-running Chromium-compatible CDP endpoint.

Before a release or cross-cutting handoff, run `bun run check && bun run test && bun run build` and
`git diff --check`. Report the outcome only after those commands complete in the current worktree.
