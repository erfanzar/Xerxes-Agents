# Codebase summary

Xerxes is a Bun-native TypeScript multi-agent runtime. Runtime and terminal-client source share the
single `src/typescript/` package.

```text
src/typescript/
  src/
    cli.ts             command dispatch
    core/              validation, paths, typed errors
    llms/              provider registry and clients
    streaming/         turn loop and event parsing
    runtime/           bootstrap, diagnostics, profiles, sessions
    executors/, tools/ tool contracts and implementations
    daemon/, api-server/, acp/, mcp/
    session/, memory/, context/
    channels/, extensions/, cortex/, skills/
    ui/                 React + OpenTUI terminal client
      __tests__/        Vitest component and behavior coverage
  test/                Bun unit, contract, and integration tests
examples/               runnable TypeScript examples
docs/                   Markdown sources and generated TypeScript API site
```

The CLI, daemon, ACP path, session export, OpenAI-compatible API library, and OpenTUI gateway all
run through Bun. Remote capabilities retain explicit native host boundaries where credentials or
a privileged integration are required. Browser automation is an explicit attachment to a running
Chromium CDP endpoint, not an implicit browser launch. Document those requirements as configuration
steps rather than treating them as local features.
