# Code standards

The active runtime target is strict TypeScript on Bun.

## Baseline

```sh
bun run check
bun run test
bun run build
```

- Use ESM with explicit `.js` extensions for local TypeScript imports.
- Start native source files with the repository Apache-2.0 header.
- Prefer discriminated unions, immutable values, and narrow interfaces at integration boundaries.
- Validate untrusted input once at the boundary; let typed errors propagate with useful context.
- Keep public types and wire formats serializable.

## Design rules

- Prefer small concrete functions over abstraction without reuse.
- Keep host I/O separate from deterministic computation. Inject filesystems, clocks, process ports,
  providers, and transports into testable functions.
- Do not add a compatibility layer merely to preserve an obsolete internal shape. Update native call
  sites together when a contract intentionally changes.
- Do not use an ambient credential, browser session, accelerator, or network result in a test. Make
  the boundary explicit and inject it.
- Keep high-power tools behind policy and permission checks. Never turn a failed or unavailable
  external operation into a success-shaped result.

## Repository hygiene

Use `apply_patch` for source edits, preserve unrelated worktree changes, and do not reset or restore
files just to simplify a migration. Before deleting an old path, require a native implementation,
focused parity coverage, and a verified consumer entrypoint. Update docs, examples, and scripts in
the same slice.

## Terminal UI

The OpenTUI interface must remain keyboard-first, readable at narrow widths, accessible in plain
text, and compatible with the v35 gateway. Use original Xerxes visual assets and wording; do not
copy a third party’s marks, artwork, or product text.
