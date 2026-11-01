# Deepening

How to deepen a cluster of shallow modules safely, given its dependencies. Uses the vocabulary in
SKILL.md: **module**, **interface**, **seam**, **adapter**.

## Dependency categories

Classify each dependency of a deepening candidate. The category decides how the deepened module is
tested across its seam.

### 1. In-process

Pure computation, in-memory state, no I/O. Always deepenable: merge the modules and test through
the new interface directly. No adapter needed.

### 2. Local-substitutable

Dependencies with a local test stand-in (SQLite or PGLite for a database, a temporary directory
for the filesystem). Deepenable when the stand-in exists. Tests run against the stand-in; the seam
stays internal and no port appears at the module's interface.

### 3. Remote but owned (ports and adapters)

Your own services across a network. Define a **port** at the seam. The deep module owns the logic;
the transport is injected as an **adapter**. Tests use an in-memory adapter; production uses an
HTTP, gRPC, or queue adapter.

### 4. True external

Third-party services you do not control (payment, messaging, model providers). The deepened module
takes the external dependency as an injected port; tests provide a fake adapter.

## Seam discipline

- **One adapter means a hypothetical seam; two adapters mean a real one.** Do not introduce a port
  unless at least two adapters are justified (typically production plus test). A single-adapter
  seam is just indirection.
- **Internal seams versus external seams.** A deep module may have internal seams used by its own
  tests. Do not expose them through the interface just because tests use them.

## Testing strategy: replace, do not layer

- Once tests exist at the deepened module's interface, old unit tests on the removed shallow
  modules are waste; delete them.
- Tests assert observable outcomes through the interface, not internal state.
- Tests should survive internal refactors. If a test must change when the implementation changes,
  it is testing past the interface.

---

Adapted from `codebase-design/DEEPENING.md` in
[mattpocock/skills](https://github.com/mattpocock/skills) (MIT License, copyright 2026 Matt
Pocock).
