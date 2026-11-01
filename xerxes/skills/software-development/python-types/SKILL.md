---
name: python-types
description: Proportionate, checker-backed typing for Python. Use when writing, refactoring, or reviewing typed Python, choosing between dataclass, Enum, Literal, NewType, TypedDict and Protocol, fixing mypy or pyright errors, or when code is over-typed (Any, cast, ignore comments, Protocols with one implementation, generic gymnastics).
version: 1.0.0
author: Xerxes Agent
platforms: [linux, macos, windows]
tags: [python, typing, mypy, pyright, design, dataclasses, protocol]
---

# Python Types

Types exist to make wrong programs fail early and to tell the next reader what a value is. They
are not decoration. Add precision where a wrong value would do real harm; leave the rest plain.

## Metadata

- Related skills: codebase-design, codebase-rescue, test-driven-development, simplify-code
- Ideas credited to: jpablo/vibe-types (Apache-2.0), "Effective TypeScript" item 34, Alexis King's
  "Parse, don't validate"

## Rules

1. **Follow the project's checker.** Find its mypy or pyright configuration (`pyproject.toml`,
   `mypy.ini`, `pyrightconfig.json`, `setup.cfg`) and run it the way the project does. Never
   weaken settings or add per-file exemptions to get green.
2. **Annotate every public signature** (parameters and return). Let the checker infer locals; do
   not annotate `x: int = 0` or `items: list[str] = []` style obvious values.
3. **Make illegal states unrepresentable.** When fields only make sense together, use one frozen
   dataclass per state and a union of them, not one class with many optional fields and a status
   string.
4. **Parse, don't validate.** At the boundary (HTTP, CLI, files, environment, model output) turn
   raw data into a typed value once, and pass that value inward. Core code should never re-check
   what the boundary already proved.
5. **Prefer the narrowest type that holds every legal value**: `Literal` or `Enum` over `str`,
   `bool` over `int` flags, `NewType` for IDs that must not be swapped (`UserId` vs `OrderId`).
6. **Handle every case.** Match on a union or enum and end with `assert_never(value)` so a new
   variant becomes a checker error instead of a silent fall-through.
7. **Accept broad, return precise.** Take `Iterable[T]`/`Mapping[K, V]`/`Sequence[T]`; return
   concrete types (`list[T]`, a dataclass).
8. **Keep effects at the edge.** Pure functions take and return values; I/O, clocks, randomness and
   network calls live in a thin outer layer that calls the core.

## Proportion: what not to do

- **No `Any` to silence an error.** If data is truly unknown, take `object` and narrow it with
  `isinstance` or a parser.
- **No `cast(...)` or `# type: ignore` as a fix.** If one is unavoidable (a broken third-party
  stub), confine it to one small wrapper, use `# type: ignore[<code>]` with the specific code, and
  write why on the same line.
- **No Protocol or ABC with one implementation.** A seam needs two adapters (production plus a
  test fake) to be real; otherwise depend on the concrete class.
- **No class hierarchies for data.** Use `@dataclass(frozen=True, slots=True)` and unions; reach
  for inheritance only for real shared behaviour.
- **No type gymnastics.** If a signature needs nested `TypeVar`s, `ParamSpec` and overloads to
  express, simplify the design instead.
- **No DTO per function.** Reuse one domain type across layers that share its meaning.
- **Plain values stay plain.** Loop counters, local accumulators, private helpers with one caller
  and throwaway scripts do not need `NewType` or wrappers.

## Choosing a construct

| Need | Use |
|---|---|
| Record with fixed fields, owned by your code | `@dataclass(frozen=True, slots=True)` |
| JSON-shaped dict crossing an API you do not own | `TypedDict` (parse it into a dataclass at the edge if the core uses it) |
| Validated external input with coercion | the project's existing validator (pydantic, attrs, msgspec); do not add a new one |
| Small closed set of values | `Literal[...]` for internal flags, `Enum` when values carry behaviour or are persisted |
| Same runtime type, must not be mixed up | `NewType` |
| Variants with different data | union of frozen dataclasses with a `kind: Literal[...]` field or distinct classes, matched exhaustively |
| Dependency that really has two implementations | `Protocol` at the seam |
| Error the caller must handle | a specific exception class, or a result union when failure is ordinary data |

## Example: status flags versus states

```python
# Over-loose: invalid combinations type-check
@dataclass
class Job:
    status: str                      # "queued" | "running" | "done" | "failed"
    started_at: datetime | None = None
    result: bytes | None = None
    error: str | None = None

# Precise: each state carries exactly its data
@dataclass(frozen=True, slots=True)
class Queued: ...

@dataclass(frozen=True, slots=True)
class Running:
    started_at: datetime

@dataclass(frozen=True, slots=True)
class Done:
    started_at: datetime
    result: bytes

@dataclass(frozen=True, slots=True)
class Failed:
    started_at: datetime
    error: str

JobState = Queued | Running | Done | Failed

def describe(state: JobState) -> str:
    match state:
        case Queued():
            return "waiting"
        case Running(started_at=t):
            return f"running since {t:%H:%M}"
        case Done():
            return "done"
        case Failed(error=e):
            return f"failed: {e}"
        case _:
            assert_never(state)
```

## Workflow

1. Read the project's typing configuration and existing patterns; match them.
2. Type the boundary first, then the public signatures of the module you touch.
3. Run the project's checker on the project (not on hand-picked files that bypass its config).
4. Report consequential typing decisions and any remaining `ignore` with its reason. Do not turn a
   small change into a typing migration of untouched code.
