---
name: codebase-rescue
description: Take over a messy inherited codebase (no tests, tangled modules, broken parts, sloppy types) and restructure or rebuild it incrementally without breaking what works. Use when the user says a codebase is a mess, legacy, unmaintainable, or needs a rescue, rebuild, re-architecture, or cleanup across many files.
version: 1.0.0
author: Xerxes Agent
platforms: [linux, macos, windows]
tags: [legacy-code, refactor, architecture, rescue, rewrite, characterization-tests, strangler-fig]
---

# Codebase Rescue

Turn a codebase you inherited and do not trust into one you can change safely. The method is
**survey, safety net, slice by slice**: nothing changes until its current behaviour is pinned,
and each step is small enough to verify and revert.

**Core rule:** a rescue that breaks working behaviour is worse than the mess it replaced. Make
everything run, then safe, then testable, then modular, then clean, in that order.

## Metadata

- Related skills: codebase-design (module vocabulary and seams), python-types (typing ratchet),
  test-driven-development, systematic-debugging, simplify-code, writing-plans
- Ideas credited to: Michael Feathers, "Working Effectively with Legacy Code" (seams,
  characterization tests); Martin Fowler (strangler fig, refactoring in small steps)

## When to use

- The user inherited a codebase and calls it a mess, legacy, or broken, or asks you to rebuild,
  re-architect, or clean it up.
- A change you were asked for keeps getting blocked by tangled structure in many places.

Do not use it for one bad function; use simplify-code or plain refactoring. A rescue is
multi-session work: agree on a scope with the user before the first edit.

## Phase 0: agree on the goal and baseline

1. Ask the user what "fixed" means: which behaviour must keep working, what hurts most (bugs,
   speed of change, onboarding, performance), and what is off limits: public APIs, database
   schemas, file formats, CLI flags, configuration keys, deployment.
2. Confirm work happens on a branch or a clean working tree. Do not commit unless asked.
3. Record the baseline and keep the exact command output:
   - Does it install, build, and start? Which commands work today?
   - Test suite: how many pass, fail, or skip, and how long it takes.
   - Type checker and linter: the error count at the project's current settings.
   - Known-broken features the user already knows about.

A failing baseline is data, not something to hide: write it down so later steps are compared
against it, not against a clean slate that never existed.

## Phase 1: survey (read-only)

Map the codebase before judging it. Delegate in parallel when agent tools are available (one
agent per area, read-only, each returning `file:line` evidence):

- **Entry points:** main modules, CLI commands, HTTP routes, jobs, workers, scheduled tasks.
- **Dependency map:** which modules import which; cycles; god modules that everything imports;
  `utils`/`helpers`/`common` dumping grounds.
- **Hot spots:** files that change most often (`git log --format= --name-only | sort | uniq -c |
  sort -rn | head -30`) intersected with the most complex files. Rescue effort pays off there
  first.
- **State and effects:** global mutable state, singletons, import-time side effects, hidden I/O
  in constructors, environment reads scattered through the code.
- **Data shapes:** untyped dicts passed through many layers, stringly-typed status fields,
  duplicated models of the same thing.
- **Tests:** what exists, what runs, what is skipped or flaky, what coverage looks like.
- **Breakage:** crashes, swallowed exceptions (empty or blanket `except`/`catch`), TODO/FIXME
  clusters, dead code, security problems (hard-coded credentials, injection, unsafe
  deserialization).

Write the result as a plan (see PLAN-TEMPLATE.md) in a file the user agrees to, for example
`docs/rescue-plan.md`. Show the user the top findings and the proposed order before editing.

## Phase 2: triage

Classify every finding, then work top-down:

| Class | Examples | Handling |
|---|---|---|
| Dangerous | data loss, security holes, leaked credentials | fix first, each with a test; tell the user immediately |
| Broken | crashes, wrong results, dead features | fix as separate behaviour changes, each starting from a failing test |
| Blocking | structure that makes the next change impossible to test or reason about | restructure in slices (Phase 4) |
| Structural | shallow layers, god modules, duplicated models, cycles | restructure where it pays off: hot spots first |
| Cosmetic | naming, formatting, style | last, in separate mechanical commits, or not at all |

Credentials found in the code are reported by location only; never copy their values into
notes, plans, or messages.

## Phase 3: safety net

Before restructuring a slice, pin its current behaviour:

- **Characterization tests:** call the code through the outermost seam you can reach (the CLI,
  an HTTP route, a public function) and assert what it does today, including odd outputs. These
  tests record behaviour; they do not judge it. Mark known bugs with a comment and a separate
  test that expects the correct behaviour, skipped or expected-to-fail until the fix lands.
- **Golden master** for large outputs (reports, generated files, rendered pages): capture
  current output for a set of representative inputs and compare byte-for-byte or with a
  normalizer for timestamps and IDs.
- **Break the dependencies only as far as the test needs:** pass in a clock, a temporary
  directory, a fake HTTP client, or a local database stand-in. Prefer the smallest seam
  (a parameter with a default) over new abstraction layers.
- Make flaky tests deterministic or quarantine them with a reason; do not delete them silently.

## Phase 4: restructure slice by slice

Pick one slice (a hot-spot module or one vertical feature) at a time. For each:

1. **Name the target shape** in codebase-design terms: the deep module, its small interface,
   where the seam goes, what hides behind it.
2. **Choose the move:**
   - **Refactor in place** when the code mostly works and the shape is close: extract, inline,
     move, and rename in small steps.
   - **Strangler fig** when the module is beyond repair: build the new module beside the old one
     behind the same interface, route callers to it one at a time, then delete the old one. Both
     paths stay runnable throughout.
   - **Rewrite the module outright** only when all of these hold: its behaviour is fully pinned
     by tests, it is small enough to finish in one slice, and nothing else depends on its
     internals.
3. **One kind of change per step.** A step either preserves behaviour (refactor) or changes it
   (fix), never both. Mechanical changes (formatting, renames across files) get their own steps.
4. **Run the safety net after every step.** A red test means revert the step, not adjust the
   test, unless the test pinned behaviour the user agreed to change.
5. **Delete as you go:** pass-through layers, dead code (after grepping every use, including
   dynamic lookups, configuration, and string references), and tests that only covered removed
   internals once tests at the new interface exist.

Typical fixes for common rot:

- **God module:** split by reason to change, not by kind of code; each part gets a small
  interface and its own tests.
- **Untyped dicts everywhere:** parse once at the edge into a frozen dataclass or typed object,
  then migrate callers inward (python-types).
- **Global state and singletons:** pass the dependency in; keep a default at the composition
  root so callers migrate one at a time.
- **Import-time side effects:** move them into an explicit start function called from the entry
  point.
- **Cycles:** move the shared concept down into its own module, or invert the dependency at a
  seam that has two real adapters.
- **Swallowed errors:** let them propagate or handle a specific error type; log with context
  where it must be contained.
- **Copy-paste variants:** unify only after tests pin each variant's behaviour; differences are
  often intentional.

## Phase 5: types and gates as a ratchet

- Turn on the project's type checker if it has none, at a lenient level, and record the error
  baseline.
- New and fully restructured modules meet the strict level; untouched code keeps the baseline.
  The error count may only go down.
- Add the checker and the test suite to the project's normal check command or CI if missing, so
  the rescue cannot silently regress.

## Delegation

When agent tools are available:

- Survey: one read-only agent per area in parallel, a fast model for mechanical mapping, a
  strong model for the synthesis and the plan.
- Restructuring: parallel agents only for slices with disjoint files, each in its own worktree
  when supported, each given the plan, the slice's target shape, and the exact verification
  commands. Integrate and run the full safety net yourself.
- Review: an independent reviewer per finished slice, checking behaviour preservation and that
  no public contract changed.

## Reporting

After each slice, report: what changed, the before and after test, type, and lint counts against
the Phase 0 baseline, behaviour changes the user approved, deliberate skips, and the next slice
in the plan. Update the plan file so a later session can continue from it.

## Pitfalls

- **The big-bang rewrite.** Rewriting everything at once loses undocumented behaviour that users
  depend on, and nothing works until everything works. Slice it.
- **Chesterton's fence.** Check `git blame` and the commit message before removing odd code;
  it may be a workaround for a real production problem.
- **Fixing bugs inside a refactor.** It makes both unverifiable. Note the bug, finish the
  refactor, then fix it with its own test.
- **Changing public contracts.** API shapes, schemas, persisted formats, environment variables
  and CLI flags stay stable unless the user approves a migration plan.
- **Reformatting the world.** A repository-wide formatter run hides every real change in the
  diff. Run it separately, and only if the user wants it.
- **Abstractions as the cure.** Adding interfaces, factories, and base classes to messy code
  usually makes it messier. Deep modules come from merging and hiding, not layering.
- **Disabling tests or checks to get green.** Never; record and fix, or quarantine with a reason
  the user can see.
