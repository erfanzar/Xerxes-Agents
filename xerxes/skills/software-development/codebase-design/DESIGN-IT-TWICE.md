# Design It Twice

Use this when an interface matters enough that the first idea should not win by default (based on
Ousterhout's "Design It Twice"). Uses the vocabulary in SKILL.md and the dependency categories in
DEEPENING.md.

## 1. Frame the problem space

Before delegating, write a short explanation for the user:

- the constraints any new interface must satisfy;
- the dependencies it relies on and their category;
- a rough code sketch that makes the constraints concrete (not a proposal).

Show it, then continue to step 2 while the user reads.

## 2. Spawn designers in parallel

When `SpawnAgents` is available, start three or four subagents in one batch, each with a fresh
context and a self-contained technical brief (file paths, coupling details, dependency categories,
what sits behind the seam, the project's own terms). Give each a different constraint:

- "Minimize the interface: one to three entry points. Maximize leverage per entry point."
- "Maximize flexibility: support many use cases and extension."
- "Optimize for the most common caller: make the default case trivial."
- "Design around ports and adapters for the cross-seam dependencies." (when relevant)

Each designer returns:

1. the interface (types, entry points, parameters, invariants, ordering, error modes);
2. a usage example from a caller's point of view;
3. what the implementation hides behind the seam;
4. the dependency strategy and adapters;
5. trade-offs: where leverage is high and where it is thin.

These are read-only design tasks; designers must not edit files. Without delegation, draft the
alternatives yourself in sequence and say so.

## 3. Present and compare

Present the designs one at a time, then compare them on **depth**, **locality**, and **seam
placement**. Finish with an opinionated recommendation; propose a hybrid when parts of different
designs combine well.

---

Adapted from `codebase-design/DESIGN-IT-TWICE.md` in
[mattpocock/skills](https://github.com/mattpocock/skills) (MIT License, copyright 2026 Matt
Pocock).
