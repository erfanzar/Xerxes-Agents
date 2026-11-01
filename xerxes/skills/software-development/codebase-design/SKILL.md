---
name: codebase-design
description: Design deep modules with small interfaces at real seams. Use when creating or restructuring a module, package, or class, deciding where an abstraction or seam belongs, removing pass-through layers, or making code easier to test and navigate. Language-agnostic; examples in Python and TypeScript.
version: 1.0.0
author: Matt Pocock (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [architecture, modularity, deep-modules, refactor, design, testability]
source: https://github.com/mattpocock/skills/tree/main/skills/engineering/codebase-design
---

# Codebase Design

Design **deep modules**: a lot of behaviour behind a small interface, placed at a clean seam,
testable through that interface. The aim is leverage for callers, locality for maintainers, and
testability for everyone.

## Metadata

- License: MIT (see LICENSE.txt in this skill folder)
- Related skills: python-types, codebase-rescue, simplify-code, test-driven-development

## Glossary

Use these terms exactly. Consistent language is the point.

**Module**: anything with an interface and an implementation. Scale-agnostic: a function, class,
package, or a slice across tiers.

**Interface**: everything a caller must know to use the module correctly: the signature, but also
invariants, ordering constraints, error modes, required configuration, and performance
characteristics.

**Implementation**: what is inside a module. Distinct from **adapter**: a small adapter can have a
large implementation (a Postgres repository) and a large adapter a small one (an in-memory fake).

**Depth**: leverage at the interface, the amount of behaviour a caller or test can exercise per
unit of interface it has to learn. **Deep**: much behaviour behind a small interface. **Shallow**:
the interface is nearly as complex as the implementation.

**Seam** (Michael Feathers): a place where behaviour can change without editing that place; the
location at which a module's interface lives. Where the seam goes is its own decision.

**Adapter**: a concrete thing that satisfies an interface at a seam. Names a role, not substance.

**Leverage**: what callers get from depth. **Locality**: what maintainers get from depth: change,
bugs and verification concentrate in one place.

## Principles

- **Depth is a property of the interface, not the implementation.** A deep module may be built
  from small internal parts; they just are not part of its interface.
- **The deletion test.** Imagine deleting the module. If complexity vanishes, it was a
  pass-through. If complexity reappears across its callers, it was earning its keep.
- **The interface is the test surface.** Callers and tests cross the same seam. If you want to
  test past the interface, the module is probably the wrong shape.
- **One adapter means a hypothetical seam; two adapters mean a real one.** Do not introduce an
  interface, abstract base class, Protocol, factory, or plugin point until something actually
  varies across it (production plus a test fake counts as two).

## Smells that mean the design is shallow

- A wrapper class or function that forwards to one other thing and adds nothing.
- Understanding one concept requires hopping through many tiny files or classes.
- A base class or Protocol with exactly one implementation and no test fake.
- Parameter lists that grow a flag per caller; boolean flags that select between behaviours.
- Pure helpers extracted only "for testability" while the bugs live in how they are wired.
- `utils`/`helpers`/`manager` modules that collect unrelated functions.

## Designing for testability

1. **Accept dependencies, do not create them.**

   ```python
   # Testable
   def process_order(order: Order, gateway: PaymentGateway) -> Receipt: ...

   # Hard to test
   def process_order(order: Order) -> Receipt:
       gateway = StripeGateway()
   ```

2. **Return results, do not produce side effects.**

   ```typescript
   // Testable
   function calculateDiscount(cart: Cart): Discount {}

   // Hard to test
   function applyDiscount(cart: Cart): void { cart.total -= discount }
   ```

3. **Small surface area.** Fewer entry points mean fewer tests; fewer parameters mean simpler
   setup.

When designing an interface, ask: can I reduce the number of entry points, simplify the
parameters, and hide more complexity inside?

## Process for a design or restructuring task

1. Read how the code is called today. List the callers and what each needs.
2. Apply the deletion test to every layer on the path. Merge pass-throughs into the module that
   does the work.
3. Decide where the seam goes and what sits behind it. Classify each dependency (see
   DEEPENING.md) to decide whether a port is justified.
4. For a significant interface, compare alternatives first (see DESIGN-IT-TWICE.md).
5. Write tests at the new interface, then delete tests that only exercised removed internals.
6. Verify with the project's own test and type-check commands.

## Rejected framings

- **Depth as lines of implementation per line of interface**: rewards padding. Depth is leverage.
- **Interface as only the type-level surface** (a `class`, `Protocol` or `interface` keyword):
  too narrow; it includes every fact a caller must know.
- **"Boundary"**: overloaded with domain-driven design's bounded context. Say seam or interface.

## Going deeper

- DEEPENING.md: dependency categories, seam discipline, and replace-don't-layer testing.
- DESIGN-IT-TWICE.md: have parallel subagents design the interface in radically different ways,
  then compare on depth, locality, and seam placement.

---

Adapted from the `codebase-design` skill in
[mattpocock/skills](https://github.com/mattpocock/skills) (MIT License, copyright 2026 Matt
Pocock). Changes: Python examples, a smell list, a process section, and Xerxes tool names.
