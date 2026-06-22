---
name: docstring-swarm
description: Coordinate a docstring-only documentation pass across a codebase using disjoint file batches and verification. Use to add, repair, or expand module/class/function docstrings without changing behavior.
tags: [software-development, documentation, workflow]
---

# Skill: Docstring Swarm

Use this to document a codebase at scale. The output is a docstring-only diff: accurate documentation and no behavior changes.

## What Gets Documented

For the scoped target, document modules, classes, functions, and methods, public and private. For every function or method, cover inputs, outputs, raised errors, and non-obvious behavior. If `**kwargs` is typed from a config, dataclass, `TypedDict`, or `Unpacked[...]`, open that definition and document the meaningful fields.

## Hard Rules

1. Docstring-only diffs. Never change logic, signatures, imports, decorators, type annotations, or formatting unrelated to docstrings.
2. Never add standalone string literals after variable or attribute assignments.
3. Read implementation before writing. Do not invent parameters, return values, or behavior.
4. Update outdated docstrings instead of duplicating them.
5. Match the project's dominant docstring style.
6. Skip generated, vendored, migration, build, and cache files unless the user explicitly includes them.
7. Tests are opt-in unless the user asks to document tests.

## Workflow

1. Detect docstring convention by sampling existing files.
2. Build a worklist from in-scope source files, respecting `.gitignore`.
3. Partition files into disjoint batches; one agent owns exactly one batch.
4. Give each agent its file list, convention, scope, and hard rules.
5. Each agent reads whole files first, edits only docstrings, and verifies parseability.
6. After fan-out, run a syntax gate for every changed source file.
7. Confirm the diff shape is docstring-only.
8. Review a sample for accuracy and consistency; feed findings into a fix-up pass.

## Per-Agent Contract

Each batch agent must return:

- Files edited.
- Units documented, updated, or left as-is.
- Parse/compile result.
- Ambiguous or buggy code noticed but not modified.

## Definition Of Done

- Every changed file parses.
- Diff contains no code changes.
- No variable/attribute string-literal docstrings were introduced.
- Sampled docstrings match implementation.
- Final report includes counts, skipped files, verification, and remaining risk.
