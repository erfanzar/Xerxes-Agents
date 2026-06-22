---
name: test-workspace
description: Select and run the right repository checks for changed code. Use for test planning, affected-area verification, pre-commit behavior, and rejecting weak tests.
tags: [software-development, testing, workflow]
---

# Skill: Test Workspace

Load this when the task is choosing or running tests rather than designing new behavior. For multi-step debugging, load `run-research` first and use this as the verification layer.

## First Reads

- Project root docs such as `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, or `WORKSPACE.md` when present.
- `.agents/AGENTS.md`, `.agents/SKILL_MAP.md`, and `.agents/ops/OPS.md` when present.
- The touched package or module config: `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, or equivalent.
- Existing CI files under `.github/workflows/` or the project-native CI directory.

## Verification Selection

1. Identify changed files with `git diff --name-only` and `git status --short`.
2. Map each changed file to the nearest package/module test target.
3. Prefer the cheapest command that exercises the changed behavior.
4. Add broader checks only when the change touches shared runtime paths, public APIs, provider boundaries, migrations, or security.
5. If a claimed behavior depends on hardware, network, credentials, or an external service, say whether that path actually ran.

## Test Quality

Prefer tests that assert:

- Public API outputs and exceptions.
- Numerical, parsing, serialization, or protocol behavior against independent references.
- Shape, dtype, schema, state transition, cache, or persistence behavior.
- CLI parsed arguments or produced artifacts.
- Regression behavior tied to the bug or contract being changed.

Reject tests that only assert private helper calls, incidental log strings, constructors not raising, permanent skips, or production logic compared with itself.

## Reporting

Report exact commands and outcomes. If a command auto-fixes files, inspect and mention the resulting diff before treating the check as complete.
