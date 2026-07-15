---
name: fix-license-headers
description: Normalize Xerxes Apache-2.0 headers with the native Bun maintenance command and verify TypeScript sources.
version: 2.0.0
tags: [license, headers, hygiene, typescript, bun, xerxes]
required_tools: [exec_command, ReadFile]
---

# When to use

Use this skill when a source, shell, YAML, or Docker file needs the repository
Apache-2.0 copyright header, or when a review reports a malformed header.

# How to use

## 1. Run the native maintenance command in preview mode

The maintenance entry point is TypeScript and runs with Bun:

```bash
bun run --cwd src/typescript fix-license-headers -- --root . --dry-run
```

Review the output, then run it without `--dry-run` only when the listed changes
are in scope:

```bash
bun run --cwd src/typescript fix-license-headers -- --root .
```

The command normalizes the repository's comment-style Apache header in the file
kinds it manages and skips generated directories such as `node_modules`,
`dist`, `build`, and `.git`.

## 2. Add a TypeScript header deliberately

Every TypeScript source file begins with the same compact native header used by
adjacent modules:

```ts
// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
```

The maintenance scanner does not insert TypeScript headers automatically, so
copy this header into a new `.ts` or `.tsx` file at creation time and review it
as part of the source change.

## 3. Verify the result

For a focused TypeScript tree, list files that lack the canonical copyright
line:

```bash
rg -L "^// Copyright 2026 The Xerxes-Agents Author" src/typescript/src -g '*.ts' -g '*.tsx'
bun test src/typescript/test/maintenanceScripts.test.ts
git diff --check
```

The `rg` command should produce no paths for a fully normalized tree. Inspect
the diff before any commit; the maintenance command must not rewrite generated
output or unrelated user changes.

## Common pitfalls

- Preserve a shell shebang as the first line; the comment header follows it.
- Preserve YAML document separators: the comment header belongs before `---`.
- Do not add an alternate license text or a different copyright year without an
  explicit repository-wide decision.
- Do not use a non-Bun script or external formatter to make header changes.
- If the scanner omits a file type, make the minimal manual header change and
  add a native test before broadening scanner behavior.
