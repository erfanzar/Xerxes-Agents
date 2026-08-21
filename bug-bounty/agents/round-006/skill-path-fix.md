# Round 006 — SkillBundleStore path containment

## Fix

- Added a `WorkspacePathResolver` rooted at the host-provided skill directory.
- Routed `get`, `create`, `update`, and `delete` targets through resolver containment.
- Rechecked mutation targets immediately before atomic write or removal.
- Added regression coverage using skill-directory symlinks that point outside the configured root; all four operations reject the escape and preserve the external file.

## Verification

- `bun test xerxes/test/agentMetaTools.test.ts` — 5 pass, 0 fail, 23 assertions.
- `bun run --cwd xerxes check` — runtime and UI TypeScript checks passed.
- `git diff --check -- xerxes/src/tools/agentMetaTools.ts xerxes/test/agentMetaTools.test.ts` — passed.
