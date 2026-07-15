---
name: author-skill
description: Author a Bun-native Xerxes SKILL.md bundle with valid metadata, safe assets, and discovery tests.
version: 2.0.0
tags: [skills, authoring, markdown, typescript, bun, xerxes]
required_tools: [ReadFile, WriteFile, FileEditTool]
---

# When to use

Use this skill when authoring a reusable Xerxes workflow in `SKILL.md` format.
A skill is an instruction bundle with YAML frontmatter, Markdown directions,
and optional safe references or templates. It is not a substitute for a native
tool implementation.

# How to use

## 1. Choose a discovery location

Read `src/typescript/src/extensions/skills.ts` before creating a skill.

- Framework-bundled skills live in `src/typescript/skills/<skill-name>/`.
- Project skills can live in `skills/<skill-name>/` or
  `.agents/skills/<skill-name>/`.
- User skills live in `~/.xerxes/skills/<skill-name>/`.

The registry discovers roots in priority order; the first skill with a given
frontmatter `name` wins. Choose a unique kebab-case name and never leave a
duplicate bundled name behind.

## 2. Write valid frontmatter

Create `SKILL.md` with a concise, factual description:

```markdown
---
name: release-notes
description: Draft release notes from a verified native repository diff.
version: 1.0.0
tags: [release, docs, bun]
required_tools: [ReadFile, exec_command]
resources: [references]
platforms: [macos, linux]
---

# When to use

Use this skill when preparing release notes after the relevant Bun checks pass.

# How to use

1. Inspect the intended diff and test results.
2. Summarize only verified changes.
3. Link any reference material from `references/`.
```

`dependencies` names other discoverable skills, not operating-system or
package-manager dependencies. `required_tools` is checked against the host's
available tool surface. Keep `setup_command` absent unless the workflow truly
requires an explicit, user-authorized setup step.

## 3. Keep executable behavior native and explicit

Instructions may invoke `bun`, a documented native CLI, or a host-provided
tool. If a workflow needs browser automation, credentials, cloud APIs, or
hardware, state the explicit host boundary and expected configuration. Do not
include a Python script, package installation, hidden credential discovery, or
unsafe bypass workflow in a bundled skill.

Place reference Markdown under `references/` and reusable templates under
`templates/`. Preserve only safe assets. Bundled skills are copied recursively
to the runtime distribution by `src/typescript/scripts/copyBundledSkills.ts`.

## 4. Validate discovery

Add or update the closest Bun test. For a bundled skill, verify its name is
unique and its required safe assets survive the copy step.

```bash
bun test src/typescript/test/skills.test.ts
bun test src/typescript/test/bundledSkillAssets.test.ts
bun run --cwd src/typescript build
```

Run `git diff --check` after adding templates or copied assets. Do not edit the
generated `src/typescript/dist/skills/` tree manually.

## Common pitfalls

- Frontmatter must start and end with `---` on its own line.
- The registry accepts a small YAML-like metadata grammar; prefer scalar values
  and string lists over complex nested structures.
- A duplicate name shadows later roots, so test discovery rather than assuming
  directory names are enough.
- Keep instructions specific enough to execute, but do not hard-code a private
  home directory, token, or machine-specific endpoint.
- A skill that requires a real integration must say so and return an actionable
  failure if the required host port is unavailable.
