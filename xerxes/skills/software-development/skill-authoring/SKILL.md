---
name: skill-authoring
description: Author Xerxes bundled SKILL.md files that pass the linter.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [skills, authoring, documentation, conventions]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/software-development/hermes-agent-skill-authoring/SKILL.md
---

# Authoring Xerxes Skills (in-repo)

## Overview

This skill covers writing a bundled skill for this repository: a
`SKILL.md` at `xerxes/skills/<category>/<name>/SKILL.md`, committed and
shipped with the runtime (`xerxes/scripts/copyBundledSkills.ts` copies
`xerxes/skills/` recursively into the distribution). Xerxes skills are
agentskills.io-compatible and ship **no script assets** — everything is
self-contained instruction text.

A strict linter (`xerxes/src/extensions/skillLint.ts`, exercised by
`xerxes/test/skills.test.ts`) validates every bundled skill. Meeting its
rules up front is cheaper than a salvage pass later.

## When to Use

- The user asks to add or edit a skill inside this repository
- A reusable workflow should ship with Xerxes itself

Do not use for user-local skills in a user's own skills directory.

## Required Frontmatter

Allowed keys — exactly these, no others, no duplicates:

```yaml
---
name: my-skill-name
description: One-line capability statement.
version: 1.0.0
author: Real Name (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [short, descriptive, tags]
source: <original skill URL when adapted, otherwise the origin>
---
```

Hard rules enforced by the linter:

- `name` MUST equal the directory basename (lowercase, hyphens).
- `description` is required and must be one line.
- Unknown keys, duplicate keys, nested mappings, block scalars, and
  multiline lists all fail.
- `platforms` and `tags` are single-line inline `[a, b]` lists.
- Frontmatter starts at byte 0 with `---` (no leading blank line or BOM)
  and closes with a `---` line before the body.
- The body must be non-empty, and prompt-injection-style phrasings are
  blocked: never write "ignore previous/all/above/prior instructions" or
  "disregard ... instructions/rules/guidelines" anywhere in the file.

## Body Structure

```
# <Skill>
2-3 sentence intro: what it does, what it does not do.

## When to Use       — bulleted triggers (+ "Do not use for" counter-triggers)
## Prerequisites     — exact installs, credentials, host requirements
## How to Run        — canonical invocation
## Procedure         — numbered steps, each with a checkable completion criterion
## Pitfalls          — known limits, things that look broken but are not
## Verification      — how to prove the skill worked
```

When to Use + actionable body + Pitfalls + Verification are the minimum.
Cut marketing intros and no-op "be careful" lines that do not change
behavior. Keep the file 60-140 lines.

## Tool References

Name the agent's tools, not raw shell equivalences: shell/terminal tool,
file read tool, file write tool, web fetch tool, screenshot/vision
analysis, scheduled trigger, and the agent spawning tools. Reference only
skills that exist as Xerxes bundled skills — check `xerxes/skills/` before
adding a cross-reference, and never point at a skill that exists only in
another branch or plan.

## No Script Assets

If adapting a source skill that ships `scripts/*.py` or relies on helper
CLIs living next to its `SKILL.md`, rewrite those invocations as equivalent
inline snippets in the body. Executable or privileged integrations belong
in native TypeScript behind an explicit host port, not in a skill bundle.

## Writing Quality Principles

1. Optimize for process predictability: if a line does not change behavior,
   cut it.
2. The description is paid for every turn; details go in the body.
3. End steps with completion criteria — "every modified file accounted for"
   beats "summarize changes".
4. Co-locate rules with the concept they govern.
5. Use strong leading words ("tight loop", "root cause", "regression test").
6. Never bake in machine-local paths like `/home/<you>/...`; use
   repo-relative paths.

## Workflow

1. **Survey peers** in the target category under `xerxes/skills/` and read
   2-3 peer SKILL.md files to match tone and structure. Prefer extending an
   existing skill over creating a narrow sibling. Do not write router or
   index skills whose content is only pointers to siblings.
2. **Draft** the file at `xerxes/skills/<category>/<name>/SKILL.md` with the
   file write tool.
3. **Validate locally** by running the focused test:
   ```bash
   bun test xerxes/test/skills.test.ts
   ```
   The linter checks frontmatter keys, name/directory match, injection
   phrasings, and resource containment. Fix every named diagnosis.
4. **Run the broader gate** for cross-cutting changes:
   `bun run check && bun run test && bun run build`.
5. **Commit on the active branch** with a Conventional Commit message — but
   only when the user explicitly asks; never stage or commit unprompted.

Note: the running session's skill loader is initialized at session start;
it will not see a newly written skill until a new session. That is
expected, not a bug.

## Common Pitfalls

1. Adding a frontmatter key the strict linter does not know (for example
   `license`, `metadata`, or `prerequisites`) — use only the seven allowed
   keys.
2. Duplicate keys: validation fails closed instead of letting the last
   occurrence win.
3. Leading whitespace before `---` or a fenced ```yaml wrapper around the
   frontmatter — both fail parsing.
4. `name` not matching the directory basename.
5. Cross-referencing a skill that does not exist under `xerxes/skills/`.
6. Bundling scripts or binaries next to the SKILL.md — inline the
   equivalent commands instead.
7. Injection-style phrasings left over from a source skill — rephrase
   neutrally.
8. Duplicating a peer skill; survey the category first.

## Verification Checklist

- [ ] File at `xerxes/skills/<category>/<name>/SKILL.md`, name matches dir
- [ ] Frontmatter uses only the seven allowed keys, no duplicates
- [ ] `platforms` and `tags` are single-line inline lists
- [ ] No injection-style phrasings anywhere in the file
- [ ] Cross-references resolve to existing bundled skills
- [ ] No script assets; external helpers rewritten as inline snippets
- [ ] 60-140 lines; ends with the hermes-agent adaptation attribution
- [ ] `bun test xerxes/test/skills.test.ts` passes

---

Adapted from the `hermes-agent-skill-authoring` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Hermes Agent.
