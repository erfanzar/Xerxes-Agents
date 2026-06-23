---
name: author-skill
description: Create a new SKILL.md skill bundle for the Xerxes framework or for personal use in ~/.xerxes/skills/. Covers YAML frontmatter, markdown body, discovery, and validation.
version: 1.0.0
tags: [skills, authoring, markdown, xerxes]
required_tools: [ReadFile, WriteFile, FileEditTool]
---

# When to use

Use this skill when authoring a new skill for the Xerxes framework. A "skill" is a self-contained markdown instruction set with YAML frontmatter that teaches the agent how to perform a specific repeated workflow.

Examples:
- A skill for creating GitHub PRs with proper formatting and checklist
- A skill for analyzing Python code for security issues (bandit, ruff, etc.)
- A skill for writing Sphinx documentation from code docstrings
- A skill for deploying a Docker container to a specific cloud provider

Do NOT use this for:
- Adding a new tool module (use `add-tool-module` skill)
- Adding a new agent spec (use `add-agent-spec` skill)
- Writing general documentation (skills are executable instructions, not narrative docs)

# How to use

## 1. Inspect an existing skill

Read `src/python/xerxes/skills/xerxes-agent/SKILL.md` or any skill in `src/python/xerxes/extensions/skill_sources/` to understand the format.

A skill is a single `SKILL.md` file with YAML frontmatter and a markdown body:

```markdown
---
name: my-skill
description: One-line description of what this skill does.
version: 1.0.0
tags: [python, security, xerxes]
required_tools: [ReadFile, ExecuteShell]
---

# When to use

Use this skill when you need to ...

# How to use

## Step 1: Do the first thing
...

## Step 2: Do the second thing
...

## Common pitfalls
...
```

## 2. Choose a skill directory

Skills can live in two places:

**A. Bundled skills (shipped with the framework):**
```
src/python/xerxes/skills/<skill-name>/SKILL.md
```

**B. User skills (personal, auto-discovered):**
```
~/.xerxes/skills/<skill-name>/SKILL.md
```

**Rules:**
- For framework-bundled skills: use `src/python/xerxes/skills/<skill-name>/`.
- For personal/user skills: use `~/.xerxes/skills/<skill-name>/`.
- The directory name should be `kebab-case` and match the `name` field in frontmatter.

## 3. Write the YAML frontmatter

```yaml
---
name: my-skill                    # Required. Kebab-case, unique.
description: One-line summary.   # Required. Shown in skill listings.
version: 1.0.0                   # Required. SemVer.
tags: [python, lint, security]    # Optional. Array of strings.
required_tools:                   # Optional. List of tool names.
  - ReadFile
  - ExecuteShell
  - GrepTool
author: Your Name                # Optional.
dependencies:                    # Optional. Python packages needed.
  - ruff>=0.7.0
platforms: [linux, macos]        # Optional. OS constraints.
setup_command: pip install ruff  # Optional. Command to run before first use.
subcommands:                     # Optional. Auto-detected sub-skills.
  - check
  - fix
  - format
---
```

**Rules:**
- `name` must be kebab-case and unique across all discovered skills.
- `description` should be a single sentence, <120 characters.
- `version` must follow SemVer (`MAJOR.MINOR.PATCH`).
- `required_tools` lists the tool class names (e.g., `ReadFile`, `ExecuteShell`) that this skill will use. The framework uses this for permission filtering.
- `subcommands` are auto-detected if you create `references/<subcommand>-workflow.md` files alongside `SKILL.md`.

## 4. Write the markdown body

The body must contain at least two sections:

### `# When to use`
Describe the trigger condition — when should an agent invoke this skill? Be specific:

- Bad: "Use this when you need to do things."
- Good: "Use this skill when you need to run a security scan on a Python file and generate a SARIF report."

### `# How to use`
Step-by-step instructions. Use `##` subsections for each step. Include:

- Exact file paths to read/edit
- Exact commands to run
- Exact code snippets to insert
- Verification steps

Example:

```markdown
# How to use

## 1. Read the target file

```python
from xerxes.tools import ReadFile
ReadFile(file_path="src/python/xerxes/my_module.py")
```

## 2. Run the security scan

```bash
uv run bandit -r src/python/xerxes/my_module.py -f json
```

## 3. Report findings

Summarize the bandit output in a concise table.
```

### Optional sections

- `## Common pitfalls` — mistakes users make when using this skill
- `## Example` — a concrete before/after example
- `## References` — links to external docs, RFCs, standards

## 5. Add subcommands (optional)

If the skill has multiple modes, create `references/<subcommand>-workflow.md` files:

```
~/.xerxes/skills/my-skill/
├── SKILL.md
└── references/
    ├── check-workflow.md
    └── fix-workflow.md
```

Each subcommand file follows the same markdown structure but focuses on one mode. The `SkillRegistry` auto-detects these and exposes them as `my-skill:check` and `my-skill:fix`.

## 6. Validate the skill

Run the framework's skill validation (if available) or manually check:

```bash
uv run python -c "
from xerxes.extensions.skills import SkillRegistry, parse_skill_md
skill_md = open('~/.xerxes/skills/my-skill/SKILL.md').read()
meta = parse_skill_md(skill_md)
print(meta.name)
print(meta.version)
print(meta.tags)
"
```

## 7. Test discovery

Trigger skill discovery and confirm your skill appears:

```bash
uv run python -c "
from xerxes.extensions.skills import SkillRegistry
registry = SkillRegistry()
registry.discover()
print('my-skill' in [s.name for s in registry.list_skills()])
"
```

## 8. Run lint on the skill file

```bash
uv run ruff check --fix ~/.xerxes/skills/my-skill/SKILL.md
```

(Note: ruff won't lint markdown, but it will catch any Python code blocks with syntax errors if you run `ruff` on extracted code.)

## Common pitfalls

- **Missing `---` delimiters:** The frontmatter must be enclosed in `---` on its own lines. Missing delimiters cause `parse_skill_md()` to fail.
- **Invalid YAML in frontmatter:** Common mistakes include unquoted strings with colons (e.g., `description: Use this: when ...`), trailing commas, and tabs instead of spaces.
- **Duplicate skill names:** If two skills have the same `name`, the one discovered first wins. Use unique names.
- **File outside the skill directory:** The `SKILL.md` must be inside the skill directory. A standalone `SKILL.md` in `~/.xerxes/skills/` will not be discovered.
- **Hardcoded absolute paths:** Skill instructions should use relative paths or explain how to determine the correct path. Don't hardcode `/home/username/...`.
- **Missing `# When to use` or `# How to use`:** These sections are required by the `Skill` dataclass validation. Missing them causes the skill to be rejected.
