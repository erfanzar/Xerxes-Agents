---
name: add-agent-spec
description: Scaffold a new YAML agent specification in the Xerxes agents/default/ directory. Covers inheritance, system prompts, tool whitelists, and subagent hierarchies.
version: 1.0.0
tags: [agents, yaml, spec, orchestration, xerxes]
required_tools: [ReadFile, WriteFile, FileEditTool, GlobTool]
---

# When to use

Use this skill when creating a new built-in agent definition for the Xerxes framework. Agent specs are YAML files that define role, system prompt, model, tools, and subagent hierarchies.

Examples:
- A new specialized coding agent (e.g., "rust-expert", "frontend-reviewer")
- A new research agent (e.g., "medical-researcher", "legal-analyst")
- A new orchestration agent (e.g., "ci-pipeline-agent", "docs-writer")

Do NOT use this for:
- Adding a new LLM provider (use `add-llm-provider` skill)
- Adding a new tool (use `add-tool-module` skill)
- Adding runtime agent logic (agent specs are declarative YAML, not code)

# How to use

## 1. Inspect the base agent spec

Read `src/python/xerxes/agents/default/agent.yaml` to understand the base spec:

```yaml
version: 1
agent:
  name: xerxes
  description: General-purpose agent
  system_prompt_path: ./system.md
  system_prompt_args:
    ROLE: "You are Xerxes, a helpful AI assistant."
    ROLE_ADDITIONAL: ""
  model: claude-sonnet-4-6
  max_depth: 5
  isolation: false
  tools: []
  allowed_tools: []
  exclude_tools: []
  subagents: []
```

Also read `src/python/xerxes/agents/default/coder.yaml` and `planner.yaml` for examples of extending the base agent.

## 2. Read the system prompt template

Read `src/python/xerxes/agents/default/system.md` to understand the prompt template syntax. Agent specs reference `system_prompt_path` and inject `system_prompt_args` into the template.

## 3. Create the new agent spec

Create a new YAML file in `src/python/xerxes/agents/default/` (e.g., `reviewer.yaml`):

```yaml
version: 1
agent:
  name: reviewer
  description: Code review specialist
  extend: ./agent.yaml
  system_prompt_args:
    ROLE_ADDITIONAL: >
      You are a senior code reviewer. Focus on correctness, performance,
      security, and maintainability. Be concise but thorough.
  model: claude-sonnet-4-6
  max_depth: 3
  allowed_tools:
    - ReadFile
    - WriteFile
    - FileEditTool
    - GrepTool
    - GlobTool
  exclude_tools: []
  subagents: []
```

**Rules:**
- `version: 1` is mandatory.
- `extend: ./agent.yaml` inherits the base tool list, system prompt template, and defaults. Most specs should extend the base agent.
- `name` must be unique across all agent specs. Use `snake_case`.
- `description` is shown to users in the `/agents` slash command.
- `system_prompt_args.ROLE_ADDITIONAL` overrides or extends the base prompt. Keep it focused and role-specific.
- `model` is the default model for this agent. Users can override it at runtime.
- `max_depth` controls how many levels of subagent delegation this agent can trigger (default 5).
- `isolation: true` means the agent runs in a subprocess with its own memory. Use for untrusted or high-risk agents.

## 4. Tool configuration

The tool list is the most important part of an agent spec. There are three ways to control it:

### Option A: Inherit everything (default)
```yaml
agent:
  extend: ./agent.yaml
  # No tool keys → inherits full base tool list
```

### Option B: Whitelist specific tools
```yaml
agent:
  extend: ./agent.yaml
  allowed_tools:
    - ReadFile
    - GrepTool
    - GlobTool
```

### Option C: Inherit all but exclude specific tools
```yaml
agent:
  extend: ./agent.yaml
  exclude_tools:
    - exec_command
    - WriteFile
    - FileEditTool
```

**Rules:**
- `allowed_tools` is an **exclusive** allow-list. If present, the agent can ONLY use those tools.
- `exclude_tools` subtracts from the inherited list. Use this for "safe" agents that shouldn't have write access.
- Tool names must match the class names exactly (e.g., `ReadFile`, not `read_file` or `readfile`).
- See `src/python/xerxes/tools/__init__.py` `TOOL_CATEGORIES` for the full list of available tools.

## 5. Subagent hierarchy

If this agent can delegate to child agents, define them in `subagents`:

```yaml
agent:
  subagents:
    - name: junior_reviewer
      description: Junior code reviewer for style and formatting
      model: gpt-4o-mini
      allowed_tools:
        - ReadFile
        - GrepTool
      max_depth: 2
```

**Rules:**
- Subagent definitions are inline YAML objects, not references to external files.
- Subagents inherit the parent's `model` if not specified.
- Subagents cannot have their own `subagents` (max one level of nesting in YAML; deeper nesting is handled by the `max_depth` budget).
- Keep subagent lists small (2-5). Too many subagents degrade the orchestration quality.

## 6. Verify the spec loads correctly

Run:

```bash
uv run python -c "
from xerxes.agents.agentspec import load_agent_spec
spec = load_agent_spec('src/python/xerxes/agents/default/reviewer.yaml')
print(spec.name)
print(spec.allowed_tools)
print(spec.subagents)
"
```

If `load_agent_spec()` fails, the error will tell you which YAML key is invalid.

## 7. Add a test

Add a test in `tests/agents/test_agentspec.py` or create `tests/agents/test_reviewer_agent.py`:

```python
import pytest
from xerxes.agents.agentspec import load_agent_spec


class TestReviewerAgent:
    def test_loads_successfully(self):
        spec = load_agent_spec("src/python/xerxes/agents/default/reviewer.yaml")
        assert spec.name == "reviewer"
        assert "ReadFile" in spec.allowed_tools
        assert "exec_command" not in spec.allowed_tools

    def test_inherits_from_base(self):
        spec = load_agent_spec("src/python/xerxes/agents/default/reviewer.yaml")
        assert spec.system_prompt_path is not None
        assert spec.max_depth <= 5
```

## 8. Update the agent registry (if needed)

Some parts of the framework may require explicit registration of new agent specs. Check:
- `src/python/xerxes/agents/__init__.py` for `__all__` re-exports.
- `src/python/xerxes/xerxes.py` for hardcoded agent lists or fallback logic.
- `src/python/xerxes/agents/default/` for an index or manifest file.

## Common pitfalls

- **YAML syntax errors:** Indentation must be consistent (2 spaces per level). Mixed tabs/spaces will cause cryptic `yaml.scanner.ScannerError`.
- **Missing `extend`:** If you don't extend `./agent.yaml`, you must define ALL required fields (system_prompt_path, model, max_depth, etc.) yourself. It's easier to extend.
- **Tool name typos:** `allowed_tools` and `exclude_tools` use the exact Python class name. `readfile` or `read_file` will silently be ignored because the tool doesn't exist.
- **Overly broad tool lists:** Giving an agent `exec_command` or `WriteFile` is high-risk. Use `exclude_tools` to remove dangerous tools from inherited lists.
- **Circular subagent references:** Don't create A → B → A delegation loops. The framework's `LoopDetector` will catch them at runtime, but it's better to design hierarchies as trees.
- **System prompt bloat:** `ROLE_ADDITIONAL` should be 2-5 sentences. Long system prompts consume context window and degrade performance.
