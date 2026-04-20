---
name: xerxes-agent
description: Multi-agent delegation skill for Xerxes. Instructs the model on when and how to spawn sub-agents, delegate parallel tasks, manage background jobs, and coordinate multi-agent workflows using Xerxes's built-in agent spawning tools.
version: 3.0.0
author: Xerxes Agent
license: MIT
metadata:
  tags: [xerxes, multi-agent, delegation, spawning, subagent, cortex, parallel]
  homepage: https://github.com/erfanzar/Xerxes-Agents
  related_skills: [claude-code, codex, deepscan]
---

# Xerxes Multi-Agent Delegation

You are running inside Xerxes, a multi-agent orchestration framework. You have access to agent-spawning tools that let you delegate work to specialized sub-agents. **Use them aggressively.** Do not try to do everything yourself in a single monolithic response.

## Core Principle

If a task has **more than 2 distinct sub-tasks**, **spawns files in multiple directories**, or **requires research + implementation + review**, spawn sub-agents. Parallel execution is faster and produces better results than sequential monolithic work.

## Available Agent-Spawn Tools

### 1. AgentTool — Spawn a single sub-agent

Use this for one-off delegation.

```
AgentTool(
  prompt="Write a FastAPI router for user CRUD with JWT auth",
  subagent_type="coder",
  name="backend-dev",
  wait=true
)
```

Parameters:
- `prompt` (required): Task description. Be specific. Include file paths, constraints, and success criteria.
- `subagent_type`: `general-purpose` (default), `coder`, `reviewer`, `researcher`, `tester`, `planner`, `data-analyst`
- `isolation`: `""` (normal) or `"worktree"` (git worktree isolation — use when the sub-agent will edit code to avoid git conflicts)
- `name`: Human-readable name. Use it later with SendMessageTool.
- `model`: Model override. Leave empty to inherit from parent.
- `wait`: `true` (default) blocks until done. `false` runs in background.

**When to use `wait=false`:**
- Long-running tasks (research, data processing, builds)
- Tasks where you want to spawn multiple agents in parallel, then gather results
- Fire-and-forget background jobs

### 2. SpawnAgents — Spawn multiple agents in parallel

Use this when you need **3+ agents** working simultaneously.

```
SpawnAgents(
  agents=[
    {"prompt": "Design REST API schema for a todo app", "name": "api-designer", "subagent_type": "planner"},
    {"prompt": "Write React components for todo list UI", "name": "frontend-dev", "subagent_type": "coder"},
    {"prompt": "Write pytest tests for todo CRUD", "name": "tester", "subagent_type": "tester"}
  ],
  wait=true
)
```

Returns a JSON array of results (name, status, result) when `wait=true`, or task snapshots when `wait=false`.

### 3. TaskCreateTool — Fire-and-forget background task

Alias for `AgentTool(..., wait=false)`. Use when you don't need the result immediately.

```
TaskCreateTool(
  prompt="Scrape https://example.com/docs and summarize key endpoints",
  name="doc-scraper",
  subagent_type="researcher"
)
```

Returns a task snapshot with `id`, `name`, `status`, `result`. Save the `id` or `name` for later checks.

### 4. SendMessageTool — Message a running agent

Use this to send follow-up instructions to a background agent that is still running.

```
SendMessageTool(
  target="backend-dev",
  message="Add rate limiting middleware to the auth router"
)
```

**Important:** Agents spawned with `wait=true` finish and cannot receive follow-up messages. Always spawn with `wait=false` if you plan to send follow-ups.

### 5. TaskListTool — Check running tasks

```
TaskListTool()
```

Returns all tasks with their status (`pending`, `running`, `completed`, `failed`, `cancelled`).

### 6. TaskGetTool — Get a specific task's result

```
TaskGetTool(task_id="abc123")
```

### 7. TaskStopTool — Cancel a running task

```
TaskStopTool(task_id="abc123")
```

---

## Agent Types & When to Use Them

| Type | Best for |
|------|----------|
| `general-purpose` | Default. Jack of all trades. |
| `coder` | Writing code, refactoring, debugging. Use with `isolation="worktree"` for parallel code work. |
| `reviewer` | Code review, PR feedback, security audits. |
| `researcher` | Web research, documentation reading, data gathering. |
| `tester` | Writing tests, reproducing bugs, QA. |
| `planner` | Architecture design, task breakdown, API design. |
| `data-analyst` | Data processing, SQL, pandas, visualization. |

---

## Delegation Patterns

### Pattern A: Parallel Feature Team

Spawn 3 agents simultaneously for a full-stack feature:

```
SpawnAgents(
  agents=[
    {"prompt": "Design DB schema and FastAPI routes for /users", "name": "backend", "subagent_type": "coder", "isolation": "worktree"},
    {"prompt": "Build React components for user profile page", "name": "frontend", "subagent_type": "coder", "isolation": "worktree"},
    {"prompt": "Write integration tests for user flows", "name": "qa", "subagent_type": "tester"}
  ],
  wait=true
)
```

Then review all results and merge.

### Pattern B: Research → Implement → Review Pipeline

Sequential delegation with handoffs:

```
# Step 1: Research
research = AgentTool(prompt="Research OAuth2 PKCE best practices for SPA auth", subagent_type="researcher", name="researcher", wait=true)

# Step 2: Implement based on research
AgentTool(prompt=f"Implement OAuth2 PKCE login flow. Research notes: {research}", subagent_type="coder", name="implementer", wait=true)

# Step 3: Review
AgentTool(prompt="Review the OAuth implementation for security issues", subagent_type="reviewer", name="reviewer", wait=true)
```

### Pattern C: Background Worker + Follow-ups

```
# Spawn background researcher
TaskCreateTool(prompt="Monitor https://news.ycombinator.com for AI breakthroughs. Save summaries to ~/ai-news/", name="news-monitor", subagent_type="researcher")

# Later...
SendMessageTool(target="news-monitor", message="Also track LLM pricing changes")

# Check progress
TaskListTool()
```

### Pattern D: Hierarchical Manager

Use yourself as a manager that spawns workers, collects results, and synthesizes:

```
# You (manager) break down the project
results = SpawnAgents(
  agents=[
    {"prompt": "Task A...", "name": "worker-a"},
    {"prompt": "Task B...", "name": "worker-b"},
    {"prompt": "Task C...", "name": "worker-c"}
  ],
  wait=true
)

# You synthesize the results and present final output to user
```

---

## Worktree Isolation

When spawning multiple `coder` agents that edit the same git repo, use `isolation="worktree"`. Each agent gets its own git worktree branch, preventing file conflicts. After they finish, merge the branches or cherry-pick changes.

```
AgentTool(prompt="Refactor auth module", subagent_type="coder", isolation="worktree", name="auth-refactor")
AgentTool(prompt="Add caching layer", subagent_type="coder", isolation="worktree", name="cache-layer")
```

**Without worktree isolation**, two agents editing the same file will overwrite each other's changes.

---

## Best Practices

1. **Be specific in prompts.** Include file paths, expected output format, constraints, and success criteria. Vague prompts produce vague results.

2. **Name your agents.** Use descriptive names (`backend-dev`, `api-designer`, `bug-repro`) so you can reference them with SendMessageTool.

3. **Use `wait=false` for exploration.** If you're not sure how long a task takes, spawn it in the background and check later.

4. **Check `TaskListTool()` before spawning duplicates.** Don't spawn 5 identical agents.

5. **Set `isolation="worktree"` for parallel code work.** This is the #1 cause of lost work in multi-agent setups.

6. **One agent = one concern.** Don't ask a single agent to "design the API, implement the backend, build the frontend, and write tests." That's 4 agents.

7. **Aggregate results yourself.** Sub-agents return raw results. You (the parent) should synthesize, deduplicate, and format the final answer for the user.

8. **Fail fast.** If a sub-agent fails, don't retry blindly. Inspect the error, fix the prompt or constraints, and respawn.

---

## When NOT to Delegate

- **Simple Q&A** ("What is the capital of France?") — answer directly.
- **Single-file edits** — use WriteFile/ReadFile directly.
- **Context-free math** — use Calculator directly.
- **User wants to chat** — stay in conversational mode.

---

## Xerxes Reference (for context)

### Install
```bash
curl -fsSL https://raw.githubusercontent.com/erfanzar/Xerxes-Agents/main/scripts/install.sh | sh
```

### Paths
- `~/.xerxes/` — config, skills, sessions, memory
- `~/.xerxes/skills/` — user skills (SKILL.md directories)

### In-Session Slash Commands
```
/skills              List available skills
/skill <name>        Invoke a skill
/skill-create <name> Create a new skill
/agents              List registered agents
/plan <objective>    Create multi-step plan
/tools               List active tools
/provider            Setup provider profile
/model               Switch model
/config              Show config
```

### Environment Variables
- `XERXES_HOME` — override data directory
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, etc.

### Docs
- https://erfanzar.github.io/Xerxes-Agents
- https://github.com/erfanzar/Xerxes-Agents
