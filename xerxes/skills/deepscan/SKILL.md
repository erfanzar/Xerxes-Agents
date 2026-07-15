---
name: deepscan
description: >-
  Spawns a swarm of specialized agents to deeply analyze the current project
  and compile comprehensive findings into project-scoped agent memory.
  ALWAYS runs agents in parallel — sequential analysis is unacceptable.
version: "2.2"
tags: [analysis, swarm, project, deepscan, multi-agent, parallel]
required_tools: [SpawnAgents, agent_memory_status, agent_memory_read, agent_memory_write, agent_memory_append, agent_memory_journal]
---

# DeepScan Skill

**PARALLEL IS MANDATORY.** DeepScan ALWAYS spawns all analysis agents simultaneously. Running them sequentially is a bug that wastes time and produces worse results.

Spawns a coordinated swarm of 8 specialized agents to perform comprehensive analysis of the current project, then compiles all findings into the project's persistent agent memory at `deepscan/AGENT_NOTES.md`.

## Project Memory Contract

DeepScan output is project-specific memory, not a workspace temp artifact.

- Final report: `agent_memory_write("project", "deepscan/AGENT_NOTES.md", report)`
- Required per-agent findings: `agent_memory_write("project", "deepscan/findings/<agent-name>.md", findings)`
- Discovery index: append a short pointer to project `MEMORY.md` so future agents know the report exists.
- Journal: add a one-line project journal entry after saving the report.

Before spawning agents, call `agent_memory_status()` and confirm project memory is available. If project memory is unavailable or any memory write fails, stop and report the memory error. **Do not fall back to `tmp-files`, repo-local report files, or shell-created scratch files.**

## Subagent Context and Return Contract

Each subagent must preserve detailed findings in project memory and keep its final response small.

- There is **no artificial tool-call cap** for DeepScan subagents. Use as many tool calls as needed to produce a useful project analysis.
- Do not dump full files by default. Use inventory commands (`rg --files`, `find ... -maxdepth`, `wc -l`) and chunked file reads (`ReadFile(file_path=..., offset=..., limit=...)`). Continue with the next offset when more context is needed. Use `limit=-1` only when the whole file is intentionally required.
- Do not read massive source files, lockfiles, generated files, or full glob output unless the prompt specifically requires it.
- If token pressure, timeout risk, or uncertainty appears: compact current findings into the assigned memory path immediately, then continue from memory or mark the report partial and return the path plus what remains unknown.
- The subagent's final response must be **latest agent content only**:
  - `memory_path: deepscan/findings/<agent-name>.md`
  - `status: complete|partial`
  - 5-8 concise bullets with key findings and gaps
  - basic stats such as files sampled, commands used, confidence
- Do **not** return raw tool logs, chain-of-thought, file dumps, huge command output, or the full findings body. The parent reads the full report from project memory when compiling.

## Execution Rule

**DO NOT** analyze the project yourself. **DO NOT** read files one by one. **DO NOT** write the report incrementally.

Your ONLY job is to:
1. Confirm project agent memory is available
2. Spawn **all 8 agents in parallel** via `SpawnAgents`
3. Wait for all results
4. Compile the final report
5. Save the report to project agent memory and index it

If you find yourself reading source code or writing analysis paragraphs directly, you have FAILED this skill. Stop and spawn the agents.

## Prerequisites

```
agent_memory_status()
```

The status must show project memory availability. Use the returned `project_dir` in the report overview when present, but always address writes through the `agent_memory_*` tools with `scope="project"`.

## Swarm Architecture — ALL 8 AGENTS IN PARALLEL

Use `SpawnAgents` with `wait=true`. Every agent gets the `researcher` subtype. Each subagent writes full findings to its assigned project-memory path and returns only the compact contract above. Subagents must not write to `tmp-files` or workspace files.

```
SpawnAgents(
  agents=[
    {
      "prompt": "Analyze project structure. Scan all directories up to 4 levels deep using inventory commands and chunked reads. Document tree, key dirs, config files, module structure. Write full findings to project memory at `deepscan/findings/structure-analyzer.md`, then return only `memory_path`, `status`, 5-8 concise bullets, stats, confidence, and gaps. No artificial tool-call cap: use as many tool calls as needed, but use `ReadFile(offset=..., limit=...)` by default and `limit=-1` only for intentional full-file reads. Do not return raw tool logs, reasoning, file dumps, or large command output. Do not write tmp-files or workspace files.",
      "title": "Analyze project structure",
      "name": "structure-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze technology stack. Identify languages, frameworks, dependencies, build systems, type checkers using config files and representative manifests with chunked reads. Write full findings to project memory at `deepscan/findings/tech-analyzer.md`, then return only `memory_path`, `status`, 5-8 concise bullets, stats, confidence, and gaps. No artificial tool-call cap: use as many tool calls as needed, but use `ReadFile(offset=..., limit=...)` by default and `limit=-1` only for intentional full-file reads. Do not return raw tool logs, reasoning, file dumps, or large command output. Do not write tmp-files or workspace files.",
      "title": "Analyze technology stack",
      "name": "tech-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze code patterns and architecture. Identify design patterns, architectural style, module relationships, separation of concerns using targeted grep and chunked representative file reads. Write full findings to project memory at `deepscan/findings/arch-analyzer.md`, then return only `memory_path`, `status`, 5-8 concise bullets, stats, confidence, and gaps. No artificial tool-call cap: use as many tool calls as needed, but use `ReadFile(offset=..., limit=...)` by default and `limit=-1` only for intentional full-file reads. Do not return raw tool logs, reasoning, file dumps, or large command output. Do not write tmp-files or workspace files.",
      "title": "Analyze code architecture",
      "name": "arch-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze configuration and environment. Document env files, Docker, CI/CD, deployment configs, database setup, security configs using file discovery and chunked reads. Write full findings to project memory at `deepscan/findings/config-analyzer.md`, then return only `memory_path`, `status`, 5-8 concise bullets, stats, confidence, and gaps. No artificial tool-call cap: use as many tool calls as needed, but use `ReadFile(offset=..., limit=...)` by default and `limit=-1` only for intentional full-file reads. Do not return raw tool logs, reasoning, file dumps, or large command output. Do not write tmp-files or workspace files.",
      "title": "Analyze configuration",
      "name": "config-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze testing and quality. Identify test frameworks, coverage, linting, formatting tools, quality configs using manifests, config files, and chunked representative tests. Write full findings to project memory at `deepscan/findings/quality-analyzer.md`, then return only `memory_path`, `status`, 5-8 concise bullets, stats, confidence, and gaps. No artificial tool-call cap: use as many tool calls as needed, but use `ReadFile(offset=..., limit=...)` by default and `limit=-1` only for intentional full-file reads. Do not return raw tool logs, reasoning, file dumps, or large command output. Do not write tmp-files or workspace files.",
      "title": "Analyze test quality",
      "name": "quality-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze documentation. Check README completeness, API docs, inline comments, CHANGELOG, LICENSE using discovery and chunked docs reads. Write full findings to project memory at `deepscan/findings/docs-analyzer.md`, then return only `memory_path`, `status`, 5-8 concise bullets, stats, confidence, and gaps. No artificial tool-call cap: use as many tool calls as needed, but use `ReadFile(offset=..., limit=...)` by default and `limit=-1` only for intentional full-file reads. Do not return raw tool logs, reasoning, file dumps, or large command output. Do not write tmp-files or workspace files.",
      "title": "Analyze documentation",
      "name": "docs-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze security. Check auth patterns, secrets exposure, .gitignore, dependency vulnerability signals, encryption usage using targeted searches and chunked reads. Write full findings to project memory at `deepscan/findings/security-analyzer.md`, then return only `memory_path`, `status`, 5-8 concise bullets, stats, confidence, and gaps. No artificial tool-call cap: use as many tool calls as needed, but use `ReadFile(offset=..., limit=...)` by default and `limit=-1` only for intentional full-file reads. Do not return raw tool logs, reasoning, file dumps, or large command output. Do not write tmp-files or workspace files.",
      "title": "Analyze security posture",
      "name": "security-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze data and APIs. Identify database types, ORMs, data models, API endpoints, external integrations, caching using searches and chunked representative reads. Write full findings to project memory at `deepscan/findings/data-analyzer.md`, then return only `memory_path`, `status`, 5-8 concise bullets, stats, confidence, and gaps. No artificial tool-call cap: use as many tool calls as needed, but use `ReadFile(offset=..., limit=...)` by default and `limit=-1` only for intentional full-file reads. Do not return raw tool logs, reasoning, file dumps, or large command output. Do not write tmp-files or workspace files.",
      "title": "Analyze data and APIs",
      "name": "data-analyzer",
      "subagent_type": "researcher"
    }
  ],
  wait=true,
  timeout=1800
)
```

Do not compile the report while any returned agent has a non-terminal status.
If the 30-minute wait expires, use `AwaitAgents(wake_on="all")` for the pending
agent IDs; if any agent ultimately fails or is cancelled, report the incomplete
scan explicitly instead of presenting a partial report as complete.

## Step 3: Compile Final Report

After all agents complete, parse the returned `memory_path` values and read each full finding with `agent_memory_read("project", path)`. Compile those memory files, not raw tool output or subagent logs, into a single report using this structure:

```markdown
# Project DeepScan Analysis Report

## Overview
- Project Path: [path]
- Scan Timestamp: [date]
- Total Files: [count]
- Total Directories: [count]

## Executive Summary
[Brief high-level overview]

---

## 1. Project Structure Analysis
[Findings from structure-analyzer]

## 2. Technology Stack Analysis
[Findings from tech-analyzer]

## 3. Code Patterns & Architecture
[Findings from arch-analyzer]

## 4. Configuration & Environment
[Findings from config-analyzer]

## 5. Testing & Quality Assurance
[Findings from quality-analyzer]

## 6. Documentation Analysis
[Findings from docs-analyzer]

## 7. Security Analysis
[Findings from security-analyzer]

## 8. Data & API Analysis
[Findings from data-analyzer]

---

## Key Insights & Recommendations
[Synthesized recommendations]

## File Inventory
[Complete file list]

---

*Report generated by DeepScan Agent Swarm*
```

Then persist it:

```
agent_memory_write("project", "deepscan/AGENT_NOTES.md", report)
agent_memory_append(
  "project",
  "MEMORY.md",
  "- DeepScan report for this project lives at `deepscan/AGENT_NOTES.md`; use `agent_memory_read(\"project\", \"deepscan/AGENT_NOTES.md\")` for the full report.",
  section="DeepScan index",
)
agent_memory_journal("project", "DeepScan report updated at deepscan/AGENT_NOTES.md")
```

The final chat response should not paste the full report. Return the saved report path, a brief key-finding summary, and any partial agent statuses.

## Output Format for Each Agent

Full findings saved at `deepscan/findings/<agent-name>.md`:

```markdown
## [Agent Name]: [Focus Area]

### Findings
- [Detailed finding 1]
- [Detailed finding 2]

### Statistics
- Files analyzed: [count]
- Patterns detected: [list]
- Issues/Notes: [list]

### Recommendations
- [Actionable recommendation 1]
```

Final subagent response returned to the parent:

```markdown
memory_path: deepscan/findings/<agent-name>.md
status: complete|partial
- [Key finding]
- [Key finding]
- [Gap or risk]
stats: files_sampled=[n], commands_used=[n], confidence=[low|medium|high]
```

## Verification Checklist

After compilation, verify:
- [ ] All 8 agent sections populated
- [ ] Directory tree representation
- [ ] Technology stack complete list
- [ ] File inventory
- [ ] Key insights section
- [ ] Recommendations section
- [ ] Timestamp and metadata
- [ ] Final report saved with `agent_memory_write("project", "deepscan/AGENT_NOTES.md", ...)`
- [ ] Project `MEMORY.md` includes the DeepScan index pointer

## Cleanup

No workspace cleanup is required. DeepScan must not create `tmp-files` or repo-local report files.

## Final Output Location

Project agent memory: `deepscan/AGENT_NOTES.md` — complete project analysis report
