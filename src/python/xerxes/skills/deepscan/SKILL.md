---
name: deepscan
description: >-
  Spawns a swarm of specialized agents to deeply analyze the current project
  and compile comprehensive findings into project-scoped agent memory.
  ALWAYS runs agents in parallel — sequential analysis is unacceptable.
version: "2.1"
tags: [analysis, swarm, project, deepscan, multi-agent, parallel]
---

# DeepScan Skill

**PARALLEL IS MANDATORY.** DeepScan ALWAYS spawns all analysis agents simultaneously. Running them sequentially is a bug that wastes time and produces worse results.

Spawns a coordinated swarm of 8 specialized agents to perform comprehensive analysis of the current project, then compiles all findings into the project's persistent agent memory at `deepscan/AGENT_NOTES.md`.

## Project Memory Contract

DeepScan output is project-specific memory, not a workspace temp artifact.

- Final report: `agent_memory_write("project", "deepscan/AGENT_NOTES.md", report)`
- Optional raw findings: `agent_memory_write("project", "deepscan/findings/<agent-name>.md", findings)`
- Discovery index: append a short pointer to project `MEMORY.md` so future agents know the report exists.
- Journal: add a one-line project journal entry after saving the report.

Before spawning agents, call `agent_memory_status()` and confirm project memory is available. If project memory is unavailable or any memory write fails, stop and report the memory error. **Do not fall back to `tmp-files`, repo-local report files, or shell-created scratch files.**

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

Use `SpawnAgents` with `wait=true`. Every agent gets the `researcher` subtype and returns findings in its final response. Subagents must not write to `tmp-files` or workspace files.

```
SpawnAgents(
  agents=[
    {
      "prompt": "Analyze project structure. Scan all directories up to 4 levels deep. Document tree, key dirs, config files, module structure. Return findings in the required DeepScan agent output format. Do not write tmp-files or workspace files.",
      "name": "structure-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze technology stack. Identify languages, frameworks, dependencies, build systems, type checkers. Return findings in the required DeepScan agent output format. Do not write tmp-files or workspace files.",
      "name": "tech-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze code patterns and architecture. Identify design patterns, architectural style, module relationships, separation of concerns. Return findings in the required DeepScan agent output format. Do not write tmp-files or workspace files.",
      "name": "arch-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze configuration and environment. Document env files, Docker, CI/CD, deployment configs, database setup, security configs. Return findings in the required DeepScan agent output format. Do not write tmp-files or workspace files.",
      "name": "config-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze testing and quality. Identify test frameworks, coverage, linting, formatting tools, quality configs. Return findings in the required DeepScan agent output format. Do not write tmp-files or workspace files.",
      "name": "quality-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze documentation. Check README completeness, API docs, inline comments, CHANGELOG, LICENSE. Return findings in the required DeepScan agent output format. Do not write tmp-files or workspace files.",
      "name": "docs-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze security. Check auth patterns, secrets exposure, .gitignore, dependency vulnerabilities, encryption usage. Return findings in the required DeepScan agent output format. Do not write tmp-files or workspace files.",
      "name": "security-analyzer",
      "subagent_type": "researcher"
    },
    {
      "prompt": "Analyze data and APIs. Identify database types, ORMs, data models, API endpoints, external integrations, caching. Return findings in the required DeepScan agent output format. Do not write tmp-files or workspace files.",
      "name": "data-analyzer",
      "subagent_type": "researcher"
    }
  ],
  wait=true
)
```

## Step 3: Compile Final Report

After all agents complete, compile their returned results into a single report using this structure:

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

If you save raw agent findings too, write them under `deepscan/findings/` in project memory after the final report is saved. Do not require raw findings for success.

## Output Format for Each Agent

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
