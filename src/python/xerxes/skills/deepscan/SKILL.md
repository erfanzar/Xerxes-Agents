<think>
The user wants me to create a skill called 'deepscan' that will:
1. Create and spawn a swarm of agents
2. Analyze the current project
3. Gather information
4. Write detailed notes to tmp-files/AGENT_NOTES.md

I need to create a comprehensive SKILL.md file that provides detailed instructions for the agent to accomplish this task. The skill should define how to:
- Create multiple specialized agents (a swarm)
- Have them analyze different aspects of the project
- Coordinate their findings
- Compile everything into a detailed AGENT_NOTES.md file

Let me structure this properly with YAML frontmatter and clear markdown instructions.
</think>

```yaml
---
name: deepscan
description: Spawns a swarm of specialized agents to deeply analyze the current project and compile comprehensive findings into tmp-files/AGENT_NOTES.md
version: "1.0"
tags: [analysis, swarm, project, deepscan, multi-agent]
---

# DeepScan Skill

Spawns a coordinated swarm of specialized agents to perform comprehensive analysis of the current project, then compiles all findings into a detailed report at `tmp-files/AGENT_NOTES.md`.

## Prerequisites

Before spawning the swarm, ensure the `tmp-files/` directory exists:
```bash
mkdir -p tmp-files
```

## Swarm Architecture

Spawn **8 specialized agents** in parallel, each focusing on a specific analysis domain:

### Agent 1: Project Structure Analyzer
**Focus:** Directory structure, file organization, naming conventions

**Instructions:**
- Scan the entire project recursively
- Document the directory tree structure (up to 4 levels deep)
- Identify key directories (src, lib, components, tests, docs, config, etc.)
- Note any unusual directory naming patterns
- List all configuration files (.json, .yaml, .toml, .env, etc.)
- Identify the project's module/package structure

### Agent 2: Technology Stack Analyzer
**Focus:** Languages, frameworks, libraries, dependencies

**Instructions:**
- Identify all programming languages used (check file extensions)
- Detect framework usage (React, Vue, Angular, Express, Django, FastAPI, etc.)
- List dependency files (package.json, requirements.txt, Gemfile, go.mod, Cargo.toml, etc.)
- Document the build system (webpack, vite, gradle, make, etc.)
- Note any notable libraries or tools
- Check for type checking systems (TypeScript, mypy, Flow)

### Agent 3: Code Patterns & Architecture Analyzer
**Focus:** Design patterns, architectural styles, coding conventions

**Instructions:**
- Identify architectural patterns (MVC, microservices, monolith, event-driven, etc.)
- Detect common design patterns (singleton, factory, observer, etc.)
- Document coding style conventions
- Analyze component/module relationships
- Note any interesting implementation patterns
- Identify separation of concerns

### Agent 4: Configuration & Environment Analyzer
**Focus:** Environment setup, configurations, deployment

**Instructions:**
- Document all environment configuration files
- Identify environment variables in use
- Check for Docker/container configurations
- Analyze CI/CD configurations (.github, .gitlab-ci.yml, Jenkinsfile, etc.)
- Document deployment configurations
- Identify database configurations
- Note any security-related configurations

### Agent 5: Testing & Quality Analyzer
**Focus:** Testing frameworks, coverage, quality metrics

**Instructions:**
- Identify testing frameworks (Jest, pytest, JUnit, RSpec, etc.)
- Document test directory structure
- Note testing patterns (unit, integration, e2e)
- Identify linting/formatter tools (ESLint, Prettier, Pylint, Black, etc.)
- Check for code quality configurations
- Note any coverage report configurations

### Agent 6: Documentation & README Analyzer
**Focus:** Documentation quality, API docs, inline comments

**Instructions:**
- Check for README files and their completeness
- Identify API documentation (Swagger, OpenAPI, JSDoc, Sphinx, etc.)
- Note inline documentation patterns
- Check for CHANGELOG, CONTRIBUTING, LICENSE files
- Identify any wiki or external documentation
- Assess documentation coverage

### Agent 7: Security & Dependencies Analyzer
**Focus:** Vulnerabilities, sensitive data, access patterns

**Instructions:**
- Check for security configurations
- Identify authentication/authorization patterns
- Note any hardcoded secrets or sensitive data exposure risks
- Check .gitignore completeness
- Identify dependency security tools
- Note any encryption or security libraries in use

### Agent 8: Data & API Analyzer
**Focus:** Data models, API endpoints, external integrations

**Instructions:**
- Identify database types and ORMs
- Document data models/schemas
- List API endpoints (REST, GraphQL, gRPC, etc.)
- Identify external service integrations
- Note data flow patterns
- Check for caching mechanisms

## Execution Process

### Step 1: Initialize
```
echo "=== DEEPSCAN INITIATED ===" > tmp-files/AGENT_NOTES.md
echo "Timestamp: $(date)" >> tmp-files/AGENT_NOTES.md
echo "Project: $(pwd)" >> tmp-files/AGENT_NOTES.md
echo "" >> tmp-files/AGENT_NOTES.md
```

### Step 2: Spawn Agents
Spawn all 8 agents simultaneously. Each agent should:
1. Perform their specialized analysis
2. Write findings to a temporary file: `tmp-files/AGENT_N_{name}_FINDINGS.md`
3. Append summary to `tmp-files/AGENT_NOTES.md` in the format below

### Step 3: Compile Final Report
After all agents complete, compile all findings into the master document following this structure:

```markdown
# Project DeepScan Analysis Report

## Overview
- Project Path: [path]
- Scan Timestamp: [date]
- Total Files: [count]
- Total Directories: [count]

## Executive Summary
[Brief high-level overview of the project]

---

## 1. Project Structure Analysis
[Findings from Agent 1]

## 2. Technology Stack Analysis
[Findings from Agent 2]

## 3. Code Patterns & Architecture
[Findings from Agent 3]

## 4. Configuration & Environment
[Findings from Agent 4]

## 5. Testing & Quality Assurance
[Findings from Agent 5]

## 6. Documentation Analysis
[Findings from Agent 6]

## 7. Security Analysis
[Findings from Agent 7]

## 8. Data & API Analysis
[Findings from Agent 8]

---

## Key Insights & Recommendations
[Synthesized recommendations based on all findings]

## File Inventory
[Complete list of all files in the project]

---

*Report generated by DeepScan Agent Swarm*
```

## Output Format

Each agent's findings should be written in the following format:

```markdown
## [Agent Name]: [Focus Area]

### Findings
- [Detailed finding 1]
- [Detailed finding 2]
- [Detailed finding 3]

### Statistics
- Files analyzed: [count]
- Patterns detected: [list]
- Issues/Notes: [list]

### Recommendations
- [Actionable recommendation 1]
- [Actionable recommendation 2]
```

## Verification Checklist

After compilation, verify the final report includes:
- [ ] All 8 agent sections populated
- [ ] Directory tree representation
- [ ] Technology stack complete list
- [ ] File inventory
- [ ] Key insights section
- [ ] Recommendations section
- [ ] Timestamp and metadata

## Cleanup

Remove temporary agent finding files after successful compilation:
```bash
rm -f tmp-files/AGENT_N_*_FINDINGS.md
```

## Final Output Location

`tmp-files/AGENT_NOTES.md` - Complete, detailed project analysis report
```