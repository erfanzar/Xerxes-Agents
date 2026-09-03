---
name: spike
description: Throwaway experiments validating an idea before a real build.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [spike, prototype, experiment, feasibility, exploration]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/software-development/spike/SKILL.md
---

# Spike

Use this skill when the user wants to **feel out an idea** before committing
to a real build — validating feasibility, comparing approaches, or surfacing
unknowns no amount of reading will answer. Spikes are disposable by design.
Throw them away once they have paid their debt.

Load this when the user says "let me try this", "I want to see if X works",
"spike this out", "quick prototype of Z", "is this even possible?", or
"compare A vs B".

## When NOT to Use

- The answer is knowable from docs or reading code — do research instead
- The work is production path — use the `plan` skill instead
- The idea is already validated — jump straight to implementation

## Core Method

```
decompose  →  research  →  build  →  verdict
   ↑______________________________________↓
                iterate on findings
```

### 1. Decompose

Break the idea into **2-5 independent feasibility questions**. Each question
is one spike. Present them as a table with Given/When/Then framing:

| # | Spike | Validates | Risk |
|---|-------|-----------|------|
| 001 | websocket-streaming | Given a WS connection, when tokens stream, then the client receives chunks < 100ms | High |
| 002a | pdf-parse-pdfjs | Given a multi-page PDF, when parsed with pdfjs, then text is extractable | Medium |
| 002b | pdf-parse-camelot | Same question, different library | Medium |

Spike types: **standard** (one approach, one question) and **comparison**
(same question, different approaches — shared number, letter suffix).

Good spike questions are specific feasibility with observable output. Bad
ones are too broad, have no observable output, or are just "read the docs
about X". **Order by risk**: the spike most likely to kill the idea runs
first. Skip decomposition only when the user already knows exactly what to
spike.

### 2. Align

Present the spike table and let the user drop, reorder, or re-frame before
writing any code.

### 3. Research (per spike, before building)

Brief each spike in 2-3 sentences. Surface competing approaches when there
is real choice (approach, tool/library, pros, cons, maintenance status), pick
one, and state why. Use the web fetch tool for docs and the shell/terminal
tool to check what is installed locally. Skip research for pure logic with
no external dependencies.

### 4. Build

One directory per spike, standalone:

```
spikes/
├── 001-websocket-streaming/
│   ├── README.md
│   └── main.ts
└── 002a-pdf-parse-pdfjs/
    ├── README.md
    └── parse.ts
```

**Bias toward something the user can interact with.** In order of
preference: a runnable CLI with observable output, a minimal HTML demo, a
small server with one endpoint, or a test with recognizable assertions.

**Depth over speed.** Never declare "it works" after one happy-path run.
Test edge cases; follow surprising findings.

**Avoid** unless required: complex package management, bundlers, Docker, env
file systems. Hardcode everything — it is a spike.

**Parallel comparison spikes:** when two approaches can run in parallel and
both need real engineering, fan out with the agent spawning tools; each
child returns its own verdict and you write the head-to-head.

### 5. Verdict

Each spike's `README.md` closes with:

```markdown
## Verdict: VALIDATED | PARTIAL | INVALIDATED

### What worked
### What didn't
### Surprises
### Recommendation for the real build
```

VALIDATED = answered yes with evidence. PARTIAL = works under constraints
X, Y, Z — document them. INVALIDATED = does not work, for this stated
reason. An INVALIDATED spike is a successful spike.

## Comparison Spikes

Build comparison spikes back to back, then do a head-to-head:

```markdown
## Head-to-head: pdfjs vs camelot

| Dimension | pdfjs (002a) | camelot (002b) |
|-----------|--------------|----------------|
| Extraction quality | 9/10 structured | 7/10 tables only |
| Setup complexity | one install | extra system deps |
| Perf on 100 pages | 3s | 18s |

**Winner:** pdfjs for our use case.
```

## Output

- Create `spikes/` in the repo root, one directory per spike:
  `NNN-descriptive-name/`
- `README.md` per spike captures question, approach, results, verdict
- Keep the code throwaway — a spike that takes two days to "clean up for
  production" was a bad spike

## Frontier Mode

If spikes already exist and the user asks what to spike next, walk the
directories and look for: integration risks (two validated spikes touching
the same resource, tested independently), unproven data handoffs, gaps in
the vision, and alternative angles for PARTIAL or INVALIDATED results.
Propose 2-4 candidates as Given/When/Then and let the user pick.

---

Adapted from the `spike` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Hermes Agent (adapted from gsd-build/get-shit-done).
