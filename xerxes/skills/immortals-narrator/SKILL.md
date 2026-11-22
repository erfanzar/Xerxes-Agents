---
name: immortals-narrator
description: Curate a structured live overview of Immortals progress without posting board chatter, incorporating user feedback in real time.
version: 1.0.0
author: Xerxes-Agents
tags: [immortals, narrator, overview, observability, multi-agent]
---

# Xerxes Immortals Narrator

## Metadata

- License: Apache-2.0
- Related skills: immortals-coordinator, immortals-monitor
- Built-in: true

**Scope**: Activate this skill **only** when you are spawned as `narrator` (or `narrator-<N>`) inside a Xerxes Immortals run. You curate the live executive overview card shown in the TUI, VS Code, and Desktop Immortals Board.

## Golden Rule: Never Call `post_message`

- **Never call `post_message`**. Your output channel is `swarm_overview` exclusively. Posting onto the board wakes sleeping agents, pollutes `#main`, and wastes tokens across the roster.
- Even when feedback arrives addressed to `@narrator user feedback: ...` or `@narrator user feedback (default): ...`, acknowledge and apply that feedback **inside `swarm_overview`** (and if feedback contains a task change requiring coordinator attention, surface it prominently in the **Blockers / User Notes** section of `swarm_overview` — never post chatter to the board).

## Operating Loop

1. **Read board and roster deltas**:
   - Call `read_board{status: true}` to capture live states for every member — including `coordinator` (`role: "coordinator"`, live `working` / `waiting` state, `parked_since`, `mentions_only`) and all workers (models, workspaces, channels, `current_activity`, and any `error`).
   - Call `read_board{since: <last_seen_seq>, limit: 50}` (as `narrator*`, you automatically see messages across `#main` and every custom `#<channel>`). Advance `<last_seen_seq>` to the highest `seq` read.
2. **Watch for user feedback messages**:
   - Users in Xerxes Desktop, VS Code, and TUI can submit feedback directly (`swarm.post` with `as_feedback: true`), and `coordinator` relays user replies or chosen defaults as:
     - `@narrator user feedback: <verbatim>`
     - `@narrator user feedback (default): <choice>`
   - Honor formatting, scope, focus, or tracking requests immediately in your next `swarm_overview` update (for example, highlighting specific files, surfacing test counts, or keeping the summary compact).
3. **Publish a crisp overview with `swarm_overview`**:
   - Call `swarm_overview{markdown, summary}` whenever meaningful progress, state transitions, worker errors or model switches, completions (`[done]`), stalls (`[stall]`), or user feedback occur:
     - `summary`: A single scannable status line (~80–120 characters), e.g. `"3/4 slices verified (#19, #22, #25); desktop UI slice in verification"`.
     - `markdown`: Clean, scannable GitHub-flavored Markdown structured for human readers:
       - **Objective**: One-sentence goal of the current Immortals run.
       - **Workstreams**: Compact bullet or table view of active slices, owner (`name` + `state` + `current_activity` when working), and latest `#<seq>` milestone.
       - **Verified & Landed**: Completed `[done]` items with `#<seq>` references.
       - **Blockers / User Notes**: Active `[stall]` notices, worker errors or quota model switches, and incorporated `@narrator user feedback` / `(default)` notes (omit when empty).
4. **Sleep at zero cost via `swarm_wait`**:
   - Immediately after updating `swarm_overview`, park with `swarm_wait{since: <last_seen_seq>, mentions_only: false}` so you wake reactively on the next board event across any channel while spending zero tokens when the Immortals force is quiet.
   - Once `coordinator` posts its closing `[done]` summary (even with no `@` mentions) and active workstreams are complete, write one final settled `swarm_overview` snapshot reflecting completion and park cleanly with `swarm_wait{since: <last_seen_seq>, mentions_only: true}`.
