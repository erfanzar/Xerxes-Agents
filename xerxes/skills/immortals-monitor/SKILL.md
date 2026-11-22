---
name: immortals-monitor
description: Detect genuine multi-agent Immortals stalls on a bounded check cycle and wake the single member responsible for unblocking progress.
version: 1.0.0
author: Xerxes-Agents
tags: [immortals, monitor, stall-detection, reliability, multi-agent]
---

# Xerxes Immortals Stall Monitor

## Metadata

- License: Apache-2.0
- Related skills: immortals-coordinator, immortals-narrator
- Built-in: true

**Scope**: Activate this skill **only** when you are the dedicated `monitor` (or `monitor-<N>`) worker assigned by the coordinator to watch for stalls. Coordinators, authors, verifiers, and narrators must not load or execute this workflow.

## Mission

An Immortals run stalls when every active participant (`coordinator`, authors, helpers, verifiers) is parked in `waiting` (or `stopped`) while unfinished work remains — for example, an author finished a slice without tagging `@verifier`, a worker answered without `@name`, or a wake condition missed its intended recipient. Your sole responsibility is to check roster and board state periodically, stay silent while anyone is making progress, wake **one** member with concrete `#<seq>` evidence if the whole Immortals force is genuinely stuck, and shut down cleanly once `[done]` closes the run.

## Operating Loop

1. **Inspect roster and unread messages together**:
   - Call `read_board{status: true}` to inspect every member's live `state` (`working`, `waiting`, `stopped`, `killed`), `role`, `current_activity`, `parked_since`, `mentions_only`, `last_read_seq`, `last_post_seq`, and `error`.
   - `coordinator` appears directly in `read_board{status: true}` with `role: "coordinator"`: its state is `working` during an active turn and `waiting` (with `parked_since` and `mentions_only`) both when idle between turns and while parked inside `swarm_wait`.
   - Call `read_board{since: <last_seen_seq>, limit: 50}` to read new board messages since your last check and advance `<last_seen_seq>`.
2. **Check for completion first**:
   - If `coordinator` has posted a closing `[done]` message (including an unaddressed `[done] <summary>` with no `@` mentions) summarizing completed Immortals work, or every assigned deliverable and final verification gate has posted `[done]` to `@coordinator` and the coordinator has concluded, **stop monitoring immediately**: inspect `manage_task{Action: "list"}`, cancel every active timer or cron you created via `manage_task{Action: "kill", TaskId: "<task_id>"}`, and park quietly with `swarm_wait{since: <last_seen_seq>, mentions_only: true}` without posting any noise.
3. **Stay completely silent when not stalled**:
   - Exclude yourself (`monitor*`), `narrator*`, and `killed` members from the active-progress check.
   - If **any** remaining member — including `coordinator` OR any living worker — is currently `working`, the Immortals force is **not** stalled — do **not** post anything.
   - If board `head` advanced recently and the addressed recipient is about to wake or just parked normally after completing its assigned slice while another dependency is in flight, wait one more check window before declaring a stall.
4. **Diagnose a genuine stall**:
   - A stall exists only when **all** living non-monitor, non-narrator members (including `coordinator`) are `waiting` or `stopped` **and** the Immortals task is still incomplete (including silent stalls where an author parked without posting or omitted `@<recipient>`).
   - Pinpoint the **single** member who can unstick the workflow by reading the trailing `#<seq>` posts and member states:
     - Unaddressed handoff or missing `@verifier` tag -> wake `@verifier` (or the specific author if their slice was incomplete), citing `#<seq>`.
     - Verifier reported issues without naming the owning author -> wake `@<author>`, citing `#<seq>`.
     - All authors/verifiers posted `[done]` without waking `@coordinator`, or a cross-boundary blocker awaits a decision -> wake `@coordinator`, citing `#<seq>`.
     - A member is `stopped` or failed with an `error` -> wake `@coordinator` so they can revive it or switch its model in place via `swarm{action: "model", name, model, reason}`.
5. **Post a surgical `[stall]` wakeup**:
   - Call `post_message` naming **only that one member** — never `@all`, and never CC `@coordinator` when a peer worker can act directly:
     ```text
     @verifier [stall] All workers and coordinator are waiting and #18 completed the parser slice without an @verifier tag — please verify #18.
     ```
     or:
     ```text
     @coordinator [stall] All workers are waiting after final verification in #24 — ready for synthesis or next wave.
     ```
6. **Arm a silent-stall timer with `schedule` and sleep with `swarm_wait` (never busy-poll)**:
   - Before ending your turn while the Immortals run is still open, arm one bounded silent-stall timer via `schedule{DurationSeconds: 60, TimerCondition: "any", Prompt: "Check Immortals roster and board for silent stall"}`:
     - If **any** other member posts to the board before `DurationSeconds` elapses, `TimerCondition: "any"` cancels that timer early at zero token cost.
     - If the entire Immortals force goes completely silent until `DurationSeconds` elapses, `schedule` wakes you directly with `[schedule] ...` without adding any message to the shared board.
   - In the **same turn**, park cleanly with `swarm_wait{since: <last_seen_seq>, mentions_only: false}` so you consume zero tokens while idle and wake reactively on board traffic, `@monitor` nudges, or the `[schedule]` silent-stall timer.
   - Whenever you wake (whether from `[schedule]` or `swarm_wait`), cancel any stale duplicate schedules with `manage_task{Action: "kill", TaskId}` before arming a replacement, and cancel all active schedules as soon as `[done]` lands.

## Strict Anti-Noise Rules

- Never greet, acknowledge startup, or post "all healthy / no stall detected" heartbeats. Every board post wakes subscribers and consumes context across the Immortals roster.
- Never repeat the exact same `[stall]` post for the same `head` sequence number if nothing has changed since your nudge; escalate to `@coordinator` once if the nudged worker remains unresponsive or `stopped`.
- Never edit repository files, spawn code authors, or kill workers.
