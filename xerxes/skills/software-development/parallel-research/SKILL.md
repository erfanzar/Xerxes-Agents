---
name: parallel-research
description: Run research in parallel across every available accelerator (TPU, GPU, CPU). Inventory the devices, split them into independent units, give each unit one grounded idea in its own agent, then compare results on one shared harness. Use with a goal for any research, tuning or optimization task that has more than one direction worth trying.
tags: [software-development, research, parallel, accelerators, goals, artifacts]
---

# Skill: Parallel Research

Use this when a research, tuning, debugging or optimization task has several directions worth trying and more than one device (or CPU core group) to try them on. It extends `run-research`: plan first, keep a live artifact, ground every claim. The difference is that ideas run side by side, one per device unit, instead of one after another.

Pair it with a goal: the goal holds the objective and criteria, this skill decides how the work fans out.

## 1. Inventory the compute (measure, never assume)

Before planning, find out what you actually have and whether it is free.

- TPU: list the slices and their accelerator type and topology from the cloud CLI, and on a host ask the runtime how many local chips it sees. Note chips per host, number of hosts, and which hosts answer over SSH.
- GPU: list devices, memory and current utilization (for example `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv`).
- CPU: cores and memory (`nproc` or `sysctl -n hw.ncpu`, and free memory).
- Busy check: a device already used by another process is not available. Never kill a process you did not start to free a device; report it and work around it.

Write the inventory into the goal artifact as a table: device, count, unit size, free or busy, how to reach it.

## 2. Split the compute into independent units

A unit is the smallest slice that can run one experiment on its own without touching another unit.

- Default unit: one host (for TPU, the chips attached to one worker; for GPU, one GPU or one host; for CPU, a fixed group of cores).
- Compute the number of units from the inventory, not from memory. Example: a slice with H hosts of 4 chips each gives H single-host units. Check the slice's real host count before you divide; the name of a slice does not tell you the host count reliably.
- First confirm one host can run a job alone (a single-host process that sees only its local chips or GPUs). If it cannot, the unit is larger.
- Keep some units back when an idea needs more scale: an idea that wins on one unit is re-checked on several units or the full slice before it is called a result.
- Not every idea needs its own unit. Cheap ideas (code reading, analysis, CPU-only checks) run as agents without a device.

## 3. Choose the ideas

- Start with a baseline: one unit runs the current code on the shared harness, so every other number has something to compare against.
- List candidate directions, each with a hypothesis, the exact change, the metric that would confirm or kill it, and a stop rule. Different directions, not the same idea with small variations, unless a sweep is the point.
- Every idea must be grounded: name the file, function, flag or paper it comes from, after you have opened it.
- Rank by expected value per unit of compute. Assign the best ideas first.

## 4. Fan out safely

Run one agent per idea with the `Workflow` tool (or `SpawnAgents` for a few). Pass `expect_agents` with the exact count and give each agent a unique label. Before more than about 10 agents, or anything that bills cloud resources, ask the user once.

Each agent's prompt must stand alone and include:

- The one unit it owns (host name, device indices) and nothing else. It must not touch other units.
- Isolation: its own git worktree or checkout, its own output directory, its own ports, cache and log paths, and a fixed seed.
- The shared harness command and metric, exactly as the baseline ran it.
- The hypothesis, the change to make, the stop rule and a time or step budget.
- What to return: a structured result (idea, unit, command, metric value, baseline value, pass or fail, notes, path to logs). Never an unmeasured claim.

Keep a lease table in the goal artifact (unit, agent label, idea, state). One unit has one owner at a time. A unit is released only when its agent has finished or been stopped.

## 5. Collect, verify, compare

- Wait for every agent (AwaitAgents with `wake_on: all`). A failed or interrupted agent is a missing result, not a negative one: rerun it or mark it unknown.
- Put all results in one table against the baseline. Same harness, same metric, same units.
- Verify each apparent win independently: rerun it on a different unit, and on more units or the full slice when the idea depends on scale. Use a separate verifier agent with a strong model for subtle claims.
- Combine winners only after each holds on its own, then measure the combination.

## 6. Artifact after every step

Keep one goal artifact updated (action `update`, same id) after the inventory, the split, each agent result and each verification:

- the inventory and the unit split;
- the lease table;
- the results table with baseline, metric and pass or fail;
- changes to the plan and why;
- a final Result section with the winning change, its verified numbers and what was ruled out.

Record evidence on the goal's criteria right after the command that proves them.

## Rules

- Nothing breaks: no shared outputs, no shared ports, no editing the main checkout from a lane, no restarting services or hosts you did not start.
- No result without a number from a command that ran this session.
- Negative results stay in the artifact; they stop the same idea from being tried again.
- If devices disappear mid-run (preemption, maintenance), mark the affected lanes unknown and reschedule them; do not report partial runs as results.
