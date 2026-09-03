---
name: weekly-review-planning
description: Bounded weekly reset — collect commitments, flag stalled work, and build a capacity-aware next-week plan.
version: 1.0.0
author: Ben Barclay (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [weekly-review, planning, tasks, calendar, productivity]
source: https://github.com/NousResearch/hermes-agent/tree/main/skills/productivity/weekly-review-planning
---

# Weekly Review and Planning

Run a bounded weekly reset across the user's chosen systems. This is a
concrete recurring task, not a generic productivity methodology — run
it on demand, or from a scheduled trigger if the user has configured
one.

## When to Use

- "Run my weekly review."
- "What did I commit to and what is slipping?"
- "Plan next week from my calendar, tasks, and notes."
- "Find stale projects and waiting items."
- A scheduled weekly-review trigger fires.

Don't use for: single-inbox triage (`himalaya` owns thread-level email
work) or one-off document extraction (`document-to-action-items`).

## Procedure

### 1. Set systems and window

Confirm timezone, review period, planning horizon, authoritative
task/project store, calendars, inboxes, and allowed writes. Default to
recommendations/drafts, not mutations. Done when source-of-truth
conflicts have a declared winner.

### 2. Review calendar evidence

Load `google-workspace` or the relevant calendar connector. Inspect the
completed week for meetings and commitments, then the next 1-2 weeks
for deadlines, travel, preparation, and capacity. Capture follow-ups
implied by past events and conflicts ahead. Done when both retrospective
and horizon are covered.

### 3. Clear capture inboxes

Review the task inbox, notes (`obsidian`, `notion`), flagged email
(`himalaya` for mailbox access), and other declared capture points.
Convert each item to next action, project, waiting, scheduled, someday,
reference, archive, or delete proposal. Do not mutate until scope is
approved. Done when remaining unprocessed items are counted and stated.

### 4. Reconcile active projects

For each project identify desired outcome, next action, owner, deadline,
blocker, last meaningful activity, and source link. Flag projects with
no next action, missed dates, duplicate records, or contradictory
status. Done when every active project is actionable or explicitly
paused.

### 5. Review waiting and commitments

Find promises made by the user and items owed by others. Propose
follow-ups with dates and channels. Do not infer that silence means
completion. Done when each waiting item has an owner and next
review/follow-up date.

### 6. Build a capacity-aware plan

Estimate fixed calendar load and select a small set of weekly outcomes
plus near-term next actions. Rank by consequence, deadline, dependency,
and effort; do not fill every free hour. Done when the plan fits actual
capacity and names deferred work.

### 7. Apply approved updates

Update tasks/projects, create calendar holds, archive processed items,
and draft follow-ups only as approved. Read every changed record back
from the provider. Done when verified writes match the review summary.

## Output Shape

1. Wins and completed commitments
2. Overdue or at risk
3. Waiting/follow-ups
4. Stalled or ambiguous projects
5. Next week's outcomes and calendar constraints
6. Proposed updates awaiting approval
7. Coverage gaps

## Pitfalls

- Planning from tasks without calendar capacity.
- Carrying every unfinished item forward as high priority.
- Marking projects active with no next action.
- Silently deleting or rescheduling personal commitments.
- Treating silence from others as completion.

## Verification

- [ ] Both the completed week and the planning horizon were covered, or
      gaps are stated.
- [ ] Every stalled/waiting flag traces to a specific record, event, or
      thread.
- [ ] No task, event, or note was mutated without approval; approved
      writes were read back.
- [ ] The plan names what was deferred, not just what was chosen.

---

Adapted from the `weekly-review-planning` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Ben Barclay and Nous Research.
