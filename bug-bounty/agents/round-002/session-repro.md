# Round 002 — Session reproduction

Date: 2026-08-21

## 1. Incremental search leaves non-content fields stale — **confirmed bug**

`SessionIndex.indexSessionIncremental()` reads and compares only `turn_id`, `prompt`, and `response` (`xerxes/src/session/search.ts:212-230`). A save where prompt/response are unchanged but `agentId`, `startedAt`, or `metadata` changes returns `0` and never calls `insertTurn()`, although those fields are stored and returned in search hits.

Observed with a one-turn in-memory index:

```text
first 1 ... agentId: "a", timestamp: "2026-01-01T00:00:00Z", metadata: { rev: 1 }
second 0 ... agentId: "a", timestamp: "2026-01-01T00:00:00Z", metadata: { rev: 1 }
```

The source record before the second save had `agentId: "b"`, timestamp `2026-02-02T00:00:00Z`, and `{ rev: 2 }`. Search therefore exposes stale filters/result metadata after a normal `SQLiteSessionStore.saveSession()` (`xerxes/src/session/store.ts:304-333`). Existing incremental coverage checks append/content and embedding counts, but not metadata-only updates (`xerxes/test/sessionCore.test.ts:236-270`).

Impact: incorrect `agentId` filtering and stale hit attribution/timestamps/metadata. Suggested regression: mutate each non-content indexed field, save, then assert both hit data and old/new agent filters.

## 2. Empty compaction summary — **not reproduced as destructive bug**

For completion values `""` and `"   "`, `summarizeContext()` normalized to an empty string, while `summarizeMessages()` returned all four original messages unchanged (`unchanged: true`). This follows the explicit guard `if (!summary.trim()) return original` in `xerxes/src/agents/compactionAgent.ts:153-155`.

No history loss or placeholder persistence was observed. A missing regression test for empty/whitespace model output is a coverage gap, not presently a confirmed behavioral bug.

## 3. `workspaceRoot` containment semantics — **intentional per tests; do not file**

`normalizeProjectDirectory()` falls back only when persisted `cwd` equals or is contained by `workspaceRoot`; sibling prefix paths and unrelated project paths remain unchanged (`xerxes/src/session/daemonTranscript.ts:835-843`). Observed:

```text
/users/.xerxes/agents         => /projects/current
/users/.xerxes/agents/default => /projects/current
/users/.xerxes/agents-evil/default => unchanged
/projects/other               => unchanged
```

This is containment-safe (`resolved === root || startsWith(root + sep)`) and matches the explicit test expecting persisted agent-workspace cwd `/users/.xerxes/agents/default` to resume as project cwd `/projects/current` (`xerxes/test/daemonTranscript.test.ts:502-527`). Here `workspaceRoot` denotes Xerxes-managed agent workspaces to reject as a resumed project cwd, not an allowlisted project boundary. Calling the direction inverted would contradict current tests and runtime naming/usage.

## Verification

```text
bun test xerxes/test/sessionCore.test.ts xerxes/test/compactionAgent.test.ts xerxes/test/daemonTranscript.test.ts
40 pass, 0 fail, 167 expect() calls; 1.90s
```

No production or test files changed; only this report was created.
