# Round 010 — Daemon main-session resume scope

## Bug

Explicit daemon resume of main sessions ignored the requesting project's
`currentProjectDirectory` and would load a foreign transcript's `cwd` and
message history. The existing cross-project guard only applied to subagent
histories, so a main session created in project A could be resumed from a
daemon serving project B.

## Root cause

`InMemoryDaemonRuntime.initializeSession` in `xerxes/src/daemon/runtime.ts`
loaded persisted transcripts and checked project ownership only when the
stored transcript was a subagent:

```ts
if (
  transcript &&
  transcriptIsSubagent(transcript) &&
  transcriptProjectDirectory(transcript) !== cwd
) { ... }
```

Main sessions skipped this check and were passed straight to
`sessionFromTranscript`, which adopted `transcript.cwd` and `transcript.messages`
as the resumed session state.

## Fix

Generalized the project-ownership check to apply to any resumed transcript,
keeping the subagent-specific error wording where relevant:

```ts
if (
  transcript &&
  transcriptProjectDirectory(transcript) !== cwd
) {
  const kind = transcriptIsSubagent(transcript)
    ? "subagent history"
    : "main session";
  throw new ValidationError(
    "session_id",
    `belongs to a ${kind} from a different project`,
    key,
  );
}
```

This restricts `runtime.openSession` resume to transcripts whose stored
`project_root` (or legacy `cwd`) matches the requested current project
directory, enforcing project ownership for main sessions the same way it was
already enforced for subagents.

## Files changed

- `xerxes/src/daemon/runtime.ts` — project-ownership guard now covers main
  session resumes.
- `xerxes/test/daemonSessionRuntimeParity.test.ts` — added focused regression
  `main session resume rejects a transcript from a different project`.

## Verification

```bash
bun test xerxes/test/daemonSessionRuntimeParity.test.ts
bun test xerxes/test/daemonTranscript.test.ts
bun run check
```

All passed:

- `daemonSessionRuntimeParity.test.ts`: 14 pass, 0 fail
- `daemonTranscript.test.ts`: 14 pass, 0 fail
- `bun run check`: passed (repo + runtime + UI type checks)
