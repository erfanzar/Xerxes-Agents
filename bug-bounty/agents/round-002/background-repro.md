# Round 002 — daemon-global background command cross-session reproduction

## Finding: daemon sessions share unrestricted background-command state and control

- Severity: **high**
- Status: **confirmed**
- Class: cross-session authorization / ownership isolation (IDOR-like capability leak)
- Scope: native daemon tool registry and background process lifecycle

### Claim validated

The daemon constructs one `BackgroundCommandManager` outside `runnerFactory` and passes it into every rebuilt core tool registry (`xerxes/src/cli.ts:782-809`). That persistence is needed across settings-triggered runner rebuilds, but the manager stores only `proc_id -> BackgroundEntry`; it records no owning session (`xerxes/src/tools/backgroundCommands.ts:52-60,74-76,107-121`).

The process handlers receive a `ToolExecutionContext` containing `sessionId`, but discard it: all four closures operate directly on the shared manager (`xerxes/src/tools/processTools.ts:166-217`). `list()` returns every manager entry; `check(procId)` and `kill(procId)` authorize solely by possession of a valid ID (`xerxes/src/tools/backgroundCommands.ts:133-190,217-223`). The context is demonstrably available at execution time (`xerxes/src/executors/toolRegistry.ts:132-136,323-345`; populated from the turn request in `xerxes/src/streaming/loop.ts:331-335`).

### Behavioral reproduction

Executed from repository root on 2026-08-21 with Bun. The reproducer registered the production process tools against one shared manager, started `/bin/sh -c 'echo A-secret; sleep 30'` under `{sessionId: 'A'}`, then invoked `list_commands`, `check_command`, and `kill_command` under `{sessionId: 'B'}`.

```text
$ bun -e '<shared-manager production-tool reproducer>'
{"aProcId":"d51d05dc6cc6","bListed":["d51d05dc6cc6"],"bRead":"A-secret","bKilled":true}
```

Observed behavior:

1. Session B listed session A's process and learned its `proc_id`.
2. Session B consumed A's unread stdout (`A-secret`). Because `check()` uses destructive buffer reads, A would no longer receive that output.
3. Session B successfully sent `SIGKILL` to A's process.

A second reproducer using two independently constructed `ToolRegistry` instances sharing the same manager produced the same result, matching settings reload registry replacement:

```text
A_START {"procId":"2d806cd5f70c",...,"running":true,...}
B_LIST {"processes":[{"procId":"2d806cd5f70c",...}]}
B_CHECK {"procId":"2d806cd5f70c",...,"stdout":"owned-by-A\n",...}
B_KILL {"procId":"2d806cd5f70c","signalled":true,"exitCode":null}
A_POST_KILL {"processes":[]}
```

Both commands exited `0`; cleanup called `disposeAll()` and removed the temporary workspace.

### Impact

Any daemon session able to call these tools can discover all daemon-managed background commands, read and consume another session's retained stdout/stderr, and terminate another session's child process. This crosses normal session boundaries and can disclose command lines, working directories, logs, build output, or secrets printed by a process; it can also sabotage builds, servers, or training runs. The attack does not require guessing a random `proc_id`, because `list_commands` discloses it.

Subagents also use the same tool executor and supply their own history session IDs (`xerxes/src/daemon/subagentHost.ts:1020-1026`), so manager-level ownership must support all tool execution sessions, not only top-level TUI session keys.

### Root cause and lifecycle contradiction

The code documents `BackgroundCommandManager` as owning processes "for one session" and says `disposeAll()` is called on session teardown (`xerxes/src/tools/backgroundCommands.ts:67-73,201-207`). Production instead creates one daemon-global instance. Production has no call to `backgroundCommands.disposeAll()` at session eviction or daemon shutdown: `onSessionEvict` only cancels subagents and prunes memory, and `shutdown` only shuts down the subagent manager (`xerxes/src/cli.ts:990-994`). The only `disposeAll()` call sites found are unit-test cleanup.

Consequences extend beyond cross-session control:

- Evicting/deleting a session does not reclaim its background commands.
- Daemon shutdown does not explicitly reclaim manager-owned children.
- A naive fix that calls global `disposeAll()` on one session's eviction would kill every other session's processes and is therefore incorrect.

## Ownership architecture / fix boundaries

### Required invariant

Keep **one host-lifetime manager** so proc handles survive runner/settings rebuilds, but make every entry explicitly session-owned. All list/read/control/release operations must be scoped by the authenticated execution context, not by caller-provided ownership data.

Recommended manager contract:

```ts
start(ownerSessionId, options)
list(ownerSessionId)
check(ownerSessionId, procId, maxOutputChars, waitMs)
kill(ownerSessionId, procId, signal)
disposeSession(ownerSessionId)
disposeAll()
```

`proc_id` can remain opaque and daemon-global, but it must not be authorization. A mismatched owner should return the same unknown-`proc_id` validation result as a missing entry to avoid an existence oracle. `list(owner)` must filter before returning records. Ownership should be captured from `ToolExecutionContext.sessionId`; do not add a model-controlled schema argument.

### Production boundaries

1. **Manager/storage — `xerxes/src/tools/backgroundCommands.ts`**
   - Add immutable owner session ID to `BackgroundEntry` (or maintain a manager-private owner index).
   - Enforce owner equality in list/check/kill.
   - Add `disposeSession(sessionId)`; keep `disposeAll()` for daemon shutdown.
   - Decide explicitly what to do when `sessionId` is absent. For daemon safety, fail closed when a host selected session-scoped operation; non-daemon direct users may need a private sentinel owner or a separate unscoped wrapper.

2. **Tool binding — `xerxes/src/tools/processTools.ts`**
   - Use the currently ignored handler context for `exec_command`, `check_command`, `list_commands`, and `kill_command`.
   - Preserve the shared manager injection; do **not** revert to a manager per rebuilt registry, which would reintroduce lost handles after reload.

3. **Daemon lifecycle composition — `xerxes/src/cli.ts`**
   - Extend `onSessionEvict` to asynchronously reclaim only that session's background commands.
   - Extend daemon `shutdown` to await both subagent shutdown and global background disposal.
   - Current `InMemoryDaemonRuntimeOptions.onSessionEvict` is synchronous (`xerxes/src/daemon/runtime.ts:315-316`) while process disposal is async. Either widen/await the lifecycle hook (larger runtime contract change) or define an intentionally fire-and-observe host cleanup path. Avoid unhandled rejections.

4. **Runtime lifecycle contract — `xerxes/src/daemon/runtime.ts`**
   - If deterministic cleanup on eviction is required, `evictSession` and callers must become async or the runtime needs a tracked cleanup queue drained by `shutdown()`.
   - Do not silently fire-and-forget resource teardown without error handling.

### Test boundaries

- `xerxes/test/backgroundCommands.test.ts`: prove owner A can list/check/kill its process; owner B sees no list entry and receives unknown-process errors for check/kill; B cannot consume A's output; `disposeSession(B)` leaves A alive; `disposeSession(A)` reclaims A only.
- Daemon integration/lifecycle test: two sessions share a production-like manager; reload preserves A's handle; evicting A kills only A's processes; shutdown disposes all remaining processes.
- Include a subagent/history-session case or explicitly normalize subagent ownership to the intended parent lifecycle. The present context uses the subagent history session ID, so parent eviction cannot reclaim it unless parent-child ownership is modeled or subagent cleanup owns it.

### Architecture decision still required

Choose whether a subagent's background command is owned by:

- its own history session (strong isolation, but subagent/session lifecycle must dispose it), or
- the top-level source session (parent eviction naturally reclaims the whole delegation tree).

The current `ToolExecutionContext` exposes only the executing `sessionId`; parent/source identity would need trusted metadata or a dedicated ownership field populated by the daemon, never model input. This decision does not block fixing top-level cross-session authorization, but it affects complete lifecycle cleanup.

## Verification evidence

- `git status --short` before writing: clean.
- Static search: production daemon has one manager created in `cli.ts`; process handlers do not inspect execution context; no production `disposeAll()` call exists.
- Dynamic shared-registry reproduction: exit code `0`, session B listed/read/killed session A's process.
- Dynamic two-registry reproduction: exit code `0`, same cross-session result across registry instances.
- No source or test files edited; only this report was created.
