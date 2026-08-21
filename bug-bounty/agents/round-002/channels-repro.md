# Round 002 — channel and skill hub reproduction

Date: 2026-08-21

## Results

| Claim | Verdict | Severity signal |
|---|---|---|
| Failed channel stop clears running state | **Confirmed** | State becomes falsely disabled after teardown throws; retry paths and status reporting can no longer identify the adapter as active. |
| Forced skill reinstall deletes local data | **Confirmed** | The entire installed skill directory is recursively removed, including files not supplied by the source. |
| Concurrent skill lock updates lose records | **Confirmed** | Two successful installs can leave both skill directories present while `lock.json` contains only one entry. |

## 1. Failed channel stop state

A standalone Bun reproduction registered a channel whose `start()` succeeds and whose `stop()` throws `Error("stop failed")`. After `start()` and rejected `stop()`:

```json
{"error":"stop failed","isStarted":false,"startedNames":[],"failures":[{"channel":"broken","operation":"stop","error":"stop failed"}]}
```

This is deterministic from `ChannelRegistry.stopOne()`: `xerxes/src/channels/registry.ts` calls `channel.stop()`, records/rethrows the error, but unconditionally executes `this.started.delete(name)` in `finally`. Consequently `ChannelManager.status()` reports `enabled` from `registry.isStarted()`, and `ChannelManager.disable()` will not retry stop once this false state is recorded (`xerxes/src/channels/manager.ts`).

Impact depends on the adapter's failure semantics, but the registry cannot assert that a transport which failed teardown is stopped. It reports disabled and omits it from later `stopAll()` attempts.

## 2. Force reinstall data loss

Reproduction:

1. Install local skill `demo` with `Version one.`.
2. Add `skills/demo/user-data.txt` containing `irreplaceable`.
3. Change the source manifest to `Version two.`.
4. Call `install(..., { force: true })`.

Observed output:

```text
Installed skill 'demo' ...
Installed skill 'demo' ...
{"manifest":"---\nname: demo\n---\nVersion two.","userData":"ENOENT"}
```

Confirmed cause: on an existing directory, force install invokes `removeDirectoryTree(directories.skills, existing.path)` before recreating the target (`xerxes/src/extensions/skillsHub.ts`). This deletes all colocated state, not only managed bundle files. There is also no rollback if a later write or security scan fails.

## 3. Concurrent lock update loss

A deterministic race used two independent `SkillsHub` instances sharing one `skillsDirectory`, installing `alpha` and `beta`. Both instances were paused immediately before their private `saveLock()` calls so both had already loaded the same empty lock snapshot; they were then released together.

Observed output (the surviving key can vary with final-writer ordering):

```json
{"results":["Installed skill 'alpha' ...","Installed skill 'beta' ..."],"lockKeys":["alpha"],"alphaManifest":"---\nname: alpha\n---\nAlpha.","betaManifest":"---\nname: beta\n---\nBeta."}
```

Both API calls reported success and both manifests existed, but one lock record was lost. The install path performs an unlocked read-modify-write (`loadLock`; mutate object; `saveLock`). `writeDirectFile()` may protect individual replacement integrity, but it does not serialize the transaction. The only in-memory lock in `SkillsHub` is `auditLocks`, used solely by `appendAudit()` and scoped to one hub instance; it does not cover `lock.json` or coordinate separate instances/processes.

## Verification

All three reproductions were run with `bun -e` against the current TypeScript source and exited with code 0. No production or test files were edited. Temporary reproduction directories were removed by each script.
