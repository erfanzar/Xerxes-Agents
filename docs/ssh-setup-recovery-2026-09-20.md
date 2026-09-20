# SSH setup recovery — September 20, 2026

The reported desktop failure was an abandoned `~/.xerxes/remote-runtime/setup.lock`
directory, encountered while preparing commit `9fc87343d44ad964f651c78e19842b0a2b84933c`.
It was not an SSH host-key mismatch. No SSH trust or credential settings were changed.

## Capability and workflow matrix

| User expectation | Current behavior and source | Confirmed defect / limitation | Implemented correction | Automated and actual-use evidence | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Reconnect after interrupted setup without manual lock deletion | `xerxes/src/ui/lib/remoteBootstrap.ts`, shared by desktop and TUI | Directory existence was treated as active ownership indefinitely. The reported host had an empty marker and abandoned staging checkout. | Use kernel-owned `setup.guard` locking: Linux `flock`, macOS/BSD `lockf -k`. The guard file stays in place; ownership ends when its holders exit. Old markers are ignored, not deleted. | Eight real-shell regressions pass with macOS lockf and Linux flock over SSH. Abrupt termination leaves the guard file present, and the next setup succeeds. Actual target bootstrap returned `XERXES_REMOTE_READY`. | Network filesystems and other BSD systems were not exercised. Hosts lacking either locking utility get an installation instruction. |
| Open two connections while setup is in progress | Same bootstrap, guarded setup body | The second connection immediately failed; the old body did not wait for the first build. | Wait up to 240 seconds, then recheck the ready release under the lock. A completed build is reused. Timeout explains reconnect and preserves the owner. | Native concurrent-process tests assert exactly one install, waiter cancellation, bounded timeout, and successful reconnect. All pass on macOS and the Linux host. | Legacy clients still use their old directory marker and can build concurrently during migration. Unique staging directories and atomic publication protect the winning release. |
| Keep a valid release when an older client finishes first | Atomic release publication in `remoteBootstrap.ts` | Waiting alone does not coordinate with already-running old bootstrap code. Shell `mv` can nest a stage inside an existing directory. | Bun `renameSync` publishes atomically. Accept a concurrently published ready release; do not merge into a populated release. | A simulated legacy publication is retained; a conflicting populated incomplete release is preserved and reports an actionable error. | A populated incomplete revision directory still requires inspection. Abandoned staging directories and legacy markers are deliberately retained rather than guessing ownership and deleting them. |
| Resolve daemon build mismatch without interrupting work or losing history | Existing `ui/lib/managedRuntime.ts`, `runtime.restart_if_idle` | The stale setup marker prevented reaching the existing safe update path. An older acceptance fixture incorrectly expected an empty, unsent session to persist. | Bootstrap can reach the existing idle guard again. Acceptance now seeds a completed exchange and verifies session identity, message count and saved content after replacement. | Actual isolated daemons on macOS and Linux pass both local and managed-remote replacement: new PID/build, same saved session/content. Existing busy/cancellation runtime tests passed in the full gate. | No new live provider call was needed or performed. Active-user daemon replacement was not forced; real busy behavior is covered by existing automated tests, not a new user-task experiment. |
| Recover the reported desktop workspace | Installed app's Retry SSH connection action | Desktop showed “Could not open SSH workspace” with the stale-marker diagnostic. | Ran the new bootstrap on the designated host, then clicked the existing retry action. Rebuilt app installed for the next launch. | Native accessibility state changed to Connecting, then removed the failure view. Screenshot showed Connected and saved EasyDeL sessions, including Block Indexer and mHC Models. | Specific saved-conversation reopening in the desktop was not exercised afterward; the Mac locked during follow-up inspection. The running app was not restarted. |

## Captured terminal evidence

Actual SSH target: `n_server_spot_m`, workspace `/home/erfan/EasyDeL`.
Bootstrap source was supplied over SSH stdin. No credentials, configuration
folders, SSH keys, or environment dumps were copied.

```text
Xerxes · checking remote installation…
Xerxes · waiting for remote setup, then checking the build…
Xerxes · installing/updating remote build 9fc87343d44ad964f651c78e19842b0a2b84933c…
Xerxes · installing dependencies and building for this machine…
Xerxes · ready. Opening remote workspace…
XERXES_REMOTE_READY {"projectDir":"/home/erfan/EasyDeL","socketPath":"/home/erfan/.xerxes/daemon/global-5ec677887c85cd46.sock","expectedBuildId":"69c465cafd68c6a7","busy":false}
```

Linux shell regressions (Bun 1.3.12, isolated temporary homes):

```text
8 pass
0 fail
46 expect() calls
Ran 8 tests across 1 file. [2.64s]
```

Actual isolated daemon acceptance, independently completed on macOS and Linux:

```text
PASS local: aaaaaaaaaaaaaaaa -> bbbbbbbbbbbbbbbb; new PID; original session preserved
PASS managed-remote: aaaaaaaaaaaaaaaa -> bbbbbbbbbbbbbbbb; new PID; original session preserved
```

Only owned test daemons were shut down. The repaired user workspace uses the
published main revision above; the bootstrap correction is in the local source
and rebuilt local client, not yet committed or published.

## Completed checks

- `bun run check`: Bun-only repository policy, catalog checks, runtime, TUI and desktop typechecks passed.
- `bun run test`: 4,110 runtime tests and 1,493 UI tests passed; three pre-existing environment-dependent runtime skips; zero failures.
- `bun run build`: runtime and TUI bundles built and verified.
- `bun run --cwd xerxes build:desktop`: packaged and signed local app successfully.
- `codesign --verify --deep --strict`: passed for the prepared installed app.
- `git diff --check`: passed.
- `bun xerxes/test/fixtures/daemon/runtimeUpdate.ts`: passed locally and, using the same fixture with paths adjusted to the remote installation, over SSH. These exercise real daemon processes with seeded history, not live provider traffic.

Raw captures for this run are `/tmp/xerxes-ssh-repair.log`,
`/tmp/xerxes-ssh-lock-linux-tests.log`, `/tmp/xerxes-local-daemon-update.log`,
`/tmp/xerxes-ssh-daemon-update.log`, `/tmp/xerxes-ssh-fix-{check,test,build}.log`,
`/tmp/xerxes-ssh-fix-desktop-build.log`, and `/tmp/xerxes-ssh-fix-install.log`.
The installed app is `/Applications/Xerxes Agents.app`; its previous bundle was
retained at `/Applications/.Xerxes-Agents-before-ssh-recovery-1789892914203.app`.
The running app was left alone; the new bootstrap loads on its next launch.

This checkpoint verifies the reported setup failure and its automatic recovery
path. It does not declare the broader repository/TUI capability audit complete.
