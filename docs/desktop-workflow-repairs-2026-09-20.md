# Desktop and TUI workflow repairs — 2026-09-20

This checkpoint covers the reported desktop activity, agent roster, fullscreen,
file-tree and new-file review defects. It does not close the broader repository
and remote-integration audit. Prior acceptance claims were not used as proof of
these changes.

| User expectation | Observed behavior and source | Confirmed defect | Implemented correction | Verification | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Work appears immediately in a closed, expandable group | `desktop/renderer/activityGroups.ts`, `blocks.ts`, `App.tsx` | Operational notices and agent cards escaped grouping; notices finalized running tools and broke result pairing | Group every operational block from the first event; preserve run/group identity and click state; keep prose outside; keep approvals separately accessible; notices no longer complete tools | Bun desktop execution/shell regressions; native Electron sequence: thinking, tool, notice, expand, result, completion, collapse | Live provider streaming was not called; native sequence uses injected events through the production block builder |
| Running commands are compact and readable | `DesktopPanels.tsx` used a wrapping strong title with a generic Output action | Long commands filled most of the activity panel | `CommandActivity.tsx`: three-line summary, state, elapsed time, cwd; click reveals bounded full command and only that command's output, refresh and authorized interrupt; errors retain output; disconnect disables controls | Native Electron with a long command, 36 historical jobs, 58 past agents and 3 running agents; collapsed row measured 91 CSS px; output error/retry and disconnect/reconnect passed; 720px and 1440px captures inspected | Native command presentation uses injected terminal data; no throughput or CPU improvement claimed |
| Running agents appear before failed attempts | `AgentRoster.tsx` placed failed agents among current rows | Old failures dominated the panel | Active/queued agents first, attention next; failed, stopped and completed agents under collapsed Past agents; statistics initially collapsed | Bun ordering/render regressions; native Electron confirms 3 active rows before 58 collapsed past rows | Very large histories still use the existing DOM roster rather than virtualization |
| Chat and panel agree about spawned agents | `store.ts` matched titles and relied on snapshots; polling stopped after about six minutes | Array spawn receipts were ignored, renamed agents could not be joined, pending requests disappeared from the panel, long work stopped polling | Reconcile receipt IDs with chat members; adopt returned agent rows; prevent an older in-flight snapshot from overriding a receipt; display unmatched requests as Awaiting runtime status without fabricated controls; continue polling while work remains; genuine resumed turn events reactivate terminal rows | Store regression with three differently titled spawn results, stale manifest and subsequent completion; event regression for failed-agent resume; native populated roster | The user's three original agents were not inspected or restarted; pending labels intentionally do not assert a live process |
| SSH agent worktrees work with the host's Git | `runtime/subagentWorktrees.ts` used `ls-tree --format`, matching screenshot error | Git 2.34.1 on `n_server_spot_m` lacks that option | Parse longstanding NUL-delimited `ls-tree -r -z`; retain submodule/nested-repository rejection | Bun real-Git worktree/index/submodule tests; actual SSH Git 2.34.1 isolated working-tree capture includes edits and new files, byte-identical parent index | User's existing failed attempts are retained and not automatically retried; other Git versions not individually exercised |
| Fullscreen toolbar aligns without an empty traffic-light inset | `desktop/main.ts`, `preload.ts`, `App.tsx`, `atelier.css` used a fixed macOS inset | Windowed inset persisted in fullscreen | Narrow host capability reports traffic-light visibility; renderer follows events and ignores an older query response | Native production Electron: 88px windowed, 12px fullscreen, restored 88px | Actual verification on macOS only |
| Open multiple folder branches without losing siblings | `DesktopPanels.tsx` replaced the current folder on click | Navigation drilled into one directory | `WorkspaceFileTree.tsx` keeps independent lazy branches, descendants, keyboard access and 50-entry paging; additive `complete.path_offset` | Real local daemon/native desktop: f4 and f7 open together, nested children, collapse/reopen state, 65-file paging; real SSH daemon paging 50+15 | Tree state belongs to the mounted panel; older daemons need updating for paging |
| Read newly added/untracked files in GUI and TUI Changes | `workspace/gitDiff.ts`, `desktopRpc.ts`, `ReviewPanel`, TUI `diffPanel.tsx` | Overview budget could omit file bodies; initial 50-file cap made later files unreachable | Keep complete overview index, load individual literal paths, paginate new files, expose working-tree review, preserve overview when returning; shared bounded old-daemon filePreview fallback for known untracked files; scroll long TUI file indexes | Real local daemon/native desktop after 5,000 changed tracked rows; 65 new files and Unicode path; actual 140x40 TUI F7/M/last-file select/back/exit; real SSH-host daemon selected diff and filePreview; automated old-daemon fallback; Bun error/cancellation/bounds/index tests | Up to 10,000 untracked entries; previews remain bounded; old daemons require updating for list pagination |

All source references above are under `xerxes/src/` unless stated otherwise.
Public v35 contracts remain compatible; optional path, untracked-limit and
completion-offset fields are documented in `xerxes/src/ui/PROTOCOL.md`.

## Actual-use evidence

These are local verification artifacts, not repository fixtures represented as
live services. Paths may expire when temporary storage is cleared.

- Native work monitor: `/var/folders/fh/c3_87ccs6zlbc4g3dm2nxr3h0000gn/T/xerxes-desktop-qa-FKNv8B/`.
  `monitor-result.json`, collapsed/expanded/narrow, output-error and disconnected
  PNGs and text captures. Run again with
  `bun xerxes/scripts/previewDesktop.ts --verify-work-monitor`.
- Real production desktop and local daemon:
  `/tmp/xerxes-ui-native-2UqY8K/`; `result.json`, recursive-tree, new-file-diff,
  fullscreen and windowed PNGs/text. Reproduce after desktop build with
  `bun xerxes/scripts/verifyDesktopWorkspace.ts`.
- Immediate work grouping native capture:
  `/var/folders/fh/c3_87ccs6zlbc4g3dm2nxr3h0000gn/T/xerxes-desktop-qa-FKNv8B/`;
  `activity-result.json` and five event-stage captures.
- Real TUI terminal capture: `/tmp/xerxes-tui-review-final.raw`.
- Real isolated SSH daemon file paging and preview:
  `/tmp/xerxes-ssh-files-final.log`.
- Actual SSH worktree compatibility: `/tmp/xerxes-git-compat-ssh.log`.
  Git 2.34.1, tracked edit plus new file captured, parent index unchanged.

Only owned fixture processes and temporary workspaces were used. Active user
sessions, daemons, commands and SSH tasks were not stopped. Credentials were not
copied and external provider calls were not made.

## Checks and limits

Final root `bun run check && bun run test && bun run build` passed, followed by
`bun run --cwd xerxes build:desktop` and `git diff --check`. Runtime: 4,122
passed, 3 environment-dependent skips (Windows PTY, Windows directory fsync,
installed clangd), zero failed. TUI: 1,495 passed, zero failed. Logs:
`/tmp/xerxes-monitor-final-{check,test,build,desktop}.log`.

The installed `/Applications/Xerxes Agents.app` was replaced with the verified
build; its main bundle SHA-256 matches the build, and strict code-signature
verification passed. The previous app is retained at
`/Applications/.Xerxes-Agents-before-workflow-update-20260920.app`. The running
user app was not restarted. The remote worktree correction was verified in an
isolated checkout; the user's remote runtime was not deployed or restarted.
At this verification checkpoint, the changes were uncommitted and had not been
pushed or published. A subsequent Git push does not deploy or restart that remote runtime.

One repeat of native fullscreen acceptance timed out waiting for the fullscreen
event. A subsequent complete run passed on the identical build. The failed
capture remains at `/tmp/xerxes-ui-native-Blqmoe/`; the final successful run is
`/tmp/xerxes-ui-native-2UqY8K/`. The cause of that intermittent transition timeout
is not established; this is a remaining native-window verification uncertainty.

 An earlier gate caught
an outdated assertion requiring a pending agent to be absent from the entire
window; the corrected test separately checks closed chat content and discoverable
pending-agent status in the panel.

The native Electron test required a short isolated `XERXES_HOME`: a long macOS
temporary path failed to connect to its Unix socket while a Bun client could
connect. That environment boundary remains unresolved. Windows, live external
providers, and the user's actual current three-agent runtime state remain
unverified by this checkpoint. No broad performance acceptance is claimed.
