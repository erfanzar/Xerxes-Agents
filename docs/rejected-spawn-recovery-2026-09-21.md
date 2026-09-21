# Rejected spawn status and host-specific provider names

The screenshot showed an `AgentTool` call rejected for provider profile `zai`
while its provisional inspector still said “Awaiting runtime status.” The remote
profile is named `zai-glm`; local and remote names are distinct configuration.

| User expectation | Current behavior/source | Confirmed defect or limitation | Correction | Verification | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| A rejected spawn stops appearing as running | `desktop/renderer/store.ts`, `blocks.ts`, `AgentRoster.tsx` | The agent card checked only a separate error field; native failure text and denied results could leave it working without a runtime identity | Share failure interpretation between transcript and agent state. Retain the failure on unconfirmed members. Reconcile stale cards with their tool evidence, bounded by user-turn boundaries | Store tests cover explicit errors, native error text, permission denial, later turns, and partial batches with a confirmed running child. Renderer tests cover stale cards and failure association across turns | A partial failure does not prove an already-identified child stopped; its real runtime status remains authoritative |
| Inspect what was requested and why it failed | `desktop/renderer/types.ts`, `store.ts:spawnMembersOf`, `AgentInspector.tsx` | Model/profile arguments disappeared from provisional records; the inspector promised an identity that would never arrive | Retain requested model/profile, show failed status and error, label unconfirmed settings as requested, and remove the “not assigned yet” claim for terminal requests | Focused inspector rendering checks requested settings, failure guidance, absent waiting claims and absent controls | Older records without requested settings cannot reconstruct them. No fake runtime ID or control authority is created |
| Recover from using a local profile name on an SSH host | `daemon/agentProvider.ts`, `daemon/server.ts`, `runtime/profileInventory.ts`, `runtime/modelInventory.ts` | Early route resolution rejected the name before catalog validation and returned no recovery path | All unavailable-profile paths suggest exact matching configured profiles on the execution host and direct the agent to session-scoped inventory. Never alias profiles or change credential sources automatically | Resolver tests assert the hint, no client construction, and no credentials/endpoints in errors. Daemon/standalone tests validate suggestions. A read-only check on `n_server_spot_m` rejects `zai`, suggests `zai-glm`, and successfully validates `zai-glm` with `glm-5.3-flash` against its live catalog | Catalog validation is not model inference, quota acceptance, or successful child execution. Local forwarding retains its existing explicit scope |

The initial native acceptance run caught an additional early route resolver that
still emitted the old error, despite validator tests passing. That resolver was
corrected and covered separately. Its earlier failed run is not final acceptance.

This work includes the
[provider inventory and history fixes](model-discovery-recovery-2026-09-21.md).
No user session, default model, credentials, or running daemon was changed.
The broader workflow audit remains in progress.

Final validation passed: `bun run check`, `bun run test`, `bun run build`, and
`bun run --cwd xerxes build:desktop`. Runtime: **4,136 passed, three skipped,
zero failed**. TUI: **1,495 passed across 151 files**. Build ID:
`922b810b8986f968`. Logs: `/tmp/xerxes-spawn-final-{check,test,build,desktop}.log`.
The strengthened cross-turn regression also passed separately in
`/tmp/xerxes-spawn-cross-turn.log`. `git diff --check` passed.

Final native acceptance used the packaged production main/preload/renderer and
an isolated production CLI daemon with a synthetic HTTP provider. An actual
`AgentTool` request failed during provider routing, then provider inventory and
the correct profile's model catalog succeeded. Clicking the failed agent showed
its requested model/profile and exact host suggestion, with no running claim or
controls. Work groups began closed. Wide and narrow native captures and results
are retained at
`/Users/erfan/Documents/Projects/xerxes-desktop-verification/rejected-spawn-2026-09-21/`:

- `gui-result.json`, `rejected-spawn-wide.png`, `rejected-spawn-narrow.png`:
  actual GUI interaction and visible failed inspector.
- `result.json`, `frames.json`, `provider-results.json`: real daemon/tool flow
  and same-session resume evidence. The provider was synthetic.
- `ssh-profile-validation.json`: separate real configured SSH profile validation,
  without model generation or credential copying.

Owned fixture processes were stopped and the temporary SSH verification bundle
was removed. On September 21, the verified build was installed at
`/Applications/Xerxes Agents.app` at the user's request. All 614 file and symlink
entries matched the packaged build; strict deep signature verification passed.
The previous app is retained at
`/Applications/.Xerxes-Agents-before-provider-recovery-20260921.app`.
The app was not relaunched and active daemons were not restarted. The new client
loads on reopening; runtime corrections require the matching daemon update when
the user's running work permits it.
