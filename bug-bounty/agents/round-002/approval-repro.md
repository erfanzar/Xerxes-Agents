# Round 002 — legacy empty `args_hash` grants remain tool-wide

## Verdict

**Confirmed; security-relevant compatibility behavior.** A persisted legacy `ALWAYS` grant with `args_hash: ""` authorizes every future argument payload for the same tool, across sessions and daemon restarts, without another prompt. This is explicitly intentional backward compatibility, but for argument-sensitive tools such as `exec_command` it preserves the confused-deputy behavior that argument scoping was added to prevent.

Suggested severity: **High when a legacy empty-hash grant exists for a command/file/network-capable tool; otherwise dormant.** The condition is not remotely created by the demonstrated path: an attacker/model must encounter a pre-existing legacy grant (or separately gain the ability to modify the owner-only approval file).

## Reproduction evidence

Current production flow:

1. Production loads the durable file at Xerxes home `approvals.json` through `ApprovalStore` (`xerxes/src/daemon/productionInteractions.ts`).
2. `ApprovalRecord.fromRecord()` maps missing or empty `args_hash` to the empty internal hash (`xerxes/src/security/approvals.ts:69-81`, `:229-232`).
3. Every live request is correctly hashed before lookup (`xerxes/src/daemon/interactions.ts:220-227`).
4. Lookup nevertheless treats a stored empty hash as a wildcard: `record.argsHash === '' || record.argsHash === hash` for `ALWAYS` and `SESSION` (`xerxes/src/security/approvals.ts:133-138`).

A temporary on-disk legacy record for `exec_command`, followed by a fresh-session request for `rm -rf /tmp/valuable`, produced:

```json
{"decision":"approve","pending":[],"legacyArgsHash":""}
```

Thus the destructive argv was approved immediately and no permission request remained pending.

Focused existing tests also passed:

```text
(pass) always and session approvals only match the arguments they were granted for
(pass) records written before argument scoping keep granting tool-wide
2 pass, 0 fail, 7 expect() calls
```

Command: `bun test xerxes/test/securityRedactApprovals.test.ts --test-name-pattern "records written before argument scoping keep granting tool-wide|always and session approvals only match"` (Bun 1.3.12).

The legacy test explicitly proves one empty-hash `exec_command` grant accepts both `ls` and `rm -rf /` (`xerxes/test/securityRedactApprovals.test.ts`).

## Compatibility intent

Intent is unambiguous rather than accidental:

- The inline comment says empty hashes represent pre-argument-scoping records and remain tool-wide to avoid silently revoking prior approvals (`xerxes/src/security/approvals.ts:133-135`).
- Commit `34ae92e` introduced argument-scoped `ALWAYS`/`SESSION` checks while deliberately retaining empty hashes as wildcard legacy records. Its message states: “Legacy records with an empty hash still grant, so existing approvals are not revoked.”
- The same commit added the regression test named `records written before argument scoping keep granting tool-wide`.

New interactive decisions are not affected: the daemon unconditionally hashes current arguments and persists that hash (`xerxes/src/daemon/interactions.ts:163-171`, `:220-227`).

## Exploitability and boundaries

- **Required state:** an old durable `ALWAYS` grant with empty/missing `args_hash` for the exact tool name. Durable loading keeps only `ALWAYS` records, so a legacy `SESSION` record does not survive restart (`xerxes/src/security/approvals.ts:166-173`).
- **Trigger:** induce any later call to that tool with more dangerous arguments. Matching is tool-name-wide, independent of session or argument hash.
- **Impact:** bypass of the current per-argument approval prompt. For `exec_command`, this can turn approval of a benign historical invocation into arbitrary command execution under Xerxes' process privileges. Equivalent impact depends on the affected tool's capabilities.
- **Persistence:** `ALWAYS` records are shared across daemon restarts; no fresh user gesture is required.
- **Limitations:** the behavior does not cross tool names. A normal newly created approval carries a SHA-256 argument hash. Directly planting the record requires local write access to the approvals file or another vulnerability; the store writes it owner-only (`0600`).
- **Denials:** empty-hash legacy denials are also wildcard. They fail closed rather than escalating privilege, though record ordering can affect which newest matching decision wins.

## Safe migration

Do **not** synthesize a hash for a legacy grant: the historical arguments are unavailable, so any chosen hash would be fabricated.

Recommended migration:

1. Add an explicit persisted format/version or scope discriminator; stop overloading `args_hash: ""` as a wildcard.
2. On load, classify empty/missing-hash records as legacy.
3. Preserve legacy wildcard **denials**, but quarantine/disable legacy wildcard **grants** for argument-sensitive tools and require one fresh approval for the exact displayed arguments. Prefer applying this to all executable/write/network tools through capability metadata rather than a hard-coded `exec_command` list.
4. Surface a one-time notice listing disabled tool names and the approval-file path; retain an owner-only backup for rollback/audit. Never silently broaden or guess scope.
5. Persist replacements atomically only after an explicit new user decision. New records must contain a non-empty argument hash; reject empty hashes from current writers.
6. If compatibility requires temporary opt-in, gate legacy tool-wide grants behind an explicit, time-bounded user setting disabled by default, with conspicuous tool-wide wording. Do not auto-enable it during upgrade.
7. Add migration tests for missing and empty `args_hash`, grant versus denial, restart behavior, exact-argument reapproval, and ordering with newer scoped records.

This intentionally trades silent retention of old grants for a one-time re-prompt on privileged operations, which is the safe failure mode.
