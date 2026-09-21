# Model discovery and failure replay recovery

The reported screenshot contained a real inventory failure and a misleading
successful tool label. The September 19 empty-profile correction covered the
inventory core and locally forwarded routes, but missed the daemon and standalone
hosts' post-discovery profile identity checks. This follow-up verifies those hosts.

| User expectation | Current behavior and source | Confirmed defect | Correction | Verification evidence | Remaining uncertainty |
| --- | --- | --- | --- | --- | --- |
| Discover configured providers with omitted or empty optional fields | `daemon/server.ts:modelInventoryToolRequest`, `runtime/profileInventory.ts:profileInventoryHost`, `runtime/modelInventory.ts` | The core normalized empty input, then the host looked up the original empty string and threw after successful provider enumeration. Padded valid names also failed post-validation | Apply the same trimming and omission semantics to the final identity check. Keep nonempty unknown-profile rejection and credential-change checks | Daemon and standalone regressions cover the screenshot-shaped request, whitespace, padded valid names, invalid names, usage requirements and cancellation. Existing identity-replacement tests pass. Actual production daemon and native tool execution recover from a failed lookup into successful provider and model discovery | Enumeration does not grant access or switch the conversation's provider. Local forwarding retains its approved scope |
| See failed tools as failed after reconnect or history restoration | `desktop/renderer/blocks.ts:tool_result/blocksFromStoredMessages` | Native persisted exceptions carry the reserved `Tool execution failed:` prefix with `permitted:true`. Replay treated permission as success when no separate error field survived | Preserve explicit stored errors, recognize the native failure prefix and handle permission denial at the common event fold | Regression covers live events, saved messages with and without execution metadata, explicit errors, cancellation and ordinary text mentioning failures. Packaged GUI restored actual daemon-generated history, showed the failed row and successful retry, and kept work groups initially closed | A legacy result with neither error metadata nor the native failure marker cannot reliably recover its outcome |
| Read an error without identical extra panels | `desktop/renderer/Execution.tsx` | Inferred errors duplicated the exact output in an additional error panel | Render a separate error panel only when it adds information beyond output/stderr | Renderer regression and final packaged native interaction assertion; wide/narrow captures | Raw details still retain the original result for inspection |
| Find GLM without guessing a profile name | Existing local and SSH profile stores; `profileInventoryHost` | The failed empty-profile listing prevented discovery of the actual configured profile names | Corrected inventory lists existing configuration without altering it | Live catalog reads returned `glm-5.3-flash` for local `zai` and remote `zai-glm` on `n_server_spot_m`. The remote check ran on that host using its existing credentials; no credential was copied or printed | Catalog visibility is verified. Paid generation, subscription allowance and delegated GLM execution were not tested |

Final full gate passed: `bun run check`, `bun run test`, `bun run build`, and
`bun run --cwd xerxes build:desktop`. Runtime: **4,131 passed, three skipped,
zero failed**. TUI: **1,495 passed across 151 files**. Build ID:
`a68af04871b006ae`. Logs are `/tmp/xerxes-discovery-final-{check,test,build,desktop}.log`.
`git diff --check` passed. The skips remain environment-dependent Windows and
installed-clangd checks.

Actual-use evidence:

- `/tmp/xerxes-discovery-final-20260921/result.json`, `frames.json` and
  `provider-results.json`: production CLI daemon, native tool execution, a failing
  request, successful empty/padded-profile discovery and same-session resume.
  The provider in this end-to-end test was an isolated synthetic HTTP service.
- `gui-result.json` and `discovery-replay-{wide,narrow}.png` in that directory:
  packaged production main/preload/renderer reading the real daemon's saved tool
  history, with initially closed groups and correct expanded status/output.
- `tui-replay.raw`: actual 100-by-28 OpenTUI resumed the same saved session,
  displayed three retained tools and `DISCOVERY_RECOVERED`, then exited normally.
- `/tmp/xerxes-discovery-live-{local,ssh}.json`: separate live catalog checks
  against the configured GLM service. These are catalog acceptance, not inference
  or an end-to-end SSH daemon/TUI trial.

Earlier disposable harness attempts used an unsupported interactive CLI argument,
mismatched builds, an incorrect Electron import, or a stale fixture daemon. They
are not acceptance evidence. The final isolated run above used the matching build.
No active user daemon, session, provider selection or credential configuration was
changed. These fixes were subsequently included in the installed
[rejected-spawn recovery build](rejected-spawn-recovery-2026-09-21.md).
The broader workflow audit remains in progress; no performance gain is claimed.

Captures and result records are also retained at
`/Users/erfan/Documents/Projects/xerxes-desktop-verification/model-discovery-2026-09-21/`.
All owned test processes were stopped, and the temporary SSH catalog-check bundle
was removed. No credential material was included in the retained evidence.
