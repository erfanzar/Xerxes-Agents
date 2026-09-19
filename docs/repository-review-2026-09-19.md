# Repository review — September 19–20, 2026

This review covers the repository inventory, pending product changes, public
boundaries and the repository-wide validation gate. It is a risk-based source
review, not a claim that every line or external integration was exercised.
The existing [TUI capability matrix](tui-workflow-acceptance.md) records detailed
workflow findings, terminal captures, performance observations and remaining
uncertainty. Its broader acceptance goal remains unfinished.

## Findings corrected in this review

| Finding | Correction | Evidence |
| --- | --- | --- |
| Permission changes acknowledged before persistence could revert to a less restrictive daemon default after restart | Serialize a staged permission/pin/context-delta write before publishing; preserve the previous policy on failure | `xerxes/test/permissionPersistence.test.ts`: restart without another flush, failed save/retry, concurrent writes and turn cancellation |
| Bare `/permissions` reported the daemon default instead of the selected task | Resolve the current session policy first | `xerxes/src/ui/__tests__/gatewayPermissionPersistence.test.ts`: two isolated tasks, invalid selection and task-scoped reporting |
| Initial status after resume overwrote the correct task permission with the daemon default in the footer | Use the resumed session policy in the initial status event | Same real-socket test fails before the fix with both manual and auto events; passes after. Actual terminal restart shows plan-only policy, preserved draft, transcript and session identity |
| Locked test-tool dependencies included advisory-affected Vitest/mocker, PostCSS and nanoid versions | Vitest 4.1.11; root overrides keep PostCSS on patched 8.x and nanoid on patched 3.x, without adding them as runtime dependencies | `bun audit --json` changed from four affected package entries to `{}`; frozen-lockfile install passes. Locked PostCSS 8.5.28 and nanoid 3.3.18 |

Dependency advisories: [Vitest GHSA-82fw-gwwq-j7x9](https://github.com/vitest-dev/vitest/security/advisories/GHSA-82fw-gwwq-j7x9),
[PostCSS GHSA-fxqj-rqcc-2cmp](https://github.com/postcss/postcss/security/advisories/GHSA-fxqj-rqcc-2cmp),
[nanoid GHSA-2v37-7h3g-55p8](https://github.com/advisories/GHSA-2v37-7h3g-55p8).
Audit results establish dependency status, not evidence of exploitation or a
complete security assessment.

## Coverage

| Area | Review and verification scope | Limits |
| --- | --- | --- |
| TUI, daemon, sessions and workspace routing | Pending changes, v35 contract, persistence queues, history/outcomes, command restoration, local/remote routing and appearance; full automated suites; isolated source-TUI trials | See capability matrix for per-workflow acceptance and unresolved external environments |
| SSH and provider reuse | Grant ownership, model/destination/workspace scope, revocation/expiry, fixed errors, request/output limits, private broker, routing for parent/child/auxiliary calls and SSH configuration | Earlier actual-use loopback OpenSSH evidence is retained. External destination, proxy-chain and live-provider acceptance remain unverified |
| Desktop | All pending desktop changes: production React build, lazy execution details, memoized markdown/tool rows, bounded preview requests/retry bookkeeping and child-view focus ordering; desktop typecheck/tests and bundle build | No new installed-app visual or performance acceptance claimed by this review |
| Providers, streaming and orchestration | Changed provider adapters, output caps, partial output/outcome persistence, native child route inheritance and fail-closed recovery; full runtime suites | Real external provider behavior and paid calls not exercised in this review |
| API, ACP, MCP, authentication and security | Targeted source inspection of authentication, body/frame bounds, cancellation, credential storage, MCP parsing, URL/SSH policy and private relay authority; associated tests in full runtime gate | This is not a penetration test or an exhaustive cross-process race proof |
| Channels, schedules, tools, skills and extensions | Inventory plus targeted signature, lease, skill-path/trust and execution-boundary review; repository runtime tests | No live messaging, email, cloud or third-party account operations |
| Build, docs, distribution and CI | Bun-only repository assertion, model/image catalog checks, all TypeScript projects, frozen dependencies, runtime/TUI/desktop bundles, documentation and isolated smoke/release checks | Linux/Windows/macOS-x64 and Docker runtime coverage depend on CI or those environments |

The existing pending product work is included in this handoff: secure local
provider reuse for SSH, truthful task outcomes, persistence and recovery fixes,
TUI navigation/status improvements, `/appearance chrome|transparent`, and desktop
performance fixes. Transparent appearance uses the terminal's default background;
actual translucency is controlled by the terminal application. The setting is
saved locally and does not alter daemon configuration.

## Actual-use evidence

Evidence root: `/tmp/xerxes-tui-audit-20260919/` on the review machine. These local
captures are not embedded in the repository and may be removed by OS temporary
storage cleanup.

`permission-terminal.raw` records the 80×24 source TUI against an isolated real
Unix-socket daemon. The trial selected manual permissions, blocked its transcript
file to force a save failure, confirmed the readable failure and retained policy,
restored storage, selected plan permissions and restarted the owned daemon twice.
The first restart exposed the footer defect; the corrected fresh daemon rendered
`plan only, no writes`. The same renderer retained its draft and task identity.
`permission-acceptance.json` contains eight passing assertions. No external
provider call occurred. Owned renderer/daemon processes were stopped.

Earlier appearance, cancellation, reconnect, local-provider capability and
loopback SSH captures are indexed by the TUI matrix. They are not relabelled as
external-host or real-service acceptance. The user's active daemon was not stopped.

Machine-specific `.codex/hooks.json` and `.github/hooks/impeccable.json` are left
local, outside the product commit.

## Final checks

The following completed against the final source, without source edits during
the gate:

- `bun install --frozen-lockfile`: passed.
- `bun audit --json`: `{}` (no reported advisories).
- `bun run check`: passed repository, catalogs, runtime, TUI and desktop checks.
- `bun run test`: **4,095 runtime passed, 3 skipped, 0 failed** across 570 files;
  **1,475 TUI passed** across 151 files. Skips cover two Windows-only cases and
  the unavailable installed-clangd integration trial.
- `bun run build`: passed; runtime build ID `0b70fa1a43fbf0bb`, verified TUI bundle
  694,039 bytes.
- `bun run smoke` with disposable `XERXES_HOME`: both daemon RPC and completion
  smokes passed; no provider call required.
- `bun run docs:build`: passed.
- `bun xerxes/scripts/buildDesktop.ts`: passed after matching runtime build.
- `bun run release:prepare`, `bun pm pack --ignore-scripts`, `bun run
  release:check` and `bun run release:smoke`: passed against a package staged in
  `/tmp`; the packed CLI and OpenTUI runtime loaded in an empty install. Nothing
  was published.
- `git diff --cached --check`: passed. High-confidence credential-pattern scan
  of added staged lines: no matches.

Logs in the evidence root: `review-full-{check,test,build}.log`,
`review-smoke-isolated.log`, `review-docs.log`, `review-desktop-final.log`,
`review-release-{prepare,pack,check,smoke}.log`, and
`review-dependency-audit-after.json`. Failed preliminary smoke/build attempts
remain recorded: smoke safely refused the existing user-daemon build mismatch;
desktop build rejected a stale runtime bundle. Both passed with the corrected
verification setup. Docker is unavailable on this machine; container and other
operating-system acceptance remain unverified. Remote CI results are not implied
by this local gate.
