# Bug bounty Round 002 — TUI reproduction

Date: 2026-08-21

## Scope and inventory

Inventory was taken before chunked reads. Review excluded generated output, dependencies, and lockfiles. Relevant inspected files:

- `xerxes/src/ui/app/useSessionLifecycle.ts`
- `xerxes/src/ui/gatewayClient.ts`
- `xerxes/src/ui/opentui/sessionPicker.tsx`
- `xerxes/src/ui/__tests__/sessionLifecycle.test.ts`
- `xerxes/src/ui/__tests__/sessionPicker.test.tsx`
- `xerxes/src/ui/__tests__/gatewayClientLifecycle.test.ts`

The Round 001 source was read from project memory at `/Users/erfan/.xerxes/projects/6577d2152bdc/memory/bug-bounty/agents/round-001/tui.md`. No production or test source was edited.

## R001-TUI-01 — overlapping session switches desynchronize visible and transport-active sessions — **confirmed**

### Source evidence

The reported race remains present.

- `useSessionLifecycle` uses one `switchGenerationRef` and checks it only around asynchronous continuations. `activateLiveSession()` starts `gw.request('session.activate', ...)`, then ignores a stale completion; `resumeById()` similarly starts `gw.request('session.resume', ...)` after `setup.status`, then ignores a stale completion (`xerxes/src/ui/app/useSessionLifecycle.ts`, `activateLiveSession` and `resumeById`). This protects React-visible state but does not cancel or serialize the client operations.
- `GatewayClient.sessionResume()` commits transport-global `activeSessionKey = nextSessionKey` after daemon `initialize` succeeds. `GatewayClient.sessionActivate()` commits the same global key after `session.open` succeeds (`xerxes/src/ui/gatewayClient.ts`, `sessionResume` and `sessionActivate`). Those commits occur before the lifecycle hook receives and generation-checks the result.
- Resume transcript capture is still singleton state. `captureInitializeInfo(true)` throws `cannot initialize two resumed sessions concurrently` whenever another resume owns `initializeTranscriptCapture` (`xerxes/src/ui/gatewayClient.ts`, `captureInitializeInfo`).

Consequently, two resume selections produce the exact Round 001 failure ordering:

1. Resume A passes `setup.status`, begins daemon `initialize`, and owns transcript capture.
2. Resume B is the newest UI generation, passes setup, and rejects immediately at the singleton capture guard.
3. B's current-generation catch reports an error and restores status to `ready`, but retains the previously visible session.
4. A later succeeds and commits the client's `activeSessionKey` to A; A's lifecycle continuation is stale and therefore does not update visible UI state.

The visible session and keyless/active-key daemon operations are then scoped to different sessions. The analogous out-of-order `session.activate` case does not need the capture singleton: an older request may commit its key after the newer request has already won visible state.

### Precise regression designs

**Resume race, hook plus real client boundary**

1. Mount a probe around `useSessionLifecycle` with initial UI `sid = old`.
2. Use a `GatewayClient` whose `rawRequest('initialize', ...)` is deferred for A; make `setup.status` resolve immediately for both calls.
3. Call `resumeById('A')`, wait until A's initialize request is observed, then call `resumeById('B')`.
4. Assert B rejects through the current-generation error path with `cannot initialize two resumed sessions concurrently` and visible `sid` remains `old`.
5. Resolve A successfully.
6. Assert visible `sid` is still `old`, then issue a keyless request such as `slash.exec` and assert it uses the transport/daemon active session A. This should fail until lifecycle switching and transport attachment are made atomic or serialized.

**Activation out-of-order race**

1. Seed client session-key mappings for live sessions A and B through `session.active_list`.
2. Defer `session.open` independently for `activateLiveSession('A')` and then `activateLiveSession('B')`.
3. Resolve B first and assert UI `sid === B`.
4. Resolve A second and assert UI remains B.
5. Execute `slash.exec`; assert the active daemon attachment is B, not A. Current code permits A's late `session.open`/client commit to win transport state.

These tests should assert both halves of the invariant, not merely that stale React continuations are discarded:

```text
visible session id === GatewayClient/daemon active session key
```

### Existing coverage assessment

`sessionLifecycle.test.ts` contains one test for overlapping **new-session setup** calls. It ensures only one `session.create` occurs and the newest visible state wins; it does not start overlapping `session.resume` or `session.activate` client operations and does not observe transport attachment.

`gatewayClientLifecycle.test.ts` verifies a single activation attaches before a following slash command and separately verifies ordinary resume behavior. It has no overlapping or out-of-order session-switch case.

**Disposition:** confirmed, critical. Existing passing tests do not exercise the failing interleaving.

## R001-TUI-02 — stale peek completion overrides later picker navigation/closure — **confirmed**

### Source evidence

The reported stale completion remains present in `SessionPicker`.

- `openPeek()` captures `rows[selected]`, sets `notice` to `loading preview…`, starts `session.peek`, and unconditionally calls `setPeek()` and `setNotice('')` on resolution (`xerxes/src/ui/opentui/sessionPicker.tsx`, `openPeek`). There is no generation/request id, mounted/open check, selected-row validation, or abort signal.
- Loading is represented only by notice text; `peek` remains `null`. Therefore the key handler still permits Up/Down/Home/End while the request is pending.
- Escape checks only `peek`. During loading, Escape takes the `close()` branch instead of invalidating the pending request. A later fulfillment still executes the stale state setters.

Deterministic navigation ordering:

1. With A highlighted, press Space and hold A's `session.peek` promise unresolved.
2. Press Down; because `peek` is still null, selection changes to B.
3. Resolve A's request.
4. The component enters preview state for `rowId: A` while the user's latest highlighted choice was B. Preview reply/steer uses `peek.rowId`, so a subsequent submission targets A.

This is not rejected by React merely because selection changed: the picker remains mounted and the closure intentionally retains A's row. Escape/close additionally leaves an unresolved callback capable of updating stale component state if the picker remains mounted through overlay hiding or is reopened before settlement, depending on parent mounting behavior.

### Precise regression designs

**Navigation during deferred peek (primary reproduction)**

1. Render `SessionPicker` with two live rows A and B and inject `Promise.withResolvers<SessionPeekResponse>()` for A's `session.peek`.
2. Focus A and press Space.
3. Assert the frame contains `loading preview…` and no preview text.
4. Press Down and verify B is highlighted (or press Right and verify B would be the attach target before resolving A).
5. Resolve A with a unique transcript marker, then flush.
6. Required behavior: A's marker must not render and the picker must remain on B's list row. Current behavior renders A's preview.
7. Type a unique draft and press Enter; additionally assert neither `session.steer` nor `prompt.submit` targets A. Current code targets `peek.rowId === A`.

**Competing peeks**

1. Start deferred peek A.
2. Navigate to B and start peek B (or, if loading is intentionally made modal, validate that the second request cannot start).
3. Resolve B, then A.
4. Assert only the newest request may control preview and notices. This catches out-of-order completion independently of row highlighting.

**Escape/close invalidation**

1. Start deferred peek A and press Escape before it resolves.
2. Resolve A after close; reopen the picker if the parent preserves the component.
3. Assert no A preview or cleared/new notice is restored and no state-update-after-unmount warning occurs. The request generation should be invalidated on Escape, refresh replacement, and unmount.

### Existing coverage assessment

`sessionPicker.test.tsx` covers a promptly resolved peek followed by steer and a promptly resolved idle peek followed by reply. Its mock resolves `session.peek` immediately. It has no deferred peek, navigation while loading, Escape while loading, competing request, or stale completion assertion.

**Disposition:** confirmed, high. Existing passing tests cover only the non-racing happy path.

## Verification

Fresh focused runs against the current worktree:

```text
bun test xerxes/src/ui/__tests__/sessionLifecycle.test.ts
1 pass, 0 fail, 3 expect() calls

bun test xerxes/src/ui/__tests__/sessionPicker.test.tsx
7 pass, 0 fail, 26 expect() calls

bun test xerxes/src/ui/__tests__/gatewayClientLifecycle.test.ts
29 pass, 0 fail, 83 expect() calls
```

The commands confirm the existing suites are green; inspection shows none contains the race interleavings needed to reject either report. Only this reproduction report was created.
