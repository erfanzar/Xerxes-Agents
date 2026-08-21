# Bug Bounty Round 002/020 — Runtime Reproduction

Date: 2026-08-21  
Scope: validate Round 1 cancellation-misclassification and concurrent-shutdown early-return findings.  
Method: source inspection plus isolated Bun repros; no production or test source was edited.

## Result

Both Round 1 findings reproduce on the current worktree.

| Finding | Verdict | Severity assessment |
| --- | --- | --- |
| `AbortError` is unconditionally labeled a user interrupt | Confirmed | Medium: destroys error provenance and can suppress recovery/retry decisions |
| concurrent `shutdown()` callers return before the first shutdown completes | Confirmed in both runtime managers | Medium: callers can proceed while runners/resources remain active |

## R002-1: cancellation misclassification

### Source evidence

`ErrorClassifier.classify()` branches only on the error name:

- `xerxes/src/runtime/errorClassifier.ts:76-81` obtains error details and maps every `AbortError` or `InterruptedError` to `{ kind: fatal, message: "user interrupt" }`.
- The classifier receives no `AbortSignal` or cancellation-origin context, so it cannot distinguish a caller/user abort from a provider, transport, timeout wrapper, or other internally generated `AbortError`.

### Reproduction

The isolated repro classified both:

1. a normal `DOMException(..., "AbortError")`; and
2. an internally described `Error` whose name was `AbortError` and message was `provider transport aborted its own request`.

Observed output for **both** inputs:

```json
{
  "kind": "fatal",
  "message": "user interrupt",
  "retryable": false
}
```

The second input's original message survives only inside `original`; the public classification message falsely asserts user action. This validates the misclassification, not merely a cosmetic message issue: `fatal` is non-retryable (`errorClassifier.ts:36-41,135-146`).

### Coverage gap

`xerxes/test/runtimeResilience.test.ts` exercises status codes, patterns, retry hints, and connection failures, but has no assertion that cancellation origin is preserved or that an uncorrelated `AbortError` is not called a user interrupt.

## R002-2: concurrent shutdown early return

### Source evidence

The same state-flag pattern exists in two owners:

- `BackgroundSessionManager.shutdown()` at `xerxes/src/runtime/backgroundSessions.ts:221-235` immediately returns at line 222 when `shuttingDown` is already true. Only the first caller reaches the active-runner wait at lines 232-235.
- `ExecutionRegistry.shutdown()` at `xerxes/src/runtime/executionRegistry.ts:423-439` immediately returns at line 424 when `shuttingDown` is already true. Only the first caller reaches the active-runner wait at lines 434-439.

Neither class stores a shared shutdown promise. Therefore, the boolean prevents duplicate setup but does not make concurrent callers join the in-progress shutdown.

### Reproduction

For each class, an injected runner was held on an unresolved promise. The first `shutdown({ timeoutMs: 1000 })` was started, then a second shutdown was awaited before releasing the runner.

Observed:

```json
{"label":"BackgroundSessionManager","secondMs":0.01,"firstStillPending":true,"runningCount":1}
{"label":"ExecutionRegistry","secondMs":0.004,"firstStillPending":true,"runningCount":1}
```

Thus each second caller resolved effectively immediately while:

- the first shutdown promise was still pending; and
- the manager still reported one physically active runner.

This violates the methods' documented wait semantics (`backgroundSessions.ts:214-219`; `executionRegistry.ts:419-421`) for concurrent callers and creates an observable cleanup race.

### Coverage gap

Existing shutdown tests are single-caller only:

- `xerxes/test/backgroundSessions.test.ts` checks rejection of post-shutdown submissions after completed work.
- `xerxes/test/executionRegistry.test.ts` checks pending cancellation and a zero-timeout shutdown.

Neither asserts that two concurrent shutdown calls settle together or that the second remains pending while cleanup is active.

## Verification run

```text
bun test xerxes/test/runtimeResilience.test.ts
5 pass, 0 fail, 42 expect() calls

bun test xerxes/test/backgroundSessions.test.ts xerxes/test/executionRegistry.test.ts
9 pass, 0 fail, 60 expect() calls
```

The passing suites establish current baseline behavior; they do not invalidate either reproduced finding because the concurrency and cancellation-origin cases are absent.

## Suggested regression tests (not added)

1. Classifier: provide an `AbortError` without a correlated aborted caller signal and assert it is not represented as `user interrupt`; separately test a correlated user-abort path.
2. Each shutdown owner: hold one active runner, invoke shutdown twice, assert the second promise remains pending while the first remains pending, release the runner, then assert both promises settle and `runningCount === 0`.

## Files changed

- Added this report only: `bug-bounty/agents/round-002/runtime-repro.md`.
