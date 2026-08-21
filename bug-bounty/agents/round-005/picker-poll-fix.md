# Session picker pending-peek polling fix

## Bug

`SessionPicker.refresh()` invalidated every in-flight `session.peek` request. Because refresh runs on a 1.5-second polling interval, a valid slow preview response was discarded even when the highlighted session was unchanged and still present.

## Fix

- Removed unconditional peek invalidation from periodic/manual refresh.
- Track the row ID associated with the pending peek request.
- Preserve that request while refreshed rows still contain the same selected/pending session.
- Continue invalidating stale responses when:
  - the selection actually changes;
  - the selected or pending session disappears during refresh;
  - the picker closes or unmounts;
  - a newer peek starts.
- Clear pending tracking when the current peek resolves or fails.

## Tests

Updated `xerxes/src/ui/__tests__/sessionPicker.test.tsx` to verify:

- a deferred peek survives a polling refresh and renders;
- a deferred peek is rejected when its session disappears during polling;
- only the newest deferred peek applies;
- selection changes and picker close still invalidate deferred peeks.

## Verification

- `bun test xerxes/src/ui/__tests__/sessionPicker.test.tsx` — 10 passed, 0 failed (React emitted existing async `act(...)` warnings during the close/selection invalidation test).
- `bun run --cwd xerxes check:ui` — passed.
- `git diff --check -- xerxes/src/ui/opentui/sessionPicker.tsx xerxes/src/ui/__tests__/sessionPicker.test.tsx` — passed.
