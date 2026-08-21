# Round 003 — Session picker stale peek fix

## Fix

Added a monotonic peek generation in `xerxes/src/ui/opentui/sessionPicker.tsx`. Every peek captures its generation and may update preview/notice state only while mounted and still current. Selection changes, picker close/unmount, refresh, and a subsequent peek invalidate older completions.

## Coverage

Added deferred-promise tests in `xerxes/src/ui/__tests__/sessionPicker.test.tsx` for stale completion after selection, Escape/close, refresh, and a second peek. The latest peek remains authoritative.

## Verification

- Focused Vitest: 1 file passed, 9 tests passed.
- `bun run check:ui`: passed.
- Scoped `git diff --check`: passed.
