// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Why 400ms: the window must be wide enough that a deliberate double-tap —
 * two distinct key events at a natural human cadence — reliably lands inside
 * it, even on terminals that deliver keystrokes with noticeable latency, yet
 * narrow enough that two *unrelated* space presses (e.g. dismissing a pager
 * with space, then pressing space again a second later) are not mistaken for
 * the refocus gesture. 400ms sits comfortably inside the range users perceive
 * as "one action" while excluding incidental repeats.
 */
export const COMPOSER_DOUBLE_SPACE_MS = 400

/**
 * Structural slice of the composer textarea that the focus tracker needs.
 *
 * Kept deliberately minimal and structural (rather than importing the
 * renderer's textarea type) so this module stays renderer-agnostic and cheap
 * to fake in tests — anything with a `focused` flag and a `focus()` method
 * qualifies.
 */
export interface ComposerFocusTarget {
  readonly focused: boolean
  focus(): void
}

/**
 * Module-level singleton state: exactly one composer textarea exists per app
 * instance, so the tracker keeps a single registered target plus the
 * monotonic clock of the most recent unfocused space press (`0` = disarmed).
 */
let target: ComposerFocusTarget | null = null
let lastSpaceAt = 0

/**
 * Register the live composer textarea; pass null when it unmounts.
 *
 * The Composer wires this from a mount effect (see `opentui/appLayout.tsx`):
 * on mount it calls `registerComposerFocusTarget(ref.current)` so the global
 * key handler can reach the real textarea, and the effect cleanup calls
 * `registerComposerFocusTarget(null)` so the gesture degrades to a no-op
 * instead of acting on a detached component after unmount.
 *
 * The double-space clock is reset on every registration change: a stale
 * `lastSpaceAt` left over from a previous textarea instance must never arm
 * the gesture for the new one.
 */
export function registerComposerFocusTarget(next: ComposerFocusTarget | null): void {
  target = next
  lastSpaceAt = 0
}

export function isComposerFocused(): boolean {
  return target?.focused ?? false
}

/**
 * Focus the composer when it exists and is not already focused.
 *
 * The refusal when already focused matters: the gesture is a *return to the
 * prompt* affordance, never a focus steal. If focus is already where the user
 * wants it (or there is no composer at all), this is a strict no-op.
 */
export function focusComposer(): boolean {
  if (!target || target.focused) {
    return false
  }
  target.focus()
  return true
}

/** Reset the double-space clock (tests, session switches). */
export function resetComposerFocusTracking(): void {
  lastSpaceAt = 0
}

/**
 * Double-space gesture: when the composer is not focused (focus drifted to the
 * transcript, a selection, or a freshly dismissed overlay), two space presses
 * within {@link COMPOSER_DOUBLE_SPACE_MS} refocus it so the user can type again.
 * While the composer is focused the gesture stays inert and typed spaces pass
 * through untouched.
 *
 * Inert-while-focused design: when the textarea owns focus, every space is
 * ordinary text input, so the handler must not consume it. Returning false
 * lets the caller fall through and deliver the keystroke to the textarea.
 * The clock is also cleared on every focused press — otherwise the first
 * space typed after a subsequent blur could pair with a stale timestamp and
 * refocus from a single press.
 *
 * The gesture also never fires during modal overlays, by construction rather
 * than by any check here: the global input handler (`useInputHandlers.ts`)
 * consumes keys (or narrows them to scroll input) while an approval, clarify,
 * confirm, or pager overlay is up, and returns before ever reaching the
 * double-space branch. Only keys that fall all the way through that gate can
 * reach this function from the live key path.
 */
export function refocusComposerOnDoubleSpace(now: number = Date.now()): boolean {
  if (isComposerFocused()) {
    lastSpaceAt = 0
    return false
  }
  const isDouble = lastSpaceAt > 0 && now - lastSpaceAt <= COMPOSER_DOUBLE_SPACE_MS
  lastSpaceAt = now
  return isDouble ? focusComposer() : false
}
