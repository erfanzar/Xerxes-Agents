// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Maximum gap between two space presses that still counts as one double-tap gesture. */
export const COMPOSER_DOUBLE_SPACE_MS = 400

/** Structural slice of the composer textarea that the focus tracker needs. */
export interface ComposerFocusTarget {
  readonly focused: boolean
  focus(): void
}

let target: ComposerFocusTarget | null = null
let lastSpaceAt = 0

/** Register the live composer textarea; pass null when it unmounts. */
export function registerComposerFocusTarget(next: ComposerFocusTarget | null): void {
  target = next
  lastSpaceAt = 0
}

export function isComposerFocused(): boolean {
  return target?.focused ?? false
}

/** Focus the composer when it exists and is not already focused. */
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
