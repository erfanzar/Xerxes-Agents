// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface DraftPosition {
  cursor: number
  value: string
}

/**
 * Preserve edits made while an asynchronous clipboard read was in flight.
 * Paste resolvers return an insertion applied to the captured draft. When the
 * textarea has since changed, recover that insertion and apply it without
 * replacing the newer text.
 */
export function rebasePasteResult(
  captured: DraftPosition,
  resolved: DraftPosition,
  current: DraftPosition
): DraftPosition {
  if (current.value === captured.value) {
    return resolved
  }

  const cursor = Math.max(0, Math.min(captured.cursor, captured.value.length))
  const prefix = captured.value.slice(0, cursor)
  const suffix = captured.value.slice(cursor)
  const resolvedSuffixAt = resolved.value.length - suffix.length
  const isCapturedInsertion = resolved.value.startsWith(prefix) && resolvedSuffixAt >= prefix.length &&
    resolved.value.slice(resolvedSuffixAt) === suffix

  if (!isCapturedInsertion) {
    return resolved
  }

  const inserted = resolved.value.slice(prefix.length, resolvedSuffixAt)
  const currentSuffixAt = current.value.length - suffix.length
  const stillAtCapturedBoundary = current.value.startsWith(prefix) && currentSuffixAt >= prefix.length &&
    current.value.slice(currentSuffixAt) === suffix

  if (stillAtCapturedBoundary) {
    const middle = current.value.slice(prefix.length, currentSuffixAt)

    return {
      cursor: current.cursor >= prefix.length ? current.cursor + inserted.length : current.cursor,
      value: `${prefix}${inserted}${middle}${suffix}`
    }
  }

  const liveCursor = Math.max(0, Math.min(current.cursor, current.value.length))

  return {
    cursor: liveCursor + inserted.length,
    value: `${current.value.slice(0, liveCursor)}${inserted}${current.value.slice(liveCursor)}`
  }
}
