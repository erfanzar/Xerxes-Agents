// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface TranscriptScrollPort {
  y(): number
  height(): number
  viewport(): number
  top(): number
  blocks(): readonly { readonly id: string; readonly top: number }[]
  scrollTo(y: number): void
}

/** Document-scrolling chat: preserve a visible block, not a changing page height. */
export function createTranscriptScroll(port: TranscriptScrollPort) {
  let initial = true
  let pinned = true
  let lastY = port.y()
  let anchor: { readonly id: string; readonly top: number } | undefined
  let prependY: number | undefined
  let prependHeight = 0

  function remember(): void {
    anchor = port.blocks().find(block => block.top >= 40) ?? port.blocks().at(-1)
  }

  return {
    /** Only upward reader movement triggers older loading, not our own bottom scroll. */
    moved(): boolean {
      const y = port.y()
      const upward = y < lastY
      pinned = port.height() - y - port.viewport() <= 100
      lastY = y
      remember()
      return !initial && upward && (y <= 140 || (port.top() >= -140 && port.top() <= 140))
    },
    beforePrepend(): void {
      remember()
      prependY = port.y()
      prependHeight = port.height()
    },
    changed(): void {
      if (initial || (pinned && prependY === undefined)) {
        port.scrollTo(Math.max(0, port.height() - port.viewport()))
        initial = false
      } else {
        const found = anchor && port.blocks().find(block => block.id === anchor?.id)
        if (found && anchor) port.scrollTo(Math.max(0, port.y() + found.top - anchor.top))
        else if (prependY !== undefined) port.scrollTo(Math.max(0, prependY + port.height() - prependHeight))
      }
      prependY = undefined
      lastY = port.y()
      remember()
    },
  }
}
