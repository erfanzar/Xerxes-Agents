// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Holds the active CliRenderer so imperative (non-hook) call sites in the
// controller — forceRedraw() from a slash command handler and
// withTerminalSuspended() during an $EDITOR shell-out — can reach it.
// entry.tsx sets it right after createCliRenderer(). Keeping this in a tiny
// module avoids pulling the React/OpenTUI hook surface into plain logic.
import type { CliRenderer } from '@opentui/core'

import { resetTerminalModes } from '../lib/terminalModes.js'

let active: CliRenderer | null = null
let completedTeardown: ActiveRendererTeardownResult | null = null

export interface ActiveRendererTeardownResult {
  destroyError: unknown | null
  hadRenderer: boolean
  rendererDestroyed: boolean
  terminalReset: boolean
}

export function setActiveRenderer(renderer: CliRenderer): void {
  active = renderer
  completedTeardown = null
}

export function getActiveRenderer(): CliRenderer | null {
  return active
}

/**
 * Forget a renderer that OpenTUI destroyed on its own. Supplying the expected
 * instance prevents an old renderer's delayed destroy event from clearing a
 * newer renderer.
 */
export function clearActiveRenderer(renderer?: CliRenderer): boolean {
  if (renderer && active !== renderer) {
    return false
  }

  const hadRenderer = active !== null

  active = null
  completedTeardown = null

  return hadRenderer
}

/**
 * Tear down the process-wide renderer exactly once, then reset terminal
 * protocols. Clearing the singleton before destroy() makes this safe against
 * re-entrant OpenTUI destroy events and racing signal/exit handlers.
 *
 * Renderer cleanup is best-effort: a native teardown failure must never skip
 * the terminal reset or prevent the caller from exiting.
 */
export function destroyActiveRenderer(
  resetModes: () => boolean = () => resetTerminalModes()
): ActiveRendererTeardownResult {
  if (completedTeardown) {
    return completedTeardown
  }

  const renderer = active
  let destroyError: unknown | null = null
  let rendererDestroyed = false
  let terminalReset = false

  active = null

  try {
    if (renderer) {
      try {
        renderer.destroy()
        rendererDestroyed = true
      } catch (error) {
        destroyError = error
      }
    }
  } finally {
    try {
      terminalReset = resetModes()
    } catch {
      terminalReset = false
    }
  }

  completedTeardown = {
    destroyError,
    hadRenderer: renderer !== null,
    rendererDestroyed,
    terminalReset
  }

  return completedTeardown
}
