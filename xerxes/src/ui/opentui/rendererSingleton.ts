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

const DEFAULT_RECOVERY_HEARTBEAT_MS = 1_000
const DEFAULT_WAKE_GAP_MS = 5_000

interface ForceRepaintRenderer {
  forceFullRepaintRequested: boolean
  requestRender: () => void
}

interface RecoverySignalSource {
  off: (event: 'SIGCONT', listener: () => void) => unknown
  on: (event: 'SIGCONT', listener: () => void) => unknown
}

type RecoveryTimer = ReturnType<typeof setInterval>

export interface RendererRecoveryOptions {
  clearInterval?: (timer: RecoveryTimer) => void
  heartbeatMs?: number
  now?: () => number
  setInterval?: (callback: () => void, milliseconds: number) => RecoveryTimer
  signalSource?: RecoverySignalSource
  wakeGapMs?: number
}

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
 * Invalidate OpenTUI's retained terminal buffer before scheduling a frame.
 *
 * `requestRender()` alone only emits changed cells. After a terminal loses
 * its backing surface (system sleep, focus restoration, or an out-of-band
 * terminal write), that retained buffer no longer describes what the user
 * sees and an ordinary diff leaves large blank/stale bands behind. OpenTUI
 * 0.4.x has the invalidation latch internally but does not expose a public
 * full-repaint method, so keep this pinned-version bridge in one place.
 */
export function forceRendererRepaint(renderer: CliRenderer | null = active): boolean {
  if (!renderer) {
    return false
  }

  const repaintable = renderer as unknown as ForceRepaintRenderer

  repaintable.forceFullRepaintRequested = true
  repaintable.requestRender()

  return true
}

/**
 * Recover the physical terminal after focus/resize/job-control changes and
 * after a sleeping laptop wakes without delivering any of those signals.
 * The heartbeat never repaints during normal operation; it only notices a
 * multi-second event-loop gap and invalidates the next frame once.
 */
export function installRendererRecovery(
  renderer: CliRenderer,
  options: RendererRecoveryOptions = {}
): () => void {
  const now = options.now ?? Date.now
  const heartbeatMs = Math.max(250, Math.floor(options.heartbeatMs ?? DEFAULT_RECOVERY_HEARTBEAT_MS))
  const wakeGapMs = Math.max(heartbeatMs * 2, Math.floor(options.wakeGapMs ?? DEFAULT_WAKE_GAP_MS))
  const schedule = options.setInterval ?? setInterval
  const cancel = options.clearInterval ?? clearInterval
  const signalSource = options.signalSource ?? (process as unknown as RecoverySignalSource)
  let lastHeartbeatAt = now()
  let cleaned = false

  const recover = () => {
    lastHeartbeatAt = now()
    forceRendererRepaint(renderer)
  }
  const heartbeat = () => {
    const current = now()
    const elapsed = current - lastHeartbeatAt

    lastHeartbeatAt = current

    if (elapsed >= wakeGapMs) {
      forceRendererRepaint(renderer)
    }
  }
  const timer = schedule(heartbeat, heartbeatMs)

  timer.unref?.()
  renderer.on('focus', recover)
  renderer.on('resize', recover)
  signalSource.on('SIGCONT', recover)

  const cleanup = () => {
    if (cleaned) {
      return
    }

    cleaned = true
    cancel(timer)
    renderer.off('focus', recover)
    renderer.off('resize', recover)
    renderer.off('destroy', cleanup)
    signalSource.off('SIGCONT', recover)
  }

  renderer.once('destroy', cleanup)

  return cleanup
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
