// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import type { CliRenderer } from '@opentui/core'

import {
  clearActiveRenderer,
  destroyActiveRenderer,
  forceRendererRepaint,
  getActiveRenderer,
  installRendererRecovery,
  setActiveRenderer
} from './rendererSingleton.js'

const mockRenderer = (destroy: () => void): CliRenderer =>
  ({ destroy }) as unknown as CliRenderer

describe('active OpenTUI renderer lifecycle', () => {
  it('invalidates the retained native buffer before requesting a full repaint', () => {
    const renderer = {
      forceFullRepaintRequested: false,
      requestRender: vi.fn(() => {
        expect(renderer.forceFullRepaintRequested).toBe(true)
      })
    }

    expect(forceRendererRepaint(renderer as unknown as CliRenderer)).toBe(true)
    expect(renderer.requestRender).toHaveBeenCalledOnce()
  })

  it('recovers on focus, resize, SIGCONT, and a delayed wake heartbeat', () => {
    const events = new EventEmitter()
    const signals = new EventEmitter()
    const requestRender = vi.fn()
    const renderer = Object.assign(events, {
      forceFullRepaintRequested: false,
      requestRender
    }) as unknown as CliRenderer
    let now = 0
    let heartbeat = () => {}
    const timer = { unref: vi.fn() } as unknown as ReturnType<typeof setInterval>
    const clearTimer = vi.fn()
    const cleanup = installRendererRecovery(renderer, {
      clearInterval: clearTimer,
      heartbeatMs: 1_000,
      now: () => now,
      setInterval: callback => {
        heartbeat = callback

        return timer
      },
      signalSource: signals,
      wakeGapMs: 5_000
    })

    events.emit('focus')
    events.emit('resize', 120, 40)
    signals.emit('SIGCONT')
    expect(requestRender).toHaveBeenCalledTimes(3)

    now = 6_000
    heartbeat()
    expect(requestRender).toHaveBeenCalledTimes(4)

    now = 7_000
    heartbeat()
    expect(requestRender).toHaveBeenCalledTimes(4)
    expect(timer.unref).toHaveBeenCalledOnce()

    cleanup()
    events.emit('focus')
    signals.emit('SIGCONT')
    heartbeat()
    expect(requestRender).toHaveBeenCalledTimes(4)
    expect(clearTimer).toHaveBeenCalledWith(timer)
  })

  it('destroys before resetting the terminal and is idempotent across exit paths', () => {
    const events: string[] = []
    const renderer = mockRenderer(() => events.push('destroy'))

    setActiveRenderer(renderer)

    const first = destroyActiveRenderer(() => {
      events.push('reset')
      return true
    })
    const second = destroyActiveRenderer(() => {
      events.push('second-reset')
      return true
    })

    expect(first).toEqual({
      destroyError: null,
      hadRenderer: true,
      rendererDestroyed: true,
      terminalReset: true
    })
    expect(second).toBe(first)
    expect(events).toEqual(['destroy', 'reset'])
    expect(getActiveRenderer()).toBeNull()
  })

  it('still resets and clears the singleton when renderer destruction throws', () => {
    const failure = new Error('native teardown failed')
    const resetModes = vi.fn(() => true)

    setActiveRenderer(
      mockRenderer(() => {
        throw failure
      })
    )

    expect(destroyActiveRenderer(resetModes)).toEqual({
      destroyError: failure,
      hadRenderer: true,
      rendererDestroyed: false,
      terminalReset: true
    })
    expect(resetModes).toHaveBeenCalledOnce()
    expect(getActiveRenderer()).toBeNull()
  })

  it('does not let a stale destroy event clear a newer renderer', () => {
    const oldRenderer = mockRenderer(() => {})
    const currentRenderer = mockRenderer(() => {})

    setActiveRenderer(oldRenderer)
    setActiveRenderer(currentRenderer)

    expect(clearActiveRenderer(oldRenderer)).toBe(false)
    expect(getActiveRenderer()).toBe(currentRenderer)
    expect(clearActiveRenderer(currentRenderer)).toBe(true)
    expect(getActiveRenderer()).toBeNull()
  })
})
