// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it, vi } from 'vitest'

import type { CliRenderer } from '@opentui/core'

import {
  clearActiveRenderer,
  destroyActiveRenderer,
  getActiveRenderer,
  setActiveRenderer
} from './rendererSingleton.js'

const mockRenderer = (destroy: () => void): CliRenderer =>
  ({ destroy }) as unknown as CliRenderer

describe('active OpenTUI renderer lifecycle', () => {
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
