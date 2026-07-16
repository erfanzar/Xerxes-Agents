// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it, vi } from 'vitest'

import type { CliRenderer } from '@opentui/core'

import { destroyActiveRenderer, setActiveRenderer } from '../opentui/rendererSingleton.js'
import { forceRedraw, runWithTerminalSuspended, useApp } from './terminalRuntime.opentui.js'

const renderer = () => ({
  requestRender: vi.fn(),
  resume: vi.fn(),
  suspend: vi.fn()
})

describe('runWithTerminalSuspended', () => {
  it('suspends and resumes only once across nested terminal handoffs', async () => {
    const target = renderer()

    await runWithTerminalSuspended(target, async () => {
      await runWithTerminalSuspended(target, async () => {})
    })

    expect(target.suspend).toHaveBeenCalledTimes(1)
    expect(target.resume).toHaveBeenCalledTimes(1)
    expect(target.requestRender).toHaveBeenCalledTimes(1)
  })

  it('restores and repaints the renderer when the external process fails', async () => {
    const target = renderer()

    await expect(
      runWithTerminalSuspended(target, async () => {
        throw new Error('editor failed')
      })
    ).rejects.toThrow('editor failed')

    expect(target.resume).toHaveBeenCalledTimes(1)
    expect(target.requestRender).toHaveBeenCalledTimes(1)
  })

  it('preserves the handoff failure if renderer restoration also fails', async () => {
    const target = renderer()

    target.resume.mockImplementationOnce(() => {
      throw new Error('resume failed')
    })

    await expect(
      runWithTerminalSuspended(target, async () => {
        throw new Error('editor failed')
      })
    ).rejects.toThrow('editor failed')
    expect(target.requestRender).toHaveBeenCalledTimes(1)
  })

  it('does not run the handoff or poison later calls when suspend fails', async () => {
    const broken = renderer()
    const run = vi.fn(async () => {})

    broken.suspend.mockImplementationOnce(() => {
      throw new Error('cannot suspend')
    })

    await expect(runWithTerminalSuspended(broken, run)).rejects.toThrow('cannot suspend')
    expect(run).not.toHaveBeenCalled()

    const healthy = renderer()

    await runWithTerminalSuspended(healthy, run)
    expect(healthy.suspend).toHaveBeenCalledTimes(1)
    expect(healthy.resume).toHaveBeenCalledTimes(1)
    expect(run).toHaveBeenCalledTimes(1)
  })
})

describe('forceRedraw', () => {
  it('requests a native full repaint instead of an incremental frame', () => {
    const target = {
      destroy: vi.fn(),
      forceFullRepaintRequested: false,
      requestRender: vi.fn()
    }

    setActiveRenderer(target as unknown as CliRenderer)

    expect(forceRedraw()).toBe(true)
    expect(target.forceFullRepaintRequested).toBe(true)
    expect(target.requestRender).toHaveBeenCalledOnce()

    destroyActiveRenderer(() => true)
  })
})

describe('OpenTUI app exit', () => {
  it('tears down the active renderer before exiting the process', () => {
    const events: string[] = []
    const exit = vi.spyOn(process, 'exit').mockImplementation(code => {
      events.push(`exit:${code}`)

      return undefined as never
    })

    setActiveRenderer({
      destroy: () => events.push('destroy')
    } as unknown as CliRenderer)

    try {
      useApp().exit(new Error('fatal'))
      expect(events).toEqual(['destroy', 'exit:1'])
    } finally {
      exit.mockRestore()
    }
  })
})
