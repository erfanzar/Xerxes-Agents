// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it, vi } from 'vitest'

import {
  createTerminalWatchdog,
  TERMINAL_MODE_RESET,
  type SttyRunner,
  withTerminalWatchdog
} from './terminalModes.js'

const success = (stdout = '') => ({ exitCode: 0, stdout })

describe('terminal mode reset', () => {
  it('covers the mouse, alternate-screen, paste, focus, and keyboard protocols', () => {
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1003l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1006l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1049l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?2004l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1004l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?2026l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[<u')
  })
})

describe('parent terminal watchdog', () => {
  it('captures and restores the exact opaque termios state before resetting escape modes', () => {
    const events: string[] = []
    const runStty: SttyRunner = arguments_ => {
      events.push(`stty:${arguments_.join(' ')}`)

      return arguments_[0] === '-g' ? success('gfmt1:abc123\n') : success()
    }
    const watchdog = createTerminalWatchdog({
      stdinIsTTY: true,
      platform: 'darwin',
      runStty,
      resetModes: () => {
        events.push('reset')
        return true
      }
    })

    expect(watchdog.snapshot).toBe('gfmt1:abc123')
    expect(watchdog.restore()).toEqual({
      exactStateRestored: true,
      fallbackStateRestored: false,
      modesReset: true
    })
    expect(events).toEqual(['stty:-g', 'stty:gfmt1:abc123', 'reset'])
  })

  it('restores a sane terminal when exact restoration fails and emits escape reset last', () => {
    const events: string[] = []
    const runStty: SttyRunner = arguments_ => {
      events.push(`stty:${arguments_.join(' ')}`)

      if (arguments_[0] === '-g') return success('saved-state\n')
      if (arguments_[0] === 'saved-state') return { exitCode: 1, stdout: '' }
      return success()
    }
    const watchdog = createTerminalWatchdog({
      stdinIsTTY: true,
      platform: 'linux',
      runStty,
      resetModes: () => {
        events.push('reset')
        return true
      }
    })

    expect(watchdog.restore()).toEqual({
      exactStateRestored: false,
      fallbackStateRestored: true,
      modesReset: true
    })
    expect(events).toEqual(['stty:-g', 'stty:saved-state', 'stty:sane', 'reset'])
  })

  it('still disables protocols when exact termios restoration throws', () => {
    const events: string[] = []
    const runStty: SttyRunner = arguments_ => {
      events.push(`stty:${arguments_.join(' ')}`)

      if (arguments_[0] === '-g') return success('captured')
      if (arguments_[0] === 'captured') throw new Error('controlling tty disappeared')
      return success()
    }
    const watchdog = createTerminalWatchdog({
      stdinIsTTY: true,
      platform: 'darwin',
      runStty,
      resetModes: () => {
        events.push('reset')
        return true
      }
    })

    expect(watchdog.restore()).toEqual({
      exactStateRestored: false,
      fallbackStateRestored: true,
      modesReset: true
    })
    expect(events.at(-1)).toBe('reset')
  })

  it('uses escape cleanup without invoking stty on unsupported or non-TTY hosts', () => {
    const runStty = vi.fn<SttyRunner>(() => success())
    const resetModes = vi.fn(() => true)
    const watchdog = createTerminalWatchdog({
      stdinIsTTY: true,
      platform: 'win32',
      runStty,
      resetModes
    })

    expect(watchdog.snapshot).toBeNull()
    expect(watchdog.restore()).toEqual({
      exactStateRestored: false,
      fallbackStateRestored: false,
      modesReset: true
    })
    expect(runStty).not.toHaveBeenCalled()
    expect(resetModes).toHaveBeenCalledOnce()
  })

  it('restores only once when signal and child-exit paths race', () => {
    const runStty = vi.fn<SttyRunner>(arguments_ =>
      arguments_[0] === '-g' ? success('captured') : success()
    )
    const resetModes = vi.fn(() => true)
    const watchdog = createTerminalWatchdog({
      stdinIsTTY: true,
      platform: 'linux',
      runStty,
      resetModes
    })

    const first = watchdog.restore()
    const second = watchdog.restore()

    expect(second).toBe(first)
    expect(runStty).toHaveBeenCalledTimes(2)
    expect(resetModes).toHaveBeenCalledOnce()
  })

  it('restores after an abnormal child wait rejects without replacing that failure', async () => {
    const events: string[] = []
    const runStty: SttyRunner = arguments_ => {
      events.push(`stty:${arguments_.join(' ')}`)

      return arguments_[0] === '-g' ? success('captured') : success()
    }

    await expect(
      withTerminalWatchdog(
        async () => {
          events.push('child-died')
          throw new Error('SIGTRAP')
        },
        {
          stdinIsTTY: true,
          platform: 'darwin',
          runStty,
          resetModes: () => {
            events.push('reset')
            return true
          }
        }
      )
    ).rejects.toThrow('SIGTRAP')
    expect(events).toEqual(['stty:-g', 'child-died', 'stty:captured', 'reset'])
  })
})
