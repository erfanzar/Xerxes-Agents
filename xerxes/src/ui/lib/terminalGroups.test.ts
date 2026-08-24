// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { orderTerminals, terminalGroup, terminalHeading } from './terminalGroups.js'
import type { TerminalSummary } from './terminals.js'

const shell = (over: Partial<TerminalSummary>): TerminalSummary => ({
  canInterrupt: false,
  canKill: false,
  canWrite: false,
  command: 'bun test',
  cwd: '/repo',
  exitCode: 0,
  id: Math.random().toString(36).slice(2),
  kind: 'background',
  label: 'bun test',
  outputChars: 0,
  running: false,
  startedAt: 0,
  ...over
})

describe('terminalGroup', () => {
  it('classifies by what you can still act on', () => {
    expect(terminalGroup(shell({ running: true }))).toBe('running')
    expect(terminalGroup(shell({ exitCode: 0 }))).toBe('succeeded')
    expect(terminalGroup(shell({ exitCode: 2 }))).toBe('failed')
  })

  it('separates a shell waiting on YOU from one the agent is waiting on', () => {
    // A live PTY that accepts input is, by construction, waiting for someone
    // to type into it — and a root shell in /etc nobody remembers opening is
    // exactly the case that separation exists to catch.
    expect(terminalGroup(shell({ canWrite: true, kind: 'pty', running: true }))).toBe('interactive')
    // A background command is alive too, but nobody has to do anything.
    expect(terminalGroup(shell({ kind: 'background', running: true }))).toBe('running')
    // A PTY the daemon will not let you write to is not waiting on you.
    expect(terminalGroup(shell({ canWrite: false, kind: 'pty', running: true }))).toBe('running')
    // …and once it exits it is judged by its exit code like anything else.
    expect(terminalGroup(shell({ canWrite: true, exitCode: 0, kind: 'pty', running: false }))).toBe('succeeded')
  })

  it('orders running, failed, then the shells waiting on a human, then the dead', () => {
    const ordered = orderTerminals([
      shell({ exitCode: 0, label: 'done' }),
      shell({ canWrite: true, kind: 'pty', label: 'zsh', running: true }),
      shell({ exitCode: 137, label: 'oom' }),
      shell({ label: 'dev', running: true })
    ])

    expect(ordered.map(entry => entry.label)).toEqual(['dev', 'oom', 'zsh', 'done'])
  })

  it('treats a settled shell with no exit code as failed', () => {
    // Killed, or it never reported — closer to a failure than a success, and
    // silently filing it under "succeeded" would hide it in the fold.
    expect(terminalGroup(shell({ exitCode: null, running: false }))).toBe('failed')
  })
})

describe('orderTerminals', () => {
  it('puts running first, then failures, then successes', () => {
    const ordered = orderTerminals([
      shell({ endedAt: 90, exitCode: 0, label: 'ok-old' }),
      shell({ exitCode: 1, endedAt: 50, label: 'boom' }),
      shell({ label: 'live', running: true, startedAt: 10 }),
      shell({ endedAt: 99, exitCode: 0, label: 'ok-new' })
    ])

    expect(ordered.map(e => e.label)).toEqual(['live', 'boom', 'ok-new', 'ok-old'])
  })

  it('orders by recency inside a group', () => {
    const ordered = orderTerminals([
      shell({ endedAt: 10, exitCode: 1, label: 'older' }),
      shell({ endedAt: 80, exitCode: 1, label: 'newer' })
    ])

    expect(ordered.map(e => e.label)).toEqual(['newer', 'older'])
  })

  it('does not mutate the array it was given', () => {
    const input = [shell({ exitCode: 0, label: 'a' }), shell({ label: 'b', running: true })]
    orderTerminals(input)

    expect(input.map(e => e.label)).toEqual(['a', 'b'])
  })
})

describe('terminalHeading', () => {
  it('labels the first row of each group with its size', () => {
    const rows = orderTerminals([
      shell({ label: 'live', running: true }),
      shell({ exitCode: 1, label: 'boom' }),
      shell({ exitCode: 0, label: 'ok1' }),
      shell({ exitCode: 0, label: 'ok2' })
    ])

    expect(terminalHeading(rows, 0)).toBe('RUNNING · 1')
    expect(terminalHeading(rows, 1)).toBe('FAILED · 1')
    expect(terminalHeading(rows, 2)).toBe('SUCCEEDED · 2')
    expect(terminalHeading(rows, 3)).toBe('')
  })
})
