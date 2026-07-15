// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  ApprovalChoice,
  ApprovalCountdown,
  ApprovalPanelState,
  DEFAULT_APPROVAL_OPTIONS,
  PanelSelection
} from './panelState.js'

describe('PanelSelection', () => {
  it('wraps through options in both directions and accepts absolute indices', () => {
    const selection = new PanelSelection(['a', 'b', 'c'])

    expect(selection.down()).toBe(1)
    expect(selection.down()).toBe(2)
    expect(selection.down()).toBe(0)
    expect(selection.up()).toBe(2)
    expect(selection.set(5)).toBe(2)
    expect(selection.set(-1)).toBe(2)
    expect(selection.selected()).toBe('c')
  })

  it('keeps an empty panel inert', () => {
    const selection = new PanelSelection<string>([])

    expect(selection.down()).toBe(0)
    expect(selection.set(7)).toBe(0)
    expect(selection.selected()).toBe('')
  })
})

describe('ApprovalPanelState', () => {
  it('uses the full five-choice approval layout', () => {
    const state = new ApprovalPanelState()

    expect(DEFAULT_APPROVAL_OPTIONS).toHaveLength(5)
    expect(DEFAULT_APPROVAL_OPTIONS).toContain(ApprovalChoice.APPROVE_ONCE)
    expect(DEFAULT_APPROVAL_OPTIONS).toContain(ApprovalChoice.APPROVE_ALWAYS)
    expect(state.current()).toBe(ApprovalChoice.APPROVE)
    expect(state.down()).toBe(ApprovalChoice.APPROVE_ONCE)
  })

  it('fails clearly if a caller creates an approval panel with no choices', () => {
    expect(() => new ApprovalPanelState([]).current()).toThrow('Approval panel has no choices')
  })
})

describe('ApprovalCountdown', () => {
  afterEach(() => vi.useRealTimers())

  it('fires once at its deadline and reports elapsed/remaining time', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-01-01T00:00:00.000Z'))
    const timeout = vi.fn()
    const countdown = new ApprovalCountdown(0.05)

    countdown.start(timeout)

    expect(countdown.isActive()).toBe(true)
    expect(countdown.remaining()).toBe(0.05)
    vi.advanceTimersByTime(25)
    expect(countdown.elapsed()).toBe(0.025)
    expect(countdown.remaining()).toBe(0.025)
    vi.advanceTimersByTime(25)
    expect(timeout).toHaveBeenCalledOnce()
    expect(countdown.isActive()).toBe(false)
    expect(countdown.remaining()).toBe(0)
  })

  it('replaces and cancels in-flight callbacks safely', () => {
    vi.useFakeTimers()
    const first = vi.fn()
    const second = vi.fn()
    const countdown = new ApprovalCountdown(1)

    countdown.start(first)
    countdown.start(second)
    vi.advanceTimersByTime(1_000)

    expect(first).not.toHaveBeenCalled()
    expect(second).toHaveBeenCalledOnce()

    countdown.start(first)
    countdown.cancel()
    vi.advanceTimersByTime(1_000)

    expect(first).not.toHaveBeenCalled()
    expect(countdown.isActive()).toBe(false)
  })
})
