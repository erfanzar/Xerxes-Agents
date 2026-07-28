// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { busyInputLabels } from '../domain/busyInputLabels.js'

describe('composer labels while a turn is running', () => {
  it('promises steering in the default mode rather than queueing', () => {
    // The composer said "Queue a follow-up…" and "Enter queue" in every mode
    // while the default is steer, so pressing Enter injected the message into the
    // running turn and the interface reported it had been held for later. Typing
    // something to redirect the agent and being told it was queued is
    // indistinguishable from steering being broken.
    const labels = busyInputLabels('steer')
    expect(labels.enter).toBe('steer')
    expect(labels.placeholder).toContain('Steer')
    expect(labels.placeholder).not.toContain('Queue')
  })

  it('still says queue when queueing is what is configured', () => {
    const labels = busyInputLabels('queue')
    expect(labels.enter).toBe('queue')
    expect(labels.placeholder).toContain('Queue a follow-up')
  })

  it('says interrupt in interrupt mode', () => {
    expect(busyInputLabels('interrupt').enter).toBe('interrupt')
    expect(busyInputLabels('interrupt').placeholder).toContain('Interrupt')
  })

  it('describes what Escape actually does, which depends on whether text is held', () => {
    expect(busyInputLabels('steer', 0).escape).toBe('interrupt')
    expect(busyInputLabels('steer', 2).escape).toBe('clear queue')
    expect(busyInputLabels('queue', 1).escape).toBe('clear queue')
  })
})
