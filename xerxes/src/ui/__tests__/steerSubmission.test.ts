// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { steerWasAccepted } from '../app/useSubmission.js'

describe('steer submission acknowledgement', () => {
  it('accepts the native daemon ok response produced when Enter sends a steer', () => {
    expect(steerWasAccepted({ ok: true })).toBe(true)
  })

  it('accepts the legacy queued response and rejects explicit failures', () => {
    expect(steerWasAccepted({ status: 'queued' })).toBe(true)
    expect(steerWasAccepted({ ok: false, status: 'rejected' })).toBe(false)
    expect(steerWasAccepted(null)).toBe(false)
  })
})
