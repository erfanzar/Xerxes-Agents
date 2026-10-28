// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { vi } from 'vitest'

import { tierDownloadBytes } from '../../client/voiceModelManifest.js'
import type { CallSnapshot } from '../voice/call/engine.js'
import { VoiceCallService, type CallEngineControl } from '../voice/call/service.js'

export function voiceServiceFixture() {
  let state: CallSnapshot = { phase: 'idle', tier: 'standard', muted: false, heard: '', reply: '' }
  const observers = new Set<() => void>()
  const emit = (patch: Partial<CallSnapshot>): void => { state = { ...state, ...patch }; for (const observer of observers) observer() }
  const engine: CallEngineControl = {
    getSnapshot: () => state,
    subscribe: observer => { observers.add(observer); return () => { observers.delete(observer) } },
    start: vi.fn(async tier => { emit({ phase: 'listening', tier }) }),
    hangup: vi.fn(() => emit({ phase: 'idle' })),
    mute: vi.fn(muted => emit({ muted })),
    interrupt: vi.fn(() => emit({ phase: 'listening' })),
    visibility: vi.fn(),
  }
  const createEngine = vi.fn(() => engine)
  const modelsStatus = vi.fn(async () => ({ standard: { bytes: tierDownloadBytes('standard'), ready: true }, light: { bytes: tierDownloadBytes('light'), ready: true } }))
  const beforeStart = vi.fn()
  const service = new VoiceCallService({ engine: createEngine, modelsStatus, beforeStart, canStart: () => true })
  return { service, engine, emit, createEngine, modelsStatus, beforeStart }
}
