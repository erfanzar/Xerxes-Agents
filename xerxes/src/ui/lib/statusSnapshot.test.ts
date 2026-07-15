// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import {
  compactStatusNumber,
  createStatusSnapshot,
  formatStatus,
  formatStatusDuration,
  isYoloEnabled,
  statusCacheHitRate,
  statusContextPercent,
  statusContextUsed,
  statusSnapshotRecord
} from './statusSnapshot.js'

describe('status snapshots', () => {
  it('supplies complete defaults with zero derived utilization', () => {
    const snapshot = createStatusSnapshot()

    expect(statusContextUsed(snapshot)).toBe(0)
    expect(statusContextPercent(snapshot)).toBe(0)
    expect(statusCacheHitRate(snapshot)).toBe(0)
    expect(isYoloEnabled()).toBe(true)
    expect(isYoloEnabled(snapshot.permissionMode)).toBe(true)
    expect(isYoloEnabled('auto')).toBe(false)
  })

  it('includes cache reads in context and cache hit calculations', () => {
    const snapshot = createStatusSnapshot({ cacheReadTokens: 80_000, contextWindow: 200_000, inputTokens: 20_000 })

    expect(statusContextUsed(snapshot)).toBe(100_000)
    expect(statusContextPercent(snapshot)).toBe(50)
    expect(statusCacheHitRate(snapshot)).toBe(0.8)
    expect(statusSnapshotRecord(snapshot)).toMatchObject({
      cacheHitRate: 0.8,
      contextPercent: 50,
      contextUsed: 100_000
    })
  })

  it('caps context percentage and handles an unavailable context window', () => {
    expect(statusContextPercent(createStatusSnapshot({ contextWindow: 100, inputTokens: 200 }))).toBe(100)
    expect(statusContextPercent(createStatusSnapshot({ contextWindow: 0, inputTokens: 200 }))).toBe(0)
  })
})

describe('formatStatus', () => {
  it('renders model, token totals, cost, context, and duration', () => {
    const line = formatStatus(
      createStatusSnapshot({
        contextWindow: 200_000,
        costUsd: 0.05,
        durationSeconds: 42,
        inputTokens: 1_000,
        model: 'claude-opus-4-7',
        outputTokens: 500
      })
    )

    expect(line).toContain('claude-opus-4-7')
    expect(line).toContain('1.0Kin/500out')
    expect(line).toContain('$0.0500')
    expect(line).toContain('00:42')
    expect(line).toContain('YOLO ON')
  })

  it('includes cache, skill, queue, and non-auto permission extras', () => {
    const line = formatStatus(
      createStatusSnapshot({
        activeSkill: 'plan',
        cacheReadTokens: 500,
        cacheWriteTokens: 10,
        model: 'm',
        permissionMode: 'manual',
        queueDepth: 2
      })
    )

    expect(line).toContain('/500c/10cw')
    expect(line).toContain('skill=plan')
    expect(line).toContain('queued=2')
    expect(line).toContain('manual')
  })

  it('uses stable compact count and duration formats', () => {
    expect(compactStatusNumber(1_250)).toBe('1.3K')
    expect(compactStatusNumber(1_250_000)).toBe('1.3M')
    expect(formatStatusDuration(-3)).toBe('00:00')
    expect(formatStatusDuration(3_661.9)).toBe('61:01')
  })
})
