// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
import { describe, expect, it } from 'vitest'

import { agentGroup, agentHeading, orderAgentRecords } from './agentGroups.js'
import type { SubagentStatus } from '../types.js'

const rec = (status: SubagentStatus, startedAt = 0, index = 0) => ({ item: { index, startedAt, status } })

describe('agentGroup', () => {
  it('treats queued work as working — it is still ahead of you', () => {
    expect(agentGroup('running')).toBe('working')
    expect(agentGroup('queued')).toBe('working')
  })

  it('separates completion from every other way a run can end', () => {
    expect(agentGroup('completed')).toBe('review')

    // A run that broke is over; a run you stopped is a decision you still owe.
    for (const status of ['failed', 'error', 'timeout'] as SubagentStatus[]) {
      expect(agentGroup(status)).toBe('failed')
    }

    expect(agentGroup('interrupted')).toBe('input')
  })
})

describe('orderAgentRecords', () => {
  it('orders by action: unblock, monitor, review, then the dead', () => {
    const ordered = orderAgentRecords([
      rec('completed', 50), rec('failed', 40), rec('running', 10), rec('queued', 20), rec('interrupted', 30)
    ])

    expect(ordered.map(r => r.item.status)).toEqual(['interrupted', 'queued', 'running', 'completed', 'failed'])
  })

  it('falls back to spawn index when a batch shares a timestamp', () => {
    const ordered = orderAgentRecords([rec('failed', 7, 2), rec('failed', 7, 1)])

    expect(ordered.map(r => r.item.index)).toEqual([1, 2])
  })

  it('does not mutate its input', () => {
    const input = [rec('completed', 1), rec('running', 2)]
    orderAgentRecords(input)

    expect(input.map(r => r.item.status)).toEqual(['completed', 'running'])
  })
})

describe('agentHeading', () => {
  it('labels each group once, with its size, in action order', () => {
    const rows = orderAgentRecords([
      rec('completed'), rec('failed'), rec('completed'), rec('running'), rec('interrupted')
    ])

    expect(agentHeading(rows, 0)).toBe('NEEDS INPUT · 1')
    expect(agentHeading(rows, 1)).toBe('WORKING · 1')
    expect(agentHeading(rows, 2)).toBe('READY TO REVIEW · 2')
    expect(agentHeading(rows, 3)).toBe('')
    expect(agentHeading(rows, 4)).toBe('FAILED · 1')
  })
})
