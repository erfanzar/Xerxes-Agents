// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { buildToolTrailLine } from './text.js'
import { collapsedRunHeight, groupToolRun, TOOL_RUN_MIN } from './toolRun.js'

const ok = (name: string, arg: string, secs: number) => buildToolTrailLine(name, arg, false, '', secs)
const bad = (name: string, arg: string, note: string) => buildToolTrailLine(name, arg, true, note, 1)

describe('groupToolRun', () => {
  it('leaves a short run as individual rows', () => {
    const lines = [ok('read_file', 'a.ts', 0.1), ok('read_file', 'b.ts', 0.1)]

    expect(groupToolRun(lines).map(g => g.kind)).toEqual(['row', 'row'])
  })

  it('folds a long run of successes into one group', () => {
    const lines = Array.from({ length: TOOL_RUN_MIN }, (_, i) => ok('read_file', `f${i}.ts`, 0.1))
    const groups = groupToolRun(lines)

    expect(groups).toHaveLength(1)
    expect(groups[0]!.kind).toBe('run')
  })

  it('summarizes by verb, total time, and slowest call', () => {
    const lines = [
      ok('read_file', 'a.ts', 0.2),
      ok('read_file', 'b.ts', 0.1),
      ok('grep', 'retryBudget', 0.3),
      ok('exec', 'bun test ./test', 12.4)
    ]
    const group = groupToolRun(lines)[0]!

    if (group.kind !== 'run') { throw new Error('expected a folded run') }

    expect(group.summary.total).toBe(4)
    expect(group.summary.tally).toBe('read ×2 · grep ×1 · exec ×1')
    expect(group.summary.duration).toBeCloseTo(13, 1)
    expect(group.summary.slowest).toContain('bun test ./test')
    expect(group.summary.slowestDuration).toBeCloseTo(12.4, 1)
  })

  it('never folds a failure away', () => {
    const lines = [
      ok('read_file', 'a.ts', 0.1),
      ok('read_file', 'b.ts', 0.1),
      ok('read_file', 'c.ts', 0.1),
      ok('read_file', 'd.ts', 0.1),
      bad('exec', 'bun test', '3 tests failed'),
      ok('read_file', 'e.ts', 0.1)
    ]
    const groups = groupToolRun(lines)

    expect(groups.map(g => g.kind)).toEqual(['run', 'row', 'row'])
    expect(groups[1]).toMatchObject({ kind: 'row' })
    expect((groups[1] as { line: string }).line).toContain('✗')
  })

  it('never folds a call that is still in flight', () => {
    const lines = [
      ok('read_file', 'a.ts', 0.1),
      ok('read_file', 'b.ts', 0.1),
      ok('read_file', 'c.ts', 0.1),
      ok('read_file', 'd.ts', 0.1),
      'drafting a patch…'
    ]
    const groups = groupToolRun(lines)

    expect(groups.map(g => g.kind)).toEqual(['run', 'row'])
  })

  it('preserves order exactly — the sequence is the record', () => {
    const lines = [bad('exec', 'first', 'boom'), ok('read_file', 'a.ts', 0.1), bad('exec', 'last', 'boom')]
    const flat = groupToolRun(lines).map(g => (g.kind === 'row' ? g.line : '<run>'))

    expect(flat[0]).toContain('first')
    expect(flat[2]).toContain('last')
  })

  it('reports a collapsed height the estimator can trust', () => {
    const folded = groupToolRun(Array.from({ length: 6 }, (_, i) => ok('read_file', `f${i}.ts`, 0.2)))

    // Summary row plus the surviving slowest-call row.
    expect(collapsedRunHeight(folded[0]!)).toBe(2)
    const fleet = groupToolRun([
      ok('read_file', 'a.ts', 0.1),
      ok('spawn_agents', '2 agents: structure, security', 4),
      ok('task_list', '', 0.1),
      ok('read_file', 'b.ts', 0.1)
    ])
    // Summary + slowest + one row per visible fleet member.
    expect(collapsedRunHeight(fleet[0]!)).toBe(4)
    expect(collapsedRunHeight({ kind: 'row', line: 'x' })).toBe(1)
  })
})
