// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import type { WorkspaceContext } from '../src/desktop/main/contextNavigation.js'
import { COLLAPSED_SECTIONS_KEY, groupByWorkspace, hostOfScope, limitRows, loadCollapsedSections, saveCollapsedSections, sidebarOrder, sidebarSections, workspaceName } from '../src/desktop/renderer/workspaceGroups.js'

describe('workspace grouping', () => {
  test('workspace display name is the folder basename', () => {
    expect(workspaceName('/Users/erfan/Documents/Projects/Xerxes-Agents')).toBe('Xerxes-Agents')
    expect(workspaceName('/home/dev/EasyDeL/')).toBe('EasyDeL')
    expect(workspaceName('/')).toBe('')
    expect(workspaceName('')).toBe('')
    expect(workspaceName(undefined)).toBe('')
  })

  test('sessions group under their folder, first-seen order preserved', () => {
    const groups = groupByWorkspace([
      { id: 'a', cwd: '/repo/Xerxes-Agents' },
      { id: 'b', cwd: '/code/EasyDeL' },
      { id: 'c', cwd: '/repo/Xerxes-Agents' },
      { id: 'd', cwd: '' },
    ])
    expect(groups.map(g => g.name)).toEqual(['Xerxes-Agents', 'EasyDeL', 'Other'])
    expect(groups[0]!.rows.map(r => r.id)).toEqual(['a', 'c'])
    expect(groups[1]!.rows.map(r => r.id)).toEqual(['b'])
    expect(groups[2]!.rows.map(r => r.id)).toEqual(['d'])
  })

  test('recency order of the incoming list survives grouping', () => {
    const groups = groupByWorkspace([
      { id: 'newest', cwd: '/w/B' },
      { id: 'older', cwd: '/w/A' },
      { id: 'older2', cwd: '/w/A' },
    ])
    expect(groups.map(g => g.name)).toEqual(['B', 'A'])
    expect(groups[1]!.rows.map(r => r.id)).toEqual(['older', 'older2'])
  })

  test('the workspace the user is in leads the list', () => {
    const groups = groupByWorkspace([
      { id: 'a', cwd: '/code/EasyDeL' },
      { id: 'b', cwd: '/repo/Xerxes-Agents' },
      { id: 'c', cwd: '/w/B' },
    ], '/repo/Xerxes-Agents')
    expect(groups.map(g => g.name)).toEqual(['Xerxes-Agents', 'EasyDeL', 'B'])
    // An unknown current workspace leaves recency order untouched.
    expect(groupByWorkspace([{ id: 'x', cwd: '/w/A' }], '/nowhere').map(g => g.name)).toEqual(['A'])
  })
})

test('empty saved folders remain reachable and identical basenames stay separate', () => {
  const groups = groupByWorkspace([{id:'a',cwd:'/first/repo'},{id:'b',cwd:'/second/repo'}], '/second/repo', ['/empty', '/first/repo'])
  expect(groups.map(g => g.cwd)).toEqual(['/second/repo','/first/repo','/empty'])
  expect(groups[0]!.rows.map(r=>r.id)).toEqual(['b'])
  expect(groups[2]!.rows).toEqual([])
})

describe('sidebar order', () => {
  // Working in EasyDeL, clicking a chat in eyvan used to send eyvan to the
  // bottom: the open chat left its place and was appended last.
  const rows = [
    { id: 'e1', cwd: '/p/EasyDeL', activeAt: 3000 },
    { id: 'y1', cwd: '/p/eyvan', activeAt: 2000 },
    { id: 'e2', cwd: '/p/EasyDeL', activeAt: 1000 },
  ]
  const layout = (list: ReadonlyArray<{ id: string; cwd: string; activeAt?: number }>, activity: Record<string, number> = {}) =>
    groupByWorkspace(sidebarOrder(list, activity), '').map(group => `${group.name}:${group.rows.map(row => row.id).join(',')}`)

  test('opening a chat moves nothing: the open row is placed by its message time', () => {
    const before = layout(rows)
    // The store drops the open chat from its lists and the sidebar re-adds
    // it last, with no time of its own — its time comes from sessionActivity.
    const opened = [rows[0]!, rows[2]!, { id: 'y1', cwd: '/p/eyvan' }]
    expect(layout(opened, { y1: 2000 })).toEqual(before)
    expect(before).toEqual(['EasyDeL:e1,e2', 'eyvan:y1'])
  })

  test('a new message moves its chat (and folder) to the top', () => {
    expect(layout(rows, { y1: 4000 })).toEqual(['eyvan:y1', 'EasyDeL:e1,e2'])
  })

  test('a chat that has not spoken yet leads the list', () => {
    expect(sidebarOrder([...rows, { id: 'fresh', cwd: '/p/eyvan' }]).map(row => row.id)[0]).toBe('fresh')
  })
})

describe('sidebar sections', () => {
  const local: WorkspaceContext = { id: 1, label: 'This Mac', workspace: '/p/Xerxes-Agents', remote: false, sessions: [
    { id: 'x1', cwd: '/p/Xerxes-Agents', title: 'X one', status: 'idle' },
    { id: 'y1', cwd: '/p/eyvan', title: 'Y one', status: 'working' },
    { id: 'x2', cwd: '/p/Xerxes-Agents', title: 'X two', status: 'idle' },
  ] }
  const tpu: WorkspaceContext = { id: -1, label: 'v5p · prismcore', alias: 'v5p', workspace: '/home/e/prismcore', workspacePath: '/home/e/prismcore', remote: true, saved: true, sessions: [
    { id: 'r1', cwd: '/home/e/prismcore', title: 'Remote', status: 'idle' },
  ] }

  test('the host of a view comes from its storage scope', () => {
    expect(hostOfScope('')).toBe('This Mac')
    expect(hostOfScope(undefined)).toBe('This Mac')
    expect(hostOfScope('ssh:v5p:')).toBe('v5p')
  })

  test('a local context seen from an SSH view is split per folder, never one "This Mac" heap', () => {
    const here = groupByWorkspace([{ id: 'r1', cwd: '/home/e/prismcore' }], '')
    const sections = sidebarSections(here, [local], 'v5p')
    expect(sections.map(section => section.label)).toEqual(['This Mac · eyvan', 'This Mac · Xerxes-Agents', 'v5p · prismcore'])
    expect(sections.map(section => section.rows.map(row => row.id).join(','))).toEqual(['y1', 'x1,x2', 'r1'])
  })

  test('the order is the same whichever view is open', () => {
    const fromLocal = sidebarSections(groupByWorkspace([
      { id: 'x1', cwd: '/p/Xerxes-Agents' }, { id: 'y1', cwd: '/p/eyvan' }, { id: 'x2', cwd: '/p/Xerxes-Agents' },
    ], ''), [tpu], 'This Mac')
    const fromRemote = sidebarSections(groupByWorkspace([{ id: 'r1', cwd: '/home/e/prismcore' }], ''), [local], 'v5p')
    expect(fromLocal.map(section => section.label)).toEqual(fromRemote.map(section => section.label))
    expect(fromLocal.map(section => section.kind)).toEqual(['here', 'here', 'context'])
  })

  test('a section lists its last five tasks; expanding lists all; the open task is never hidden', () => {
    const rows = Array.from({ length: 8 }, (_, i) => ({ id: `t${i}`, cwd: '/p' }))
    expect(limitRows(rows, 5, false)).toEqual({ shown: rows.slice(0, 5), hidden: 3 })
    expect(limitRows(rows, 5, true)).toEqual({ shown: rows, hidden: 0 })
    const withOpen = limitRows(rows, 5, false, 't7')
    expect(withOpen.shown.map(row => row.id)).toEqual(['t0', 't1', 't2', 't3', 't4', 't7'])
    expect(withOpen.hidden).toBe(2)
    expect(limitRows(rows.slice(0, 3), 5, false)).toEqual({ shown: rows.slice(0, 3), hidden: 0 })
  })

  test('folded sections are remembered, and bad stored data is ignored', () => {
    const values = new Map<string, string>()
    const storage = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value) } }
    saveCollapsedSections(new Set(['This Mac\u0000/p/eyvan']), storage)
    expect([...loadCollapsedSections(storage)]).toEqual(['This Mac\u0000/p/eyvan'])
    values.set(COLLAPSED_SECTIONS_KEY, '{"not":"a list"}')
    expect(loadCollapsedSections(storage).size).toBe(0)
    values.set(COLLAPSED_SECTIONS_KEY, 'not json')
    expect(loadCollapsedSections(storage).size).toBe(0)
  })

  test('a task listed by this view is not repeated under a context', () => {
    const here = groupByWorkspace([{ id: 'x1', cwd: '/p/Xerxes-Agents' }], '')
    const sections = sidebarSections(here, [local], 'This Mac')
    expect(sections.flatMap(section => section.rows.map(row => row.id)).sort()).toEqual(['x1', 'x2', 'y1'])
  })
})
