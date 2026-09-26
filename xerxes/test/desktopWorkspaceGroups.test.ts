// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { groupByWorkspace, sidebarOrder, workspaceName } from '../src/desktop/renderer/workspaceGroups.js'

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
