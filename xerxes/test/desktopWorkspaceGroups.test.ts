// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { groupByWorkspace, workspaceName } from '../src/desktop/renderer/workspaceGroups.js'

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
