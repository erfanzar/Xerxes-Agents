// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'bun:test'

import { statusView } from '../src/client/orbMapping.js'
import { checkImageFile, fitWithin } from '../src/client/imagePrep.js'
import { createDraftStore } from '../src/client/drafts.js'
import { inboxTasks, initialHubState, reduceHub } from '../src/client/hubStore.js'
import { EMPTY_TAIL, foldSessionEvent } from '../src/client/sessionFold.js'
import { NO_ACTIVITY } from '../src/client/taskTypes.js'
import { fetchDecision, safeTaskPath } from '../src/mobile/pwa/swCore.js'
import { parsePairFragment } from '../src/mobile/pairing.js'

describe('mobile client modules under Bun', () => {
  it('maps a working task with no evidence to "Working", never invented thinking', () => {
    const view = statusView({ connection: 'live', status: 'working', activity: NO_ACTIVITY, pendingCount: 0, stale: false })
    expect(view).toMatchObject({ kind: 'orb', label: 'Working' })
  })

  it('orders needs_you before working before idle', () => {
    let state = initialHubState()
    const base = { workspaceId: 'w', model: '', goalSummary: '', pendingCount: 0, interactionIds: [], agents: { total: 0, active: 0 }, stale: false, interrupted: false, activity: NO_ACTIVITY }
    state = reduceHub(state, {
      type: 'summaries',
      workspaceId: 'w',
      tasks: [
        { ...base, id: 'i', title: 'i', status: 'idle', updatedAt: '2026-10-21T12:00:00.000Z' },
        { ...base, id: 'n', title: 'n', status: 'needs_you', pendingCount: 1, updatedAt: '2026-10-21T10:00:00.000Z' },
        { ...base, id: 'w', title: 'w', status: 'working', updatedAt: '2026-10-21T11:00:00.000Z' },
      ],
    })
    expect(inboxTasks(state).map(task => task.id)).toEqual(['n', 'w', 'i'])
  })

  it('limits and resizes images', () => {
    expect(fitWithin(4000, 3000)).toEqual({ width: 1600, height: 1200 })
    expect(checkImageFile({ name: 'a.heic', type: 'image/heic', size: 10 }, 0).ok).toBe(false)
  })

  it('never lets the service worker touch /api/', () => {
    const shell = new Set(['/index.html'])
    expect(fetchDecision({ method: 'GET', url: 'https://h.example/api/hub/v1/ws', origin: 'https://h.example', mode: 'navigate', shell })).toBe('network')
    expect(safeTaskPath('https://evil.example/w/a/s/b', 'https://h.example')).toBe('/')
  })

  it('reads a pairing fragment', () => {
    expect(parsePairFragment('#pairing_id=p&code=c')).toEqual({ pairingId: 'p', code: 'c' })
  })

  it('keeps a tool call’s exact command, diff, todo list and duration under Bun', () => {
    let tail = foldSessionEvent(EMPTY_TAIL, { id: 'e1', type: 'tool_call', payload: { id: 'c1', name: 'exec_command', arguments: JSON.stringify({ cmd: 'bun', args: ['test'] }) } })
    tail = foldSessionEvent(tail, { id: 'e2', type: 'tool_result', payload: { tool_call_id: 'c1', result: 'ok', permitted: true, duration_ms: 7 } })
    tail = foldSessionEvent(tail, { id: 'e3', type: 'tool_call', payload: { id: 'c2', name: 'FileEditTool', arguments: { file_path: 'a.ts', old_string: 'x', new_string: 'y' } } })
    tail = foldSessionEvent(tail, { id: 'e4', type: 'tool_call', payload: { id: 'c3', name: 'TodoWriteTool', arguments: { todos: [{ content: 'Ship it', status: 'in_progress' }] } } })
    expect(tail.blocks[0]).toMatchObject({ kind: 'tool', args: 'bun test', output: 'ok', durationMs: 7, state: 'done' })
    expect(tail.blocks[1]).toMatchObject({ kind: 'tool', diff: { path: 'a.ts', before: 'x', after: 'y' } })
    expect(tail.todos).toEqual([{ content: 'Ship it', status: 'in_progress' }])
  })

  it('persists drafts through an injected storage', () => {
    const data = new Map<string, string>()
    const storage = { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) } }
    createDraftStore(50, { storage }).set('w1\u001fs1', 'half-typed')
    expect(createDraftStore(50, { storage }).get('w1\u001fs1')).toBe('half-typed')
  })
})
