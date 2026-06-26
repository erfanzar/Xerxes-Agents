// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { initialState, reduce, type UiState, type WireEvt } from '../app/gatewayState.js'

function run(events: WireEvt[], from: UiState = initialState): UiState {
  return events.reduce(reduce, from)
}

describe('gatewayState.reduce', () => {
  it('init_done populates the session', () => {
    const s = run([
      {
        type: 'init_done',
        payload: {
          model: 'claude-code/opus',
          agent_name: 'Xerxes',
          git_branch: 'main',
          context_limit: 1_000_000,
          session_id: 'abc'
        }
      }
    ])
    expect(s.connected).toBe(true)
    expect(s.session.model).toBe('claude-code/opus')
    expect(s.session.gitBranch).toBe('main')
    expect(s.session.contextLimit).toBe(1_000_000)
  })

  it('status_update updates the status rule fields', () => {
    const s = run([
      {
        type: 'status_update',
        payload: { context_tokens: 4200, max_context: 200000, mode: 'plan', plan_mode: true, reasoning_effort: 'high' }
      }
    ])
    expect(s.status.contextTokens).toBe(4200)
    expect(s.status.planMode).toBe(true)
    expect(s.status.reasoningEffort).toBe('high')
  })

  it('accepts PascalCase aliases (bridge transport)', () => {
    const s = run([{ type: 'StatusUpdate', payload: { mode: 'code' } }])
    expect(s.status.mode).toBe('code')
  })

  it('streams text_part then flushes to an assistant row on turn_end', () => {
    const s = run([
      { type: 'turn_begin', payload: { user_input: 'hi' } },
      { type: 'text_part', payload: { text: 'Hel' } },
      { type: 'text_part', payload: { text: 'lo.' } },
      { type: 'turn_end', payload: {} }
    ])
    expect(s.busy).toBe(false)
    expect(s.streaming).toBe('')
    expect(s.transcript).toHaveLength(1)
    expect(s.transcript[0]).toMatchObject({ role: 'assistant', text: 'Hello.' })
  })

  it('turn_begin sets busy and clears stale stream buffers', () => {
    const mid = run([
      { type: 'turn_begin', payload: {} },
      { type: 'text_part', payload: { text: 'partial' } }
    ])
    expect(mid.busy).toBe(true)
    expect(mid.streaming).toBe('partial')
    const next = reduce(mid, { type: 'turn_begin', payload: {} })
    expect(next.streaming).toBe('')
  })

  it('tool_call adds a tool row', () => {
    const s = run([{ type: 'tool_call', payload: { id: 't1', name: 'exec_command', arguments: '{"cmd":"ls"}' } }])
    expect(s.transcript[0]).toMatchObject({ role: 'tool' })
    expect(s.transcript[0]!.text).toContain('exec_command')
  })

  it('__user synthetic adds an optimistic user row', () => {
    const s = run([{ type: '__user', payload: { text: 'do the thing' } }])
    expect(s.transcript[0]).toMatchObject({ role: 'user', text: 'do the thing' })
  })

  it('gateway.ready / gateway.closed toggle connection', () => {
    const open = reduce(initialState, { type: 'gateway.ready', payload: {} })
    expect(open.connected).toBe(true)
    const closed = reduce({ ...open, busy: true }, { type: 'gateway.closed', payload: {} })
    expect(closed.connected).toBe(false)
    expect(closed.busy).toBe(false)
  })

  it('think_part accumulates separately from streaming', () => {
    const s = run([
      { type: 'think_part', payload: { think: 'reason ' } },
      { type: 'think_part', payload: { think: 'more' } }
    ])
    expect(s.thinking).toBe('reason more')
    expect(s.streaming).toBe('')
  })

  it('ignores unknown events without mutating state', () => {
    const s = reduce(initialState, { type: 'mcp_loading_begin', payload: { server_name: 'x' } })
    expect(s).toEqual(initialState)
  })
})
