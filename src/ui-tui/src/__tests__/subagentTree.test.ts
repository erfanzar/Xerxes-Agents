// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { emptySubagents, listSubagents, reduceSubagentEvent, subagentSummary } from '../lib/subagentTree.js'
import { initialState, reduce } from '../app/gatewayState.js'

const sub = (agentId: string, type: string, event: { type: string; payload: Record<string, unknown> }) => ({
  agent_id: agentId,
  subagent_type: type,
  parent_tool_call_id: 'tc1',
  event
})

describe('reduceSubagentEvent', () => {
  it('creates a node and accumulates streamed text', () => {
    let s = emptySubagents
    s = reduceSubagentEvent(s, sub('a1', 'researcher', { type: 'text_part', payload: { text: 'look' } }))
    s = reduceSubagentEvent(s, sub('a1', 'researcher', { type: 'text_part', payload: { text: 'ing' } }))
    const node = s.byId.a1!
    expect(node.subagentType).toBe('researcher')
    expect(node.text).toBe('looking')
    expect(node.active).toBe(true)
  })

  it('tracks tool calls and turn completion', () => {
    let s = emptySubagents
    s = reduceSubagentEvent(s, sub('a1', 'coder', { type: 'tool_call', payload: { name: 'Edit' } }))
    s = reduceSubagentEvent(s, sub('a1', 'coder', { type: 'tool_call', payload: { name: 'Bash' } }))
    s = reduceSubagentEvent(s, sub('a1', 'coder', { type: 'turn_end', payload: {} }))
    expect(s.byId.a1!.tools).toEqual(['Edit', 'Bash'])
    expect(s.byId.a1!.active).toBe(false)
  })

  it('keeps multiple subagents in insertion order', () => {
    let s = emptySubagents
    s = reduceSubagentEvent(s, sub('a1', 't1', { type: 'turn_begin', payload: {} }))
    s = reduceSubagentEvent(s, sub('a2', 't2', { type: 'turn_begin', payload: {} }))
    expect(listSubagents(s).map(n => n.agentId)).toEqual(['a1', 'a2'])
  })

  it('accepts PascalCase nested events', () => {
    const s = reduceSubagentEvent(emptySubagents, sub('a1', 'x', { type: 'TextPart', payload: { text: 'hi' } }))
    expect(s.byId.a1!.text).toBe('hi')
  })

  it('falls back to parent_tool_call_id when agent_id is missing', () => {
    const s = reduceSubagentEvent(emptySubagents, {
      parent_tool_call_id: 'tcX',
      subagent_type: 'x',
      event: { type: 'turn_begin', payload: {} }
    })
    expect(s.order).toEqual(['tcX'])
  })
})

describe('subagentSummary', () => {
  it('prefers latest text, then last tool, then status', () => {
    expect(
      subagentSummary({
        agentId: 'a',
        subagentType: 't',
        parentToolCallId: '',
        text: 'line1\nline2',
        tools: [],
        active: true
      })
    ).toBe('line2')
    expect(
      subagentSummary({
        agentId: 'a',
        subagentType: 't',
        parentToolCallId: '',
        text: '',
        tools: ['Grep'],
        active: true
      })
    ).toBe('⚙ Grep')
    expect(
      subagentSummary({ agentId: 'a', subagentType: 't', parentToolCallId: '', text: '', tools: [], active: false })
    ).toBe('done')
  })
})

describe('subagent_event reducer integration', () => {
  it('feeds the gatewayState subagents tree and resets on a new turn', () => {
    let s = reduce(initialState, {
      type: 'subagent_event',
      payload: sub('a1', 'researcher', { type: 'text_part', payload: { text: 'hi' } })
    })
    expect(listSubagents(s.subagents)).toHaveLength(1)
    s = reduce(s, { type: 'turn_begin', payload: {} })
    expect(listSubagents(s.subagents)).toHaveLength(0)
  })
})
