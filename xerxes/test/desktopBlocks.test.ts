// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { BlockBuilder, blocksFromStoredMessages } from '../src/desktop/renderer/blocks.js'

// Pure fold tests: wire events in, display blocks out. No sockets, no React.

describe('BlockBuilder', () => {
  test('text deltas accumulate into one agent block', () => {
    const b = new BlockBuilder()
    b.push('text_part', { text: 'hel' })
    b.push('text_part', { text: 'lo' })
    b.finalize()
    const blocks = [...b.all()]
    expect(blocks).toHaveLength(1)
    expect(blocks[0]).toMatchObject({ kind: 'agent', text: 'hello', streaming: false })
  })

  test('tool rows retain exact names, arguments, and outputs for expansion', () => {
    const b = new BlockBuilder()
    b.push('tool_call', {
      tool_call_id: 'call-1',
      name: 'exec_command',
      arguments: '{"command":"ls","workdir":"/tmp"}',
    })
    b.push('tool_result', {
      tool_call_id: 'call-1',
      name: 'exec_command',
      return_value: { files: ['README.md'] },
      duration_ms: 150,
    })

    b.finalize()
    const block = [...b.all()][0]
    expect(block).toMatchObject({
      kind: 'tools',
      items: [{
        arg: 'ls',
        input: '{\n  "command": "ls",\n  "workdir": "/tmp"\n}',
        name: 'exec_command',
        output: '{\n  "files": [\n    "README.md"\n  ]\n}',
        state: 'done',
      }],
    })
  })

  test('consecutive tool calls collapse into a single trail', () => {
    const b = new BlockBuilder()
    b.push('tool_call', { id: 't1', name: 'ext.fs.read', arguments: '{"path":"a.ts"}' })
    b.push('tool_call', { id: 't2', name: 'grep', arguments: '{"pattern":"x"}' })
    b.push('tool_result', { tool_call_id: 't1', duration_ms: 400 })
    b.push('tool_result', { tool_call_id: 't2', duration_ms: 15_000 })
    b.finalize()
    const blocks = [...b.all()]
    expect(blocks).toHaveLength(1)
    const trail = blocks[0]!
    expect(trail.kind).toBe('tools')
    if (trail.kind !== 'tools') return
    expect(trail.items.map(i => i.verb)).toEqual(['read', 'grep'])
    expect(trail.items.map(i => i.dur)).toEqual(['0.4s', '15s'])
    expect(trail.running).toBe(false)
  })

  test('a working tool still running at turn end is closed out honestly', () => {
    const b = new BlockBuilder()
    b.push('tool_call', { id: 't1', name: 'bash', arguments: '{"cmd":"ls"}' })
    b.finalize()
    const trail = [...b.all()][0]!
    expect(trail.kind).toBe('tools')
    if (trail.kind !== 'tools') return
    expect(trail.items[0]!.state).toBe('done')
  })

  test('streaming snapshot interleaves scratch state without mutating the fold', () => {
    const b = new BlockBuilder()
    b.pushUser('map the repo')
    b.push('tool_call', { id: 't1', name: 'read', arguments: '{"p":"x"}' })
    b.push('think_part', { think: 'analyzing' })
    b.push('text_part', { text: 'start' })

    const live = b.snapshot(true)
    expect(live.map(b2 => b2.kind)).toEqual(['user', 'tools', 'thinking', 'agent'])

    b.finalize()
    const done = [...b.all()]
    // Arrival order: the tool call preceded the thinking buffer.
    expect(done.map(b2 => b2.kind)).toEqual(['user', 'tools', 'thinking', 'agent'])
    // Scratch is drained exactly once: a second finalize adds nothing.
    b.finalize()
    expect(b.all().length).toBe(done.length)
  })

  test('error notifications become error notices', () => {
    const b = new BlockBuilder()
    b.push('notification', { severity: 'error', body: 'quota exhausted' })
    b.push('notification', { level: 'info', message: 'resumed session' })
    const blocks = [...b.all()]
    expect(blocks).toEqual([
      expect.objectContaining({ kind: 'notice', error: true, text: 'quota exhausted' }),
      expect.objectContaining({ kind: 'notice', error: false, text: 'resumed session' }),
    ])
  })
})

describe('blocksFromStoredMessages', () => {
  test('role/content pairs hydrate in order; system rows drop; parts fold sequentially', () => {
    const blocks = blocksFromStoredMessages([
      { role: 'user', content: 'fix the loop' },
      { role: 'system', content: 'hidden' },
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'on it' },
          { type: 'think', think: 'plan' },
        ],
      },
    ])
    // The live grammar survives replay: the think part opens its own block.
    expect(blocks.map(b => b.kind)).toEqual(['user', 'agent', 'thinking'])
    expect(blocks[1]).toMatchObject({ kind: 'agent', text: 'on it' })
    expect(blocks[2]).toMatchObject({ kind: 'thinking', text: 'plan' })
  })

  test('stored tool messages replay as tool rows from tool_executions, not thinking', () => {
    const blocks = blocksFromStoredMessages(
      [
        { role: 'user', content: 'inspect changes' },
        { role: 'assistant', content: 'running the diff' },
        { role: 'tool', content: ' file | +12 -4' },
        { role: 'tool', content: 'M src/loop.ts' },
        { role: 'assistant', content: 'large change set' },
      ],
      {
        executions: [
          { name: 'exec_command', inputs: { args: ['diff', '--stat'], cmd: 'git' }, toolCallId: 'c1', durationMs: 420 },
          { name: 'exec_command', inputs: { args: ['status', '--porcelain'], cmd: 'git' }, toolCallId: 'c2', durationMs: 180 },
        ],
        thinking: ['think about the diff'],
      },
    )
    expect(blocks.map(b => b.kind)).toEqual(['user', 'thinking', 'agent', 'tools', 'agent'])
    const trail = blocks[3]
    if (trail?.kind !== 'tools') return
    expect(trail.items.map(item => item.verb)).toEqual(['exec_command', 'exec_command'])
    expect(trail.items.map(item => item.dur)).toEqual(['0.4s', '0.2s'])
    expect(trail.items[0]?.arg).toBe('git')
  })

  test('structured tool_use parts in assistant content replay as call+result rows', () => {
    const blocks = blocksFromStoredMessages([
      { role: 'user', content: 'read a.ts' },
      {
        role: 'assistant',
        content: [
          { type: 'tool_use', id: 'u1', name: 'read', input: { path: 'a.ts' } },
          { type: 'tool_result', tool_use_id: 'u1', content: 'const x = 1' },
          { type: 'text', text: 'it is a constant' },
        ],
      },
    ])
    expect(blocks.map(b => b.kind)).toEqual(['user', 'tools', 'agent'])
    if (blocks[1]?.kind !== 'tools') return
    expect(blocks[1].items[0]).toMatchObject({ verb: 'read', arg: 'a.ts', state: 'done' })
  })

  test('alternating think/tools/text streams open new blocks instead of appending to the first', () => {
    const b = new BlockBuilder()
    // think → tool, tool → think → tool, tool, tool → think → tool → think → reply
    b.push('think_part', { think: 'need the files' })
    b.push('tool_call', { id: 't1', name: 'ls', arguments: '{"cmd":"ls src"}' })
    b.push('tool_call', { id: 't2', name: 'grep', arguments: '{"pattern":"x"}' })
    b.push('tool_result', { tool_call_id: 't1', duration_ms: 100 })
    b.push('tool_result', { tool_call_id: 't2', duration_ms: 200 })
    b.push('think_part', { think: 'found it, checking more' })
    b.push('tool_call', { id: 't3', name: 'read', arguments: '{"path":"a.ts"}' })
    b.push('tool_call', { id: 't4', name: 'read', arguments: '{"path":"b.ts"}' })
    b.push('tool_call', { id: 't5', name: 'edit', arguments: '{"file_path":"a.ts","old_string":"x","new_string":"y"}' })
    b.push('tool_result', { tool_call_id: 't3', duration_ms: 100 })
    b.push('tool_result', { tool_call_id: 't4', duration_ms: 100 })
    b.push('tool_result', { tool_call_id: 't5', duration_ms: 300 })
    b.push('think_part', { think: 'one more check' })
    b.push('tool_call', { id: 't6', name: 'bash', arguments: '{"cmd":"bun test"}' })
    b.push('tool_result', { tool_call_id: 't6', duration_ms: 900 })
    b.push('think_part', { think: 'writing the answer' })
    b.push('text_part', { text: 'here is the summary' })
    b.finalize()

    expect(b.all().map(block => block.kind)).toEqual([
      'thinking', // think 1
      'tools', // tool 1, tool 2
      'thinking', // think 2
      'tools', // tool 3, tool 4, tool 5
      'thinking', // think 3
      'tools', // tool 6
      'thinking', // think 4
      'agent', // response
    ])
    const firstTrail = b.all()[1]
    if (firstTrail?.kind !== 'tools') return
    expect(firstTrail.items.map(item => item.id)).toEqual(['t1', 't2'])
    const secondTrail = b.all()[3]
    if (secondTrail?.kind !== 'tools') return
    expect(secondTrail.items.map(item => item.id)).toEqual(['t3', 't4', 't5'])
    expect(b.all()[0]).toMatchObject({ kind: 'thinking', text: 'need the files', streaming: false })
    expect(b.all()[2]).toMatchObject({ kind: 'thinking', text: 'found it, checking more' })
  })

  test('mid-turn snapshots render runs in stream order, caret on the trailing run only', () => {
    const b = new BlockBuilder()
    b.push('think_part', { think: 'thinking first' })
    b.push('tool_call', { id: 't1', name: 'ls', arguments: '{"cmd":"ls"}' })
    b.push('tool_result', { tool_call_id: 't1', duration_ms: 50 })
    b.push('think_part', { think: 'thinking again' })
    b.push('text_part', { text: 'partial answer' })

    const live = [...b.snapshot(true)]
    expect(live.map(block => block.kind)).toEqual(['thinking', 'tools', 'thinking', 'agent'])
    // Only the trailing run still receives deltas, so only it may stream —
    // every run carrying a caret kept the "|" blinking on text the model had
    // already finished.
    expect(live[0]).toMatchObject({ kind: 'thinking', text: 'thinking first', streaming: false })
    expect(live[2]).toMatchObject({ kind: 'thinking', text: 'thinking again', streaming: false })
    expect(live[3]).toMatchObject({ kind: 'agent', text: 'partial answer', streaming: true })

    b.finalize()
    expect(b.all().map(block => block.kind)).toEqual(['thinking', 'tools', 'thinking', 'agent'])
  })

  test('a still-working tool keeps its live marker when a later run takes the tail', () => {
    const b = new BlockBuilder()
    b.push('tool_call', { id: 't1', name: 'bash', arguments: '{"cmd":"bun test"}' })
    b.push('text_part', { text: 'while that runs…' })

    const live = [...b.snapshot(true)]
    const trail = live[0]!
    expect(trail.kind).toBe('tools')
    if (trail.kind !== 'tools') return
    // The caret moved on to the text run; the tool row must still read live.
    expect(trail.running).toBe(true)
    expect(live[1]).toMatchObject({ kind: 'agent', streaming: true })

    b.finalize()
    const done = [...b.all()][0]!
    expect(done.kind).toBe('tools')
    if (done.kind !== 'tools') return
    // Turn end closes the unanswered call honestly: not live, not failed.
    expect(done.running).toBe(false)
    expect(done.items[0]!.state).toBe('done')
  })

  test('the agents card trails live runs, updates in place, and resets per turn', () => {
    const b = new BlockBuilder()
    b.pushUser('map the repo')
    b.push('tool_call', { id: 't1', name: 'SpawnAgents', arguments: '{"agents":[]}' })
    b.pushAgents([
      { key: 't1:0', title: 'Map entry points', status: 'working' },
      { key: 't1:1', title: 'Map hot paths', status: 'working' },
    ])

    // Mid-turn: the card trails the live tool run.
    const live = [...b.snapshot(true)]
    expect(live.map(block => block.kind)).toEqual(['user', 'tools', 'agents'])
    const card = live[2]!
    expect(card.kind === 'agents' && card.members).toHaveLength(2)

    // In-place status update (the daemon snapshots landed).
    b.pushAgents([
      { key: 't1:0', title: 'Map entry points', status: 'completed' },
      { key: 't1:1', title: 'Map hot paths', status: 'working' },
    ])
    const updated = [...b.snapshot(true)][2]!
    expect(updated.kind === 'agents' && updated.members[0]?.status).toBe('completed')

    // Finalize commits the card behind the drained runs; a post-turn
    // terminal sweep still finds it in place.
    b.finalize()
    const done = [...b.all()]
    expect(done.map(block => block.kind)).toEqual(['user', 'tools', 'agents'])
    b.pushAgents([
      { key: 't1:0', title: 'Map entry points', status: 'completed' },
      { key: 't1:1', title: 'Map hot paths', status: 'completed' },
    ])
    const settled = [...b.all()][2]!
    expect(settled.kind === 'agents' && settled.members.every(m => m.status === 'completed')).toBe(true)

    // The next turn's spawn opens a NEW card — the old one is history.
    b.closeAgentsCard()
    b.pushAgents([{ key: 't2:0', title: 'Next batch', status: 'working' }])
    const cards = b.all().filter(block => block.kind === 'agents')
    expect(cards).toHaveLength(2)
  })
})
