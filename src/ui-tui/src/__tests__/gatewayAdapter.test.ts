// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { adaptDaemonEvent, sessionInfoFromInit, transcriptFromStoredMessages } from '../gatewayAdapter.js'

describe('gatewayAdapter', () => {
  it('maps init payloads into session info', () => {
    const info = sessionInfoFromInit({
      agent_name: 'xerxes',
      context_tokens: 12,
      cwd: '/repo',
      head_hash: 'abc123',
      max_context: 200,
      mode: 'researcher',
      model: 'claude-code/opus',
      session_id: 'sid123',
      skills: ['deep-scan'],
      skill_descriptions: { 'deep-scan': 'scan deeply' },
      version: '0.2.6'
    })

    expect(info).toMatchObject({
      cwd: '/repo',
      head_hash: 'abc123',
      mode: 'researcher',
      model: 'claude-code/opus',
      profile_name: 'xerxes',
      session_id: 'sid123',
      version: '0.2.6'
    })
    expect(info.usage.context_used).toBe(12)
    expect(info.usage.context_max).toBe(200)
    expect(info.skills).toEqual({ skills: ['deep-scan'] })
    expect(info.skillDescriptions).toEqual({ 'deep-scan': 'scan deeply' })
  })

  it('caps large tool result previews', () => {
    const events = adaptDaemonEvent('tool_result', {
      duration_ms: 1,
      name: 'ReadFile',
      return_value: 'x'.repeat(1000),
      tool_call_id: 'call_1'
    })
    const payload = events[0]?.payload as { result_text?: string }

    expect(payload.result_text).toHaveLength(600)
    expect(String(payload.result_text)).toMatch(/…$/)
  })

  it('normalizes stored transcript messages', () => {
    expect(
      transcriptFromStoredMessages([
        { content: 'hi', role: 'user' },
        { content: [{ text: 'hello' }, { content: 'again' }], role: 'assistant' },
        { content: 'skip', role: 'bad' }
      ])
    ).toEqual([
      { role: 'user', text: 'hi' },
      { role: 'assistant', text: 'hello\nagain' }
    ])
  })

  it('maps stream and tool events to UI events', () => {
    expect(adaptDaemonEvent('text_part', { text: 'hello' })).toEqual([
      { payload: { text: 'hello' }, type: 'message.delta' }
    ])

    expect(adaptDaemonEvent('tool_call', { arguments: '{"q":"x"}', id: 'call_1', name: 'ReadFile' })).toEqual([
      {
        payload: {
          args_text: '{"q":"x"}',
          name: 'ReadFile',
          tool_id: 'call_1'
        },
        type: 'tool.start'
      }
    ])

    expect(
      adaptDaemonEvent('tool_result', {
        display_blocks: [{ body: 'short', type: 'brief' }],
        duration_ms: 250,
        name: 'ReadFile',
        return_value: 'full output',
        tool_call_id: 'call_1'
      })
    ).toEqual([
      {
        payload: {
          duration_s: 0.25,
          inline_diff: '',
          name: 'ReadFile',
          result_text: 'full output',
          summary: 'short',
          todos: undefined,
          tool_id: 'call_1'
        },
        type: 'tool.complete'
      }
    ])
  })

  it('keeps clarify responses addressable by daemon request id and question id', () => {
    expect(
      adaptDaemonEvent('question_request', {
        id: 'req_1',
        questions: [{ id: 'q1', options: ['yes', 'no'], question: 'Continue?' }]
      })
    ).toEqual([
      {
        payload: {
          choices: ['yes', 'no'],
          question: 'Continue?',
          request_id: 'req_1:q1'
        },
        type: 'clarify.request'
      }
    ])
  })

  it('maps nested subagent progress', () => {
    expect(
      adaptDaemonEvent('subagent_event', {
        agent_id: 'a1',
        event: { payload: { text: 'working' }, type: 'text_part' },
        subagent_type: 'reviewer',
        task_index: 2
      })
    ).toEqual([
      {
        payload: {
          goal: 'reviewer',
          status: 'running',
          subagent_id: 'a1',
          task_index: 2,
          text: 'working'
        },
        type: 'subagent.progress'
      }
    ])
  })

  it('falls back to subagent ids when nested subagent type is empty', () => {
    expect(
      adaptDaemonEvent('subagent_event', {
        agent_id: 'r01-runtime',
        event: { payload: { name: 'ListDir' }, type: 'tool_call' },
        subagent_type: '',
        task_index: 2
      })
    ).toEqual([
      {
        payload: {
          goal: 'r01-runtime',
          status: 'running',
          subagent_id: 'r01-runtime',
          task_index: 2,
          tool_name: 'ListDir',
          tool_preview: ''
        },
        type: 'subagent.tool'
      }
    ])
  })

  it('maps daemon subagent stream notifications into live subagent events', () => {
    expect(
      adaptDaemonEvent('notification', {
        body: 'ReadFile README.md',
        category: 'subagent_stream',
        payload: {
          action: 'ReadFile README.md',
          agent_type: 'researcher',
          count: 3,
          parent: 'call_spawn',
          status: 'running',
          task_id: 'task_1'
        },
        type: 'subagent_stream'
      })
    ).toEqual([
      {
        payload: {
          goal: 'researcher',
          parent_id: 'call_spawn',
          status: 'running',
          subagent_id: 'task_1',
          task_index: 0,
          text: 'ReadFile README.md',
          tool_count: 3
        },
        type: 'subagent.start'
      },
      {
        payload: {
          goal: 'researcher',
          parent_id: 'call_spawn',
          status: 'running',
          subagent_id: 'task_1',
          task_index: 0,
          text: 'ReadFile README.md',
          tool_count: 3
        },
        type: 'subagent.progress'
      }
    ])
  })

  it('uses daemon stream labels when subagent agent_type is empty', () => {
    expect(
      adaptDaemonEvent('notification', {
        body: 'ListDir',
        category: 'subagent_stream',
        payload: {
          action: 'ListDir',
          agent_type: '',
          count: 20,
          label: 'r01-runtime#agent123',
          parent: 'call_spawn',
          status: 'running',
          task_id: 'agent123456789'
        },
        type: 'subagent_stream'
      })
    ).toEqual([
      {
        payload: {
          goal: 'r01-runtime#agent123',
          parent_id: 'call_spawn',
          status: 'running',
          subagent_id: 'agent123456789',
          task_index: 0,
          text: 'ListDir',
          tool_count: 20
        },
        type: 'subagent.start'
      },
      {
        payload: {
          goal: 'r01-runtime#agent123',
          parent_id: 'call_spawn',
          status: 'running',
          subagent_id: 'agent123456789',
          task_index: 0,
          text: 'ListDir',
          tool_count: 20
        },
        type: 'subagent.progress'
      }
    ])
  })

  it('maps terminal daemon subagent stream notifications into completion events', () => {
    expect(
      adaptDaemonEvent('notification', {
        body: 'done',
        category: 'subagent_stream',
        payload: {
          agent_type: 'reviewer',
          count: 8,
          result: 'final notes',
          status: 'completed',
          task_id: 'task_2'
        },
        type: 'subagent_stream'
      })
    ).toEqual([
      {
        payload: {
          goal: 'reviewer',
          parent_id: null,
          status: 'completed',
          subagent_id: 'task_2',
          summary: 'final notes',
          task_index: 0,
          text: 'done',
          tool_count: 8
        },
        type: 'subagent.complete'
      }
    ])
  })
})
