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
      permission_mode: 'accept-all',
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
      permission_mode: 'accept-all',
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

  it('forwards a cancelled turn_end as a daemon-confirmed interruption', () => {
    expect(adaptDaemonEvent('turn_end', { cancelled: true, session_id: 's1' })).toEqual([
      { payload: { interrupted: true }, type: 'message.complete' }
    ])
    expect(adaptDaemonEvent('turn_end', { cancelled: false, session_id: 's1' })).toEqual([
      { payload: {}, type: 'message.complete' }
    ])
    expect(adaptDaemonEvent('turn_end', {})).toEqual([{ payload: {}, type: 'message.complete' }])
  })

  it('marks denied and failed tool results without exposing successful payloads as errors', () => {
    const denied = adaptDaemonEvent('tool_result', {
      name: 'ExecCommand',
      permitted: false,
      return_value: 'Cancelled before execution.',
      tool_call_id: 'call_denied'
    })[0]?.payload as { error?: string }
    const failed = adaptDaemonEvent('tool_result', {
      name: 'ExecCommand',
      permitted: true,
      return_value: 'Tool execution failed: command not found',
      tool_call_id: 'call_failed'
    })[0]?.payload as { error?: string }
    const successful = adaptDaemonEvent('tool_result', {
      name: 'ExecCommand',
      permitted: true,
      return_value: '{"stdout":"ok"}',
      tool_call_id: 'call_ok'
    })[0]?.payload as { error?: string }

    expect(denied.error).toBe('Cancelled before execution.')
    expect(failed.error).toBe('Tool execution failed: command not found')
    expect(successful.error).toBeUndefined()
  })

  it('reads flat token usage from the native daemon status payload', () => {
    const events = adaptDaemonEvent('status_update', {
      context_tokens: 21,
      input_tokens: 8,
      max_context: 1000,
      mode: 'code',
      output_tokens: 13,
      permission_mode: 'auto',
      total_input_tokens: 8,
      total_output_tokens: 13,
      total_tokens: 21
    })
    const payload = events[0]?.payload as { usage?: Record<string, unknown> }

    expect(payload.usage).toMatchObject({
      context_max: 1000,
      context_used: 21,
      input: 8,
      output: 13,
      total: 21
    })
    expect((events[1]?.payload as Record<string, unknown>).permission_mode).toBe('auto')
    expect(events.some(event => event.type === 'transcript.append')).toBe(false)
  })

  it('does not invent a code mode for usage-only turn-end status updates', () => {
    const events = adaptDaemonEvent('status_update', {
      context_tokens: 21,
      input_tokens: 8,
      output_tokens: 13,
      total_tokens: 21
    })
    const status = events[0]?.payload as Record<string, unknown>
    const info = events[1]?.payload as Record<string, unknown>

    expect(status).not.toHaveProperty('mode')
    expect(info).not.toHaveProperty('mode')
    expect(status).not.toHaveProperty('reasoning_effort')
    expect(info).not.toHaveProperty('reasoning_effort')
  })

  it('normalizes stored transcript messages', () => {
    expect(
      transcriptFromStoredMessages([
        { content: '<attached_files>hidden</attached_files>\n\nhi', role: 'user', text: 'hi' },
        { content: [{ text: 'hello' }, { content: 'again' }], role: 'assistant' },
        { content: 'internal system prompt', role: 'system' },
        { content: 'x'.repeat(700), role: 'tool' },
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

    expect(adaptDaemonEvent('think_part', { think: 'hidden chain' })).toEqual([
      { payload: { text: 'hidden chain' }, type: 'thinking.delta' }
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

  it('maps replay notifications to transcript append events instead of live streaming', () => {
    expect(
      adaptDaemonEvent('notification', {
        body: 'old answer',
        category: 'history',
        type: 'replay_assistant'
      })
    ).toEqual([{ payload: { role: 'assistant', text: 'old answer' }, type: 'transcript.append' }])

    expect(
      adaptDaemonEvent('notification', {
        body: '✨ old prompt',
        category: 'history',
        type: 'replay_user'
      })
    ).toEqual([{ payload: { role: 'user', text: 'old prompt' }, type: 'transcript.append' }])
  })

  it('keeps clarify responses addressable by daemon request id and question id', () => {
    expect(
      adaptDaemonEvent('question_request', {
        flow: 'provider',
        id: 'req_1',
        tool_call_id: 'tool-clarify-1',
        questions: [
          {
            allow_free_form: false,
            id: 'q1',
            options: ['yes', 'no'],
            placeholder: 'Pick one',
            question: 'Continue?'
          }
        ]
      })
    ).toEqual([
      {
        payload: {
          allow_free_form: false,
          choices: ['yes', 'no'],
          placeholder: 'Pick one',
          question: 'Continue?',
          question_id: 'q1',
          request_id: 'req_1:q1',
          source: 'provider',
          tool_id: 'tool-clarify-1'
        },
        type: 'clarify.request'
      }
    ])
  })

  it('keeps approval responses addressable by their daemon request id', () => {
    expect(
      adaptDaemonEvent('approval_request', {
        action: 'ExecCommand',
        allow_permanent: false,
        description: 'Run the test suite',
        request_id: 'permission-42'
      })
    ).toEqual([
      {
        payload: {
          allow_permanent: false,
          command: 'ExecCommand',
          description: 'Run the test suite',
          request_id: 'permission-42'
        },
        type: 'approval.request'
      }
    ])
  })

  it('maps nested subagent progress', () => {
    expect(
      adaptDaemonEvent('subagent_event', {
        agent_id: 'a1',
        agent_name: 'security-review',
        event: { payload: { text: 'working' }, type: 'text_part' },
        subagent_type: 'reviewer',
        task_index: 2
      })
    ).toEqual([
      {
        payload: {
          agent_name: 'security-review',
          agent_type: 'reviewer',
          depth: 0,
          goal: 'reviewer',
          parent_id: null,
          status: 'running',
          subagent_id: 'a1',
          task_index: 2,
          text: 'working'
        },
        type: 'subagent.progress'
      }
    ])
  })

  it('preserves agent-panel metadata and completion metrics', () => {
    const [event] = adaptDaemonEvent('subagent_event', {
      agent_id: 'policy-child',
      agent_name: 'policy-review',
      agent_type: 'reviewer',
      creator_id: 'runtime-parent',
      event: {
        payload: {
          api_calls: 4,
          duration_seconds: 12.5,
          files_read: ['src/auth.ts'],
          files_written: ['src/policy.ts'],
          input_tokens: 1200,
          output_tokens: 320,
          reasoning_tokens: 90,
          status: 'completed',
          summary: 'Policy review complete.',
          tool_count: 7
        },
        type: 'turn_end'
      },
      model: 'grok-code-fast',
      parent_id: 'runtime-parent',
      rules: ['read-only audit'],
      task_index: 1,
      title: 'Policy audit',
      toolsets: ['ReadFile', 'Grep']
    })

    expect(event).toMatchObject({
      payload: {
        api_calls: 4,
        creator_id: 'runtime-parent',
        duration_seconds: 12.5,
        files_read: ['src/auth.ts'],
        files_written: ['src/policy.ts'],
        input_tokens: 1200,
        model: 'grok-code-fast',
        output_tokens: 320,
        reasoning_tokens: 90,
        rules: ['read-only audit'],
        status: 'completed',
        summary: 'Policy review complete.',
        title: 'Policy audit',
        tool_count: 7,
        toolsets: ['ReadFile', 'Grep']
      },
      type: 'subagent.complete'
    })
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
          agent_type: '',
          depth: 0,
          goal: 'r01-runtime',
          parent_id: null,
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
          agent_type: 'researcher',
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
          agent_type: 'researcher',
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
          agent_name: 'r01-runtime#agent123',
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
          agent_name: 'r01-runtime#agent123',
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
          agent_type: 'reviewer',
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

  it('normalizes PascalCase nested tool events while preserving parallel agent identity', () => {
    const first = adaptDaemonEvent('subagent_event', {
      agent_id: 'research-child',
      depth: 1,
      event: {
        payload: { arguments: '{"file_path":"README.md"}', name: 'ReadFile' },
        type: 'ToolCall'
      },
      goal: 'inspect documentation',
      parent_id: 'root-agent',
      subagent_type: 'researcher',
      task_index: 0
    })
    const second = adaptDaemonEvent('subagent_event', {
      agent_id: 'review-child',
      depth: 1,
      event: {
        payload: { name: 'ExecCommand', permitted: true, return_value: 'checks passed\nmore output' },
        type: 'ToolResult'
      },
      goal: 'run verification',
      parent_id: 'root-agent',
      subagent_type: 'reviewer',
      task_index: 1
    })

    expect(first).toEqual([
      {
        payload: {
          agent_type: 'researcher',
          depth: 1,
          goal: 'inspect documentation',
          parent_id: 'root-agent',
          status: 'running',
          subagent_id: 'research-child',
          task_index: 0,
          tool_name: 'ReadFile',
          tool_preview: '{"file_path":"README.md"}'
        },
        type: 'subagent.tool'
      }
    ])
    expect(second).toEqual([
      {
        payload: {
          agent_type: 'reviewer',
          depth: 1,
          goal: 'run verification',
          parent_id: 'root-agent',
          status: 'running',
          subagent_id: 'review-child',
          task_index: 1,
          text: '✓ ExecCommand — checks passed'
        },
        type: 'subagent.progress'
      }
    ])
  })
})
