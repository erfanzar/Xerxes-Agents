// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  INTERRUPTED_TOOL_RESULT,
  daemonTranscriptRecord,
  normalizeDaemonTranscript,
} from '../src/session/daemonTranscript.js'

test('transcript normalization preserves complete model history while tail-capping auxiliary buffers', () => {
  const messages = [
    ...Array.from({ length: 12 }, (_, index) => ({ role: 'user', content: `before-${index}` })),
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        { id: 'call-a', name: 'ReadFile', input: { path: 'README.md' } },
        { id: 'call-b', name: 'GrepTool', input: { pattern: 'TODO' } },
      ],
    },
    { role: 'tool', tool_call_id: 'call-a', name: 'ReadFile', content: 'read result' },
    { role: 'tool', tool_call_id: 'call-b', name: 'GrepTool', content: 'grep result' },
    ...Array.from({ length: 12 }, (_, index) => ({ role: 'assistant', content: `after-${index}` })),
  ]
  const thinking = Array.from({ length: 40 }, (_, index) => `thought-${index}`)
  const executions = Array.from({ length: 205 }, (_, index) => ({ name: `tool-${index}` }))
  const normalized = normalizeDaemonTranscript({
    session_id: 'a1b2c3d4',
    messages,
    thinking_content: thinking,
    tool_executions: executions,
  }, {
    currentProjectDirectory: '/workspace/project',
    requestedSessionKey: 'a1b2c3d4',
  })
  if (!normalized) {
    throw new Error('expected a valid native daemon transcript')
  }

  expect(normalized.messages).toEqual(messages)
  expect(normalized.thinkingContent).toEqual(thinking.slice(-32))
  expect(normalized.toolExecutions).toEqual(executions.slice(-200))

  const record = daemonTranscriptRecord(normalized)
  expect(record.messages).toEqual(messages)
  expect(record.thinking_content).toEqual(thinking.slice(-32))
  expect(record.tool_executions).toEqual(executions.slice(-200))
})

test('resume safety backfills only an interrupted tool call with an explicit sentinel', () => {
  const normalized = normalizeDaemonTranscript({
    session_id: 'b1c2d3e4',
    messages: [
      { role: 'user', content: 'inspect the repository' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          { id: 'complete', name: 'ReadFile', input: { path: 'README.md' } },
          { id: 'interrupted', name: 'GrepTool', input: { pattern: 'TODO' } },
        ],
      },
      { role: 'tool', tool_call_id: 'complete', name: 'ReadFile', content: 'actual file contents' },
    ],
  }, {
    currentProjectDirectory: '/workspace/project',
    requestedSessionKey: 'b1c2d3e4',
  })
  if (!normalized) {
    throw new Error('expected a valid native daemon transcript')
  }

  expect(normalized.messages).toEqual([
    { role: 'user', content: 'inspect the repository' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        { id: 'complete', name: 'ReadFile', input: { path: 'README.md' } },
        { id: 'interrupted', name: 'GrepTool', input: { pattern: 'TODO' } },
      ],
    },
    { role: 'tool', tool_call_id: 'complete', name: 'ReadFile', content: 'actual file contents' },
    { role: 'tool', tool_call_id: 'interrupted', content: INTERRUPTED_TOOL_RESULT },
  ])
  expect(normalized.pendingResumeReplays).toEqual([
    { tool_call_id: 'interrupted', name: 'GrepTool', arguments: '{"pattern":"TODO"}' },
  ])
})
