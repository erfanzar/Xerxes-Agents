// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { buildBootstrapSystemPrompt } from '../src/runtime/bootstrap.js'
import {
  PromptContextBuilder,
  type PromptContextHost,
} from '../src/runtime/promptContext.js'
import { PromptProfile } from '../src/runtime/promptProfiles.js'

const PROMPT_CONTEXT_HOST: PromptContextHost = {
  captureRuntimeInfo: () => ({
    platform: 'darwin',
    runtimeVersion: 'Bun test',
    timestamp: '2026-07-13T12:00:00.000Z',
    timezone: 'Europe/Istanbul',
    workingDirectory: '/workspace',
    workspaceName: 'workspace',
    xerxesVersion: '0.2.0',
  }),
}

test('bootstrap prompt retains guidance only for the supplied terminal, mode, and memory tools', () => {
  const prompt = buildBootstrapSystemPrompt({
    cwd: '/workspace',
    date: '2026-07-13 Sunday',
    model: 'gpt-4o',
    platform: 'darwin',
  }, '', [
    promptTool('exec_command', {
      cmd: { type: 'string' },
      args: { type: 'array', items: { type: 'string' } },
    }, ['cmd']),
    promptTool('write_stdin', { session_id: { type: 'string' } }, ['session_id']),
    promptTool('close_terminal_session', { session_id: { type: 'string' } }, ['session_id']),
    promptTool('agent_memory_read', { namespace: { type: 'string' }, path: { type: 'string' } }, ['namespace', 'path']),
    promptTool('SetInteractionModeTool', { mode: { type: 'string' } }, ['mode']),
  ])

  expect(prompt).toContain('- exec_command: Test definition for exec_command. (required: cmd)')
  expect(prompt).toContain('exec_command uses direct argv: cmd is one executable and each argument belongs in args')
  expect(prompt).not.toContain('"required":["cmd"]')
  expect(prompt).toContain('Poll or interact with a live terminal session through write_stdin.')
  expect(prompt).toContain('Close terminal sessions with close_terminal_session')
  expect(prompt).toContain(
    '- objective: hard-goal loop for measurable outcomes.',
  )
  expect(prompt).toContain(
    'SetInteractionModeTool schedules a mode for the next user turn',
  )
  expect(prompt).toContain(
    'Do not final-answer in objective mode while acceptance criteria are unmet',
  )
  expect(prompt).toContain(
    'Oversized tool results are stored in project agent memory',
  )
  expect(prompt).toContain(
    '`[Large tool result stored outside model context]` as a valid tool result',
  )
  expect(prompt).toContain('Read stored-result pointers with agent_memory_read')
  expect(prompt).not.toContain('apply_patch')
})

test('full runtime prompt preserves native tool reinvocation and web-search grounding rules', async () => {
  const prompt = await new PromptContextBuilder({
    host: PROMPT_CONTEXT_HOST,
  }).assembleSystemPromptPrefix({
    profile: PromptProfile.FULL,
    toolNames: ['ReadFile', 'web.search_query', 'web.open'],
  })

  expect(prompt).toContain(
    'Do not use or simulate tools for greetings, simple arithmetic',
  )
  expect(prompt).toContain(
    'Do not repeat the same tool call with identical arguments',
  )
  expect(prompt).toContain('If a tool result answers the task, use it directly')
  expect(prompt).toContain('Generic web-search follow-ups')
  expect(prompt).toContain('do not force an unrelated search')
  expect(prompt).toContain(
    'do not claim that you cannot browse or access current information',
  )
  expect(prompt).toContain(
    'Search snippets and result titles are leads, not verification',
  )
  expect(prompt).toContain(
    'Do not simulate tool calls or wrap normal answers in tool/XML markup.',
  )
})

function promptTool(
  name: string,
  properties: Readonly<Record<string, unknown>>,
  required: readonly string[],
): Readonly<Record<string, unknown>> {
  return {
    type: 'function',
    function: {
      name,
      description: `Test definition for ${name}.`,
      parameters: { type: 'object', properties, required },
    },
  }
}
