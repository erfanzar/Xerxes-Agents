// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { buildBootstrapSystemPrompt } from '../src/runtime/bootstrap.js'

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
  // These assertions previously pinned a claim the runtime did not honor —
  // nothing constructed the store, so every agent was told its large results
  // were preserved when they were simply sent whole. The prompt now describes
  // the envelope the turn runner actually emits.
  expect(prompt).toContain('`<persisted-output>` envelope carrying a bounded preview')
  expect(prompt).toContain('absolute path of the full output on disk')
  expect(prompt).toContain('successful result, not a failure')
  expect(prompt).not.toContain('stored in project agent memory')
  expect(prompt).not.toContain('apply_patch')
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
