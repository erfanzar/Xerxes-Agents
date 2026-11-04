// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { createEscalationFactory } from '../src/daemon/turnEscalation.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { DaemonEvent, DaemonSession } from '../src/daemon/runtime.js'
import type { LlmDelta } from '../src/llms/client.js'
import type { CheckFinding } from '../src/runtime/resultChecks.js'
import type { DataBoundaryMode } from '../src/security/boundaryTypes.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

const ERROR: CheckFinding = { code: 'xlsx_is_text', severity: 'error', path: 'out.xlsx', message: 'not a workbook' }
const WARN: CheckFinding = { code: 'csv_bad_date', severity: 'warn', path: 'a.csv', message: 'odd date' }
const session = { id: 's1', cwd: '/tmp' }

function factory(runtime: Record<string, unknown>, options: { advisor?: boolean; boundary?: DataBoundaryMode; touchesPrivate?: boolean } = {}) {
  return createEscalationFactory<{ id: string; cwd: string }>({
    runtime: () => runtime,
    env: {},
    boundaryMode: () => options.boundary ?? 'off',
    touchesPrivate: () => options.touchesPrivate ?? false,
    advisorAvailable: () => options.advisor ?? false,
  })
}

test('escalation off never produces a step', () => {
  const next = factory({})
  expect(next(session, [ERROR])).toBeUndefined()
  expect(next(session, [ERROR])).toBeUndefined()
})

test('the ladder retries locally, then asks the advisor, then stops', () => {
  const next = factory({ escalation: { mode: 'on', max_local_retries: 2, max_advisor_calls: 1 } }, { advisor: true })
  // A fresh finding each attempt so the no-improvement stop does not end it early.
  const attempt = (errors: number): CheckFinding[] => Array.from({ length: errors }, (_, index) => ({ ...ERROR, path: `out${index}.xlsx` }))
  expect(next(session, attempt(5))?.action).toBe('retry_local')
  expect(next(session, attempt(4))?.action).toBe('retry_local')
  const advisor = next(session, attempt(3))
  expect(advisor?.action).toBe('ask_advisor')
  expect(advisor?.feedback).toContain('AskAdvisor')
  const stop = next(session, attempt(2))
  expect(stop?.action).toBe('stop')
  expect(stop?.feedback).toBeUndefined()
})

test('a clean or warn-only check resets the ladder', () => {
  const next = factory({ escalation: { mode: 'on', max_local_retries: 1 } })
  expect(next(session, [ERROR])?.action).toBe('retry_local')
  expect(next(session, [WARN])).toBeUndefined()
  expect(next(session, [ERROR])?.action).toBe('retry_local')
})

test('private work never escalates to the cloud advisor', () => {
  const next = factory({ escalation: { mode: 'on', max_local_retries: 0 } }, { advisor: true, boundary: 'hybrid', touchesPrivate: true })
  const step = next(session, [ERROR])
  expect(step?.action).toBe('stop')
  expect(step?.message).toContain('private')
})

const WRITE_TOOL: ToolDefinition = {
  type: 'function',
  function: { name: 'WriteFile', description: 'Write a file.', parameters: { type: 'object', properties: { file_path: { type: 'string' }, content: { type: 'string' } } } },
}

function daemonSession(cwd: string): DaemonSession {
  return {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd, extra: {}, id: 'esc-turn',
    interactionMode: 'code', sessionKey: 'esc-turn', lastActive: 0, messages: [], metadata: {},
    model: 'gpt-4o', planMode: false, status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0, workspace: cwd,
  }
}

test('in a turn, escalation feedback replaces the plain enforce message and is not duplicated', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-turn-escalation-'))
  try {
    let round = 0
    const runner = new AgentTurnRunner({
      llm: {
        async *stream(): AsyncGenerator<LlmDelta> {
          round += 1
          if (round === 1) {
            yield { toolCalls: [{ id: 'w', type: 'function', function: { name: 'WriteFile', arguments: { file_path: 'out.xlsx', content: 'a,b\n1,2\n' } } }] }
            return
          }
          yield { content: 'Wrote out.xlsx.' }
        },
      },
      model: 'gpt-4o',
      permissionMode: 'accept-all',
      toolExecutor: { async execute(call) { const args = call.function.arguments; await writeFile(join(root, String(args.file_path)), String(args.content)); return 'written' } },
      tools: [WRITE_TOOL],
      resultChecksForSession: () => 'enforce',
      escalationForSession: factory({ escalation: { mode: 'on' } }),
    })
    const target = daemonSession(root)
    const events: DaemonEvent[] = []
    for await (const event of runner.run(target, 'Make a workbook.', new AbortController().signal)) events.push(event)
    const harness = target.messages.filter(message => {
      const record = message as { role?: unknown; origin?: unknown }
      return record.role === 'user' && record.origin === 'harness'
    })
    expect(harness).toHaveLength(1)
    expect(String((harness[0] as { content?: unknown }).content)).toContain('retry')
    expect(events.some(event => event.type === 'notification' && event.payload.escalation !== undefined)).toBe(true)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
