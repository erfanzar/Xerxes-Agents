// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { createEscalationFactory } from '../src/daemon/turnEscalation.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { DaemonEvent, DaemonSession } from '../src/daemon/runtime.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { CheckFinding } from '../src/runtime/resultChecks.js'
import type { DataBoundaryMode } from '../src/security/boundaryTypes.js'
import { createOfficeZip } from '../src/skills/powerpoint/zip.js'
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

const WORKBOOK = createOfficeZip([
  { name: '[Content_Types].xml', data: new TextEncoder().encode('<Types/>') },
  { name: 'xl/workbook.xml', data: new TextEncoder().encode('<workbook/>') },
  { name: 'xl/worksheets/sheet1.xml', data: new TextEncoder().encode('<worksheet/>') },
])

/** A model scripted by call number; `write` puts CSV text or a real workbook at out.xlsx. */
function scriptedModel(script: ReadonlyArray<'csv' | 'workbook' | 'done'>) {
  const requests: CompletionRequest[] = []
  const llm: LlmClient = {
    async *stream(request): AsyncGenerator<LlmDelta> {
      requests.push(request)
      const step = script[Math.min(requests.length, script.length) - 1] ?? 'done'
      if (step === 'done') {
        yield { content: 'Wrote out.xlsx.' }
        return
      }
      yield { toolCalls: [{ id: `w${requests.length}`, type: 'function', function: { name: 'WriteFile', arguments: { file_path: 'out.xlsx', content: step } } }] }
    },
  }
  return { llm, requests }
}

function runnerFor(root: string, llm: LlmClient, extra: Partial<ConstructorParameters<typeof AgentTurnRunner>[0]> = {}): AgentTurnRunner {
  return new AgentTurnRunner({
    llm,
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    toolExecutor: {
      async execute(call) {
        const target = join(root, String(call.function.arguments.file_path))
        if (call.function.arguments.content === 'workbook') await Bun.write(target, WORKBOOK)
        else await writeFile(target, 'a,b\n1,2\n')
        return 'written'
      },
    },
    tools: [WRITE_TOOL],
    resultChecksForSession: () => 'enforce',
    ...extra,
  })
}

const harnessMessages = (target: DaemonSession): string[] => target.messages
  .filter(message => (message as { role?: unknown }).role === 'user' && (message as { origin?: unknown }).origin === 'harness')
  .map(message => String((message as { content?: unknown }).content))

const escalationActions = (events: readonly DaemonEvent[]): unknown[] => events
  .filter(event => event.type === 'notification' && event.payload.escalation !== undefined)
  .map(event => (event.payload.escalation as { action?: unknown }).action)

test('a failed check re-runs the model inside the same turn, and a passing fix ends the cascade', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-cascade-retry-'))
  try {
    // Writes CSV text into .xlsx, claims done, then fixes it on the retry.
    const model = scriptedModel(['csv', 'done', 'workbook', 'done'])
    const runner = runnerFor(root, model.llm, { escalationForSession: factory({ escalation: { mode: 'on' } }) })
    const target = daemonSession(root)
    const events: DaemonEvent[] = []
    for await (const event of runner.run(target, 'Make a workbook.', new AbortController().signal)) events.push(event)
    // One user turn, four model calls: the retry happened inside it.
    expect(model.requests).toHaveLength(4)
    expect(escalationActions(events)).toEqual(['retry_local'])
    const harness = harnessMessages(target)
    expect(harness).toHaveLength(1)
    expect(harness[0]).toContain('failed these checks')
    // The model saw the feedback on the very next call.
    expect(JSON.stringify(model.requests[2]!.messages)).toContain('failed these checks')
    const bytes = new Uint8Array(await Bun.file(join(root, 'out.xlsx')).arrayBuffer())
    expect(bytes[0]).toBe(0x50)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('when the ladder reaches the advisor, the runner consults it itself and passes the answer on', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-cascade-advisor-'))
  try {
    const model = scriptedModel(['csv', 'done', 'workbook', 'done'])
    const consulted: Array<{ question: string; context: string }> = []
    const runner = runnerFor(root, model.llm, {
      escalationForSession: factory({ escalation: { mode: 'on', max_local_retries: 0 } }, { advisor: true }),
      consultAdvisor: async (_session, request) => {
        consulted.push({ ...request })
        return 'Write the file with XlsxWriter so it is a real workbook.'
      },
    })
    const target = daemonSession(root)
    const events: DaemonEvent[] = []
    for await (const event of runner.run(target, 'Make a workbook.', new AbortController().signal)) events.push(event)
    expect(consulted).toHaveLength(1)
    expect(consulted[0]!.context).toContain('xlsx_is_text')
    expect(escalationActions(events)).toEqual(['ask_advisor'])
    const harness = harnessMessages(target)
    expect(harness).toHaveLength(1)
    expect(harness[0]).toContain('Write the file with XlsxWriter')
    expect(harness[0]).not.toContain('Call AskAdvisor')
    expect(model.requests).toHaveLength(4)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('a cascade that does not improve stops instead of looping', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-cascade-stop-'))
  try {
    // Never fixes the file.
    const model = scriptedModel(['csv', 'done'])
    const runner = runnerFor(root, model.llm, { escalationForSession: factory({ escalation: { mode: 'on', max_local_retries: 5 } }) })
    const target = daemonSession(root)
    const events: DaemonEvent[] = []
    for await (const event of runner.run(target, 'Make a workbook.', new AbortController().signal)) events.push(event)
    const actions = escalationActions(events)
    expect(actions.at(-1)).toBe('stop')
    // Two retries without improvement end it, well under the five allowed.
    expect(actions.filter(action => action === 'retry_local').length).toBeLessThanOrEqual(3)
    expect(model.requests.length).toBeLessThanOrEqual(8)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('unset, escalation follows the host default, and an explicit mode wins', () => {
  const withDefault = (runtime: Record<string, unknown>, mode: 'on' | 'off') => createEscalationFactory<{ id: string; cwd: string }>({
    runtime: () => runtime, env: {}, boundaryMode: () => 'off', advisorAvailable: () => false, defaultMode: () => mode,
  })
  expect(withDefault({}, 'on')(session, [ERROR])?.action).toBe('retry_local')
  expect(withDefault({}, 'off')(session, [ERROR])).toBeUndefined()
  expect(withDefault({ escalation: { mode: 'off' } }, 'on')(session, [ERROR])).toBeUndefined()
})
