// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { DaemonEvent, DaemonSession } from '../src/daemon/runtime.js'
import type { LlmDelta } from '../src/llms/client.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

function sessionFor(cwd: string, id: string): DaemonSession {
  return {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd, extra: {}, id,
    interactionMode: 'code', sessionKey: id, lastActive: 0, messages: [], metadata: {},
    model: 'gpt-4o', planMode: false, status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: cwd,
  }
}

const WRITE_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'WriteFile',
    description: 'Write a file.',
    parameters: { type: 'object', properties: { file_path: { type: 'string' }, content: { type: 'string' } } },
  },
}

interface Outcome {
  readonly events: DaemonEvent[]
  readonly session: DaemonSession
}

async function runWith(mode: 'off' | 'warn' | 'enforce' | undefined): Promise<Outcome> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-result-checks-'))
  try {
    let round = 0
    const runner = new AgentTurnRunner({
      llm: {
        async *stream(): AsyncGenerator<LlmDelta> {
          round += 1
          if (round === 1) {
            yield {
              toolCalls: [{
                id: 'write-xlsx',
                type: 'function',
                function: { name: 'WriteFile', arguments: { file_path: 'out.xlsx', content: 'a,b\n1,2\n' } },
              }],
            }
            return
          }
          yield { content: 'Wrote out.xlsx.' }
        },
      },
      model: 'gpt-4o',
      permissionMode: 'accept-all',
      // A model that writes CSV text into an .xlsx: the failure the checks exist for.
      toolExecutor: {
        async execute(call) {
          const args = call.function.arguments
          await writeFile(join(root, String(args.file_path)), String(args.content))
          return 'written'
        },
      },
      tools: [WRITE_TOOL],
      ...(mode === undefined ? {} : { resultChecksForSession: () => mode }),
    })
    const session = sessionFor(root, `checks-${mode ?? 'none'}`)
    const events: DaemonEvent[] = []
    for await (const event of runner.run(session, 'Make a workbook.', new AbortController().signal)) events.push(event)
    return { events, session }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

const checkNotices = (events: readonly DaemonEvent[]) =>
  events.filter(event => event.type === 'notification' && event.payload.result_checks !== undefined)
const harnessFindings = (session: DaemonSession) =>
  session.messages.filter(message => {
    const record = message as { role?: unknown; origin?: unknown; content?: unknown }
    return record.role === 'user' && record.origin === 'harness' && String(record.content).includes('Result checks found')
  })

test('without result checks configured, or with them off, nothing runs', async () => {
  for (const mode of [undefined, 'off'] as const) {
    const { events, session } = await runWith(mode)
    expect(checkNotices(events)).toEqual([])
    expect(harnessFindings(session)).toEqual([])
  }
})

test('warn tells the user that the .xlsx is text, and never tells the model', async () => {
  const { events, session } = await runWith('warn')
  const notices = checkNotices(events)
  expect(notices).toHaveLength(1)
  expect(String(notices[0]?.payload.message)).toContain('xlsx_is_text')
  expect(harnessFindings(session)).toEqual([])
})

test('enforce also hands the error findings to the model as a harness message', async () => {
  const { events, session } = await runWith('enforce')
  expect(checkNotices(events)).toHaveLength(1)
  const delivered = harnessFindings(session)
  expect(delivered).toHaveLength(1)
  expect(String((delivered[0] as { content?: unknown }).content)).toContain('Fix these before you report the task as done.')
})
