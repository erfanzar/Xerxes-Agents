// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { gateLlmClient, outboundApprovalPort, OUTBOUND_DENY, OUTBOUND_SEND } from '../src/daemon/outboundGate.js'
import { createOutboundGateFactory, type BoundarySession } from '../src/daemon/sessionBoundary.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { DaemonEvent, DaemonSession } from '../src/daemon/runtime.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'
import { DataBoundaryError, type OutboundRequestSummary } from '../src/security/boundaryTypes.js'
import { DisclosureLog } from '../src/security/disclosureLog.js'

const CANARY = 'CANARY-outbound-gate-9931'

function recordingLlm(): { llm: LlmClient; requests: CompletionRequest[] } {
  const requests: CompletionRequest[] = []
  return { requests, llm: { async *stream(request) { requests.push(request); yield { content: 'done' } } } }
}

const request: CompletionRequest = { model: 'opus', messages: [{ role: 'user', content: `hello ${CANARY}` }] }

async function drain(llm: LlmClient): Promise<void> {
  for await (const delta of llm.stream(request)) void delta
}

test('with no gate context the client is returned unchanged', () => {
  const { llm } = recordingLlm()
  expect(gateLlmClient(llm, undefined)).toBe(llm)
})

test('a refused request never reaches the provider and throws DataBoundaryError', async () => {
  const { llm, requests } = recordingLlm()
  const gated = gateLlmClient(llm, { gate: { check: async () => ({ allow: false, reason: 'private' }) }, trust: 'cloud', provider: 'p' })
  const error = await drain(gated).catch(caught => caught)
  expect(error).toBeInstanceOf(DataBoundaryError)
  expect(requests).toEqual([])
})

test('an allowed request is sent once, with the exact text summarized', async () => {
  const { llm, requests } = recordingLlm()
  const seen: OutboundRequestSummary[] = []
  const gated = gateLlmClient(llm, { gate: { check: async summary => { seen.push(summary); return { allow: true, decision: 'sent' } } }, trust: 'cloud', provider: 'p', sessionId: 's1' })
  await drain(gated)
  expect(requests).toHaveLength(1)
  expect(seen[0]?.text).toContain(CANARY)
  expect(seen[0]?.bytes).toBe(new TextEncoder().encode(seen[0]!.text).length)
  expect(seen[0]?.sessionId).toBe('s1')
})

test('the approval card sends only on an explicit Send', async () => {
  const summary: OutboundRequestSummary = { provider: 'p', model: 'm', trust: 'cloud', purpose: 'turn', sessionId: 's', text: 'x', bytes: 1 }
  const asked: unknown[] = []
  const port = (answer: string) => outboundApprovalPort(async (_id, question) => { asked.push(question); return answer })
  expect(await port(OUTBOUND_SEND)(summary)).toBe(true)
  expect(await port(OUTBOUND_DENY)(summary)).toBe(false)
  expect(await port('')(summary)).toBe(false)
  expect(asked[0]).toMatchObject({ options: [OUTBOUND_SEND, OUTBOUND_DENY], allowFreeform: false })
  // No session to show the card in: refused.
  const { sessionId: _unused, ...noSession } = summary
  expect(await port(OUTBOUND_SEND)(noSession)).toBe(false)
})

async function withRoot<T>(body: (root: string) => Promise<T>): Promise<T> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-outbound-'))
  try { return await body(root) } finally { await rm(root, { recursive: true, force: true }) }
}

const session = (cwd: string): BoundarySession => ({ id: 's1', cwd, metadata: {} })

test('factory: off mode wraps nothing; private refuses cloud and logs without the text', async () => {
  await withRoot(async root => {
    const log = new DisclosureLog({ path: join(root, 'log', 'disclosures.jsonl') })
    const off = createOutboundGateFactory({ runtime: () => ({}), env: {}, routeOf: () => ({ trust: 'cloud', provider: 'p' }), log })
    expect(off(session(root), 'opus')).toBeUndefined()
    const priv = createOutboundGateFactory({ runtime: () => ({ data_boundary: 'private' }), env: {}, routeOf: () => ({ trust: 'cloud', provider: 'p' }), log })
    const { llm, requests } = recordingLlm()
    expect(await drain(gateLlmClient(llm, priv(session(root), 'opus'))).catch(error => error)).toBeInstanceOf(DataBoundaryError)
    expect(requests).toEqual([])
    const local = createOutboundGateFactory({ runtime: () => ({ data_boundary: 'private' }), env: {}, routeOf: () => ({ trust: 'local', provider: 'bonsai' }), log })
    await drain(gateLlmClient(llm, local(session(root), 'bonsai-9b')))
    expect(requests).toHaveLength(1)
    const written = await readFile(log.path, 'utf8')
    expect(written).toContain('"denied"')
    expect(written).not.toContain(CANARY)
  })
})

test('factory: hybrid asks once per session and provider; a throwing route counts as cloud', async () => {
  await withRoot(async root => {
    let asks = 0
    const factory = createOutboundGateFactory({
      runtime: () => ({ data_boundary: 'hybrid' }),
      env: {},
      routeOf: () => { throw new Error('no profile') },
      approve: async () => { asks += 1; return true },
      log: new DisclosureLog({ path: join(root, 'd.jsonl') }),
    })
    const { llm, requests } = recordingLlm()
    await drain(gateLlmClient(llm, factory(session(root), 'opus')))
    await drain(gateLlmClient(llm, factory(session(root), 'opus')))
    expect(asks).toBe(1)
    expect(requests).toHaveLength(2)
  })
})

function daemonSession(cwd: string, id: string): DaemonSession {
  return {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd, extra: {}, id,
    interactionMode: 'code', sessionKey: id, lastActive: 0, messages: [], metadata: {},
    model: 'opus', planMode: false, status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: cwd,
  }
}

test('a turn under private mode on a cloud model ends without calling the provider, and is not retried', async () => {
  await withRoot(async root => {
    const { llm, requests } = recordingLlm()
    const runner = new AgentTurnRunner({
      llm,
      model: 'opus',
      outboundGateForSession: createOutboundGateFactory<DaemonSession>({
        runtime: () => ({ data_boundary: 'private' }),
        env: {},
        routeOf: () => ({ trust: 'cloud', provider: 'opus-profile' }),
        log: new DisclosureLog({ path: join(root, 'd.jsonl') }),
      }),
    })
    const events: DaemonEvent[] = []
    try {
      for await (const event of runner.run(daemonSession(root, 'private-turn'), 'hi', new AbortController().signal)) events.push(event)
    } catch { /* A terminal refusal may surface as a thrown turn error. */ }
    expect(requests).toEqual([])
    const text = JSON.stringify(events)
    expect(text).toContain('Data boundary')
    // Exactly one refusal: an unknown error class is not retried.
    const denied = (await readFile(join(root, 'd.jsonl'), 'utf8')).trim().split('\n').filter(line => line.includes('"denied"'))
    expect(denied).toHaveLength(1)
  })
})
