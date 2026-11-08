// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ValidationError } from '../src/core/errors.js'
import type { DaemonSession } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { modelCardReport } from '../src/runtime/modelCardReport.js'
import { DEFAULT_REPLY_BUDGET, resolveReplyBudget, type ReplyBudget } from '../src/runtime/replyBudget.js'
import { createAgentState, type AgentState, type StreamEvent } from '../src/streaming/events.js'
import { MAX_REPLY_BUDGET_STOPS, runTurn, type TurnDependencies } from '../src/streaming/loop.js'

const BONSAI = 'ternary-bonsai-2-27b-mlx'
const RUNAWAY = 'Let me add these up by hand. '

/** A model that reasons without acting on every request whose last message is not a harness note. */
function runawayModel(options: { chunks?: number; chunkChars?: number; recoverOnNote?: boolean; clock?: Clock; secondsPerChunk?: number } = {}) {
  const requests: CompletionRequest[] = []
  const abortedOnClose: boolean[] = []
  const llm: LlmClient = {
    async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
      requests.push(request)
      const last = request.messages.at(-1)
      if (options.recoverOnNote !== false && last?.role === 'user' && typeof last.content === 'string' && last.content.startsWith('[harness]')) {
        yield { content: 'Done: the total is 42.', usage: { inputTokens: 10, outputTokens: 5 } }
        return
      }
      try {
        for (let index = 0; index < (options.chunks ?? 40); index += 1) {
          if (options.clock && options.secondsPerChunk) options.clock.advance(options.secondsPerChunk * 1_000)
          yield { thinking: RUNAWAY.repeat(Math.ceil((options.chunkChars ?? 400) / RUNAWAY.length)).slice(0, options.chunkChars ?? 400) }
        }
        yield { content: 'Finally an answer.', usage: { inputTokens: 10, outputTokens: 5 } }
      } finally {
        abortedOnClose.push(signal?.aborted ?? false)
      }
    },
  }
  return { llm, requests, abortedOnClose }
}

interface Clock { now(): number; advance(ms: number): void }
function fakeClock(): Clock {
  let at = 1_000
  return { now: () => at, advance: ms => { at += ms } }
}

async function collect(
  model: string,
  llm: LlmClient,
  extra: { replyBudget?: ReplyBudget | null; now?: () => number; signal?: AbortSignal } = {},
): Promise<{ events: StreamEvent[]; state: AgentState }> {
  const state = createAgentState([])
  const dependencies: TurnDependencies = { llm, retryDelays: [], delay: async () => undefined, ...(extra.now ? { now: extra.now } : {}) }
  const events: StreamEvent[] = []
  for await (const event of runTurn({
    model,
    state,
    userMessage: 'Sum the invoices.',
    ...(extra.replyBudget === undefined ? {} : { replyBudget: extra.replyBudget }),
  }, dependencies, extra.signal)) events.push(event)
  return { events, state }
}

const budgetRetries = (events: readonly StreamEvent[]) =>
  events.filter(event => event.type === 'provider_retry' && event.cause === 'reply_budget')
const keptRunaway = (state: AgentState) =>
  state.messages.some(message => message.role === 'assistant' && JSON.stringify(message).includes(RUNAWAY))

test('the setting: unset follows the model card, off turns it off, and fields fill from the card or the default', () => {
  expect(resolveReplyBudget({}, BONSAI)).toEqual({ maxReasoningTokens: 8_192, maxSeconds: 600 })
  expect(resolveReplyBudget({}, 'gpt-4o')).toBeNull()
  expect(resolveReplyBudget({ reply_budget: 'off' }, BONSAI)).toBeNull()
  expect(resolveReplyBudget({ reply_budget: false }, BONSAI)).toBeNull()
  expect(resolveReplyBudget({ reply_budget: true }, 'gpt-4o')).toEqual(DEFAULT_REPLY_BUDGET)
  expect(resolveReplyBudget({ reply_budget: { max_seconds: 120 } }, BONSAI)).toEqual({ maxReasoningTokens: 8_192, maxSeconds: 120 })
  expect(() => resolveReplyBudget({ reply_budget: { max_minutes: 5 } }, BONSAI)).toThrow(ValidationError)
  expect(() => resolveReplyBudget({ reply_budget: { max_reasoning_tokens: 10 } }, BONSAI)).toThrow('between 256')
  expect(() => resolveReplyBudget({ reply_budget: 'sometimes' }, BONSAI)).toThrow('reply_budget')
})

test('/bonsai shows the card budget, the one in effect and what set it', () => {
  const row = (runtime: Record<string, unknown>, model = BONSAI) =>
    modelCardReport({ model, runtime, env: {} }).settings.find(setting => setting.name === 'Reply budget')
  expect(row({})).toEqual({ name: 'Reply budget', card: '8,192 reasoning tokens or 10 min', effective: '8,192 reasoning tokens or 10 min', from: 'card' })
  expect(row({ reply_budget: 'off' })).toMatchObject({ effective: 'none', from: 'runtime.reply_budget' })
  expect(row({ reply_budget: { max_seconds: 90 } })).toMatchObject({ effective: '8,192 reasoning tokens or 90 s' })
  expect(row({}, 'gpt-4o')).toMatchObject({ card: '—', effective: 'none', from: '—' })
})

test('a reply past the token budget is stopped, its reasoning dropped, and the step asked again with a note', async () => {
  const model = runawayModel({ chunks: 40, chunkChars: 400 })
  const { events, state } = await collect('test-model', model.llm, { replyBudget: { maxReasoningTokens: 1_000, maxSeconds: 600 } })

  const [retry, ...others] = budgetRetries(events)
  expect(others).toHaveLength(0)
  expect(retry).toMatchObject({ final: false, attempt: 1, maxAttempts: MAX_REPLY_BUDGET_STOPS + 1 })
  expect(retry?.type === 'provider_retry' && retry.error).toContain('reply budget 1,000 tokens')
  // The provider stream was cancelled, not read to its end.
  expect(model.abortedOnClose).toEqual([true])
  expect(model.requests).toHaveLength(2)
  const second = JSON.stringify(model.requests[1]!.messages)
  expect(second).not.toContain(RUNAWAY)
  expect(second).toContain('[harness] Your last reply was stopped after about 1,')
  expect(keptRunaway(state)).toBe(false)
  expect(state.messages.at(-1)).toMatchObject({ role: 'assistant', content: 'Done: the total is 42.' })
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'completed' })
})

test('the time budget counts from the first token, not from the request', async () => {
  const clock = fakeClock()
  const model = runawayModel({ chunks: 10, chunkChars: 20, clock, secondsPerChunk: 50 })
  const { events } = await collect('test-model', model.llm, { replyBudget: { maxReasoningTokens: 100_000, maxSeconds: 120 }, now: clock.now })
  const [retry] = budgetRetries(events)
  expect(retry?.type === 'provider_retry' && retry.error).toContain('(reply budget 2 min)')
  // First chunk at 0 s, then 50, 100, 150: the fourth chunk crosses 120 s.
  expect(retry?.type === 'provider_retry' && retry.error).toContain('over 2.5 min')
})

test('a reply that has started answering or calling a tool is not cut', async () => {
  const llm: LlmClient = {
    async *stream(): AsyncGenerator<LlmDelta> {
      yield { content: 'Here is the total: ' }
      for (let index = 0; index < 30; index += 1) yield { thinking: RUNAWAY.repeat(20) }
      yield { content: '42.', usage: { inputTokens: 10, outputTokens: 5 } }
    },
  }
  const { events } = await collect('test-model', llm, { replyBudget: { maxReasoningTokens: 256, maxSeconds: 600 } })
  expect(budgetRetries(events)).toHaveLength(0)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'completed' })
})

test(`after ${MAX_REPLY_BUDGET_STOPS} stops the next runaway reply ends the turn, with nothing of it kept`, async () => {
  const model = runawayModel({ chunks: 40, chunkChars: 400, recoverOnNote: false })
  const { events, state } = await collect('test-model', model.llm, { replyBudget: { maxReasoningTokens: 1_000, maxSeconds: 600 } })
  expect(budgetRetries(events)).toHaveLength(MAX_REPLY_BUDGET_STOPS)
  expect(model.requests).toHaveLength(MAX_REPLY_BUDGET_STOPS + 1)
  const stopText = events.filter(event => event.type === 'text').map(event => event.type === 'text' ? event.text : '').join('')
  expect(stopText).toContain('[Stopped: Reply stopped')
  expect(stopText).toContain('runtime.reply_budget')
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'output_limit' })
  expect(keptRunaway(state)).toBe(false)
  expect(state.messages.filter(message => typeof message.content === 'string' && message.content.startsWith('[harness] Your last reply'))).toHaveLength(MAX_REPLY_BUDGET_STOPS)
})

test("Bonsai's card sets a budget; another model, or the setting turned off, has none", async () => {
  // 40 chunks of 1,000 characters is about 10,000 tokens, past the card's 8,192.
  const bonsai = runawayModel({ chunks: 40, chunkChars: 1_000 })
  expect(budgetRetries((await collect(BONSAI, bonsai.llm)).events)).toHaveLength(1)

  const other = runawayModel({ chunks: 40, chunkChars: 1_000 })
  const plain = await collect('gpt-4o', other.llm)
  expect(budgetRetries(plain.events)).toHaveLength(0)
  expect(plain.state.messages.at(-1)).toMatchObject({ role: 'assistant', content: 'Finally an answer.' })

  const off = runawayModel({ chunks: 40, chunkChars: 1_000 })
  expect(budgetRetries((await collect(BONSAI, off.llm, { replyBudget: null })).events)).toHaveLength(0)
})

test('cancelling a long reply ends the turn as aborted, not as a budget stop', async () => {
  const controller = new AbortController()
  let chunks = 0
  const llm: LlmClient = {
    async *stream(_request, signal): AsyncGenerator<LlmDelta> {
      while (!signal?.aborted) {
        chunks += 1
        if (chunks === 3) controller.abort()
        yield { thinking: RUNAWAY.repeat(10) }
      }
    },
  }
  const { events } = await collect('test-model', llm, { replyBudget: { maxReasoningTokens: 100_000, maxSeconds: 600 }, signal: controller.signal })
  expect(budgetRetries(events)).toHaveLength(0)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'aborted' })
})

test('a daemon turn applies the configured budget per model and shows the stop as a warning', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-reply-budget-'))
  try {
    const model = runawayModel({ chunks: 40, chunkChars: 400 })
    const asked: string[] = []
    const runner = new AgentTurnRunner({
      llm: model.llm,
      model: BONSAI,
      replyBudgetFor: candidate => {
        asked.push(candidate)
        return { maxReasoningTokens: 1_000, maxSeconds: 600 }
      },
    })
    const session: DaemonSession = {
      activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: root, extra: {}, id: 'budget',
      interactionMode: 'code', sessionKey: 'budget', lastActive: 0, messages: [], metadata: {},
      model: BONSAI, planMode: false, status: 'working', thinkingContent: [],
      toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0, workspace: root,
    }
    const events: Array<{ type: string; payload?: Record<string, unknown> }> = []
    for await (const event of runner.run(session, 'Sum the invoices.', new AbortController().signal)) events.push(event as never)
    expect(asked).toEqual([BONSAI])
    const warning = events.find(event => event.type === 'notification' && (event.payload?.retry as { cause?: string } | undefined)?.cause === 'reply_budget')
    expect(warning?.payload).toMatchObject({ level: 'warning' })
    expect(String(warning?.payload?.message)).toContain('reply budget 1,000 tokens')
    expect(model.requests).toHaveLength(2)
    expect(JSON.stringify(session.messages)).not.toContain(RUNAWAY)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
