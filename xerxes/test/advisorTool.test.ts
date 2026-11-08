// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { LlmClient, LlmDelta } from '../src/llms/client.js'
import { AdvisorBudget, parseAdvisorConfig, type AdvisorConfig } from '../src/runtime/advisorBudget.js'
import type { OutboundGate, OutboundRequestSummary } from '../src/security/boundaryTypes.js'
import {
  ADVISOR_SYSTEM_PROMPT,
  buildCheckFailureAdvisorRequest,
  createAdvisorClientFromLlm,
  registerAdvisorTool,
  type AdvisorClient,
  type AdvisorConsultRecord,
} from '../src/tools/advisorTool.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

type ClientInput = Parameters<AdvisorClient['complete']>[0]
type ClientOutput = Awaited<ReturnType<AdvisorClient['complete']>>

function config(overrides: Partial<AdvisorConfig> = {}): AdvisorConfig {
  return {
    profile: 'big',
    maxCalls: 5,
    maxUsd: 1,
    autoOnCheckFail: false,
    timeoutMs: 5000,
    maxInputChars: 40_000,
    ...overrides,
  }
}

function call(inputs: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name: 'AskAdvisor', arguments: inputs } }
}

function fakeClient(
  respond: (input: ClientInput) => Promise<ClientOutput> | ClientOutput = () => ({
    text: 'Check the lockfile.',
    inputTokens: 120,
    outputTokens: 40,
    costUsd: 0.0123,
    model: 'big-model',
  }),
): AdvisorClient & { readonly inputs: ClientInput[] } {
  const inputs: ClientInput[] = []
  return {
    inputs,
    complete(input) {
      inputs.push(input)
      return Promise.resolve(respond(input))
    },
  }
}

function setup(options: {
  readonly config?: AdvisorConfig
  readonly client?: AdvisorClient
  readonly gate?: OutboundGate
  readonly consults?: AdvisorConsultRecord[]
} = {}) {
  const cfg = options.config ?? config()
  const registry = new ToolRegistry()
  const budget = new AdvisorBudget(cfg)
  const client = options.client ?? fakeClient()
  const registered = registerAdvisorTool(registry, {
    config: cfg,
    client,
    ...(options.gate === undefined ? {} : { gate: options.gate }),
    trust: 'cloud',
    budgetFor: () => budget,
    ...(options.consults === undefined ? {} : { onConsult: record => options.consults?.push(record) }),
  })
  return { registry, budget, registered, client }
}

const context = { metadata: {}, sessionId: 's1' }

test('the tool is not registered without an advisor profile', () => {
  const { profile: _profile, ...withoutProfile } = config()
  const { registry, registered } = setup({ config: withoutProfile })
  expect(registered).toBe(false)
  expect(registry.definitions()).toHaveLength(0)
})

test('the tool registers with a short description when a profile is set', () => {
  const { registry, registered } = setup()
  expect(registered).toBe(true)
  const [definition] = registry.definitions()
  expect(definition?.function.name).toBe('AskAdvisor')
  expect(definition?.function.description.length).toBeLessThanOrEqual(300)
})

test('happy path returns the answer and a footer with tokens, cost and calls left', async () => {
  const consults: AdvisorConsultRecord[] = []
  const { registry, client, budget } = setup({ consults })
  const result = await registry.execute(call({ question: 'Why does the install hang?', context: 'bun install output', kind: 'review' }), context)
  expect(result.startsWith('Check the lockfile.')).toBe(true)
  const lines = result.trimEnd().split('\n')
  expect(lines.at(-1)).toBe('[advisor big-model: 120/40 tokens, $0.0123, 4 calls left]')
  const sent = (client as ReturnType<typeof fakeClient>).inputs
  expect(sent).toHaveLength(1)
  expect(sent[0]?.system).toBe(ADVISOR_SYSTEM_PROMPT)
  expect(sent[0]?.user).toContain('Request type: review')
  expect(sent[0]?.user).toContain('Why does the install hang?')
  expect(sent[0]?.user).toContain('bun install output')
  expect(budget.snapshot().calls).toBe(1)
  expect(consults).toHaveLength(1)
  expect(consults[0]).toMatchObject({ sessionId: 's1', model: 'big-model', inputTokens: 120, outputTokens: 40, costUsd: 0.0123 })
})

test('a gate denial returns a tool result and never calls the client', async () => {
  const seen: OutboundRequestSummary[] = []
  const gate: OutboundGate = {
    check(request) {
      seen.push(request)
      return Promise.resolve({ allow: false, reason: 'private mode blocks cloud models' })
    },
  }
  const { registry, client, budget } = setup({ gate })
  const result = await registry.execute(call({ question: 'help' }), { ...context, agentId: 'default' })
  expect(result).toBe('Advisor not consulted: private mode blocks cloud models. Continue locally.')
  expect((client as ReturnType<typeof fakeClient>).inputs).toHaveLength(0)
  expect(budget.snapshot().calls).toBe(0)
  expect(seen).toHaveLength(1)
  expect(seen[0]).toMatchObject({ purpose: 'advisor', trust: 'cloud', sessionId: 's1', provider: 'big' })
})

test('the gate sees exactly the text the client receives, with its UTF-8 size', async () => {
  const seen: OutboundRequestSummary[] = []
  const gate: OutboundGate = {
    check(request) {
      seen.push(request)
      return Promise.resolve({ allow: true, decision: 'sent' })
    },
  }
  const { registry, client } = setup({ gate })
  await registry.execute(call({ question: 'naïve café ☕', context: 'ctx' }), context)
  const sent = (client as ReturnType<typeof fakeClient>).inputs[0]
  expect(sent).toBeDefined()
  expect(seen[0]?.text).toBe(`${sent?.system}\n\n${sent?.user}`)
  expect(seen[0]?.bytes).toBe(Buffer.byteLength(seen[0]?.text ?? '', 'utf8'))
  expect(seen[0]?.bytes).toBeGreaterThan(seen[0]?.text.length ?? 0)
})

test('a throwing gate fails closed without calling the client', async () => {
  const gate: OutboundGate = { check: () => Promise.reject(new Error('disclosure log unwritable')) }
  const { registry, client } = setup({ gate })
  const result = await registry.execute(call({ question: 'help' }), context)
  expect(result).toContain('Advisor not consulted')
  expect(result).toContain('Continue locally.')
  expect((client as ReturnType<typeof fakeClient>).inputs).toHaveLength(0)
})

test('the call limit stops further consultation', async () => {
  const { registry, client } = setup({ config: config({ maxCalls: 2 }) })
  await registry.execute(call({ question: 'one' }), context)
  const second = await registry.execute(call({ question: 'two' }), context)
  expect(second).toContain('0 calls left')
  const third = await registry.execute(call({ question: 'three' }), context)
  expect(third).toMatch(/^Advisor not consulted: advisor call limit reached .*Continue locally\.$/)
  expect((client as ReturnType<typeof fakeClient>).inputs).toHaveLength(2)
})

test('the USD limit stops further consultation', async () => {
  const expensive = fakeClient(() => ({ text: 'answer', inputTokens: 1000, outputTokens: 500, costUsd: 0.06, model: 'big-model' }))
  const { registry, budget } = setup({ config: config({ maxUsd: 0.05 }), client: expensive })
  const first = await registry.execute(call({ question: 'one' }), context)
  expect(first).toContain('0 calls left')
  const second = await registry.execute(call({ question: 'two' }), context)
  expect(second).toMatch(/^Advisor not consulted: advisor spend limit reached .*Continue locally\.$/)
  expect(expensive.inputs).toHaveLength(1)
  expect(budget.snapshot().spentUsd).toBeCloseTo(0.06)
})

test('a timeout returns an error result and aborts the client', async () => {
  let clientSignal: AbortSignal | undefined
  const hanging = fakeClient(
    input =>
      new Promise<ClientOutput>((_, reject) => {
        clientSignal = input.signal
        input.signal.addEventListener('abort', () => reject(input.signal.reason), { once: true })
      }),
  )
  const { registry, budget } = setup({ config: config({ timeoutMs: 30 }), client: hanging })
  const result = await registry.execute(call({ question: 'slow' }), context)
  expect(result).toMatch(/^Advisor failed: .*did not answer within .*continue locally\.$/)
  expect(clientSignal?.aborted).toBe(true)
  expect(budget.snapshot()).toMatchObject({ calls: 1, failedCalls: 1 })
})

test('a client that ignores its signal still times out', async () => {
  const stuck = fakeClient(() => new Promise<ClientOutput>(() => undefined))
  const { registry } = setup({ config: config({ timeoutMs: 30 }), client: stuck })
  const result = await registry.execute(call({ question: 'slow' }), context)
  expect(result).toContain('Advisor failed')
})

test('a provider error is an error result, not an exception', async () => {
  const broken = fakeClient(() => {
    throw new Error('503 overloaded')
  })
  const { registry, budget } = setup({ client: broken })
  const result = await registry.execute(call({ question: 'help' }), context)
  expect(result).toBe('Advisor failed: 503 overloaded. No answer was received; continue locally.')
  expect(budget.snapshot().failedCalls).toBe(1)
})

test('turn cancellation propagates and is not charged', async () => {
  let clientSignal: AbortSignal | undefined
  const hanging = fakeClient(
    input =>
      new Promise<ClientOutput>((_, reject) => {
        clientSignal = input.signal
        input.signal.addEventListener('abort', () => reject(input.signal.reason), { once: true })
      }),
  )
  const { registry, budget } = setup({ client: hanging })
  const handler = registry.get('AskAdvisor')
  expect(handler).toBeDefined()
  const controller = new AbortController()
  const pending = handler?.({ question: 'help' }, context, controller.signal)
  setTimeout(() => controller.abort(new Error('user cancelled')), 10)
  await expect(pending).rejects.toThrow('user cancelled')
  expect(clientSignal?.aborted).toBe(true)
  expect(budget.snapshot().calls).toBe(0)
})

test('an already-cancelled turn never reaches the gate or the client', async () => {
  const { registry, client } = setup()
  const controller = new AbortController()
  controller.abort(new Error('stop'))
  await expect(registry.get('AskAdvisor')?.({ question: 'help' }, context, controller.signal)).rejects.toThrow('stop')
  expect((client as ReturnType<typeof fakeClient>).inputs).toHaveLength(0)
})

test('input is clipped so system plus user text stays within maxInputChars', async () => {
  const seen: OutboundRequestSummary[] = []
  const gate: OutboundGate = {
    check(request) {
      seen.push(request)
      return Promise.resolve({ allow: true, decision: 'sent' })
    },
  }
  const { registry, client } = setup({ config: config({ maxInputChars: 2000 }), gate })
  const result = await registry.execute(call({ question: 'q'.repeat(3000), context: 'c'.repeat(30_000) }), context)
  const sent = (client as ReturnType<typeof fakeClient>).inputs[0]
  expect((sent?.system.length ?? 0) + (sent?.user.length ?? 0)).toBeLessThanOrEqual(2000)
  expect(seen[0]?.text.length).toBeLessThanOrEqual(2002)
  expect(result).toContain('[input clipped to 2000 characters before sending]')
  expect(result.trimEnd().split('\n').at(-1)).toMatch(/^\[advisor /)
})

test('short input is sent unclipped', async () => {
  const { registry, client } = setup()
  const result = await registry.execute(call({ question: 'short', context: 'also short' }), context)
  expect(result).not.toContain('clipped')
  expect((client as ReturnType<typeof fakeClient>).inputs[0]?.user).toContain('also short')
})

test('an unknown price is reported as unknown and counts against calls only', async () => {
  const unpriced = fakeClient(() => ({ text: 'answer', inputTokens: 10, outputTokens: 5, model: 'mystery' }))
  const { registry, budget } = setup({ client: unpriced })
  const result = await registry.execute(call({ question: 'help' }), context)
  expect(result.trimEnd().split('\n').at(-1)).toBe('[advisor mystery: 10/5 tokens, $unknown, 4 calls left]')
  expect(budget.snapshot()).toMatchObject({ calls: 1, spentUsd: 0, unknownCostCalls: 1, costComplete: false })
  expect(budget.canCall()).toEqual({ ok: true })
})

test('invalid input is rejected', async () => {
  const { registry, client } = setup()
  await expect(registry.get('AskAdvisor')?.({ question: '   ' }, context)).rejects.toThrow(ValidationError)
  await expect(registry.get('AskAdvisor')?.({ question: 'x'.repeat(8001) }, context)).rejects.toThrow(ValidationError)
  await expect(registry.get('AskAdvisor')?.({ question: 'ok', kind: 'chat' }, context)).rejects.toThrow(ValidationError)
  expect((client as ReturnType<typeof fakeClient>).inputs).toHaveLength(0)
})

test('parseAdvisorConfig applies defaults, lets env win, and rejects bad values', () => {
  expect(parseAdvisorConfig({}, {})).toEqual({
    maxCalls: 5,
    maxUsd: 1,
    autoOnCheckFail: false,
    timeoutMs: 120_000,
    maxInputChars: 40_000,
  })
  const fromFile = parseAdvisorConfig({ advisor: { profile: 'file-profile', model: 'm', max_calls: 3, max_usd: 0.5, auto_on_check_fail: true } }, {})
  expect(fromFile).toMatchObject({ profile: 'file-profile', model: 'm', maxCalls: 3, maxUsd: 0.5, autoOnCheckFail: true })
  expect(parseAdvisorConfig({ advisor: { profile: 'file-profile' } }, { XERXES_ADVISOR_PROFILE: 'env-profile' }).profile).toBe('env-profile')
  expect(() => parseAdvisorConfig({ advisor: { max_calls: 0 } }, {})).toThrow('advisor.max_calls')
  expect(() => parseAdvisorConfig({ advisor: { max_calls: 1.5 } }, {})).toThrow('advisor.max_calls')
  expect(() => parseAdvisorConfig({ advisor: { max_usd: -1 } }, {})).toThrow('advisor.max_usd')
  expect(() => parseAdvisorConfig({ advisor: { max_usd: '1' } }, {})).toThrow('advisor.max_usd')
  expect(() => parseAdvisorConfig({ advisor: { auto_on_check_fail: 'yes' } }, {})).toThrow('advisor.auto_on_check_fail')
  expect(() => parseAdvisorConfig({ advisor: 'big' }, {})).toThrow('advisor')
  expect(() => parseAdvisorConfig({ advisor: { profile: 7 } }, {})).toThrow('advisor.profile')
})

test('AdvisorBudget tracks both limits and failed calls', () => {
  const budget = new AdvisorBudget({ maxCalls: 3, maxUsd: 0.1 })
  expect(budget.canCall()).toEqual({ ok: true })
  budget.record({ inputTokens: 0, outputTokens: 0, failed: true })
  budget.record({ inputTokens: 100, outputTokens: 50, costUsd: 0.1 })
  expect(budget.snapshot()).toMatchObject({ calls: 2, failedCalls: 1, callsLeft: 0, spentUsd: 0.1 })
  const blocked = budget.canCall()
  expect(blocked.ok).toBe(false)
  expect(() => budget.record({ inputTokens: -1, outputTokens: 0 })).toThrow(ValidationError)
})

test('buildCheckFailureAdvisorRequest carries the task and findings within the tool limits', () => {
  const request = buildCheckFailureAdvisorRequest('claimed file missing: report.xlsx', 'Build the Q3 report')
  expect(request.question.length).toBeLessThanOrEqual(8000)
  expect(request.context).toContain('Build the Q3 report')
  expect(request.context).toContain('claimed file missing: report.xlsx')
  const huge = buildCheckFailureAdvisorRequest('f'.repeat(100_000), 't')
  expect(huge.context.length).toBeLessThanOrEqual(32_000)
})

function streamOnly(deltas: readonly LlmDelta[]): LlmClient {
  return {
    async *stream() {
      for (const delta of deltas) yield delta
    },
  }
}

test('createAdvisorClientFromLlm collects a stream and prices it only when usage is known', async () => {
  const llm = streamOnly([{ content: 'Use ' }, { content: 'the lockfile.' }, { usage: { inputTokens: 200, outputTokens: 30 } }])
  const client = createAdvisorClientFromLlm(llm, 'big-model', 1024, (model, input, output) => (model === 'big-model' ? (input + output) / 1000 : undefined))
  const answer = await client.complete({ system: 's', user: 'u', maxTokens: 2048, signal: new AbortController().signal })
  expect(answer).toEqual({ text: 'Use the lockfile.', inputTokens: 200, outputTokens: 30, costUsd: 0.23, model: 'big-model' })

  const noUsage = createAdvisorClientFromLlm(streamOnly([{ content: 'answer' }]), 'big-model', 1024, () => 0)
  const unpriced = await noUsage.complete({ system: 's', user: 'u', maxTokens: 2048, signal: new AbortController().signal })
  expect(unpriced.costUsd).toBeUndefined()
  expect(unpriced.inputTokens).toBe(0)
})

test('createAdvisorClientFromLlm caps maxTokens and passes system and user messages', async () => {
  const requests: unknown[] = []
  const llm: LlmClient = {
    async *stream(request) {
      requests.push(request)
      yield { content: 'ok' }
    },
  }
  const client = createAdvisorClientFromLlm(llm, 'big-model', 512, () => undefined)
  await client.complete({ system: 'SYS', user: 'USR', maxTokens: 2048, signal: new AbortController().signal })
  expect(requests[0]).toMatchObject({
    model: 'big-model',
    maxTokens: 512,
    messages: [
      { role: 'system', content: 'SYS' },
      { role: 'user', content: 'USR' },
    ],
  })
})
