// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  runWorkflowScript,
  structuredValue,
  workflowAgentRequest,
  type WorkflowAgentOutcome,
  type WorkflowAgentPort,
  type WorkflowAgentRequest,
} from '../src/tools/claudeTools/workflowScript.js'

interface FakePort extends WorkflowAgentPort {
  readonly requests: WorkflowAgentRequest[]
  readonly corrections: string[]
  closed: number
  peak: number
}

function fakePort(answer: (request: WorkflowAgentRequest, index: number) => Partial<WorkflowAgentOutcome> | Promise<Partial<WorkflowAgentOutcome>>, delayMs = 5): FakePort {
  let running = 0
  const port: FakePort = {
    requests: [],
    corrections: [],
    closed: 0,
    peak: 0,
    async run(request, signal) {
      const index = port.requests.push(request) - 1
      running++
      port.peak = Math.max(port.peak, running)
      try {
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, delayMs)
          signal.addEventListener('abort', () => { clearTimeout(timer); reject(signal.reason) }, { once: true })
        })
        const outcome = await answer(request, index)
        return { id: `agent-${index}`, status: 'completed', output: '', tokens: 10, ...outcome }
      } finally {
        running--
      }
    },
    async correct(id, message) {
      port.corrections.push(message)
      return { id, status: 'completed', output: '{"verdict":"real","score":3}', tokens: 5 }
    },
    closeAll() { port.closed++ },
  }
  return port
}

test('a script fans out by phase and gets every agent result back', async () => {
  const port = fakePort(request => ({ output: `read ${request.label}` }))
  const result = await runWorkflowScript({
    name: 'Review',
    concurrency: 4,
    maxAgents: 100,
    port,
    args: { slices: ['a', 'b', 'c'] },
    script: `
      phase('Read')
      const reads = await parallel(args.slices.map(s => () => agent('read ' + s, { label: s, model: 'fast-model' })))
      phase('Sum')
      const total = await agent('summarize ' + reads.join(', '), { label: 'sum' })
      log('done with', reads.length)
      return { reads, total }
    `,
  })
  expect(result.status).toBe('completed')
  expect(result.result).toEqual({ reads: ['read a', 'read b', 'read c'], total: 'read sum' })
  expect(result.phases).toEqual(['Read', 'Sum'])
  expect(port.requests.map(request => request.phase)).toEqual(['Read', 'Read', 'Read', 'Sum'])
  expect(port.requests[0]?.model).toBe('fast-model')
  expect(result.agents).toEqual({ started: 4, completed: 4, failed: 0, cancelled: 0 })
  expect(result.tokens).toBe(40)
  expect(result.logs).toContain('done with 3')
})

test('hundreds of agents run through the concurrency gate without a total cap', async () => {
  const port = fakePort((_request, index) => ({ output: String(index) }), 1)
  const result = await runWorkflowScript({
    name: 'Sweep',
    concurrency: 8,
    maxAgents: 10_000,
    port,
    script: `const out = await parallel(Array.from({ length: 300 }, (_, i) => () => agent('item ' + i))); return out.length`,
  })
  expect(result.status).toBe('completed')
  expect(result.result).toBe(300)
  expect(result.agents.completed).toBe(300)
  expect(port.peak).toBeLessThanOrEqual(8)
}, 30_000)

test('a structured reply is parsed, and a malformed one gets one correction round', async () => {
  const schema = { type: 'object', properties: { verdict: { type: 'string' }, score: { type: 'integer' } }, required: ['verdict', 'score'] }
  const port = fakePort((_request, index) => ({ output: index === 0 ? 'Here you go:\n```json\n{"verdict":"real","score":2}\n```' : 'not json at all' }))
  const result = await runWorkflowScript({
    name: 'Verify',
    concurrency: 2,
    maxAgents: 10,
    port,
    args: { schema },
    script: `return await parallel([() => agent('first', { schema: args.schema }), () => agent('second', { schema: args.schema })])`,
  })
  expect(result.result).toEqual([{ verdict: 'real', score: 2 }, { verdict: 'real', score: 3 }])
  expect(port.corrections).toHaveLength(1)
  expect(port.corrections[0]).toContain('JSON Schema')
})

test('a failed agent rejects agent(), becomes null in parallel, and is reported', async () => {
  const port = fakePort((_request, index) => index === 1 ? { status: 'failed', error: 'provider exploded' } : { output: 'ok' })
  const result = await runWorkflowScript({
    name: 'Mixed',
    concurrency: 4,
    maxAgents: 10,
    port,
    script: `
      const all = await parallel([() => agent('a'), () => agent('b', { label: 'bee' }), () => agent('c')])
      let caught = ''
      try { await agent('d', { label: 'dee' }) } catch (error) { caught = error.message }
      return { all, caught }
    `,
  })
  expect(result.status).toBe('completed')
  expect(result.result).toEqual({ all: ['ok', null, 'ok'], caught: '' })
  expect(result.agents.failed).toBe(1)
  expect(result.failures).toEqual([{ label: 'bee', error: 'provider exploded' }])
})

test('a script error comes back as a failed run with the message', async () => {
  const result = await runWorkflowScript({
    name: 'Broken',
    concurrency: 1,
    maxAgents: 1,
    port: fakePort(() => ({})),
    script: `const x = undefinedThing.value; return x`,
  })
  expect(result.status).toBe('failed')
  expect(result.error).toContain('undefinedThing')
})

test('stopping the turn cancels the run and every agent it started', async () => {
  const controller = new AbortController()
  const port = fakePort(() => ({ output: 'late' }), 10_000)
  const started = Date.now()
  setTimeout(() => controller.abort(new Error('Turn cancelled')), 150)
  const result = await runWorkflowScript({
    name: 'Long',
    concurrency: 4,
    maxAgents: 100,
    port,
    signal: controller.signal,
    script: `return await parallel(Array.from({ length: 10 }, (_, i) => () => agent('slow ' + i)))`,
  })
  expect(Date.now() - started).toBeLessThan(5_000)
  expect(result.status).toBe('cancelled')
  expect(result.error).toBe('Turn cancelled')
  expect(port.closed).toBe(1)
}, 15_000)

test('the runaway guard and the token budget stop further agents', async () => {
  const guarded = await runWorkflowScript({
    name: 'Guarded',
    concurrency: 2,
    maxAgents: 2,
    port: fakePort(() => ({ output: 'x' })),
    script: `return await parallel([1, 2, 3].map(i => () => agent('n' + i)))`,
  })
  expect(guarded.result).toEqual(['x', 'x', null])
  expect(guarded.logs.some(line => line.includes('agent limit (2)'))).toBe(true)

  const budgeted = await runWorkflowScript({
    name: 'Budgeted',
    concurrency: 1,
    maxAgents: 100,
    tokenBudget: 15,
    port: fakePort(() => ({ output: 'y' })),
    script: `const a = await agent('one'); const b = await agent('two'); let third = 'ran'; try { await agent('three') } catch (e) { third = e.message } return { a, b, third, total: budget.total, left: budget.remaining() }`,
  })
  expect(budgeted.result).toMatchObject({ a: 'y', b: 'y', total: 15, left: 0 })
  expect(String((budgeted.result as Record<string, unknown>).third)).toContain('token budget')
})

test('the script process holds no provider keys and console output lands in the logs', async () => {
  process.env.XERXES_TEST_SECRET_KEY = 'sk-should-not-leak'
  try {
    const result = await runWorkflowScript({
      name: 'Env',
      concurrency: 1,
      maxAgents: 1,
      port: fakePort(() => ({})),
      script: `console.log('hello', { n: 1 }); return process.env.XERXES_TEST_SECRET_KEY ?? 'absent'`,
    })
    expect(result.result).toBe('absent')
    expect(result.logs).toContain('hello {"n":1}')
  } finally {
    delete process.env.XERXES_TEST_SECRET_KEY
  }
})

test('agent options from the script are validated before anything spawns', () => {
  expect(() => workflowAgentRequest({ prompt: '' })).toThrow('non-empty prompt')
  expect(() => workflowAgentRequest({ prompt: 'x', isolation: 'container' })).toThrow('isolation')
  expect(() => workflowAgentRequest({ prompt: 'x', schema: [1] })).toThrow('schema')
  expect(workflowAgentRequest({ prompt: ' go ', title: 'T', provider_profile: 'openrouter', subagent_type: 'researcher' }))
    .toEqual({ prompt: 'go', label: 'T', profile: 'openrouter', type: 'researcher' })
  expect(structuredValue('[1,2]', { type: 'array' })).toEqual({ ok: true, value: [1, 2] })
  expect(structuredValue('nope', { type: 'object' })).toEqual({ ok: false, error: 'the reply held no JSON value' })
})

test('a script handed over as a function is called rather than rejected', async () => {
  for (const script of [
    `async function () {\n  phase('One')\n  return await agent('x')\n}`,
    `async () => { return await agent('y') }`,
  ]) {
    const result = await runWorkflowScript({ name: 'Wrapped', concurrency: 1, maxAgents: 5, port: fakePort(request => ({ output: `ran ${request.prompt}` })), script })
    expect(result.status).toBe('completed')
    expect(String(result.result)).toStartWith('ran ')
  }
})

test('a script that opens with a helper function runs as written, and a self-invoked one is not called twice', async () => {
  for (const script of [
    `function summarize(xs) {\n  return xs.join('|')\n}\nconst results = await parallel([() => agent('a'), () => agent('b')])\nreturn summarize(results)`,
    `function label(x) { return 'p-' + x }\nconst out = []\nfor (const x of ['a', 'b']) {\n  out.push(await agent(label(x)))\n}`,
    `(async () => { return await agent('c') })()`,
  ]) {
    const result = await runWorkflowScript({ name: 'Helpers', concurrency: 2, maxAgents: 5, port: fakePort(request => ({ output: `ran ${request.prompt}` })), script })
    expect({ status: result.status, error: result.error }).toEqual({ status: 'completed', error: undefined })
  }
  const helper = await runWorkflowScript({ name: 'Helpers', concurrency: 2, maxAgents: 5, port: fakePort(request => ({ output: `ran ${request.prompt}` })),
    script: `function summarize(xs) {\n  return xs.join('|')\n}\nconst results = await parallel([() => agent('a'), () => agent('b')])\nreturn summarize(results)` })
  expect(helper.result).toBe('ran a|ran b')
  const invoked = await runWorkflowScript({ name: 'Invoked', concurrency: 1, maxAgents: 5, port: fakePort(request => ({ output: `ran ${request.prompt}` })),
    script: `(async () => { return await agent('c') })()` })
  expect(invoked.result).toBe('ran c')
})

test('a run totals its published-price cost and counts agents without a price', async () => {
  const port = fakePort((_request, index) => index === 2 ? { output: 'x' } : { output: 'x', costUsd: 0.125 })
  const result = await runWorkflowScript({ name: 'Priced', concurrency: 3, maxAgents: 10, port, script: `return await parallel([1, 2, 3].map(i => () => agent('n' + i)))` })
  expect(result.cost).toEqual({ usd: 0.25, unpriced: 1 })
})
