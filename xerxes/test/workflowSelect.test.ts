// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  runWorkflowScript,
  type WorkflowAgentPort,
  type WorkflowAgentRequest,
  type WorkflowRunResult,
} from '../src/tools/claudeTools/workflowScript.js'

interface JudgePort extends WorkflowAgentPort {
  readonly requests: WorkflowAgentRequest[]
  readonly corrections: string[]
}

interface Selection {
  readonly index: number
  readonly value: unknown
  readonly letter: string
  readonly fallback: boolean
  readonly reason?: string
  readonly judgeReason?: string
  readonly mapping: readonly { readonly letter: string; readonly index: number }[]
  readonly scores?: readonly { readonly index: number; readonly ok: boolean; readonly score?: number; readonly error?: string }[]
}

function port(answer: string | ((request: WorkflowAgentRequest) => string) = '{"choice":"A","reason":"Best fit"}'): JudgePort {
  const output = (request: WorkflowAgentRequest) => typeof answer === 'function' ? answer(request) : answer
  const fake: JudgePort = {
    requests: [], corrections: [],
    async run(request) {
      fake.requests.push(request)
      return { id: 'judge', status: 'completed', output: output(request), tokens: 1 }
    },
    async correct(id, message, request) {
      fake.corrections.push(message)
      return { id, status: 'completed', output: output(request), tokens: 1 }
    },
    closeAll() {},
  }
  return fake
}

function chooseValue(request: WorkflowAgentRequest, value: string): string {
  const candidate = [...request.prompt.matchAll(/Candidate ([A-Z]):\n([^\n]*)/g)].find(match => match[2] === JSON.stringify(value))
  if (!candidate?.[1]) throw new Error(`Missing candidate ${value}`)
  return JSON.stringify({ choice: candidate[1], reason: 'Best fit' })
}

function run(script: string, judge = port(), args?: unknown): Promise<WorkflowRunResult> {
  return runWorkflowScript({ name: 'Blind selection', concurrency: 3, maxAgents: 10, port: judge, script, ...(args === undefined ? {} : { args }) })
}

function selection(result: WorkflowRunResult): Selection {
  expect({ status: result.status, error: result.error }).toEqual({ status: 'completed', error: undefined })
  return result.result as Selection
}

test('select shows one judge only stable values under anonymous letters, never candidate metadata', async () => {
  const judge = port()
  const result = await run(`return await select(args, { judge: 'Choose the clearest answer', seed: 'blind', judgeOptions: { model: 'judge-model', profile: 'judge-profile', label: 'must-not-be-used', schema: { type: 'string' } } })`, judge, [
    { label: 'original-author', model: 'secret-model', source: 'private-source', value: { z: 2, a: { z: 4, a: 3 } } },
    { label: 'new-author', model: 'other-model', source: 'other-source', value: 'Another answer' },
  ])
  const chosen = selection(result)
  expect(judge.requests).toHaveLength(1)
  const request = judge.requests[0]!
  expect(request.prompt).toContain('Choose the clearest answer')
  expect(request.prompt).toContain('Candidate A:\n')
  expect(request.prompt).toContain('Candidate B:\n')
  expect(request.prompt).toContain('{"a":{"a":3,"z":4},"z":2}')
  expect(request.prompt).toContain('"Another answer"')
  for (const hidden of ['original-author', 'secret-model', 'private-source', 'new-author', 'other-model', 'other-source', 'must-not-be-used', 'judge-model', 'judge-profile']) expect(request.prompt).not.toContain(hidden)
  expect(request).toMatchObject({ label: 'select-1-judge', model: 'judge-model', profile: 'judge-profile' })
  expect(request.schema).toEqual({ type: 'object', properties: { choice: { type: 'string', enum: ['A', 'B'] }, reason: { type: 'string' } }, required: ['choice', 'reason'], additionalProperties: false })
  expect(chosen).toMatchObject({ letter: 'A', fallback: false, judgeReason: 'Best fit' })
  expect(chosen.value).toEqual([{ z: 2, a: { z: 4, a: 3 } }, 'Another answer'][chosen.index])
  expect(chosen.scores).toBeUndefined()
})

test('an explicit seed produces the same shuffled permutation across calls and runs', async () => {
  const script = `return await select(args, { judge: 'Pick one', seed: 'reproducible' })`
  const candidates = ['zero', 'one', 'two', 'three', 'four', 'five']
  const first = selection(await run(script, port(), candidates))
  const second = selection(await run(script, port(), candidates))
  expect(first.mapping).toEqual(second.mapping)
  expect(first.mapping.map(entry => entry.letter)).toEqual(['A', 'B', 'C', 'D', 'E', 'F'])
  expect(first.mapping.map(entry => entry.index).toSorted()).toEqual([0, 1, 2, 3, 4, 5])
  expect(first.mapping.map(entry => entry.index)).not.toEqual([0, 1, 2, 3, 4, 5])
  const repeated = await run(`return await parallel([() => select(args, { judge: 'Pick one', seed: 'reproducible' }), () => select(args, { judge: 'Pick one', seed: 'reproducible' })])`, port(), candidates)
  expect((repeated.result as Selection[]).map(item => item.mapping)).toEqual([first.mapping, first.mapping])
})

test('selection commits before any scorer runs and reveals all scores only afterwards', async () => {
  const judge = port(request => chooseValue(request, 'alternative'))
  const result = await run(`
    const chosen = await select(['baseline', 'alternative'], {
      judge: 'Pick the better answer', seed: 'commit',
      score: async value => { log('scorer ran', value); await Promise.resolve(); return value === 'baseline' ? 31415 : { ok: true, score: 93412 } },
    })
    log('revealed scores', chosen.scores)
    return chosen
  `, judge)
  const chosen = selection(result)
  expect(chosen).toMatchObject({ index: 1, value: 'alternative', fallback: false, scores: [{ index: 0, ok: true, score: 31415 }, { index: 1, ok: true, score: 93412 }] })
  expect(result.logs[0]).toBe(`select: committed Candidate ${chosen.letter}`)
  expect(result.logs.slice(1, 3)).toEqual(['scorer ran baseline', 'scorer ran alternative'])
  expect(result.logs[3]).toStartWith('revealed scores ')
  expect(judge.requests).toHaveLength(1)
  expect(judge.requests[0]!.prompt).not.toContain('31415')
  expect(judge.requests[0]!.prompt).not.toContain('93412')
  expect(judge.requests[0]!.prompt).not.toContain('scorer')
})

test('a failed judge explicitly falls back to the requested baseline and still scores after commitment', async () => {
  const judge = port(() => { throw new Error('judge provider unavailable') })
  const result = await run(`return await select(['other', 'safe'], { judge: 'Pick one', baseline: 1, score: async value => { log('score', value); return 1 } })`, judge)
  const chosen = selection(result)
  expect(chosen).toMatchObject({ index: 1, value: 'safe', fallback: true })
  expect(chosen.reason).toContain('judge provider unavailable')
  expect(chosen.mapping.find(entry => entry.index === 1)?.letter).toBe(chosen.letter)
  expect(result.logs[0]).toBe(`select: committed Candidate ${chosen.letter}`)
  expect(result.logs.slice(1, 3)).toEqual(['score other', 'score safe'])
  expect(result.logs.at(-1)).toBe(`select: fell back to baseline because ${chosen.reason}`)
  expect(chosen.scores).toEqual([{ index: 0, ok: true, score: 1 }, { index: 1, ok: true, score: 1 }])
  expect(result.ledger).toMatchObject({ started: 1, failed: 1, ok: false })
})

test.each(['Z', 'a', 'A or B', ' A ', 'AA'])('an invalid judge choice %j never gets loosely parsed', async choice => {
  const judge = port(JSON.stringify({ choice, reason: 'Invalid choice' }))
  const result = await run(`return await select(['safe', 'new'], { judge: 'Pick one', seed: 'invalid' })`, judge)
  const chosen = selection(result)
  expect(chosen).toMatchObject({ index: 0, value: 'safe', fallback: true })
  expect(chosen.reason).toMatch(/invalid|schema/)
  expect(result.logs).toContain(`select: fell back to baseline because ${chosen.reason}`)
  expect(judge.requests).toHaveLength(1)
  expect(judge.corrections).toHaveLength(1)
})

test('a chosen score below the baseline explicitly falls back and retains the judge reason and all scores', async () => {
  const judge = port(request => chooseValue(request, 'weaker'))
  const result = await run(`return await select(['safe', 'weaker', 'unpicked'], { judge: 'Pick one', score: async value => value === 'safe' ? 10 : value === 'weaker' ? 2 : 20 })`, judge)
  const chosen = selection(result)
  expect(chosen).toMatchObject({ index: 0, value: 'safe', fallback: true, judgeReason: 'Best fit', scores: [{ index: 0, ok: true, score: 10 }, { index: 1, ok: true, score: 2 }, { index: 2, ok: true, score: 20 }] })
  expect(chosen.reason).toContain('below baseline')
  expect(result.logs[0]).not.toBe(`select: committed Candidate ${chosen.letter}`)
  expect(result.logs.at(-1)).toBe(`select: fell back to baseline because ${chosen.reason}`)
})

test.each([
  ['return { ok: false, score: 100 }', 'reported ok: false'],
  ['throw new Error("verification failed")', 'verification failed'],
  ['return NaN', 'finite'],
])('a failed chosen scorer (%s) falls back without discarding other scores', async (scoring, error) => {
  const judge = port(request => chooseValue(request, 'new'))
  const result = await run(`return await select(['safe', 'new', 'other'], { judge: 'Pick one', score: async value => { if (value === 'new') { ${scoring} }; return 10 } })`, judge)
  const chosen = selection(result)
  expect(chosen).toMatchObject({ index: 0, value: 'safe', fallback: true })
  expect(chosen.reason).toContain(error)
  expect(chosen.scores?.map(item => item.ok)).toEqual([true, false, true])
  expect(result.logs.at(-1)).toBe(`select: fell back to baseline because ${chosen.reason}`)
})

test('a tie keeps the judge choice, and a failed baseline scorer does not override a passing choice', async () => {
  for (const baselineScore of ['10', '{ ok: false, score: 20 }']) {
    const judge = port(request => chooseValue(request, 'new'))
    const chosen = selection(await run(`return await select(['safe', 'new'], { judge: 'Pick one', score: async value => value === 'safe' ? ${baselineScore} : 10 })`, judge))
    expect(chosen).toMatchObject({ index: 1, value: 'new', fallback: false })
    expect(chosen.reason).toBeUndefined()
  }
})

test('each select judge has a unique label and counts normally in the workflow ledger', async () => {
  const judge = port()
  const result = await runWorkflowScript({
    name: 'Unique selection judges', concurrency: 2, maxAgents: 2, expectAgents: 2, port: judge,
    script: `return await parallel([1, 2].map(() => () => select(['a', 'b'], { judge: 'Pick one', judgeOptions: { label: 'duplicate' } })))`,
  })
  expect(result.status).toBe('completed')
  expect(judge.requests.map(request => request.label)).toEqual(['select-1-judge', 'select-2-judge'])
  expect(result.ledger).toMatchObject({ expected: 2, started: 2, completed: 2, failed: 0, duplicateLabels: [], ok: true })
})

test('one candidate commits directly without a judge, then optionally scores it', async () => {
  const judge = port()
  const result = await run(`return await select([{ value: { answer: 42 }, label: 'only' }], { judge: 'Unused', score: async value => { log('scored only'); return value.answer } })`, judge)
  expect(selection(result)).toEqual({ index: 0, value: { answer: 42 }, letter: 'A', fallback: false, mapping: [{ letter: 'A', index: 0 }], scores: [{ index: 0, ok: true, score: 42 }] })
  expect(judge.requests).toEqual([])
  expect(result.logs).toEqual(['select: committed Candidate A', 'scored only'])
  expect(result.ledger).toMatchObject({ started: 0, completed: 0, ok: true })
})

test('select is appended without moving any existing sandbox parameter', async () => {
  const result = await run(`return [arguments[0] === agent, arguments[1] === parallel, arguments[2] === pipeline, arguments[3] === phase, arguments[4] === log, arguments[5] === args, arguments[6] === budget, arguments[7] === select]`, port(), { existing: 'args' })
  expect(result.result).toEqual([true, true, true, true, true, true, true, true])
})

test('invalid select inputs fail before a judge starts', async () => {
  for (const [candidates, options, message] of [
    ['[]', `{ judge: 'Pick one' }`, 'between 1 and 26'],
    ['Array.from({ length: 27 }, (_, i) => i)', `{ judge: 'Pick one' }`, 'between 1 and 26'],
    [`['a', 'b']`, `{ judge: 'Pick one', baseline: 2 }`, 'baseline'],
    [`['a', 'b']`, `{ judge: 'Pick one', baseline: 0.5 }`, 'baseline'],
    [`['a', 'b']`, `{ judge: '' }`, 'judge'],
    [`['a', 'b']`, `{ judge: 'Pick one', score: 1 }`, 'score'],
    [`['a', 'b']`, `{ judge: 'Pick one', seed: 3 }`, 'seed'],
  ]) {
    const judge = port()
    const result = await run(`return await select(${candidates}, ${options})`, judge)
    expect(result.status).toBe('failed')
    expect(result.error).toContain(message!)
    expect(judge.requests).toEqual([])
  }
})

test('cancelling a select cancels its judge and cannot report a completed fallback', async () => {
  const controller = new AbortController()
  const judge: WorkflowAgentPort = {
    ...port(),
    async run(_request, signal) {
      setTimeout(() => controller.abort(new Error('Stop blind selection')), 1)
      return new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(signal.reason), { once: true }))
    },
  }
  const result = await runWorkflowScript({ name: 'Cancelled selection', concurrency: 1, maxAgents: 1, port: judge, signal: controller.signal, script: `return await select(['a', 'b'], { judge: 'Pick one' })` })
  expect(result.status).toBe('cancelled')
  expect(result.error).toBe('Stop blind selection')
  expect(result.ledger).toMatchObject({ started: 1, completed: 0, failed: 1, ok: false })
})
