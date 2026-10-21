// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import {
  createRecallScenario,
  formatRecallPlan,
  formatRecallReport,
  gradeRecallAnswer,
  runRecallEvaluation,
  type RecallScenario,
} from '../playground/recall.js'
import { parseRecallCliOptions, runRecallCli } from '../playground/recallCli.js'
import type { EvaluationEvent, EvaluationSessionPort, EvaluationSubmitRequest } from '../playground/types.js'

const tiny = { seed: 17, facts: 10, fillerTokens: 80 }

test('recall scenarios are seeded, unique and cover each fact kind before realistic filler and questions', () => {
  const scenario = createRecallScenario(tiny)
  expect(scenario).toEqual(createRecallScenario(tiny))
  expect(scenario).not.toEqual(createRecallScenario({ ...tiny, seed: 18 }))
  expect(scenario.facts).toHaveLength(10)
  expect(new Set(scenario.facts.map(fact => fact.id)).size).toBe(10)
  expect(new Set(scenario.facts.map(fact => fact.answer)).size).toBe(10)
  expect(new Set(scenario.facts.map(fact => fact.kind)).size).toBe(5)
  expect(scenario.estimatedFillerTokens).toBeGreaterThanOrEqual(tiny.fillerTokens)
  expect(scenario.filler.length).toBeGreaterThan(0)
  for (const fact of scenario.facts) {
    expect(scenario.plantPrompt).toContain(fact.answer)
    expect(fact.question).not.toContain(fact.answer)
    for (const filler of scenario.filler) {
      expect(filler.file.content).not.toContain(fact.answer)
      expect(filler.prompt).not.toContain(fact.answer)
    }
  }
  expect(scenario.filler[0]?.prompt).toContain('read')
  expect(scenario.filler[0]?.prompt).toContain(scenario.filler[0]!.file.content)
  expect(formatRecallPlan(scenario)).toContain('not a guaranteed compaction count')
})

test('recall grading normalizes case and whitespace but requires the entire exact answer', () => {
  const fact = createRecallScenario(tiny).facts.find(fact => fact.kind === 'reason')!
  expect(gradeRecallAnswer(fact, ` \n ${fact.answer.toUpperCase().replaceAll(' ', '\t  ')}  `).ok).toBeTrue()
  expect(gradeRecallAnswer(fact, `The answer is ${fact.answer}`).ok).toBeFalse()
  expect(gradeRecallAnswer(fact, fact.answer.slice(0, -1)).ok).toBeFalse()
  expect(gradeRecallAnswer(fact, `${fact.answer}.`).ok).toBeFalse()
  expect(gradeRecallAnswer(fact, '')).toMatchObject({ ok: false, expected: fact.answer, actual: '' })
})

test('invalid or excessive sizes and non-integer seeds are refused before execution', () => {
  for (const facts of [0, -1, 1.5, 1_001]) expect(() => createRecallScenario({ ...tiny, facts })).toThrow('facts')
  for (const fillerTokens of [-1, 1.5, Infinity, 2_000_001]) expect(() => createRecallScenario({ ...tiny, fillerTokens })).toThrow('fillerTokens')
  expect(() => createRecallScenario({ ...tiny, seed: NaN })).toThrow('seed')
  expect(createRecallScenario({ ...tiny, fillerTokens: 0 }).filler).toHaveLength(0)
})

test('a remembering port scores 100%, sees only filler fixtures on disk, and uses one session', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-recall-'))
  const scenario = createRecallScenario(tiny)
  const fake = fakePort(scenario, { remembered: 10, compactions: true })
  try {
    let clock = 100
    const report = await runRecallEvaluation({
      scenario, transport: fake.port, start: startRequest(root), clock: () => { clock += 25; return clock },
    })
    expect(report.recallRate).toBe(1)
    expect(report.rows.every(row => row.ok)).toBeTrue()
    expect(report.compactionsObserved).toBe(scenario.filler.length)
    expect(report.wallTimeMs).toBe(25)
    expect(fake.starts).toBe(1)
    expect(fake.resets).toBe(0)
    expect(fake.closes).toBe(1)
    expect(fake.prompts).toHaveLength(1 + scenario.filler.length + scenario.facts.length)
    for (const filler of scenario.filler) {
      expect(await readFile(join(root, filler.file.path), 'utf8')).toBe(filler.file.content)
    }
    expect(formatRecallReport(report)).toContain('RECALL: 10/10 (100.0%)')
    expect(formatRecallReport(report)).toContain('Compactions observed before recall: 1')
  } finally { await rm(root, { force: true, recursive: true }) }
})

test('forgetting early facts lowers recall and falling context-token telemetry never guesses compactions', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-recall-forget-'))
  const scenario = createRecallScenario(tiny)
  const fake = fakePort(scenario, { remembered: 4 })
  try {
    const report = await runRecallEvaluation({ scenario, transport: fake.port, start: startRequest(root) })
    expect(report.rows.map(row => row.ok)).toEqual([false, false, false, false, false, false, true, true, true, true])
    expect(report.recallRate).toBe(0.4)
    expect(report.compactionsObserved).toBe('unknown')
    expect(formatRecallReport(report)).toContain('Compactions observed before recall: unknown')
  } finally { await rm(root, { force: true, recursive: true }) }
})

test('several filler turns measure counter deltas and zero is distinct from unknown', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-recall-count-'))
  try {
    for (const fillerTokens of [0, 20_000]) {
      const scenario = createRecallScenario({ ...tiny, fillerTokens })
      const fake = fakePort(scenario, { remembered: 10, compactions: true })
      const report = await runRecallEvaluation({ scenario, transport: fake.port, start: startRequest(root) })
      expect(report.compactionsObserved).toBe(scenario.filler.length)
      if (fillerTokens > 0) expect(report.compactionsObserved).toBeGreaterThan(1)
    }
    const scenario = createRecallScenario({ ...tiny, fillerTokens: 0 })
    const fake = fakePort(scenario, { remembered: 10 })
    fake.port.getCompactionCount = async () => fake.prompts.length === 0 ? 3 : 2
    await expect(runRecallEvaluation({ scenario, transport: fake.port, start: startRequest(root) })).rejects.toThrow('decreased')
    expect(fake.prompts).toHaveLength(1)
    expect(fake.closes).toBe(1)
  } finally { await rm(root, { force: true, recursive: true }) }
})

test('caller cancellation interrupts a blocked event stream, signals the port and prevents later turns', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-recall-abort-'))
  const controller = new AbortController()
  let submitted = 0
  let closed = 0
  let observedSignal: AbortSignal | undefined
  const port: EvaluationSessionPort = {
    approve: async () => {}, reset: async () => {},
    start: async () => ({ model: 'blocked' }), close: async () => { closed += 1 },
    submit: request => {
      submitted += 1
      observedSignal = request.signal
      return {
        [Symbol.asyncIterator]: () => ({
          next: async () => new Promise<IteratorResult<EvaluationEvent>>(() => { queueMicrotask(() => controller.abort()) }),
        }),
      }
    },
  }
  try {
    await expect(runRecallEvaluation({
      scenario: createRecallScenario(tiny), transport: port, start: startRequest(root), signal: controller.signal,
    })).rejects.toMatchObject({ name: 'AbortError' })
    expect(submitted).toBe(1)
    expect(observedSignal?.aborted).toBeTrue()
    expect(closed).toBe(1)
  } finally { await rm(root, { force: true, recursive: true }) }
})

test('pre-aborted runs start no session and transport errors and timeouts remain visible with close', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-recall-errors-'))
  const scenario = createRecallScenario({ ...tiny, fillerTokens: 0 })
  const controller = new AbortController()
  controller.abort()
  const fake = fakePort(scenario, { remembered: 10 })
  try {
    await expect(runRecallEvaluation({ scenario, transport: fake.port, start: startRequest(root), signal: controller.signal })).rejects.toMatchObject({ name: 'AbortError' })
    expect(fake.starts).toBe(0)
    expect(fake.closes).toBe(1)
    const errorPort = fakePort(scenario, { remembered: 10 })
    errorPort.port.submit = async function* () {
      yield { type: 'notification', severity: 'error', body: 'provider failed', title: undefined }
      yield { type: 'turn_end' }
    }
    await expect(runRecallEvaluation({ scenario, transport: errorPort.port, start: startRequest(root) })).rejects.toThrow('provider failed')
    expect(errorPort.closes).toBe(1)
    const timeoutPort = fakePort(scenario, { remembered: 10 })
    timeoutPort.port.submit = () => new Promise(() => {})
    await expect(runRecallEvaluation({ scenario, transport: timeoutPort.port, start: startRequest(root), turnTimeoutMs: 10 })).rejects.toThrow('timed out')
    expect(timeoutPort.closes).toBe(1)
  } finally { await rm(root, { force: true, recursive: true }) }
})

test('approval requests pass through the supplied port and incomplete streams are not scored as forgetting', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-recall-stream-'))
  const scenario = createRecallScenario({ seed: 42, facts: 1, fillerTokens: 0 })
  const approvals: string[] = []
  const fake = fakePort(scenario, { remembered: 1 })
  const submit = fake.port.submit
  fake.port.approve = async id => { approvals.push(id) }
  fake.port.submit = async function* (request) {
    yield { type: 'approval_request', id: 'read-fixture' }
    yield* await submit(request)
  }
  try {
    expect((await runRecallEvaluation({ scenario, transport: fake.port, start: startRequest(root) })).recallRate).toBe(1)
    expect(approvals).toEqual(['read-fixture', 'read-fixture'])
    const incomplete = fakePort(scenario, { remembered: 1 })
    incomplete.port.submit = async function* () { yield { type: 'text', text: 'partial' } }
    await expect(runRecallEvaluation({ scenario, transport: incomplete.port, start: startRequest(root) })).rejects.toThrow('without a turn_end')
    expect(incomplete.closes).toBe(1)
  } finally { await rm(root, { force: true, recursive: true }) }
})

test('dry-run never imports the named transport, constructs a port, or allocates isolation', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-recall-dry-'))
  const output: string[] = []
  const marker = join(root, 'imported.txt')
  try {
    await writeFile(join(root, 'transport.ts'), `await Bun.write(${JSON.stringify(marker)}, 'imported'); throw new Error('must never import')\n`)
    const code = await runRecallCli(['--dry-run', '--transport', 'transport.ts', '--facts', '5', '--filler-tokens', '64', '--seed', '7'], {
      cwd: root, write: text => { output.push(text) },
      createIsolation: async () => { throw new Error('must not allocate') },
    })
    expect(code).toBe(0)
    expect(await Bun.file(marker).exists()).toBeFalse()
    expect(output.join('')).toContain('seed=7')
    expect(output.join('')).toContain('Facts: 5')
    expect(await runRecallCli(['--dry-run', '--filler-tokens', '0'], {
      cwd: root, write: () => {}, createSessionPort: () => { throw new Error('must not construct') },
    })).toBe(0)
  } finally { await rm(root, { force: true, recursive: true }) }
})

test('CLI loads an explicit transport factory, reports scores and cleans up the private run', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-recall-cli-'))
  const output: string[] = []
  const marker = join(root, 'paths.json')
  try {
    await writeFile(join(root, 'transport.ts'), `
export async function createEvaluationSessionPort(context) {
  await Bun.write(${JSON.stringify(marker)}, JSON.stringify(context))
  let key = ''
  return {
    approve: async () => {}, close: async () => {}, reset: async () => {},
    start: async () => ({ model: 'offline-module' }),
    submit: async function* ({ prompt }) {
      if (prompt.includes('RECALL planting')) key = prompt.match(/deploy key[^\\n]* is ([^\\n]+)\\./)[1]
      yield { type: 'text', text: prompt.includes('RECALL question') ? key : 'ok' }
      yield { type: 'turn_end' }
    }
  }
}
`)
    const code = await runRecallCli(['--transport', 'transport.ts', '--facts', '1', '--filler-tokens', '0', '--run-root', 'runs'], {
      cwd: root, write: text => { output.push(text) }, writeError: text => { output.push(text) },
    })
    expect(code).toBe(0)
    expect(output.join('')).toContain('offline-module')
    expect(output.join('')).toContain('RECALL: 1/1 (100.0%)')
    const paths: { workspaceDirectory: string } = JSON.parse(await readFile(marker, 'utf8'))
    await expect(stat(paths.workspaceDirectory)).rejects.toMatchObject({ code: 'ENOENT' })
  } finally { await rm(root, { force: true, recursive: true }) }
})

test('CLI requires transport for execution, rejects invalid options, and maps abort to exit 130', async () => {
  const output: string[] = []
  const writeError = (text: string) => { output.push(text) }
  expect(await runRecallCli([], { writeError })).toBe(2)
  expect(output.join('')).toContain('--transport <module> is required')
  expect(() => parseRecallCliOptions(['--facts', '1.5'])).toThrow('integer')
  expect(() => parseRecallCliOptions(['--filler-tokens', '-1'])).toThrow()
  expect(() => parseRecallCliOptions(['--seed', 'wat'])).toThrow('seed')
  expect(() => parseRecallCliOptions(['--other', '1'])).toThrow('unknown argument')
  const controller = new AbortController()
  controller.abort()
  expect(await runRecallCli(['--filler-tokens', '0'], {
    signal: controller.signal, createSessionPort: () => { throw new Error('must not construct') },
    write: () => {}, writeError,
  })).toBe(130)
})

function startRequest(workspaceDirectory: string) {
  return { homeDirectory: workspaceDirectory, workspaceDirectory, permissionMode: 'accept-all' as const }
}

function fakePort(scenario: RecallScenario, options: { remembered: number; compactions?: boolean }) {
  const state = { starts: 0, resets: 0, closes: 0, prompts: [] as string[], count: 7 }
  const port: EvaluationSessionPort = {
    approve: async () => {}, close: async () => { state.closes += 1 },
    reset: async () => { state.resets += 1 }, start: async () => { state.starts += 1; return { model: 'recall-fake' } },
    ...(options.compactions ? { getCompactionCount: () => state.count } : {}),
    submit: async function* (request: EvaluationSubmitRequest) {
      state.prompts.push(request.prompt)
      if (scenario.filler.some(filler => filler.prompt === request.prompt)) state.count += 1
      const index = scenario.facts.findIndex(fact => fact.question === request.prompt)
      const text = index < 0 ? 'ok' : index >= scenario.facts.length - options.remembered ? scenario.facts[index]!.answer : 'unknown'
      yield { type: 'status', contextTokens: state.prompts.length % 2 ? 20_000 : 10 }
      yield { type: 'text', text }
      yield { type: 'turn_end' }
    },
  }
  return Object.assign(state, { port })
}
