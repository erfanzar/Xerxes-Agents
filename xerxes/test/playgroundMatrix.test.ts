// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import { createBenchmarkAdapter, createSyntheticPrivateAdapter } from '../playground/benchmarks/index.js'
import {
  loadSessionPortFactory,
  parseMatrixFile,
  parseRunMetrics,
  runMatrix,
  type MatrixFile,
  type MetricsReportingPort,
  type RunMetrics,
} from '../playground/matrix.js'
import { runMatrixCli } from '../playground/matrixCli.js'
import { median, renderMarkdown, summarizeConfig, type TaskRecord } from '../playground/matrixReport.js'
import type {
  EvaluationEvent,
  EvaluationSessionPort,
  EvaluationSessionPortFactory,
  EvaluationTransportContext,
} from '../playground/types.js'

const FIXTURES = resolve(import.meta.dir, 'fixtures', 'benchmarks')

let scratch: string

beforeEach(async () => {
  scratch = await mkdtemp(join(tmpdir(), 'xerxes-matrix-'))
})

afterEach(async () => {
  await rm(scratch, { force: true, recursive: true })
})

function fakePort(
  context: EvaluationTransportContext,
  script: (workspace: string, prompt: string) => Promise<string>,
  metrics: RunMetrics = {},
): EvaluationSessionPort & MetricsReportingPort {
  return {
    approve: async () => undefined,
    close: async () => undefined,
    getRunMetrics: () => metrics,
    reset: async () => undefined,
    start: async () => ({ model: 'fake-model' }),
    submit: request => (async function* (): AsyncGenerator<EvaluationEvent> {
      yield { text: await script(context.workspaceDirectory, request.prompt), type: 'text' }
      yield { type: 'turn_end' }
    })(),
  }
}

function transport(
  script: (workspace: string, prompt: string) => Promise<string>,
  metrics?: RunMetrics,
): (modulePath: string) => Promise<EvaluationSessionPortFactory> {
  return async () => context => fakePort(context, script, metrics)
}

function record(overrides: Partial<TaskRecord>): TaskRecord {
  return {
    accepted: false,
    benchmark: 'dabstep',
    configId: 'cfg',
    details: '',
    disclosureFailure: false,
    score: 0,
    status: 'rejected',
    taskId: 't',
    wallMs: 1000,
    ...overrides,
  }
}

test('report math: cost per accepted is null (never Infinity) with zero accepted or no reported cost', () => {
  const config = { id: 'cfg', privacy: 'unrestricted' } as const
  const noneAccepted = summarizeConfig(config, [
    record({ costUsd: 1, taskId: 'a' }),
    record({ costUsd: 2, taskId: 'b' }),
  ], 0)
  expect(noneAccepted.accepted).toBe(0)
  expect(noneAccepted.totalCostUsd).toBe(3)
  expect(noneAccepted.costPerAcceptedUsd).toBeNull()
  expect(noneAccepted.accuracy).toBe(0)

  const noCost = summarizeConfig(config, [record({ accepted: true, status: 'accepted' })], 0)
  expect(noCost.totalCostUsd).toBeNull()
  expect(noCost.costPerAcceptedUsd).toBeNull()

  const empty = summarizeConfig(config, [], 2)
  expect(empty).toMatchObject({ accuracy: null, costPerAcceptedUsd: null, excluded: 2, medianWallMs: null, tasks: 0 })
})

test('report math: accuracy skips needs_review, median is exact, partial costs are summed as reported', () => {
  const summary = summarizeConfig({ id: 'cfg', privacy: 'constrained' }, [
    record({ accepted: true, costUsd: 1, status: 'accepted', taskId: 'a', wallMs: 1000 }),
    record({ accepted: true, costUsd: 3, inputTokens: 10, status: 'accepted', taskId: 'b', wallMs: 3000 }),
    record({ status: 'rejected', taskId: 'c', wallMs: 2000 }),
    record({ status: 'needs_review', taskId: 'd', wallMs: 4000 }),
    record({ disclosureFailure: true, status: 'error', taskId: 'e', wallMs: 5000 }),
  ], 1)
  expect(summary).toMatchObject({
    accepted: 2,
    costPerAcceptedUsd: 2,
    costReportedTasks: 2,
    disclosureFailures: 1,
    errors: 1,
    inputTokens: 10,
    needsReview: 1,
    outputTokens: null,
    rejected: 1,
    tasks: 5,
    totalCostUsd: 4,
  })
  expect(summary.accuracy).toBeCloseTo(2 / 4)
  expect(summary.medianWallMs).toBe(3000)
  expect(median([1, 2, 3, 4])).toBe(2.5)
  expect(median([])).toBeNull()
})

test('markdown keeps privacy-constrained and unrestricted configs in separate sections and prints n/a', () => {
  const markdown = renderMarkdown({
    constrained: [summarizeConfig({ id: 'local-only', privacy: 'constrained' }, [record({ configId: 'local-only' })], 0)],
    excluded: [{ benchmark: 'terminal-bench', id: 'terminal-bench-x', reason: 'requires container grader' }],
    generatedAt: '2026-01-01T00:00:00.000Z',
    records: [],
    unrestricted: [summarizeConfig({ id: 'cloud-strong', privacy: 'unrestricted' }, [], 0)],
  })
  const constrainedAt = markdown.indexOf('## Privacy-constrained configs')
  const unrestrictedAt = markdown.indexOf('## Unrestricted configs')
  expect(constrainedAt).toBeGreaterThan(-1)
  expect(unrestrictedAt).toBeGreaterThan(constrainedAt)
  expect(markdown.indexOf('| local-only')).toBeGreaterThan(constrainedAt)
  expect(markdown.indexOf('| local-only')).toBeLessThan(unrestrictedAt)
  expect(markdown.indexOf('| cloud-strong')).toBeGreaterThan(unrestrictedAt)
  expect(markdown).toContain('n/a')
  expect(markdown).not.toContain('Infinity')
  expect(markdown).toContain('terminal-bench / terminal-bench-x: requires container grader')
})

test('parseRunMetrics keeps only well-formed measurements and invents nothing', () => {
  expect(parseRunMetrics({ cost_usd: 0.5, input_tokens: 10, outbound_texts: ['a'], output_tokens: 'x' }))
    .toEqual({ costUsd: 0.5, inputTokens: 10, outboundTexts: ['a'] })
  expect(parseRunMetrics({ costUsd: -1, outboundTexts: [1] })).toEqual({})
  expect(parseRunMetrics(null)).toEqual({})
})

test('parseMatrixFile validates configs and benchmarks and resolves relative paths', () => {
  const parsed = parseMatrixFile({
    benchmarks: [{ adapter: 'dabstep', datasetDir: 'data/dab', limit: 3 }, { adapter: 'synthetic-private' }],
    configs: [
      { id: 'local-only', kind: 'xerxes', privacy: 'constrained', transport: 'transports/local.ts' },
      { argv: ['pi', '--prompt', '{prompt_file}'], id: 'pi', kind: 'external', timeoutMs: 1000 },
    ],
  }, '/base')
  expect(parsed.configs[0]).toMatchObject({ privacy: 'constrained', transport: '/base/transports/local.ts' })
  expect(parsed.configs[1]).toMatchObject({ kind: 'external', privacy: 'unrestricted' })
  expect(parsed.benchmarks[0]).toEqual({ adapter: 'dabstep', datasetDir: '/base/data/dab', limit: 3 })

  const base = { benchmarks: [{ adapter: 'synthetic-private' }], configs: [{ id: 'a', kind: 'xerxes', transport: '/t.ts' }] }
  expect(() => parseMatrixFile({ ...base, configs: [] }, '/b')).toThrow(/configs/)
  expect(() => parseMatrixFile({ ...base, configs: [base.configs[0], base.configs[0]] }, '/b')).toThrow(/duplicated/)
  expect(() => parseMatrixFile({ ...base, configs: [{ id: '../x', kind: 'xerxes', transport: '/t' }] }, '/b')).toThrow(/id/)
  expect(() => parseMatrixFile({ ...base, configs: [{ id: 'e', kind: 'external', argv: ['x'] }] }, '/b')).toThrow(/timeoutMs/)
  expect(() => parseMatrixFile({ ...base, configs: [{ id: 'z', kind: 'other' }] }, '/b')).toThrow(/kind/)
  expect(() => parseMatrixFile({ ...base, benchmarks: [{ adapter: 'nope' }] }, '/b')).toThrow(/adapter/)
  expect(() => parseMatrixFile({ ...base, benchmarks: [{ adapter: 'dabstep' }] }, '/b')).toThrow(/datasetDir/)
  expect(() => parseMatrixFile({ ...base, configs: [{ id: 'a', kind: 'xerxes', transport: '/t', privacy: 'x' }] }, '/b')).toThrow(/privacy/)
})

test('a transport module without createEvaluationSessionPort is refused', async () => {
  await expect(loadSessionPortFactory(resolve(import.meta.dir, '..', 'src', 'tools', 'sourceLog.ts')))
    .rejects.toThrow('createEvaluationSessionPort')
})

test('the matrix runs a xerxes transport in a fresh workdir, grades it, and carries reported metrics', async () => {
  const matrix: MatrixFile = {
    benchmarks: [{ adapter: 'terminal-bench', datasetDir: join(FIXTURES, 'terminal-bench') }],
    configs: [
      { id: 'local-only', kind: 'xerxes', privacy: 'constrained', transport: '/unused' },
      { id: 'cloud-cheap', kind: 'xerxes', privacy: 'unrestricted', transport: '/unused2' },
    ],
  }
  const seenWorkdirs: string[] = []
  const report = await runMatrix(matrix, {
    loadSessionPortFactory: async modulePath => modulePath === '/unused'
      ? context => fakePort(context, async workspace => {
        seenWorkdirs.push(workspace)
        await writeFile(join(workspace, 'hello.txt'), 'hello', 'utf8')
        return 'done'
      }, { costUsd: 0.25, inputTokens: 100, outputTokens: 20 })
      : context => fakePort(context, async workspace => {
        seenWorkdirs.push(workspace)
        return 'I did nothing'
      }),
    runRoot: scratch,
  })

  expect(new Set(seenWorkdirs).size).toBe(2)
  expect(report.excluded).toEqual([{ benchmark: 'terminal-bench', id: 'terminal-bench-container-only', reason: 'requires container grader' }])
  const local = report.constrained[0]!
  expect(local).toMatchObject({ accepted: 1, costPerAcceptedUsd: 0.25, id: 'local-only', inputTokens: 100, tasks: 1, totalCostUsd: 0.25 })
  const cheap = report.unrestricted[0]!
  expect(cheap).toMatchObject({ accepted: 0, costPerAcceptedUsd: null, id: 'cloud-cheap', rejected: 1, totalCostUsd: null })
  expect(report.records.find(entry => entry.configId === 'local-only')?.costUsd).toBe(0.25)
  expect(report.records.find(entry => entry.configId === 'cloud-cheap')).not.toHaveProperty('costUsd')
})

test('a leaked canary in reported outbound text is a disclosure failure and is never accepted', async () => {
  const matrix: MatrixFile = {
    benchmarks: [{ adapter: 'synthetic-private', limit: 1 }],
    configs: [
      { id: 'leaky', kind: 'xerxes', privacy: 'constrained', transport: '/x' },
    ],
  }
  const report = await runMatrix(matrix, {
    adapterFor: id => id === 'synthetic-private' ? createSyntheticPrivateAdapter({ seed: 't' }) : createBenchmarkAdapter(id),
    loadSessionPortFactory: transport(async () => 'C-0003 and C-0004 are malformed; 3 rows are valid.', {
      outboundTexts: ['please analyze: C-0001,PRIVCANARY-t-customers Holdings'],
    }),
    runRoot: scratch,
  })
  expect(report.constrained[0]).toMatchObject({ accepted: 0, disclosureFailures: 1, tasks: 1 })
  expect(report.records[0]).toMatchObject({ accepted: false, disclosureFailure: true })
})

test('external harnesses run by argv with {prompt_file}/{workdir}/{metrics_file}, with timeout and exit handling', async () => {
  const bun = process.execPath
  const writeHello = [
    'const fs = require("fs");',
    'fs.writeFileSync("hello.txt", "hello");',
    'fs.writeFileSync("{metrics_file}", JSON.stringify({ costUsd: 0.5, outboundTexts: [] }));',
    'console.log(fs.readFileSync("{prompt_file}", "utf8").length > 0 ? "ok" : "empty");',
  ].join(' ')
  const matrix: MatrixFile = {
    benchmarks: [{ adapter: 'terminal-bench', datasetDir: join(FIXTURES, 'terminal-bench') }],
    configs: [
      { argv: [bun, '-e', writeHello], id: 'ext-good', kind: 'external', privacy: 'unrestricted', timeoutMs: 30_000 },
      { argv: [bun, '-e', 'await Bun.sleep(10000)'], id: 'ext-slow', kind: 'external', privacy: 'unrestricted', timeoutMs: 300 },
      { argv: [bun, '-e', 'process.exit(3)'], id: 'ext-crash', kind: 'external', privacy: 'unrestricted', timeoutMs: 30_000 },
    ],
  }
  const report = await runMatrix(matrix, { runRoot: scratch })
  const byConfig = (id: string) => report.records.find(entry => entry.configId === id)!
  expect(byConfig('ext-good')).toMatchObject({ accepted: true, costUsd: 0.5, status: 'accepted' })
  expect(byConfig('ext-slow')).toMatchObject({ accepted: false, status: 'error' })
  expect(byConfig('ext-slow').error).toContain('timed out')
  expect(byConfig('ext-crash').error).toContain('exited with code 3')
  expect(report.unrestricted.find(summary => summary.id === 'ext-crash')).toMatchObject({ accepted: 0, errors: 1 })
})

test('report files and review packets are written, and needs_review tasks are counted separately', async () => {
  const outputDir = join(scratch, 'out')
  const matrix: MatrixFile = {
    benchmarks: [{ adapter: 'gdpval', datasetDir: join(FIXTURES, 'gdpval'), limit: 2 }],
    configs: [{ id: 'cfg-a', kind: 'xerxes', privacy: 'unrestricted', transport: '/x' }],
  }
  const report = await runMatrix(matrix, {
    loadSessionPortFactory: transport(async () => 'deliverable summary'),
    now: () => new Date('2026-01-01T00:00:00.000Z'),
    outputDir,
    runRoot: join(scratch, 'runs'),
  })
  expect(report.generatedAt).toBe('2026-01-01T00:00:00.000Z')
  expect(report.unrestricted[0]).toMatchObject({ accepted: 1, accuracy: 1, needsReview: 1, tasks: 2 })
  expect(report.excluded.map(entry => entry.id)).toEqual(['gdpval-g3'])

  const json = JSON.parse(await readFile(join(outputDir, 'report.json'), 'utf8')) as { unrestricted: unknown[] }
  expect(json.unrestricted).toHaveLength(1)
  expect(await readFile(join(outputDir, 'report.md'), 'utf8')).toContain('## Unrestricted configs')
  const packet = await readFile(join(outputDir, 'review', 'cfg-a', 'gdpval-g2.md'), 'utf8')
  expect(packet).toContain('Draft a short cover letter.')
  expect(packet).toContain('deliverable summary')
})

test('a transport that cannot be loaded records an error instead of crashing the matrix', async () => {
  const report = await runMatrix({
    benchmarks: [{ adapter: 'synthetic-private', limit: 1 }],
    configs: [{ id: 'broken', kind: 'xerxes', privacy: 'unrestricted', transport: '/nope' }],
  }, {
    loadSessionPortFactory: async () => {
      throw new Error('cannot import transport')
    },
    runRoot: scratch,
  })
  expect(report.records[0]).toMatchObject({ accepted: false, status: 'error' })
  expect(report.records[0]?.error).toContain('cannot import transport')
})

test('the matrix CLI validates usage, prints help, and returns 1 when any run errored', async () => {
  const out: string[] = []
  const err: string[] = []
  const io = { write: (text: string) => void out.push(text), writeError: (text: string) => void err.push(text) }

  expect(await runMatrixCli(['--help'], io)).toBe(0)
  expect(out.join('')).toContain('--matrix')
  expect(await runMatrixCli([], io)).toBe(2)
  expect(await runMatrixCli(['--bogus'], io)).toBe(2)
  expect(err.join('')).toContain('--matrix <file> is required')

  const matrixPath = join(scratch, 'matrix.json')
  await writeFile(matrixPath, JSON.stringify({
    benchmarks: [{ adapter: 'synthetic-private', limit: 1 }],
    configs: [{ argv: [process.execPath, '-e', 'process.exit(1)'], id: 'bad', kind: 'external', timeoutMs: 30_000 }],
  }), 'utf8')
  const code = await runMatrixCli(['--matrix', matrixPath, '--out', join(scratch, 'o'), '--run-root', join(scratch, 'r')], io)
  expect(code).toBe(1)
  expect(out.join('')).toContain('Benchmark matrix')

  expect(await runMatrixCli(['--matrix', join(scratch, 'missing.json')], io)).toBe(1)
})
