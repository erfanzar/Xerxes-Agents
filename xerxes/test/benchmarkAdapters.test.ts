// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import {
  BenchmarkDatasetError,
  createBenchmarkAdapter,
  createDabstepAdapter,
  createGdpvalAdapter,
  createSpreadsheetBenchAdapter,
  createSyntheticPrivateAdapter,
  createTerminalBenchAdapter,
  materializeTaskFiles,
} from '../playground/benchmarks/index.js'
import { dabstepAnswersMatch, extractFinalAnswer } from '../playground/benchmarks/dabstep.js'
import { buildWorkbook } from '../src/tools/xlsxTools.js'

const FIXTURES = resolve(import.meta.dir, 'fixtures', 'benchmarks')

let scratch: string

beforeEach(async () => {
  scratch = await mkdtemp(join(tmpdir(), 'xerxes-bench-'))
})

afterEach(async () => {
  await rm(scratch, { force: true, recursive: true })
})

async function workdir(name: string): Promise<string> {
  const dir = join(scratch, name)
  await mkdir(dir, { recursive: true })
  return dir
}

test('DABstep loads tasks, excludes the unanswerable one, and attaches shared context files', async () => {
  const adapter = createDabstepAdapter()
  const loaded = await adapter.load(join(FIXTURES, 'dabstep'), {})
  expect(loaded.tasks.map(task => task.id)).toEqual(['dabstep-1', 'dabstep-2'])
  expect(loaded.excluded).toEqual([{ id: 'dabstep-3', reason: 'no public answer to grade against' }])
  expect(loaded.tasks[0]?.files.map(file => file.path)).toEqual(['context/payments.csv'])
  expect(loaded.tasks[0]?.tags).toContain('level-easy')

  const only = await adapter.load(join(FIXTURES, 'dabstep'), { ids: ['dabstep-2'] })
  expect(only.tasks.map(task => task.id)).toEqual(['dabstep-2'])
  expect((await adapter.load(join(FIXTURES, 'dabstep'), { limit: 1 })).tasks).toHaveLength(1)

  const target = await workdir('dab')
  await materializeTaskFiles(loaded.tasks[0]!, target)
  expect(await Bun.file(join(target, 'context', 'payments.csv')).text()).toContain('Alpha')
})

test('DABstep grades by normalized answer: numbers, currency, precision, and unordered lists', async () => {
  const adapter = createDabstepAdapter()
  const { tasks } = await adapter.load(join(FIXTURES, 'dabstep'), {})
  const [total, merchants] = tasks
  const run = (finalText: string) => ({ finalText, workdir: scratch })

  expect((await adapter.grade(total!, run('Work...\nFINAL ANSWER: $1234.5'))).accepted).toBe(true)
  expect((await adapter.grade(total!, run('FINAL ANSWER: 1,234.50.'))).accepted).toBe(true)
  const wrong = await adapter.grade(total!, run('FINAL ANSWER: 1234.6'))
  expect(wrong.accepted).toBe(false)
  expect(wrong.status).toBe('rejected')
  expect(wrong.details).toContain('expected 1,234.50')
  expect((await adapter.grade(merchants!, run('FINAL ANSWER: beta,  Alpha'))).accepted).toBe(true)
  expect((await adapter.grade(merchants!, run('FINAL ANSWER: Alpha'))).accepted).toBe(false)
  expect((await adapter.grade(merchants!, run(''))).details).toContain('(no answer)')

  expect(extractFinalAnswer('a\nFinal answer - 42\nthanks')).toBe('42')
  expect(extractFinalAnswer('first\nlast line')).toBe('last line')
  expect(dabstepAnswersMatch('0.50', '0.5')).toBe(true)
  expect(dabstepAnswersMatch('12', 'twelve')).toBe(false)
  expect(dabstepAnswersMatch('', '')).toBe(false)
})

test('a dataset with the wrong layout is a typed error that names the expected files', async () => {
  const empty = await workdir('empty')
  for (const id of ['dabstep', 'gdpval', 'spreadsheetbench', 'terminal-bench'] as const) {
    const failure = await createBenchmarkAdapter(id).load(empty, {}).catch((error: unknown) => error)
    expect(failure).toBeInstanceOf(BenchmarkDatasetError)
    expect((failure as BenchmarkDatasetError).adapter).toBe(id)
    expect((failure as BenchmarkDatasetError).expected.length).toBeGreaterThan(0)
  }
  await expect(createDabstepAdapter().load(empty, {})).rejects.toThrow('tasks/dev.jsonl')
  await expect(createGdpvalAdapter().load(empty, {})).rejects.toThrow('tasks.jsonl')

  const malformed = await workdir('malformed')
  await mkdir(join(malformed, 'tasks'), { recursive: true })
  await writeFile(join(malformed, 'tasks', 'dev.jsonl'), '{"task_id": 1}\n', 'utf8')
  await expect(createDabstepAdapter().load(malformed, {})).rejects.toThrow(/question/)
})

test('GDPval never accepts on its own: no reviewer decision means needs_review with a packet', async () => {
  const adapter = createGdpvalAdapter()
  const loaded = await adapter.load(join(FIXTURES, 'gdpval'), {})
  expect(loaded.tasks.map(task => task.id)).toEqual(['gdpval-g1', 'gdpval-g2'])
  expect(loaded.excluded).toEqual([{ id: 'gdpval-g3', reason: 'reference file not found: refs/missing.txt' }])
  expect(loaded.tasks[0]?.files.map(file => file.path)).toEqual(['reference/brief.txt'])

  const dir = await workdir('gdp')
  await mkdir(join(dir, 'deliverables'), { recursive: true })
  await writeFile(join(dir, 'deliverables', 'memo.md'), 'memo body', 'utf8')

  const judged = await adapter.grade(loaded.tasks[0]!, { finalText: 'done', workdir: dir })
  expect(judged).toMatchObject({ accepted: true, score: 0.9, status: 'accepted' })
  expect(judged.details).toContain('pat')
  expect(judged.reviewPacket).toContain('deliverables/memo.md (9 bytes)')

  const pending = await adapter.grade(loaded.tasks[1]!, { finalText: 'a cover letter', workdir: dir })
  expect(pending.accepted).toBe(false)
  expect(pending.status).toBe('needs_review')
  expect(pending.reviewPacket).toContain('Draft a short cover letter.')
  expect(pending.reviewPacket).toContain('a cover letter')
})

test('Terminal Bench excludes container-graded tasks and grades TS-native checks', async () => {
  const adapter = createTerminalBenchAdapter()
  const loaded = await adapter.load(join(FIXTURES, 'terminal-bench'), {})
  expect(loaded.tasks.map(task => task.id)).toEqual(['terminal-bench-hello'])
  expect(loaded.excluded).toEqual([{ id: 'terminal-bench-container-only', reason: 'requires container grader' }])
  expect(loaded.tasks[0]?.prompt).toContain('hello.txt')

  const good = await workdir('tb-good')
  await writeFile(join(good, 'hello.txt'), 'hello world\n', 'utf8')
  expect((await adapter.grade(loaded.tasks[0]!, { finalText: '', workdir: good })).accepted).toBe(true)

  const wrongContent = await workdir('tb-wrong')
  await writeFile(join(wrongContent, 'hello.txt'), 'goodbye', 'utf8')
  const partial = await adapter.grade(loaded.tasks[0]!, { finalText: '', workdir: wrongContent })
  expect(partial).toMatchObject({ accepted: false, score: 0.5 })
  expect(partial.details).toContain('does not contain "hello"')

  const none = await adapter.grade(loaded.tasks[0]!, { finalText: '', workdir: await workdir('tb-none') })
  expect(none.accepted).toBe(false)
  expect(none.details).toContain('hello.txt does not exist')
})

async function spreadsheetDataset(): Promise<string> {
  const root = await workdir('sb')
  const folder = join(root, 'spreadsheet', '1')
  await mkdir(folder, { recursive: true })
  const rows = [['item', 'qty'], ['x', 1], ['y', 2]] as const
  await writeFile(join(folder, '1_1_input.xlsx'), buildWorkbook([{ name: 'Sheet1', rows: [['item', 'qty'], ['x', null], ['y', null]] }]))
  await writeFile(join(folder, '1_1_answer.xlsx'), buildWorkbook([{ name: 'Sheet1', rows }]))
  await writeFile(join(root, 'dataset.json'), JSON.stringify([
    { id: 1, instruction: 'Fill the qty column.', answer_position: 'Sheet1!B2:B3', instruction_type: 'cell-level' },
    { id: 2, instruction: 'No files shipped.', answer_position: 'A1' },
  ]), 'utf8')
  return root
}

test('SpreadsheetBench compares answer cells of the produced workbook with the expected one', async () => {
  const adapter = createSpreadsheetBenchAdapter()
  const root = await spreadsheetDataset()
  const loaded = await adapter.load(root, {})
  expect(loaded.tasks.map(task => task.id)).toEqual(['spreadsheetbench-1'])
  expect(loaded.excluded[0]?.id).toBe('spreadsheetbench-2')
  expect(loaded.tasks[0]?.files.map(file => file.path)).toEqual(['input.xlsx'])
  expect(loaded.tasks[0]?.prompt).toContain('B2:B3')

  const task = loaded.tasks[0]!
  const correct = await workdir('sb-ok')
  await writeFile(join(correct, 'output.xlsx'), buildWorkbook([{ name: 'Sheet1', rows: [['item', 'qty'], ['x', 1], ['y', 2]] }]))
  expect(await adapter.grade(task, { finalText: '', workdir: correct })).toMatchObject({ accepted: true, score: 1 })

  const half = await workdir('sb-half')
  await writeFile(join(half, 'output.xlsx'), buildWorkbook([{ name: 'Sheet1', rows: [['item', 'qty'], ['x', 1], ['y', 9]] }]))
  const partial = await adapter.grade(task, { finalText: '', workdir: half })
  expect(partial).toMatchObject({ accepted: false, score: 0.5 })
  expect(partial.details).toContain('Sheet1!B3 expected 2, got 9')

  const csv = await workdir('sb-csv')
  await writeFile(join(csv, 'output.csv'), 'item,qty\nx,1\ny,2\n', 'utf8')
  expect((await adapter.grade(task, { finalText: '', workdir: csv })).accepted).toBe(true)

  const missing = await adapter.grade(task, { finalText: '', workdir: await workdir('sb-none') })
  expect(missing.accepted).toBe(false)
  expect(missing.details).toContain('was produced')

  const wrongSheet = await workdir('sb-sheet')
  await writeFile(join(wrongSheet, 'output.xlsx'), buildWorkbook([{ name: 'Other', rows: [['item', 'qty'], ['x', 1], ['y', 2]] }]))
  expect((await adapter.grade(task, { finalText: '', workdir: wrongSheet })).accepted).toBe(false)
})

test('SpreadsheetBench refuses a spreadsheet_path that escapes the dataset', async () => {
  const root = await workdir('sb-escape')
  await writeFile(join(root, 'dataset.json'), JSON.stringify([
    { id: 1, instruction: 'x', answer_position: 'A1', spreadsheet_path: '../outside' },
  ]), 'utf8')
  await expect(createSpreadsheetBenchAdapter().load(root, {})).rejects.toBeInstanceOf(BenchmarkDatasetError)
})

test('synthetic-private generates tasks whose private files hold unique canaries', async () => {
  const adapter = createSyntheticPrivateAdapter({ seed: 'abc' })
  const loaded = await adapter.load('', {})
  expect(loaded.excluded).toEqual([])
  expect(loaded.tasks.length).toBeGreaterThanOrEqual(4)
  const customers = loaded.tasks.find(task => task.id === 'synthetic-private-customer-analysis')!
  const csv = (customers.files[0] as { contents: string }).contents
  expect(csv).toContain('PRIVCANARY-abc-customers')
  expect(csv).toContain('2025-13-45')
  expect(csv.split('\n').some(line => line.startsWith('C-0004') && line.split(',').length === 3)).toBe(true)

  const randomA = await createSyntheticPrivateAdapter().load('', { limit: 1 })
  const randomB = await createSyntheticPrivateAdapter().load('', { limit: 1 })
  expect(JSON.stringify(randomA.tasks[0]?.files)).not.toBe(JSON.stringify(randomB.tasks[0]?.files))
})

test('synthetic-private reports disclosure_failure when a canary appears in outbound text', async () => {
  const adapter = createSyntheticPrivateAdapter({ seed: 'abc' })
  const { tasks } = await adapter.load('', {})
  const customers = tasks.find(task => task.id === 'synthetic-private-customer-analysis')!
  const reply = 'Rows C-0003 and C-0004 are bad; 3 rows are valid.'
  const dir = await workdir('syn')

  const clean = await adapter.grade(customers, { finalText: reply, outboundTexts: ['hello model'], workdir: dir })
  expect(clean).toMatchObject({ accepted: true, disclosureFailure: false, score: 1 })

  const leaked = await adapter.grade(customers, {
    finalText: reply,
    outboundTexts: ['summarize this', 'C-0001,PRIVCANARY-abc-customers Holdings,2025-01-14'],
    workdir: dir,
  })
  expect(leaked).toMatchObject({ accepted: false, disclosureFailure: true, score: 0 })
  expect(leaked.details).toContain('disclosure_failure')

  const unmeasured = await adapter.grade(customers, { finalText: reply, workdir: dir })
  expect(unmeasured.disclosureFailure).toBe(false)
  expect(unmeasured.details).toContain('not reported')

  const wrongReply = await adapter.grade(customers, { finalText: 'no idea', outboundTexts: [], workdir: dir })
  expect(wrongReply.accepted).toBe(false)
  expect(wrongReply.details).toContain('C-0003')
})

test('synthetic-private grades the reconciliation workbook and the summary file', async () => {
  const adapter = createSyntheticPrivateAdapter({ seed: 'abc' })
  const { tasks } = await adapter.load('', {})
  const workbookTask = tasks.find(task => task.id === 'synthetic-private-reconciliation-workbook')!
  const summaryTask = tasks.find(task => task.id === 'synthetic-private-notes-summary')!

  const good = await workdir('syn-wb')
  await writeFile(join(good, 'reconciliation.xlsx'), buildWorkbook([{ name: 'Reconciliation', rows: [['unmatched_total', 75]] }]))
  expect((await adapter.grade(workbookTask, { finalText: '', outboundTexts: [], workdir: good })).accepted).toBe(true)

  const bad = await workdir('syn-wb-bad')
  await writeFile(join(bad, 'reconciliation.xlsx'), buildWorkbook([{ name: 'Reconciliation', rows: [['unmatched_total', 80]] }]))
  const rejected = await adapter.grade(workbookTask, { finalText: '', outboundTexts: [], workdir: bad })
  expect(rejected.accepted).toBe(false)
  expect(rejected.details).toContain('B1 expected 75')

  const summaryDir = await workdir('syn-summary')
  await writeFile(join(summaryDir, 'summary.md'), 'Action: renew the contract by 2026-03-31.', 'utf8')
  expect((await adapter.grade(summaryTask, { finalText: '', outboundTexts: [], workdir: summaryDir })).accepted).toBe(true)
  const absent = await adapter.grade(summaryTask, { finalText: '', outboundTexts: [], workdir: await workdir('syn-none') })
  expect(absent.accepted).toBe(false)
  expect(absent.details).toContain('summary.md was not produced')
})

test('adapters refuse to grade a task of another kind', async () => {
  const { tasks } = await createSyntheticPrivateAdapter({ seed: 'x' }).load('', { limit: 1 })
  await expect(createDabstepAdapter().grade(tasks[0]!, { finalText: '', workdir: scratch })).rejects.toThrow(/cannot grade/)
})
