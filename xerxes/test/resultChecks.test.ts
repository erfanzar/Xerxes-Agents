// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import {
  parseChecksConfig,
  renderFindings,
  runResultChecks,
  snapshotInputs,
  type CheckFinding,
} from '../src/runtime/resultChecks.js'
import { createOfficeZip } from '../src/skills/powerpoint/zip.js'

async function inWorkspace(body: (workspace: string) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-checks-'))
  try {
    await body(workspace)
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

const codes = (findings: readonly CheckFinding[]): string[] => findings.map(finding => finding.code)

test('parseChecksConfig: env wins, default warn, invalid values throw', () => {
  expect(parseChecksConfig({}, {})).toBe('warn')
  expect(parseChecksConfig({ checks: 'enforce' }, {})).toBe('enforce')
  expect(parseChecksConfig({ checks: 'enforce' }, { XERXES_RESULT_CHECKS: ' OFF ' })).toBe('off')
  expect(parseChecksConfig({ checks: null }, {})).toBe('warn')
  expect(() => parseChecksConfig({ checks: 'strict' }, {})).toThrow(ValidationError)
  expect(() => parseChecksConfig({}, { XERXES_RESULT_CHECKS: 'strict' })).toThrow('XERXES_RESULT_CHECKS')
  expect(() => parseChecksConfig({ checks: true }, {})).toThrow('checks')
})

test('xlsx: CSV text in an .xlsx is xlsx_is_text, other junk is xlsx_invalid, an incomplete zip names what is missing', async () => {
  await inWorkspace(async workspace => {
    await writeFile(join(workspace, 'text.xlsx'), 'a,b\n1,2\n')
    await writeFile(join(workspace, 'junk.xlsx'), new Uint8Array([0, 1, 2, 3, 255, 254]))
    await writeFile(join(workspace, 'empty.xlsx'), '')
    await writeFile(join(workspace, 'partial.xlsx'), createOfficeZip([{ name: 'hello.txt', data: new TextEncoder().encode('hi') }]))
    const findings = await runResultChecks({
      workspaceRoot: workspace,
      changedFiles: ['text.xlsx', 'junk.xlsx', 'empty.xlsx', 'partial.xlsx', 'absent.xlsx'],
    })
    expect(findings.map(finding => [finding.path, finding.code])).toEqual([
      ['text.xlsx', 'xlsx_is_text'],
      ['junk.xlsx', 'xlsx_invalid'],
      ['empty.xlsx', 'xlsx_invalid'],
      ['partial.xlsx', 'xlsx_invalid'],
    ])
    expect(findings[3]?.message).toContain('xl/workbook.xml')
    expect(findings[3]?.message).toContain('[Content_Types].xml')
    expect(findings.every(finding => finding.severity === 'error')).toBe(true)
  })
})

test('csv: ragged rows are reported with line numbers, quoted newlines are not ragged', async () => {
  await inWorkspace(async workspace => {
    await writeFile(join(workspace, 'ragged.csv'), 'id,name,city\n1,Ada,Paris\n2,Bob\n3,Cy,Rome,extra\n4,"multi\nline",Oslo\n')
    await writeFile(join(workspace, 'good.csv'), 'id,name\n1,"a, b"\n2,"say ""hi"""\n')
    const findings = await runResultChecks({ workspaceRoot: workspace, changedFiles: ['ragged.csv', 'good.csv'] })
    expect(codes(findings)).toEqual(['csv_ragged'])
    expect(findings[0]?.path).toBe('ragged.csv')
    expect(findings[0]?.message).toContain('line 3 has 2')
    expect(findings[0]?.message).toContain('line 4 has 4')
    expect(findings[0]?.message).not.toContain('line 5')
  })
})

test('csv: an unterminated quote is reported', async () => {
  await inWorkspace(async workspace => {
    await writeFile(join(workspace, 'open.csv'), 'a,b\n1,"never closed\n2,3\n')
    const findings = await runResultChecks({ workspaceRoot: workspace, changedFiles: ['open.csv'] })
    expect(codes(findings)).toContain('csv_ragged')
    expect(findings[0]?.message).toContain('line 2')
  })
})

test('csv: invalid dates in a date column and in a mostly-ISO column are csv_bad_date', async () => {
  await inWorkspace(async workspace => {
    await writeFile(join(workspace, 'dates.csv'), 'id,order_date,when\n1,2026-01-05,2026-01-05\n2,2026-13-40,2026-02-30\n3,not a date,2026-03-01\n4,,2026-04-01\n')
    await writeFile(join(workspace, 'fine.csv'), 'id,Date\n1,2026-02-28\n2,2024-02-29\n')
    await writeFile(join(workspace, 'fine-dmy.csv'), 'id,Date\n1,31/12/2025\n2,01/02/2026\n')
    const findings = await runResultChecks({ workspaceRoot: workspace, changedFiles: ['dates.csv', 'fine.csv', 'fine-dmy.csv'] })
    expect(findings.map(finding => [finding.path, finding.code])).toEqual([
      ['dates.csv', 'csv_bad_date'],
      ['dates.csv', 'csv_bad_date'],
    ])
    expect(findings[0]?.message).toContain('order_date')
    expect(findings[0]?.message).toContain('line 3')
    expect(findings[0]?.message).toContain('line 4')
    expect(findings[1]?.message).toContain('"when"')
    expect(findings[1]?.message).toContain('line 3')
  })
})

test('csv: non-date columns are left alone', async () => {
  await inWorkspace(async workspace => {
    await writeFile(join(workspace, 'plain.csv'), 'id,name\n1,Ada\n2,Bob\n')
    expect(await runResultChecks({ workspaceRoot: workspace, changedFiles: ['plain.csv'] })).toEqual([])
  })
})

test('input_modified: snapshotInputs hashes, a changed or deleted input is reported, an untouched one is not', async () => {
  await inWorkspace(async workspace => {
    const kept = join(workspace, 'kept.csv')
    const edited = join(workspace, 'edited.csv')
    const removed = join(workspace, 'removed.csv')
    await writeFile(kept, 'a\n1\n')
    await writeFile(edited, 'a\n1\n')
    await writeFile(removed, 'a\n1\n')
    const snapshot = await snapshotInputs([kept, edited, removed, join(workspace, 'never-existed.csv')])
    expect([...snapshot.keys()]).toEqual([kept, edited, removed])
    for (const hash of snapshot.values()) expect(hash).toMatch(/^[0-9a-f]{64}$/u)

    await writeFile(edited, 'a\n2\n')
    await rm(removed)
    const findings = await runResultChecks({ workspaceRoot: workspace, changedFiles: [], inputSnapshot: snapshot })
    expect(findings.map(finding => [finding.path, finding.code, finding.severity])).toEqual([
      [edited, 'input_modified', 'error'],
      [removed, 'input_modified', 'error'],
    ])
  })
})

test('unverified_claim: a claim with no check after the last write is flagged', async () => {
  const base = { workspaceRoot: '/nonexistent-root', changedFiles: [] as string[] }
  const flagged = await runResultChecks({ ...base, finalText: 'Done. I validated the output.', toolRuns: [{ name: 'WriteFile', ok: true }] })
  expect(codes(flagged)).toEqual(['unverified_claim'])
  expect(flagged[0]?.severity).toBe('warn')

  const noRuns = await runResultChecks({ ...base, finalText: 'All tests pass.' })
  expect(codes(noRuns)).toEqual(['unverified_claim'])

  const staleCheck = await runResultChecks({
    ...base,
    finalText: 'Everything is verified.',
    toolRuns: [{ name: 'exec_command', command: 'bun test', ok: true }, { name: 'FileEditTool', ok: true }],
  })
  expect(codes(staleCheck)).toEqual(['unverified_claim'])

  const failedCheck = await runResultChecks({
    ...base,
    finalText: 'Tests passed.',
    toolRuns: [{ name: 'exec_command', command: 'bun test', ok: false }],
  })
  expect(codes(failedCheck)).toEqual(['unverified_claim'])
})

test('unverified_claim: a real check after the write, a read-back, a negation, or no claim is not flagged', async () => {
  const base = { workspaceRoot: '/nonexistent-root', changedFiles: [] as string[] }
  expect(await runResultChecks({
    ...base,
    finalText: 'All tests pass.',
    toolRuns: [{ name: 'WriteFile', ok: true }, { name: 'exec_command', command: 'bun test ./x.test.ts', ok: true }],
  })).toEqual([])
  expect(await runResultChecks({
    ...base,
    finalText: 'I verified the workbook.',
    toolRuns: [{ name: 'XlsxWriter', ok: true }, { name: 'XlsxReader', ok: true }],
  })).toEqual([])
  expect(await runResultChecks({
    ...base,
    finalText: 'I did not run the tests, so this is not verified.',
    toolRuns: [{ name: 'WriteFile', ok: true }],
  })).toEqual([])
  expect(await runResultChecks({ ...base, finalText: 'I wrote the file.', toolRuns: [{ name: 'WriteFile', ok: true }] })).toEqual([])
  expect(await runResultChecks({
    ...base,
    finalText: 'Validated.',
    toolRuns: [{ name: 'CSVProcessor', command: 'write', ok: true }],
  })).toHaveLength(1)
})

test('renderFindings lists one line per finding and states the clean case', () => {
  expect(renderFindings([])).toContain('no problems')
  const text = renderFindings([
    { code: 'csv_ragged', severity: 'error', path: 'a.csv', message: 'bad rows' },
    { code: 'unverified_claim', severity: 'warn', message: 'no check' },
  ])
  expect(text).toContain('2 problems')
  expect(text).toContain('- [error] csv_ragged a.csv: bad rows')
  expect(text).toContain('- [warn] unverified_claim: no check')
})
