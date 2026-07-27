// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { CommandResult, CommandRunner } from '../src/runtime/changeGuard.js'
import {
  EditDiagnostics,
  detectChecker,
  diagnosticKey,
  editDiagnosticsFor,
  formatEditDiagnostics,
  parseDiagnostics,
  resetEditDiagnostics,
} from '../src/runtime/editDiagnostics.js'

const CWD = '/workspace'

/** Command runner that returns queued checker output and records the git calls it saw. */
function scriptedRunner(
  checkerOutputs: readonly string[],
  gitStatus = '',
): { readonly calls: string[][], readonly runner: CommandRunner } {
  const calls: string[][] = []
  let index = 0
  const runner: CommandRunner = (args): CommandResult => {
    calls.push([...args])
    if (args[1] === 'rev-parse') return { exitCode: 0, stdout: CWD + '\n' }
    if (args[0] === 'git') return { exitCode: 0, stdout: gitStatus }
    const output = checkerOutputs[Math.min(index, checkerOutputs.length - 1)] ?? ''
    index += 1
    return { exitCode: output ? 2 : 0, stdout: output }
  }
  return { calls, runner }
}

const TSC = { source: 'tsc', command: ['tsc', '--noEmit'] } as const

test('checker detection follows workspace markers and honors an override', () => {
  expect(detectChecker(CWD, path => path.endsWith('tsconfig.json'))).toMatchObject({ source: 'tsc' })
  expect(detectChecker(CWD, path => path.endsWith('pyproject.toml'))).toMatchObject({ source: 'ruff' })
  expect(detectChecker(CWD, path => path.endsWith('Cargo.toml'))).toMatchObject({ source: 'cargo' })
  expect(detectChecker(CWD, () => false)).toBeUndefined()
  // tsconfig wins over a sibling pyproject so a polyglot repo runs one checker, not three.
  expect(detectChecker(CWD, () => true)).toMatchObject({ source: 'tsc' })

  const overridden = new EditDiagnostics(CWD, {
    checker: { source: 'ruff', command: ['ruff', 'check'] },
    fileExists: () => true,
  })
  expect(overridden.checker).toMatchObject({ source: 'ruff', command: ['ruff', 'check'] })
})

test('diagnostics parse per source into path-keyed identity tuples', () => {
  const tsc = parseDiagnostics(
    'src/a.ts(12,5): error TS2345: Argument of type string is not assignable.\n'
    + 'src/a.ts(12,5): error TS2345: Argument of type string is not assignable.\n'
    + 'src/b.ts(3,1): warning TS6133: Unused local.\n'
    + 'error TS18003: No inputs were found in config file.\n',
    'tsc',
    CWD,
  )
  // The duplicated identical row collapses: identity is a set, not a line count.
  expect(tsc.get('/workspace/src/a.ts')?.size).toBe(1)
  expect([...(tsc.get('/workspace/src/b.ts') ?? new Map()).values()][0]).toMatchObject({
    severity: 'warning',
    source: 'tsc',
    code: 'TS6133',
    range: { startLine: 3, startColumn: 1, endLine: 3, endColumn: 1 },
  })
  expect(tsc.size).toBe(2)

  const ruff = parseDiagnostics(
    'pkg/mod.py:4:1: F401 [*] `os` imported but unused\npkg/mod.py:9:80: W505 doc line too long\n',
    'ruff',
    CWD,
  )
  const ruffEntries = [...(ruff.get('/workspace/pkg/mod.py') ?? new Map()).values()]
  expect(ruffEntries.map(entry => entry.severity)).toEqual(['error', 'warning'])
  expect(ruffEntries[0]?.message).toBe('`os` imported but unused')

  const cargo = parseDiagnostics('src/main.rs:2:5: error[E0425]: cannot find value `x`\n', 'cargo', CWD)
  expect([...(cargo.get('/workspace/src/main.rs') ?? new Map()).values()][0]).toMatchObject({
    code: 'E0425',
    message: 'cannot find value `x`',
    severity: 'error',
  })

  expect(parseDiagnostics('src/a.ts(1,1): error TS1: x', 'unknown-checker', CWD).size).toBe(0)
  const key = diagnosticKey({
    severity: 'error',
    source: 'tsc',
    code: 'TS1',
    range: { startLine: 1, startColumn: 2, endLine: 1, endColumn: 2 },
    message: 'boom',
  })
  expect(key.split('\u001f')).toEqual(['error', 'tsc', 'TS1', '1:2-1:2', 'boom'])
})

test('only new diagnostics in baselined paths reach the report, and baselines refresh after it', async () => {
  const baseline = 'src/tracked.ts(1,1): error TS1000: pre-existing\n'
    + 'src/untouched.ts(1,1): error TS1001: unrelated pre-existing\n'
  const afterEdit = baseline
    + 'src/tracked.ts(9,3): error TS2345: fresh breakage\n'
    + 'src/untouched.ts(4,4): error TS2322: someone else broke this\n'
  const { runner } = scriptedRunner([baseline, afterEdit, afterEdit])

  const diagnostics = new EditDiagnostics(CWD, { checker: TSC, commandRunner: runner, includeWorkspaceRisk: false })
  await diagnostics.noteFileWillChange('src/tracked.ts')
  expect(diagnostics.trackedPaths).toEqual(['/workspace/src/tracked.ts'])

  const report = await diagnostics.report()
  expect(report.files).toHaveLength(1)
  expect(report.files[0]?.path).toBe('/workspace/src/tracked.ts')
  expect(report.files[0]?.diagnostics.map(entry => entry.code)).toEqual(['TS2345'])
  expect(report.text).toContain('src/tracked.ts:9:3 error TS2345: fresh breakage')
  // Pre-existing errors and errors in paths this turn never touched stay out of the model's face.
  expect(report.text).not.toContain('pre-existing')
  expect(report.text).not.toContain('untouched.ts')

  // Baseline refreshed: the same unfixed error is reported once, not on every later turn.
  await diagnostics.noteFileWillChange('src/tracked.ts')
  expect((await diagnostics.report()).text).toBe('')
})

test('report stays silent without a baseline, without a checker, and when the checker cannot run', async () => {
  const { runner, calls } = scriptedRunner(['src/a.ts(1,1): error TS1: boom\n'])
  const noBegin = new EditDiagnostics(CWD, { checker: TSC, commandRunner: runner })
  // No baseline was ever anchored, so nothing is attributable to this turn.
  expect((await noBegin.report()).text).toBe('')
  expect(calls).toHaveLength(0)

  const noChecker = new EditDiagnostics(CWD, { commandRunner: runner, fileExists: () => false })
  noChecker.begin()
  await noChecker.noteFileWillChange('src/a.ts')
  expect((await noChecker.report()).text).toBe('')

  const missingBinary = new EditDiagnostics(CWD, {
    checker: TSC,
    commandRunner: () => {
      throw new Error('tsc: command not found')
    },
  })
  await missingBinary.noteFileWillChange('src/a.ts')
  expect((await missingBinary.report()).text).toBe('')
})

test('unannounced edits are recovered from git status and risky changes ride along as a fact', async () => {
  const baseline = ''
  const afterEdit = 'src/daemon/server.ts(4,2): error TS2554: expected 1 argument\n'
  const { runner, calls } = scriptedRunner(
    [baseline, afterEdit, afterEdit],
    ' M src/daemon/server.ts\n D xerxes/test/regression.test.ts\n',
  )
  const diagnostics = new EditDiagnostics(CWD, { checker: TSC, commandRunner: runner })
  diagnostics.begin()

  const report = await diagnostics.report()
  expect(report.text).toContain('src/daemon/server.ts:4:2 error TS2554: expected 1 argument')
  expect(report.text).toContain('tracked test file(s) were deleted')
  // The retired nag must not come back: the checker answer is the fact, not a chore request.
  expect(report.text).not.toContain('No recent Bun test')
  expect(calls.some(call => call[0] === 'git' && call.includes('--untracked-files=all'))).toBe(true)
})

test('reports are hard bounded by file count, per-file count, and total characters', () => {
  const files = Array.from({ length: 12 }, (_unused, fileIndex) => ({
    path: `/workspace/src/file${fileIndex}.ts`,
    diagnostics: Array.from({ length: 9 }, (_ignored, index) => ({
      severity: 'error' as const,
      source: 'tsc',
      code: `TS${1000 + index}`,
      range: { startLine: index + 1, startColumn: 1, endLine: index + 1, endColumn: 1 },
      message: 'x'.repeat(400),
    })),
  }))

  const bounded = formatEditDiagnostics(files, { cwd: CWD, source: 'tsc' })
  expect(bounded.truncated).toBe(true)
  expect(bounded.text.length).toBeLessThanOrEqual(3000)
  expect(bounded.text).toContain('108 new problem(s) in 12 file(s)')

  const single = formatEditDiagnostics([{
    path: '/workspace/src/one.ts',
    diagnostics: [{
      severity: 'warning',
      source: 'tsc',
      code: 'TS6133',
      range: { startLine: 2, startColumn: 4, endLine: 2, endColumn: 4 },
      message: 'unused',
    }],
  }], { cwd: CWD, source: 'tsc', suffix: 'Risky workspace changes detected:' })
  expect(single.truncated).toBe(false)
  expect(single.text).toContain('src/one.ts:2:4 warning TS6133: unused')
  expect(single.text).toContain('Risky workspace changes detected:')
  expect(formatEditDiagnostics([], { cwd: CWD, source: 'tsc' })).toEqual({ text: '', truncated: false })
})

test('the per-workspace instance is shared so the mutation site and the turn boundary agree', () => {
  resetEditDiagnostics()
  const first = editDiagnosticsFor(CWD, { checker: TSC, fileExists: () => false })
  const second = editDiagnosticsFor('/workspace/')
  expect(second).toBe(first)
  expect(editDiagnosticsFor('/other-workspace', { fileExists: () => false })).not.toBe(first)
  resetEditDiagnostics(CWD)
  expect(editDiagnosticsFor(CWD, { fileExists: () => false })).not.toBe(first)
  resetEditDiagnostics()
})
