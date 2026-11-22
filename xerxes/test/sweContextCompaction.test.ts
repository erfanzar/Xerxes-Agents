// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import {
  COMPACTION_ARCHIVE_NOTE,
  COMPACTION_REFERENCE_PREFIX,
  COMPACTION_SUMMARY_MARKER,
  CompactionProvisioner,
  ContextCompressor,
  MAX_ACTIVE_FAILURE_COORDINATES,
  SWE_WORKING_SET_CLOSE_TAG,
  SWE_WORKING_SET_OPEN_TAG,
  SmartTokenCounter,
  compressToolResult,
  extractSweWorkingSet,
  isCompactionSummaryMessage,
  naiveSummarizer,
  parseSweWorkingSet,
  pruneToolMessages,
  pruneToolResult,
  renderSweWorkingSet,
  type ContextMessage,
} from '../src/context/index.js'

function hasUnpairedSurrogates(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index)
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1)
      if (!(next >= 0xdc00 && next <= 0xdfff)) return true
      index += 1
      continue
    }
    if (code >= 0xdc00 && code <= 0xdfff) {
      return true
    }
  }
  return false
}

function extractWorkingSetSlice(summaryText: string): string {
  const open = summaryText.indexOf(SWE_WORKING_SET_OPEN_TAG)
  const close = summaryText.indexOf(SWE_WORKING_SET_CLOSE_TAG)
  if (open < 0 || close < 0) return ''
  return summaryText.slice(open, close + SWE_WORKING_SET_CLOSE_TAG.length)
}

/**
 * Build a >30,000-char, 905-line test runner output with:
 * - Invocation/header banner at lines 1-5,
 * - Passing test noise across lines 6-500 and lines 501-897,
 * - A `TypeError` + `src/engine/file.ts:184:12` stack trace centered at line 420,
 * - Runner summary + totals + exit code at lines 898-905.
 */
function build30kTestOutput(runner: 'bun' | 'pytest' | 'tsc'): string {
  const lines: string[] = []
  if (runner === 'bun') {
    lines.push(
      '$ bun test --timeout 15000',
      'bun test v1.3.12 (workspace-root)',
      'collecting 900 test cases across 45 suites...',
      'config: tsconfig.json (strict=true, exactOptionalPropertyTypes=true)',
      '----------------------------------------------------------------------',
    )
  } else if (runner === 'pytest') {
    lines.push(
      '============================= test session starts ==============================',
      'platform darwin -- Python 3.12.7, pytest-8.3.3, pluggy-1.5.0',
      'rootdir: /workspace/project',
      'configfile: pyproject.toml',
      'collected 900 items',
    )
  } else {
    lines.push(
      '$ tsc --noEmit -p xerxes/tsconfig.json --pretty false',
      'TypeScript 5.8.2 project check starting...',
      'scanning 312 source modules in workspace...',
      'compilerOptions: strict=true, exactOptionalPropertyTypes=true',
      '----------------------------------------------------------------------',
    )
  }

  // Lines 6..905 (total 905 lines, >30,000 chars)
  for (let lineNumber = 6; lineNumber <= 905; lineNumber += 1) {
    if (lineNumber === 418) {
      lines.push('  [runner] executing suite xerxes/test/enginePipeline.test.ts case #418 ...')
    } else if (lineNumber === 419) {
      lines.push('  [runner] invoking computeWindowScore(undefined) in strict mode')
    } else if (lineNumber === 420) {
      if (runner === 'tsc') {
        lines.push("src/engine/file.ts:184:12 - error TS2532: Object is possibly 'undefined' in TypeError guard.")
      } else if (runner === 'pytest') {
        lines.push('Traceback (most recent call last):')
      } else {
        lines.push("TypeError: Cannot read properties of undefined (reading 'windowBudget')")
      }
    } else if (lineNumber === 421) {
      if (runner === 'pytest') {
        lines.push('  File "src/engine/file.ts", line 184, in compute_window_score')
      } else {
        lines.push('    at computeWindowScore (src/engine/file.ts:184:12)')
      }
    } else if (lineNumber === 422) {
      if (runner === 'pytest') {
        lines.push("TypeError: 'NoneType' object has no attribute 'window_budget' (src/engine/file.ts:184:12)")
      } else {
        lines.push('    at executePipeline (src/engine/runner.ts:77:9)')
      }
    } else if (lineNumber === 423) {
      lines.push('  [runner] teardown after failed assertion in case #418')
    } else if (lineNumber >= 898) {
      const summaryLines = [
        '=========================== short test summary info ============================',
        'FAIL xerxes/test/enginePipeline.test.ts > computeWindowScore handles empty payload',
        'FAILED xerxes/test/enginePipeline.test.ts::computeWindowScore - TypeError at src/engine/file.ts:184:12',
        'Suites: 1 failed, 44 passed, 45 total',
        'Tests:  1 failed, 899 passed, 900 total',
        'Ran 900 tests across 45 files in 4.82s',
        'Process exited with code 1',
        'EXIT_STATUS=1',
      ]
      lines.push(summaryLines[lineNumber - 898] ?? '')
    } else {
      lines.push(
        `PASS suite_${String(Math.ceil(lineNumber / 20)).padStart(2, '0')}/case_${String(lineNumber).padStart(4, '0')}.test.ts :: verifies deterministic invariant [0.42ms]`,
      )
    }
  }

  const joined = lines.join('\n')
  if (joined.length <= 30_000) {
    throw new Error(`Expected >30,000 chars fixture, got ${joined.length}`)
  }
  return joined
}

describe('Failure-Salient Tool Spill Previews (headroom & toolResultPruner)', () => {
  test('retains invocation head, exact 1-based line-numbered middle error hunk, and summary tail within 4,000 chars across bun test, pytest, and tsc outputs', () => {
    for (const [toolName, runner] of [
      ['exec_command', 'bun'],
      ['Bash', 'pytest'],
      ['check_command', 'tsc'],
      ['LSP', 'tsc'],
      ['generic_ci_output', 'bun'],
    ] as const) {
      const output = build30kTestOutput(runner)
      expect(output.length).toBeGreaterThan(30_000)

      const preview = compressToolResult(toolName, output, { maxChars: 4_000 })
      expect(preview.compressed.length).toBeLessThanOrEqual(4_000)
      expect(preview.originalChars).toBe(output.length)
      expect(hasUnpairedSurrogates(preview.compressed)).toBe(false)

      // 1. Invocation / header head preserved
      const firstLine = output.split('\n', 1)[0]!
      expect(preview.compressed).toContain(firstLine)

      // 2. Middle failure hunk around line 420 preserved with exact 1-based [lines A-B] range (+/-2 context around 420..422 -> lines 418-424)
      const rangeMatches = [...preview.compressed.matchAll(/\[lines (\d+)-(\d+)\]/g)]
      const middleRange = rangeMatches.find(match => {
        const start = Number(match[1])
        const end = Number(match[2])
        return start <= 420 && end >= 422 && start > 50 && end < 800
      })
      expect(middleRange).toBeDefined()
      expect(Number(middleRange![1])).toBe(418)
      expect(Number(middleRange![2])).toBe(424)
      expect(preview.compressed).toContain('TypeError')
      expect(preview.compressed).toContain('src/engine/file.ts:184:12')
      expect(preview.compressed).toContain('case #418')

      // 3. Summary tail around line 900 preserved
      expect(preview.compressed).toContain('1 failed, 899 passed')
      expect(preview.compressed).toContain('Process exited with code 1')
    }
  })

  test('extracts every salient failure marker (FAIL, ✕, AssertionError, Error:, error TS, SyntaxError, TypeError, ReferenceError, Traceback, === FAILURES ===, panic:, FAILED) while ignoring source code throw/extends', () => {
    const markers = [
      'FAIL test/suiteA.test.ts',
      '✕ rejects malformed header [2.1ms]',
      'AssertionError: expected 10 to equal 20',
      'Error: ECONNREFUSED 127.0.0.1:5432',
      'src/compiler/check.ts:91:7 - error TS2345: Argument of type string is not assignable.',
      'SyntaxError: Unexpected token export',
      'TypeError: undefined is not a function',
      'ReferenceError: globalState is not defined',
      'Traceback (most recent call last):',
      '=== FAILURES ===',
      'panic: runtime error: index out of range [4] with length 4',
      'FAILED tests/test_core.py::test_panic - RuntimeError',
    ]

    for (const marker of markers) {
      const lines = Array.from({ length: 300 }, (_, i) => `PASS tests/ok_${i + 1}.test.ts [0.20ms]`)
      lines[149] = marker // 1-based line 150
      lines[150] = '    at runTarget (src/target/module.ts:77:4)'
      lines[299] = '1 failed, 299 passed'
      const raw = lines.join('\n')

      const headroom = compressToolResult('exec_command', raw, { maxChars: 2_000 })
      expect(headroom.compressed.length).toBeLessThanOrEqual(2_000)
      expect(headroom.compressed).toContain(marker)
      expect(headroom.compressed).toContain('[lines 148-153]')

      const pruned = pruneToolResult(raw, { maxChars: 2_000, toolName: 'Bash' })
      expect(String(pruned.content).length).toBeLessThanOrEqual(2_000)
      expect(String(pruned.content)).toContain(marker)
      expect(String(pruned.content)).toContain('[lines 148-153]')
    }

    // Source file containing `throw new TypeError(...)` and `extends Error` should NOT be misclassified as a crash log
    const sourceLines = Array.from({ length: 200 }, (_, idx) => `export const constant_${idx} = ${idx}`)
    sourceLines[100] = 'export class DomainError extends Error {}'
    sourceLines[101] = 'export function validate(x: unknown): void { if (!x) throw new TypeError("missing") }'
    const sourcePreview = compressToolResult('ReadFile', sourceLines.join('\n'), { maxChars: 1_500 })
    expect(sourcePreview.contentType).toBe('text')
    expect(sourcePreview.compressed.length).toBeLessThanOrEqual(1_500)
  })

  test('compressToolResult preserves middle failure hunks inside JSON-serialized ProcessResult stdout/stderr', () => {
    const stdout = build30kTestOutput('bun')
    const processPayload = JSON.stringify({
      command: ['bun', 'test'],
      cwd: '/workspace',
      exitCode: 1,
      stdout,
      stderr: '',
      timedOut: false,
      truncated: false,
    })
    expect(processPayload.length).toBeGreaterThan(30_000)

    const preview = compressToolResult('exec_command', processPayload, { maxChars: 4_000 })
    expect(preview.contentType).toBe('json')
    expect(preview.compressed.length).toBeLessThanOrEqual(4_000)
    expect(preview.compressed).toContain('TypeError')
    expect(preview.compressed).toContain('src/engine/file.ts:184:12')
    expect(preview.compressed).toContain('[lines 418-424]')
    expect(preview.compressed).toContain('1 failed, 899 passed')
  })

  test('deduplicates repeated failure hunks so distinct root-cause errors in the middle both survive', () => {
    const lines = Array.from(
      { length: 700 },
      (_, index) => `PASS tests/noise_${index + 1}.test.ts :: ok [0.10ms]`,
    )
    // Repeat the same AssertionError 15 times across lines 80..360
    for (let repeat = 0; repeat < 15; repeat += 1) {
      const base = 80 + repeat * 18
      lines[base] = '✕ repeated flake test [1.23ms]'
      lines[base + 1] = 'AssertionError: expected status 200 to equal 201 at 0x7ff81234abcd'
      lines[base + 2] = '    at assertOk (src/http/client.ts:99:15)'
    }
    // Distinct root-cause ReferenceError deep at line 520
    lines[519] = 'ReferenceError: sessionTokenCache is not defined'
    lines[520] = '    at verifyAuth (src/auth/guard.ts:311:7)'
    lines[698] = '16 failed, 684 passed in 9.10s'
    lines[699] = 'exit code 1'

    const compressed = compressToolResult('bash', lines.join('\n'), { maxChars: 3_200 })
    expect(compressed.compressed.length).toBeLessThanOrEqual(3_200)
    expect(compressed.compressed).toContain('AssertionError: expected status 200 to equal 201')
    expect(compressed.compressed).toContain('similar omitted')
    expect(compressed.compressed).toContain('ReferenceError: sessionTokenCache is not defined')
    expect(compressed.compressed).toContain('src/auth/guard.ts:311:7')
    expect(compressed.compressed).toContain('16 failed, 684 passed')
  })

  test('pruneToolResult and pruneToolMessages retain middle [lines A-B] failure hunks within maxChars before compaction', () => {
    const bigOutput = build30kTestOutput('pytest')
    const pruned = pruneToolResult(bigOutput, { maxChars: 4_000, toolName: 'Bash' })
    expect(pruned.pruned).toBe(true)
    const text = String(pruned.content)
    expect(text.length).toBeLessThanOrEqual(4_000)
    expect(text).toContain('test session starts')
    expect(text).toContain('[lines 418-424]')
    expect(text).toContain('TypeError')
    expect(text).toContain('src/engine/file.ts:184:12')
    expect(text).toContain('1 failed, 899 passed')
    expect(text).toContain('lines omitted by pre-pruning')

    const messages = [
      {
        role: 'assistant',
        content: '',
        tool_calls: [{ id: 'call-pytest', type: 'function', function: { name: 'Bash', arguments: '{"command":"pytest"}' } }],
      },
      { role: 'tool', tool_call_id: 'call-pytest', content: bigOutput },
      { role: 'user', content: 'fix it' },
      { role: 'assistant', content: 'inspecting' },
    ]
    const batch = pruneToolMessages(messages, { maxChars: 4_000, protectLast: 2 })
    expect(batch.prunedCount).toBe(1)
    const prunedToolContent = String(batch.messages[1]?.content ?? '')
    expect(prunedToolContent.length).toBeLessThanOrEqual(4_000)
    expect(prunedToolContent).toContain('[lines 418-424]')
    expect(prunedToolContent).toContain('src/engine/file.ts:184:12')
  })
})

describe('SWE Working-Set Pinning Across Compaction', () => {
  test('extractSweWorkingSet tracks all mutating file tools, V4A + unified patches, background check_command, spilled <persisted-output>, 5-failure cap, and exit-0 clearing', () => {
    const v4aPatch = [
      '*** Begin Patch',
      '*** Update File: src/core/router.ts',
      '@@',
      '-const retries = 0',
      '+const retries = 2',
      '*** Add File: src/core/retryPolicy.ts',
      '+export const MAX_RETRIES = 2',
      '*** Delete File: src/core/deprecated.ts',
      '*** End Patch',
    ].join('\n')

    const unifiedPatch = [
      'diff --git a/src/utils/hash.ts b/src/utils/hash.ts',
      '--- a/src/utils/hash.ts',
      '+++ b/src/utils/hash.ts',
      '@@ -1,2 +1,2 @@',
      '-export const seed = 1',
      '+export const seed = 2',
    ].join('\n')

    const sixTscErrors = [
      'src/a.ts(10,5): error TS2322: Type A mismatch.',
      'src/b.ts(20,3): error TS2345: Argument B invalid.',
      'src/c.ts(30,9): error TS2532: Object C possibly undefined.',
      'src/d.ts(40,1): error TS2304: Cannot find name D.',
      'src/e.ts(50,7): error TS2741: Property E missing.',
      'src/f.ts(60,2): error TS2339: Property F does not exist.',
    ].join('\n')

    const messages: ContextMessage[] = [
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          { id: 'c1', type: 'function', function: { name: 'WriteFile', arguments: JSON.stringify({ file_path: 'src/newConfig.ts', content: 'export default {}\n' }) } },
          { id: 'c2', type: 'function', function: { name: 'FileEditTool', arguments: JSON.stringify({ file_path: 'src/core/router.ts', old_string: 'a', new_string: 'b' }) } },
          { id: 'c3', type: 'function', function: { name: 'ApplyPatchTool', arguments: JSON.stringify({ patch: v4aPatch }) } },
          { id: 'c4', type: 'function', function: { name: 'git_apply_patch', arguments: JSON.stringify({ patch_content: unifiedPatch, check_only: true }) } },
          { id: 'c5', type: 'function', function: { name: 'git_apply_patch', arguments: JSON.stringify({ patch_content: unifiedPatch }) } },
          { id: 'c6', type: 'function', function: { name: 'move_file', arguments: JSON.stringify({ source: 'src/newConfig.ts', destination: 'src/config/index.ts' }) } },
          { id: 'c7', type: 'function', function: { name: 'delete_file', arguments: JSON.stringify({ path: 'src/obsolete.ts' }) } },
          { id: 'c8', type: 'function', function: { name: 'FileEditTool', arguments: JSON.stringify({ file_path: 'src/brokenEdit.ts', old_string: 'missing', new_string: 'x' }) } },
        ],
      },
      { role: 'tool', tool_call_id: 'c1', content: 'Wrote src/newConfig.ts' },
      { role: 'tool', tool_call_id: 'c2', content: 'Edited src/core/router.ts' },
      { role: 'tool', tool_call_id: 'c3', content: 'Applied patch to 3 files' },
      { role: 'tool', tool_call_id: 'c4', content: 'Patch check clean' },
      { role: 'tool', tool_call_id: 'c5', content: 'Patch applied' },
      { role: 'tool', tool_call_id: 'c6', content: 'Moved file' },
      { role: 'tool', tool_call_id: 'c7', content: 'Deleted file' },
      { role: 'tool', tool_call_id: 'c8', content: 'Tool execution failed: old_string not found in src/brokenEdit.ts', is_error: true },
      // Background test run + foreground tsc check
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          { id: 't1', type: 'function', function: { name: 'exec_command', arguments: JSON.stringify({ cmd: 'bun', args: ['test', 'xerxes/test/router.test.ts'], run_in_background: true }) } },
          { id: 't2', type: 'function', function: { name: 'Bash', arguments: JSON.stringify({ command: 'tsc --noEmit -p xerxes/tsconfig.json' }) } },
        ],
      },
      { role: 'tool', tool_call_id: 't1', content: JSON.stringify({ procId: 'proc_router', running: true }) },
      { role: 'tool', tool_call_id: 't2', content: `${sixTscErrors}\nProcess exited with code 2` },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          { id: 't3', type: 'function', function: { name: 'check_command', arguments: JSON.stringify({ proc_id: 'proc_router' }) } },
        ],
      },
      {
        role: 'tool',
        tool_call_id: 't3',
        content: JSON.stringify({
          running: false,
          exitCode: 1,
          stdout: 'AssertionError: expected 2 retries\n    at src/core/router.ts:88:11\n1 fail\n',
          stderr: '',
        }),
      },
    ]

    const ledger1 = extractSweWorkingSet(messages)
    expect(ledger1.files).toEqual([
      { path: 'src/core/router.ts', operation: 'patched', status: 'applied', tool: 'ApplyPatchTool' },
      { path: 'src/core/retryPolicy.ts', operation: 'created', status: 'applied', tool: 'ApplyPatchTool' },
      { path: 'src/core/deprecated.ts', operation: 'deleted', status: 'applied', tool: 'ApplyPatchTool' },
      { path: 'src/utils/hash.ts', operation: 'patched', status: 'applied', tool: 'git_apply_patch' },
      { path: 'src/config/index.ts', movedFrom: 'src/newConfig.ts', operation: 'moved', status: 'applied', tool: 'move_file' },
      { path: 'src/obsolete.ts', operation: 'deleted', status: 'applied', tool: 'delete_file' },
      { path: 'src/brokenEdit.ts', operation: 'edited', status: 'failed', tool: 'FileEditTool' },
    ])

    expect(ledger1.verifications).toHaveLength(2)
    expect(ledger1.verifications.find(v => v.command === 'tsc --noEmit -p xerxes/tsconfig.json')?.status).toBe('failed')
    expect(ledger1.verifications.find(v => v.command === 'bun test xerxes/test/router.test.ts')?.status).toBe('failed')
    // Active failures strictly capped at 5!
    expect(ledger1.activeFailures).toHaveLength(MAX_ACTIVE_FAILURE_COORDINATES)
    expect(ledger1.activeFailures[0]?.location).toBe('src/a.ts:10')

    // Re-running `tsc --noEmit -p xerxes/tsconfig.json` with exit 0 clears its 5 errors, exposing router.ts:88!
    const afterTscFix = extractSweWorkingSet([
      ...messages,
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          { id: 't4', type: 'function', function: { name: 'Bash', arguments: JSON.stringify({ command: 'tsc --noEmit -p xerxes/tsconfig.json' }) } },
        ],
      },
      { role: 'tool', tool_call_id: 't4', content: 'exit code: 0' },
    ])

    expect(afterTscFix.verifications.find(v => v.command === 'tsc --noEmit -p xerxes/tsconfig.json')).toMatchObject({
      status: 'passed',
      exitCode: 0,
      failures: [],
    })
    expect(afterTscFix.activeFailures).toEqual([
      {
        location: 'src/core/router.ts:88',
        summary: 'AssertionError: expected 2 retries',
        command: 'bun test xerxes/test/router.test.ts',
      },
    ])

    // Also verify spilled `<persisted-output>` envelope from daemon turnRunner composes cleanly with extractSweWorkingSet
    const spilled30k = compressToolResult('exec_command', build30kTestOutput('bun'), { maxChars: 4_000 })
    const persistedEnvelope = `<persisted-output tool="exec_command" chars=31000 shown=${spilled30k.compressedChars} path="/tmp/spill.json">\n${spilled30k.compressed}\nThe full output is on disk at the path above; read it only if the preview is insufficient.\n</persisted-output>`
    const spilledLedger = extractSweWorkingSet([
      {
        role: 'assistant',
        content: '',
        tool_calls: [{ id: 'sp-1', type: 'function', function: { name: 'exec_command', arguments: JSON.stringify({ cmd: 'bun', args: ['test'] }) } }],
      },
      { role: 'tool', tool_call_id: 'sp-1', content: persistedEnvelope },
    ])
    expect(spilledLedger.verifications[0]).toMatchObject({ command: 'bun test', status: 'failed', exitCode: 1 })
    expect(spilledLedger.activeFailures[0]?.location).toBe('src/engine/file.ts:184')
  })

  test('iterative ContextCompressor & CompactionProvisioner carry <swe-working-set> forward, clear resolved test failures, preserve isCompactionSummaryMessage, and honor summaryBudget', () => {
    const counter = new SmartTokenCounter({ model: 'gpt-4o' })
    const compressor = new ContextCompressor({
      contextWindow: 300,
      threshold: 0.2,
      forceSummarize: true,
      protectFirst: 1,
      protectLast: 1,
      summaryMinTokens: 1,
      summaryMaxTokens: 500,
      summarizer: naiveSummarizer,
      tokenCounter: counter,
    })

    // Pass 1: FileEditTool on `src/calc.ts` + failing `bun test xerxes/test/calc.test.ts`
    const pass1Input: ContextMessage[] = [
      { role: 'system', content: 'You are a coding agent.' },
      { role: 'user', content: 'Fix divide-by-zero handling in calc.ts and run tests. ' + 'context '.repeat(40) },
      {
        role: 'assistant',
        content: 'Editing src/calc.ts and running tests.',
        tool_calls: [
          {
            id: 'edit-1',
            type: 'function',
            function: {
              name: 'FileEditTool',
              arguments: JSON.stringify({ file_path: 'src/calc.ts', old_string: 'a / b', new_string: 'b === 0 ? 0 : a / b' }),
            },
          },
          {
            id: 'test-1',
            type: 'function',
            function: {
              name: 'exec_command',
              arguments: JSON.stringify({ cmd: 'bun', args: ['test', 'xerxes/test/calc.test.ts'] }),
            },
          },
        ],
      },
      { role: 'tool', tool_call_id: 'edit-1', content: 'Updated src/calc.ts' },
      {
        role: 'tool',
        tool_call_id: 'test-1',
        content: JSON.stringify({
          command: ['bun', 'test', 'xerxes/test/calc.test.ts'],
          cwd: '/repo',
          exitCode: 1,
          stdout: [
            'TypeError: expected RangeError on negative divisor',
            '    at divide (src/calc.ts:42:9)',
            '1 fail',
          ].join('\n'),
          stderr: '',
          timedOut: false,
          truncated: false,
        }),
      },
      { role: 'user', content: 'Now patch the negative divisor case too.' },
    ]

    const pass1 = compressor.compress(pass1Input)
    expect(pass1.compressed).toBe(true)
    expect(pass1.metadata.strategy).toBe('first-pass')
    expect(pass1.summaryTokens).toBeLessThanOrEqual(500)

    const summaryMsg1 = pass1.messages.find(msg => msg[COMPACTION_SUMMARY_MARKER] === true)
    expect(summaryMsg1).toBeDefined()
    expect(isCompactionSummaryMessage(summaryMsg1)).toBe(true)
    // Reload simulation (without marker property): still recognized as a compaction summary!
    expect(isCompactionSummaryMessage({ role: 'user', content: summaryMsg1?.content })).toBe(true)

    const summaryText1 = String(summaryMsg1?.content ?? '')
    expect(summaryText1.startsWith(COMPACTION_REFERENCE_PREFIX)).toBe(true)
    const wsBlock1 = extractWorkingSetSlice(summaryText1)
    expect(wsBlock1).toContain('- src/calc.ts (edited, applied)')
    expect(wsBlock1).toContain('- [FAIL] bun test xerxes/test/calc.test.ts (exit 1)')
    expect(wsBlock1).toContain('src/calc.ts:42 — TypeError: expected RangeError on negative divisor')

    // Simulate daemon archive-note decoration before Pass 2
    const decoratedPass1Messages = pass1.messages.map(msg =>
      msg === summaryMsg1
        ? {
            ...msg,
            content: `${COMPACTION_REFERENCE_PREFIX}\n${COMPACTION_ARCHIVE_NOTE}${summaryText1.slice(COMPACTION_REFERENCE_PREFIX.length)}`,
          }
        : msg,
    )

    // Pass 2: ApplyPatchTool touches `src/calc.ts` + `src/validators.ts`, and re-running `bun test ./xerxes/test/calc.test.ts` exits 0!
    const patch2 = [
      '*** Begin Patch',
      '*** Update File: src/calc.ts',
      '@@',
      '-return 0',
      '+throw new RangeError("negative")',
      '*** Add File: src/validators.ts',
      '+export function assertPositive(n: number): void {}',
      '*** End Patch',
    ].join('\n')

    const pass2Input: ContextMessage[] = [
      ...decoratedPass1Messages,
      {
        role: 'assistant',
        content: 'Applying multi-file patch and re-running tests. ' + 'work '.repeat(40),
        tool_calls: [
          {
            id: 'patch-2',
            type: 'function',
            function: { name: 'ApplyPatchTool', arguments: JSON.stringify({ patch: patch2 }) },
          },
          {
            id: 'test-2',
            type: 'function',
            function: { name: 'exec_command', arguments: JSON.stringify({ cmd: 'bun', args: ['test', './xerxes/test/calc.test.ts'] }) },
          },
        ],
      },
      { role: 'tool', tool_call_id: 'patch-2', content: 'Applied patch cleanly' },
      {
        role: 'tool',
        tool_call_id: 'test-2',
        content: JSON.stringify({
          command: ['bun', 'test', './xerxes/test/calc.test.ts'],
          cwd: '/repo',
          exitCode: 0,
          stdout: '12 pass\n0 fail\n',
          stderr: '',
          timedOut: false,
          truncated: false,
        }),
      },
      { role: 'user', content: 'Summarize final verification.' },
    ]

    const pass2 = compressor.compress(pass2Input)
    expect(pass2.compressed).toBe(true)
    expect(pass2.metadata.strategy).toBe('iterative')
    expect(pass2.summaryTokens).toBeLessThanOrEqual(500)

    const summaryMsg2 = pass2.messages.find(msg => isCompactionSummaryMessage(msg))
    expect(summaryMsg2).toBeDefined()
    const summaryText2 = String(summaryMsg2?.content ?? '')

    // Exactly ONE `<swe-working-set>` block after iterative carry-forward!
    expect(summaryText2.split(SWE_WORKING_SET_OPEN_TAG).length - 1).toBe(1)
    expect(summaryText2.split(SWE_WORKING_SET_CLOSE_TAG).length - 1).toBe(1)

    const wsBlock2 = extractWorkingSetSlice(summaryText2)
    expect(wsBlock2).toContain('- src/calc.ts (patched, applied)')
    expect(wsBlock2).toContain('- src/validators.ts (created, applied)')
    expect(wsBlock2).toContain('- [PASS] bun test xerxes/test/calc.test.ts (exit 0)')
    expect(wsBlock2).not.toContain('[FAIL]')
    expect(wsBlock2).not.toContain('failure:')
    expect(wsBlock2).not.toContain('src/calc.ts:42')

    const parsedCarried = parseSweWorkingSet(summaryText2)
    expect(parsedCarried.activeFailures).toEqual([])
    expect(parsedCarried.files.map(f => f.path)).toEqual(['src/calc.ts', 'src/validators.ts'])

    // Verify CompactionProvisioner also preserves `<swe-working-set>` and carries `previousSummary` forward
    const provision = new CompactionProvisioner({
      model: 'gpt-4o',
      maxContextTokens: 1_200,
      thresholdTokens: 500,
      targetTokens: 400,
      summaryAgent: () => '## User requests\n- finalize calc module',
    }).compact(
      [
        { role: 'system', content: 'System prompt' },
        { role: 'user', content: 'Run typecheck on the repository. ' + 'padding '.repeat(260) },
        {
          role: 'assistant',
          content: 'Running tsc.',
          tool_calls: [
            { id: 'tsc-1', type: 'function', function: { name: 'Bash', arguments: JSON.stringify({ command: 'tsc --noEmit' }) } },
          ],
        },
        { role: 'tool', tool_call_id: 'tsc-1', content: 'src/validators.ts:12:5 - error TS2322: Bad return type.\nexit code 1\n' + 'tsc detail '.repeat(80) },
        { role: 'user', content: 'Fix that TS error.' },
      ],
      { force: true, previousSummary: summaryText2 },
    )

    expect(provision.compacted).toBe(true)
    const provisionSummary = provision.messages.find(msg => isCompactionSummaryMessage(msg))
    expect(provisionSummary).toBeDefined()
    const provisionWs = extractWorkingSetSlice(String(provisionSummary?.content ?? ''))
    expect(provisionWs).toContain('- src/calc.ts (patched, applied)')
    expect(provisionWs).toContain('- src/validators.ts (created, applied)')
    expect(provisionWs).toContain('- [PASS] bun test xerxes/test/calc.test.ts (exit 0)')
    expect(provisionWs).toContain('- [FAIL] tsc --noEmit (exit 1)')
    expect(provisionWs).toContain('src/validators.ts:12 — error TS2322: Bad return type.')

    // Tight summaryMaxTokens ceiling test: never blows summaryBudget even with 25 touched files!
    const tightCompressor = new ContextCompressor({
      contextWindow: 1_000,
      threshold: 0.1,
      forceSummarize: true,
      protectFirst: 1,
      protectLast: 1,
      summaryMinTokens: 1,
      summaryMaxTokens: 140,
      summarizer: () => 'Short prose summary.',
      tokenCounter: counter,
    })
    const manyFileMessages: ContextMessage[] = [
      { role: 'system', content: 'sys' },
      {
        role: 'assistant',
        content: 'Touching many files ' + 'work '.repeat(120),
        tool_calls: Array.from({ length: 25 }, (_, i) => ({
          id: `mf-${i}`,
          type: 'function',
          function: { name: 'WriteFile', arguments: JSON.stringify({ file_path: `src/deep/module_${i}.ts`, content: 'export {}' }) },
        })),
      },
      ...Array.from({ length: 25 }, (_, i) => ({ role: 'tool', tool_call_id: `mf-${i}`, content: 'ok' })),
      { role: 'user', content: 'done?' },
    ]
    const tightResult = tightCompressor.compress(manyFileMessages)
    expect(tightResult.compressed).toBe(true)
    expect(tightResult.summaryTokens).toBeLessThanOrEqual(140)
    expect(tightResult.tokensAfter).toBeLessThan(tightResult.tokensBefore)
  })
})

describe('Strict maxChars Ceiling & UTF-16 Surrogate-Pair Safety', () => {
  test('never exceeds maxChars or splits UTF-16 surrogate pairs across compressToolResult, pruneToolResult, and renderSweWorkingSet', () => {
    const emojiStream = '😀🚀🧪🔥💥'.repeat(40)
    const lines: string[] = [
      `$ bun test ${emojiStream}`,
      ...Array.from({ length: 220 }, (_, index) => `line_${index}_${'🧭⚡'.repeat(30)}`),
      `TypeError: emoji boundary failure 🚀💥🔥 at parseEmoji (src/unicode/file.ts:184:12) ${emojiStream}`,
      `    at runUnicodeCheck (src/unicode/runner.ts:51:4) ${emojiStream}`,
      ...Array.from({ length: 220 }, (_, index) => `tail_noise_${index}_${'🎯🧩'.repeat(30)}`),
      `FAILED src/unicode/file.ts — 1 failed, 400 passed ${emojiStream}`,
    ]
    const multiLineEmojiText = lines.join('\n')
    const singleLineEmojiText = `START_${emojiStream.repeat(30)}_TypeError: boom at src/u.ts:10:2_${emojiStream.repeat(30)}_END`
    const jsonEmojiPayload = JSON.stringify({
      command: ['bun', 'test'],
      exitCode: 1,
      stdout: multiLineEmojiText,
      stderr: `Error: ${emojiStream}`,
    })

    expect(hasUnpairedSurrogates(multiLineEmojiText)).toBe(false)
    expect(hasUnpairedSurrogates(singleLineEmojiText)).toBe(false)

    for (const budget of [48, 64, 127, 255, 511, 512, 999, 2_048, 4_000]) {
      for (const [tool, input] of [
        ['exec_command', multiLineEmojiText],
        ['Bash', singleLineEmojiText],
        ['exec_command', jsonEmojiPayload],
        ['ReadFile', multiLineEmojiText],
        ['read_binary', 'a\u0000b'],
      ] as const) {
        const compressed = compressToolResult(tool, input, { maxChars: budget })
        expect(compressed.compressed.length).toBeLessThanOrEqual(budget)
        expect(hasUnpairedSurrogates(compressed.compressed)).toBe(false)
        expect(compressed.compressed.isWellFormed()).toBe(true)
      }

      for (const input of [multiLineEmojiText, singleLineEmojiText]) {
        const pruned = pruneToolResult(input, { maxChars: budget, toolName: 'exec_command' })
        const prunedStr = String(pruned.content)
        expect(prunedStr.length).toBeLessThanOrEqual(budget)
        expect(hasUnpairedSurrogates(prunedStr)).toBe(false)
        expect(prunedStr.isWellFormed()).toBe(true)
      }
    }

    const emojiLedger = extractSweWorkingSet([
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          ...Array.from({ length: 12 }, (_, idx) => ({
            id: `f-${idx}`,
            type: 'function',
            function: { name: 'WriteFile', arguments: JSON.stringify({ file_path: `src/pkg_${idx}/module_${idx}.ts`, content: 'x' }) },
          })),
          {
            id: 'cmd-emoji',
            type: 'function',
            function: { name: 'Bash', arguments: JSON.stringify({ command: 'bun test' }) },
          },
        ],
      },
      ...Array.from({ length: 12 }, (_, idx) => ({ role: 'tool', tool_call_id: `f-${idx}`, content: 'ok' })),
      {
        role: 'tool',
        tool_call_id: 'cmd-emoji',
        content: `TypeError: bad emoji ${emojiStream}\n    at src/pkg_0/module_0.ts:77:3\n1 fail`,
      },
    ])

    for (const budgetChars of [16, 40, 96, 180, 320, 600, 1_600]) {
      const rendered = renderSweWorkingSet(emojiLedger, budgetChars)
      expect(rendered.length).toBeLessThanOrEqual(budgetChars)
      expect(hasUnpairedSurrogates(rendered)).toBe(false)
      expect(rendered.isWellFormed()).toBe(true)
      if (rendered.length > 0) {
        expect(rendered.startsWith(SWE_WORKING_SET_OPEN_TAG)).toBe(true)
        expect(rendered.endsWith(SWE_WORKING_SET_CLOSE_TAG)).toBe(true)
      }
    }
  })
})
