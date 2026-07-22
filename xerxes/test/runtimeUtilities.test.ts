// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ChangeGuardReport,
  analyzeStatusLines,
  analyzeWorkspaceChanges,
  formatChangeGuardNotification,
  parsePorcelainStatus,
} from '../src/runtime/changeGuard.js'
import {
  HistoryLog,
} from '../src/runtime/history.js'
import {
  buildInsightsReport,
  formatInsightsReport,
} from '../src/runtime/insights.js'
import {
  BudgetExhausted,
  IterationBudget,
  iterationBudgetFromConfig,
} from '../src/runtime/iterationBudget.js'
import {
  inspectObjectiveResponse,
  normalizeInteractionMode,
  objectiveGuardRetryLimit,
} from '../src/runtime/objectiveGuard.js'
import {
  TranscriptEntry,
  TranscriptStore,
} from '../src/runtime/transcript.js'

test('iteration budgets enforce bounded charges, refund safely, and honor injected config environment', () => {
  const unbounded = new IterationBudget(0)
  expect(unbounded.maxIterations).toBeUndefined()
  expect(unbounded.tryConsume(100)).toBe(true)
  expect(unbounded.remaining).toBeUndefined()

  const bounded = new IterationBudget(2)
  expect(bounded.consume()).toBe(1)
  expect(bounded.remaining).toBe(1)
  expect(bounded.tryConsume()).toBe(true)
  expect(bounded.tryConsume()).toBe(false)
  expect(bounded.exhausted).toBe(true)
  expect(() => bounded.consume()).toThrow(BudgetExhausted)
  expect(bounded.refund(10)).toBe(0)
  expect(() => bounded.consume(0)).toThrow('positive')
  bounded.consume(2)
  bounded.reset()
  expect(bounded.used).toBe(0)

  expect(iterationBudgetFromConfig({ max_tool_turns: 3 }).maxIterations).toBe(3)
  expect(iterationBudgetFromConfig({}, {
    environment: { XERXES_MAX_TOOL_TURNS: '4' },
  }).maxIterations).toBe(4)
  expect(iterationBudgetFromConfig({}, { environment: {} }).maxIterations).toBeUndefined()
  expect(() => iterationBudgetFromConfig({ max_tool_turns: 'bad' })).toThrow(RangeError)
})

test('objective guard accepts verified outcomes or evidenced blockers and rejects premature objective final answers', () => {
  const passingTests = {
    inputs: { cmd: 'bun', args: ['test'] },
    name: 'exec_command',
    permitted: true,
    result: JSON.stringify({ exitCode: 0, timedOut: false, stdout: '12 pass' }),
  }
  const failedTests = {
    inputs: { cmd: 'bun', args: ['test'] },
    name: 'exec_command',
    permitted: true,
    result: JSON.stringify({ exitCode: 1, stderr: 'package not installed', timedOut: false }),
  }
  const successfulWrite = {
    inputs: { file_path: 'src/answer.ts', content: 'export const answer = 42' },
    name: 'WriteFile',
    permitted: true,
    result: 'Wrote src/answer.ts.',
  }

  expect(normalizeInteractionMode('goal-runner')).toBe('objective')
  expect(normalizeInteractionMode('anything', true)).toBe('plan')
  expect(objectiveGuardRetryLimit({}, {
    environment: { XERXES_OBJECTIVE_GUARD_MAX_RETRIES: '9' },
  })).toBe(9)
  expect(objectiveGuardRetryLimit({ objective_guard_max_retries: 'not-a-number' })).toBe(6)
  expect(objectiveGuardRetryLimit({ objective_guard_max_retries: 0 }, {
    environment: { XERXES_OBJECTIVE_GUARD_MAX_RETRIES: '8' },
  })).toBe(8)

  expect(inspectObjectiveResponse('', { mode: 'objective' })).toMatchObject({
    shouldContinue: true,
    reason: 'empty assistant response',
  })
  expect(inspectObjectiveResponse('Verified complete: all tests pass.', { mode: 'objective' })).toMatchObject({
    shouldContinue: true,
    reason: expect.stringContaining('without current-turn verification evidence'),
  })
  expect(inspectObjectiveResponse('Verified complete: all tests pass.', {
    evidence: { verificationSignals: ['bun test completed with exit code 0'] },
    mode: 'objective',
  }).shouldContinue).toBe(false)
  expect(inspectObjectiveResponse('All tests pass.', {
    evidence: { toolExecutions: [passingTests] },
    mode: 'objective',
  }).shouldContinue).toBe(false)
  expect(inspectObjectiveResponse('All tests pass.', {
    evidence: { toolExecutions: [failedTests] },
    mode: 'objective',
  }).shouldContinue).toBe(true)
  expect(inspectObjectiveResponse('All tests pass.', {
    evidence: {
      toolExecutions: [{
        inputs: { cmd: 'bun test' },
        name: 'exec_command',
        permitted: true,
        result: JSON.stringify({ command: 'bun test', exit_code: null, running: true }),
      }],
    },
    mode: 'objective',
  }).shouldContinue).toBe(true)
  expect(inspectObjectiveResponse('All checks pass.', {
    evidence: {
      toolExecutions: [{
        inputs: { scope: 'tests' },
        name: 'CheckAgentMessages',
        permitted: true,
        result: JSON.stringify({ events: [] }),
      }],
    },
    mode: 'objective',
  }).shouldContinue).toBe(true)
  expect(inspectObjectiveResponse('All tests pass.', {
    evidence: {
      toolExecutions: [{
        inputs: { cmd: 'echo', args: ['tests'] },
        name: 'exec_command',
        permitted: true,
        result: JSON.stringify({ exitCode: 0, stdout: 'tests', timedOut: false }),
      }],
    },
    mode: 'objective',
  }).shouldContinue).toBe(true)
  expect(inspectObjectiveResponse('All tests pass.', {
    evidence: { toolExecutions: [passingTests, successfulWrite] },
    mode: 'objective',
  }).shouldContinue).toBe(true)
  expect(inspectObjectiveResponse('All tests pass.', {
    evidence: { toolExecutions: [passingTests, successfulWrite, passingTests] },
    mode: 'objective',
  }).shouldContinue).toBe(false)
  expect(inspectObjectiveResponse('BLOCKED: missing dependency. Evidence: command stderr says not installed.', {
    mode: 'objective',
  }).shouldContinue).toBe(true)
  expect(inspectObjectiveResponse('BLOCKED: missing dependency. Evidence: command stderr says not installed.', {
    evidence: { toolExecutions: [failedTests] },
    mode: 'objective',
  }).shouldContinue).toBe(false)
  expect(inspectObjectiveResponse('BLOCKED: command is still running. Evidence: command output is pending.', {
    evidence: {
      toolExecutions: [{
        inputs: { cmd: 'bun test' },
        name: 'exec_command',
        permitted: true,
        result: JSON.stringify({ command: 'bun test', exit_code: null, running: true }),
      }],
    },
    mode: 'objective',
  }).shouldContinue).toBe(true)
  const rejected = inspectObjectiveResponse('Here is the honest final state: ❌ still losing. Want me to continue?', {
    mode: 'objective',
  })
  expect(rejected).toMatchObject({
    shouldContinue: true,
    reason: expect.stringContaining('unresolved acceptance marker'),
    reminder: expect.stringContaining('[Objective gate]'),
  })
  expect(inspectObjectiveResponse('still losing', { mode: 'code' }).shouldContinue).toBe(false)
})

test('change guard classifies risky porcelain rows, extracts verification evidence, and uses an injected git runner', async () => {
  const report = analyzeStatusLines([
    ' D xerxes/test/standaloneTools.test.ts',
    ' M xerxes/src/tools/standalone.ts',
    ' M xerxes/src/runtime/queryEngine.ts',
  ], [{
    name: 'exec_command',
    inputs: { cmd: 'bun test test/runtimeUtilities.test.ts && bun run check' },
  }])

  expect(report.severity).toBe('error')
  expect(report.shouldNotify).toBe(true)
  expect(report.findings.map(finding => finding.code)).toEqual([
    'deleted-tests',
    'runtime-critical-changed',
  ])
  expect(report.verificationCommands).toEqual(['bun test test/runtimeUtilities.test.ts && bun run check'])
  expect(formatChangeGuardNotification(report)).toContain('Recent verification:')
  expect(report.fingerprint).toHaveLength(40)

  const parsed = parsePorcelainStatus(['R  tests/test_old.py -> tests/test_new.py'])
  expect(parsed[0]).toMatchObject({ oldPath: 'tests/test_old.py', path: 'tests/test_new.py' })
  const throughRunner = await analyzeWorkspaceChanges('/workspace', [], {
    commandRunner: async args => {
      expect(args).toEqual(['git', 'status', '--porcelain=v1', '--untracked-files=no'])
      return { exitCode: 0, stdout: ' M xerxes/src/security/policy.ts\n' }
    },
  })
  expect(throughRunner).toMatchObject({ severity: 'warning', shouldNotify: true })
  const unavailable = await analyzeWorkspaceChanges('/workspace', [], {
    commandRunner: () => ({ exitCode: 1, stdout: '' }),
  })
  expect(unavailable.statusAvailable).toBe(false)
  expect(new ChangeGuardReport().fingerprint).toBe(new ChangeGuardReport().fingerprint)
})

test('history records immutable event snapshots with helper metadata and persistence views', () => {
  const log = new HistoryLog({ now: () => new Date('2026-07-13T10:00:00.000Z') })
  const event = log.addToolCall('ReadFile', 'x'.repeat(220), 12.5)
  log.addError('boom', 'tool')
  log.addTurn('gpt-4o', 10, 20)
  log.addPermission('WriteFile', false)

  expect(event.detail).toHaveLength(200)
  expect(log.eventCount).toBe(4)
  expect(log.filterByKind('error')[0]).toMatchObject({ title: 'boom', detail: 'tool' })
  expect(log.last(2).map(item => item.kind)).toEqual(['turn', 'permission_denied'])
  expect(log.asMarkdown()).toContain('**tool_call**: ReadFile')
  expect(log.asDicts()[2]).toMatchObject({ model: 'gpt-4o', in_tokens: 10, out_tokens: 20 })
  expect(Object.isFrozen(log.events[0] as object)).toBe(true)
  log.clear()
  expect(log.eventCount).toBe(0)
})

test('transcripts preserve message order, provider metadata, flush state, and bounded markdown previews', () => {
  const store = new TranscriptStore({ now: () => new Date('2026-07-13T11:00:00.000Z') })
  store.append('user', 'hello', { request_id: 'one' })
  store.appendEntry(new TranscriptEntry({
    role: 'assistant',
    content: 'x'.repeat(205),
    timestamp: '2026-07-13T11:00:01.000Z',
    metadata: { tool_call_id: 'call-1' },
  }))

  expect(store.flushed).toBe(false)
  expect(store.turnCount).toBe(1)
  expect(store.messageCount).toBe(2)
  expect(store.replay()).toHaveLength(2)
  expect(store.toMessages()).toEqual([
    { role: 'user', content: 'hello', request_id: 'one' },
    { role: 'assistant', content: 'x'.repeat(205), tool_call_id: 'call-1' },
  ])
  expect(store.asMarkdown()).toContain('...')
  store.flush()
  expect(store.flushed).toBe(true)
  store.clear()
  expect(store).toMatchObject({ flushed: false, messageCount: 0 })
})

test('insights aggregate camel and Python-shaped cost events with time filtering, cache usage, and report formatting', () => {
  const now = new Date('2026-07-13T12:00:00.000Z')
  const report = buildInsightsReport([
    {
      model: 'gpt-4o',
      inputTokens: 200,
      outputTokens: 100,
      cacheReadTokens: 800,
      cacheCreationTokens: 5,
      costUsd: 0.02,
      label: 'turn',
      timestamp: '2026-07-13T11:00:00.000Z',
    },
    {
      model: 'claude',
      in_tokens: 50,
      out_tokens: 25,
      cost_usd: 0.01,
      label: 'turn',
      timestamp: '2026-07-03T12:00:00.000Z',
    },
  ], { days: 7, now })

  expect(report).toMatchObject({
    totalEvents: 1,
    totalCostUsd: 0.02,
    totalInputTokens: 200,
    totalOutputTokens: 100,
    totalCacheReadTokens: 800,
    totalCacheCreationTokens: 5,
    cacheHitRate: 0.8,
    projectedMonthlyCost: 0.6,
    byLabel: { turn: 1 },
  })
  expect(report.byModel['gpt-4o']).toMatchObject({ events: 1, costUsd: 0.02 })
  expect(formatInsightsReport(report)).toContain('Top models')
})
