// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, test } from 'bun:test'

import type { AgentDefinition } from '../src/agents/definitions.js'
import {
  SpawnBudgetExhaustedError,
  SubAgentManager,
  filterSubagentTools,
} from '../src/agents/subagentManager.js'
import { ValidationError } from '../src/core/errors.js'
import {
  analyzeSplit,
  assignPlacements,
  parseHybridPlan,
  rebindHybridPlan,
  runHybridPlan,
  validateHybridPlan,
  type HybridPlan,
} from '../src/cortex/hybridPlan.js'
import { executeExecutionPlan, type ExecutionPlan, type PlanStep } from '../src/cortex/planner.js'
import { DataBoundaryError } from '../src/security/boundaryTypes.js'
import {
  TaskGraphError,
  dependencyLayers,
  taskContext,
  validateContextTaskReferences,
  type CortexTask,
  type CortexTaskOutput,
} from '../src/cortex/task.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'
import {
  cascadeNext,
  latestAssistantContent,
  runCascadingTurn,
  turnResultFindings,
} from '../src/runtime/cascade.js'
import {
  DEFAULT_MAX_CONSECUTIVE_DENIALS,
  DEFAULT_MAX_SAME_TOOL_POLICY_DENIALS,
  DenialBudget,
  denialBudgetFromConfig,
  denialBudgetStopText,
} from '../src/runtime/denialBudget.js'
import { ErrorKind, classifyError } from '../src/runtime/errorClassifier.js'
import { EscalationController } from '../src/runtime/escalationPolicy.js'
import { inspectObjectiveResponse } from '../src/runtime/objectiveGuard.js'
import type { CheckFinding } from '../src/runtime/resultChecks.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'

describe('cascade & resultChecks bug fixes and cancellation handling', () => {
  test('latestAssistantContent extracts multimodal text parts without leaking image URLs into claim checks', async () => {
    const state = createAgentState()
    state.messages.push({
      role: 'assistant',
      content: [
        { type: 'text', text: 'Here is the generated chart.' },
        { type: 'image_url', image_url: { url: 'https://cdn.example.test/assets/verified-badge.png' } },
      ],
    })
    expect(latestAssistantContent(state)).toBe('Here is the generated chart.')

    state.toolExecutions.push({
      durationMs: 1,
      inputs: { file_path: 'chart.txt' },
      name: 'WriteFile',
      permitted: true,
      result: 'written',
      toolCallId: 'w1',
    })
    const findings = await turnResultFindings(state, 0, '/nonexistent-root')
    expect(findings).toEqual([])
  })

  test('turnResultFindings inspects CSVProcessor writes, Bash test commands, and backgrounded commands accurately', async () => {
    const workspace = await mkdtemp(join(tmpdir(), 'xerxes-swarm-cascade-'))
    try {
      await writeFile(join(workspace, 'out.csv'), 'id,name\n1,Ada\n2\n')

      // 1. CSVProcessor write is treated as a mutating file write (ragged CSV caught + unverified claim flagged).
      const csvWriteState = createAgentState()
      csvWriteState.toolExecutions.push({
        durationMs: 1,
        inputs: { file_path: 'out.csv', operation: 'write' },
        name: 'CSVProcessor',
        permitted: true,
        result: 'ok',
        toolCallId: 'csv-w',
      })
      csvWriteState.messages.push({ role: 'assistant', content: 'I validated out.csv.' })
      const csvFindings = await turnResultFindings(csvWriteState, 0, workspace)
      expect(csvFindings.map(finding => finding.code)).toEqual(['csv_ragged', 'unverified_claim'])

      // 2. Bash test run after a write counts as verification; backgrounded/running exec_command does not.
      const backgroundState = createAgentState()
      backgroundState.toolExecutions.push(
        { durationMs: 1, inputs: { file_path: 'a.ts' }, name: 'WriteFile', permitted: true, result: 'ok', toolCallId: 'w1' },
        {
          durationMs: 1,
          inputs: { cmd: 'bun', args: ['test'] },
          name: 'exec_command',
          permitted: true,
          result: JSON.stringify({ proc_id: 'proc-1', running: true }),
          toolCallId: 'bg-test',
        },
      )
      backgroundState.messages.push({ role: 'assistant', content: 'All tests pass.' })
      const bgFindings = await turnResultFindings(backgroundState, 0, workspace)
      expect(bgFindings.map(finding => finding.code)).toEqual(['unverified_claim'])

      // Completing via check_command or Bash clears unverified_claim.
      backgroundState.toolExecutions.push({
        durationMs: 1,
        inputs: { proc_id: 'proc-1' },
        name: 'check_command',
        permitted: true,
        result: JSON.stringify({ command: 'bun test', exit_code: 0, running: false }),
        toolCallId: 'poll-test',
      })
      expect(await turnResultFindings(backgroundState, 0, workspace)).toEqual([])

      const bashState = createAgentState()
      bashState.toolExecutions.push(
        { durationMs: 1, inputs: { file_path: 'b.ts' }, name: 'WriteFile', permitted: true, result: 'ok', toolCallId: 'w2' },
        {
          durationMs: 1,
          inputs: { command: 'bun test xerxes/test/denialBudget.test.ts' },
          name: 'Bash',
          permitted: true,
          result: JSON.stringify({ exitCode: 0 }),
          toolCallId: 'bash-test',
        },
      )
      bashState.messages.push({ role: 'assistant', content: 'All tests pass.' })
      expect(await turnResultFindings(bashState, 0, workspace)).toEqual([])
    } finally {
      await rm(workspace, { force: true, recursive: true })
    }
  })

  test('cascadeNext consults the advisor even when optional fallback feedback is omitted', async () => {
    const finding: CheckFinding = {
      code: 'xlsx_is_text',
      message: 'plain text in workbook',
      path: 'report.xlsx',
      severity: 'error',
    }
    const consulted: string[] = []
    const next = await cascadeNext(
      {
        checksMode: 'enforce',
        escalation: {
          action: 'ask_advisor',
          advisorRequest: { context: 'report.xlsx is CSV text', question: 'How to fix?' },
          level: 'info',
          message: 'Consulting advisor.',
        },
        findings: [finding],
      },
      {
        checks: () => 'enforce',
        consultAdvisor: async request => {
          consulted.push(request.question)
          return 'Use XlsxWriter instead of WriteFile.'
        },
      },
      new AbortController().signal,
    )
    expect(consulted).toEqual(['How to fix?'])
    expect(next.advisor).toEqual({ answered: true })
    expect(next.feedback).toContain('Use XlsxWriter instead of WriteFile.')
  })

  test('runCascadingTurn stops cleanly without starting another round when aborted during cascade checks', async () => {
    const controller = new AbortController()
    const requests: CompletionRequest[] = []
    const llm: LlmClient = {
      async *stream(request) {
        requests.push(request)
        yield { content: 'First attempt done.' }
      },
    }
    const state = createAgentState()
    const events: StreamEvent[] = []
    for await (const event of runCascadingTurn(
      { model: 'bonsai', state, userMessage: 'Build report' },
      { llm, toolExecutor: new ToolRegistry() },
      {
        hooks: {
          checks: () => 'enforce',
          escalate: findings => findings.length
            ? { action: 'retry_local', feedback: 'Fix the workbook.', level: 'info', message: 'retrying' }
            : undefined,
          inputFindings: async () => {
            controller.abort(new Error('user cancelled during check'))
            return [{ code: 'input_modified', message: 'input changed', path: 'in.csv', severity: 'error' }]
          },
        },
        projectRoot: '/tmp',
      },
      controller.signal,
    )) {
      events.push(event)
    }
    expect(requests).toHaveLength(1)
    expect(events.filter(event => event.type === 'turn_done')).toHaveLength(1)
    expect(state.messages.filter(msg => msg.role === 'user' && msg.origin === 'harness')).toHaveLength(1)
  })
})

describe('EscalationController high-water-mark sawtooth guard and progression context', () => {
  const makeErrors = (count: number, code: CheckFinding['code'] = 'csv_ragged'): CheckFinding[] =>
    Array.from({ length: count }, (_, index) => ({
      code,
      message: `problem ${index}`,
      path: `f${index}.csv`,
      severity: 'error',
    }))

  test('sawtooth regressions (2 -> 5 -> 3) do not reset noImprovementStreak and include regression guidance', () => {
    const controller = new EscalationController({
      advisorAvailable: false,
      alternateModelAvailable: false,
      boundaryMode: 'off',
      config: { maxAdvisorCalls: 0, maxLocalRetries: 5, maxModelChanges: 0, maxUsd: 1, mode: 'on' },
      touchesPrivate: false,
    })

    const first = controller.next({ findings: makeErrors(2) })
    expect(first.action).toBe('retry_local')

    const regressed = controller.next({ findings: makeErrors(5) })
    expect(regressed.action).toBe('retry_local')
    if (regressed.action === 'retry_local') {
      expect(regressed.feedback).toContain('Errors increased compared to an earlier attempt (2 -> 5)')
    }

    // 3 < 5 (better than the regressed attempt), but still worse than the high-water mark of 2!
    const stillWorseThanBest = controller.next({ findings: makeErrors(3) })
    expect(stillWorseThanBest.action).toBe('stop')
    if (stillWorseThanBest.action === 'stop') {
      expect(stillWorseThanBest.reason).toBe('no improvement after 3 attempts')
    }
  })

  test('advisor request carries privacy-safe attempt progression across prior local retries', () => {
    const controller = new EscalationController({
      advisorAvailable: true,
      alternateModelAvailable: false,
      boundaryMode: 'hybrid',
      config: { maxAdvisorCalls: 1, maxLocalRetries: 1, maxModelChanges: 0, maxUsd: 1, mode: 'on' },
      taskSummary: 'Clean the quarterly CSV export.',
      touchesPrivate: false,
    })

    expect(controller.next({ findings: makeErrors(3, 'xlsx_is_text') }).action).toBe('retry_local')
    const advisorStep = controller.next({ findings: makeErrors(2, 'csv_ragged') })
    expect(advisorStep.action).toBe('ask_advisor')
    if (advisorStep.action === 'ask_advisor') {
      expect(advisorStep.context).toContain('Attempt progression: attempt 1: 3 error(s)[xlsx_is_text] -> attempt 2: 2 error(s)[csv_ragged].')
    }
  })
})

describe('objectiveGuard false-positive elimination and read-only shell preservation', () => {
  const passingBunTest = {
    inputs: { cmd: 'bun', args: ['test'] },
    name: 'exec_command',
    permitted: true,
    result: JSON.stringify({ exitCode: 0, stdout: '10 pass' }),
  }

  test('does not reject verified completions mentioning domain "losses" or negated zero-failure counts', () => {
    for (const summary of [
      'All tests pass — verified cross-entropy losses and trading losses.',
      'Verified complete: 24 passing tests, 0 failing tests, no remaining issues, and zero unmet criteria.',
      'All checks pass after we fixed the failing test and resolved all remaining failures.',
    ]) {
      expect(inspectObjectiveResponse(summary, {
        evidence: { toolExecutions: [passingBunTest] },
        mode: 'objective',
      })).toMatchObject({ shouldContinue: false })
    }

    // Genuine unresolved markers and positive benchmark loss counts still block premature stop.
    expect(inspectObjectiveResponse('All tests pass except 1 failing test.', {
      evidence: { toolExecutions: [passingBunTest] },
      mode: 'objective',
    }).shouldContinue).toBe(true)
    expect(inspectObjectiveResponse('Benchmark finished: 3 wins, 2 losses.', {
      evidence: { toolExecutions: [passingBunTest] },
      mode: 'objective',
    })).toMatchObject({
      reason: 'unresolved acceptance marker `losses`',
      shouldContinue: true,
    })
  })

  test('preserves verification across read-only git/shell inspection and recognizes env-prefixed or workspace-flagged test runs', () => {
    // Running `git status` after `bun test` must not wipe out verification.
    expect(inspectObjectiveResponse('All tests pass.', {
      evidence: {
        toolExecutions: [
          passingBunTest,
          {
            inputs: { command: 'git status --short && git diff --stat' },
            name: 'Bash',
            permitted: true,
            result: JSON.stringify({ exitCode: 0, stdout: ' M src/a.ts' }),
          },
        ],
      },
      mode: 'objective',
    }).shouldContinue).toBe(false)

    // Mutating shell command after tests MUST still invalidate verification.
    expect(inspectObjectiveResponse('All tests pass.', {
      evidence: {
        toolExecutions: [
          passingBunTest,
          {
            inputs: { command: 'git status && rm -f src/a.ts' },
            name: 'Bash',
            permitted: true,
            result: JSON.stringify({ exitCode: 0 }),
          },
        ],
      },
      mode: 'objective',
    }).shouldContinue).toBe(true)

    // Env-prefixed shell tests, pnpm -C, python -m pytest, check_command, and git diff --check count as verification.
    for (const verificationExec of [
      {
        inputs: { command: 'CI=1 NO_COLOR=1 bun test xerxes/test/denialBudget.test.ts' },
        name: 'Bash',
        permitted: true,
        result: JSON.stringify({ exitCode: 0 }),
      },
      {
        inputs: { cmd: 'pnpm', args: ['-C', 'xerxes', 'test'] },
        name: 'exec_command',
        permitted: true,
        result: JSON.stringify({ exitCode: 0 }),
      },
      {
        inputs: { cmd: 'python3', args: ['-m', 'pytest', '-q'] },
        name: 'exec_command',
        permitted: true,
        result: JSON.stringify({ exitCode: 0 }),
      },
      {
        inputs: { cmd: 'git', args: ['diff', '--check'] },
        name: 'exec_command',
        permitted: true,
        result: JSON.stringify({ exitCode: 0 }),
      },
      {
        inputs: { proc_id: 'p1' },
        name: 'check_command',
        permitted: true,
        result: JSON.stringify({ command: 'cargo +nightly test', exit_code: 0, running: false }),
      },
    ]) {
      expect(inspectObjectiveResponse('All checks pass.', {
        evidence: { toolExecutions: [verificationExec] },
        mode: 'objective',
      }).shouldContinue).toBe(false)
    }
  })
})

describe('DenialBudget static same-tool policy fast-trip and blank-env handling', () => {
  test('fast-trips repeated static policy_denied on the identical tool while preserving full search budget across tools', () => {
    const sameToolBudget = new DenialBudget()
    expect(sameToolBudget.maxDenials).toBe(DEFAULT_MAX_CONSECUTIVE_DENIALS)
    expect(sameToolBudget.maxSameToolPolicyDenials).toBe(DEFAULT_MAX_SAME_TOOL_POLICY_DENIALS)

    for (let i = 1; i < DEFAULT_MAX_SAME_TOOL_POLICY_DENIALS; i += 1) {
      sameToolBudget.record('policy_denied', 'Bash')
      expect(sameToolBudget.exhausted).toBe(false)
    }
    sameToolBudget.record('policy_denied', 'Bash')
    expect(sameToolBudget.used).toBe(DEFAULT_MAX_SAME_TOOL_POLICY_DENIALS)
    expect(sameToolBudget.exhausted).toBe(true)
    expect(sameToolBudget.remaining).toBe(0)
    expect(denialBudgetStopText(sameToolBudget)).toContain(
      `${DEFAULT_MAX_SAME_TOOL_POLICY_DENIALS} consecutive tool calls were refused`,
    )

    // Cross-tool exploration or user permission rejections keep the full 25-call ceiling.
    const exploring = new DenialBudget()
    const tools = ['Bash', 'exec_command', 'WriteFile', 'FileEditTool']
    for (let i = 0; i < 12; i += 1) {
      exploring.record('policy_denied', tools[i % tools.length]!)
    }
    expect(exploring.exhausted).toBe(false)
    expect(exploring.sameToolPolicyStreak).toBe(1)

    const userPrompted = new DenialBudget()
    for (let i = 0; i < 10; i += 1) {
      userPrompted.record('permission_rejected', 'Bash')
    }
    expect(userPrompted.exhausted).toBe(false)
    expect(userPrompted.sameToolPolicyStreak).toBe(0)
  })

  test('treats whitespace-only env vars as unset and honors configured same-tool policy limits', () => {
    const fromBlankEnv = denialBudgetFromConfig({}, {
      environment: {
        XERXES_MAX_CONSECUTIVE_DENIALS: '   ',
        XERXES_MAX_SAME_TOOL_POLICY_DENIALS: '  ',
      },
    })
    expect(fromBlankEnv.maxDenials).toBe(DEFAULT_MAX_CONSECUTIVE_DENIALS)
    expect(fromBlankEnv.maxSameToolPolicyDenials).toBe(DEFAULT_MAX_SAME_TOOL_POLICY_DENIALS)

    const custom = denialBudgetFromConfig({ max_consecutive_denials: 10, max_same_tool_policy_denials: 3 }, { environment: {} })
    expect(custom.maxDenials).toBe(10)
    expect(custom.maxSameToolPolicyDenials).toBe(3)
  })
})

describe('ErrorClassifier quota/plan exhaustion and Retry-After unit accuracy', () => {
  test('classifies HTTP 400/402 credit exhaustion and stringified 429 quota messages as non-retryable QUOTA_EXCEEDED', () => {
    // Anthropic HTTP 400 low credit balance
    expect(classifyError({
      message: 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing.',
      status: 400,
    })).toMatchObject({ kind: ErrorKind.QUOTA_EXCEEDED, retryable: false })

    // OpenRouter / Together HTTP 402 Payment Required
    expect(classifyError({
      message: 'Payment Required',
      status: 402,
    })).toMatchObject({ kind: ErrorKind.QUOTA_EXCEEDED, retryable: false })

    // Stringified 429 error (as daemon/server.ts passes via `new Error(message)`)
    expect(classifyError(new Error('HTTP 429 Too Many Requests: insufficient_quota — You exceeded your current quota'))).toMatchObject({
      kind: ErrorKind.QUOTA_EXCEEDED,
      retryable: false,
    })
    expect(classifyError(new Error('429 rate_limit_error: usage_limit_reached for current subscription'))).toMatchObject({
      kind: ErrorKind.QUOTA_EXCEEDED,
      retryable: false,
    })

    // Ordinary burst 429 remains retryable RATE_LIMIT
    expect(classifyError(new Error('HTTP 429 Too Many Requests: rate limit reached, please slow down'))).toMatchObject({
      kind: ErrorKind.RATE_LIMIT,
      retryable: true,
    })
  })

  test('converts millisecond Retry-After hints to seconds and unwraps error.cause status/code', () => {
    expect(classifyError(new Error('429 Too Many Requests; please retry after 250ms')).suggestedBackoffSeconds).toBeCloseTo(0.25)
    expect(classifyError(new Error('429 rate limited (retry-after-ms: 1500)')).suggestedBackoffSeconds).toBeCloseTo(1.5)
    expect(classifyError(new Error('429 rate limited; Retry-After: 4s')).suggestedBackoffSeconds).toBe(4)
    expect(classifyError({ message: 'rate limited', retryAfterMs: 750, status: 429 }).suggestedBackoffSeconds).toBeCloseTo(0.75)

    const wrapped = new Error('provider stream failed', { cause: { code: 'ECONNRESET' } })
    expect(classifyError(wrapped)).toMatchObject({ kind: ErrorKind.PROVIDER_DOWN, retryable: true })
  })
})

describe('Cortex DAG whitespace normalization, hybrid plan hardening, and bounded concurrency', () => {
  test('dependencyLayers and context task helpers handle whitespace-padded task and step IDs without false cycle errors', () => {
    const tasks: CortexTask[] = [
      { description: 'first', expectedOutput: 'a', id: ' step-a ' },
      { contextTaskIds: [' step-a '], dependencies: [' step-a '], description: 'second', expectedOutput: 'b', id: ' step-b ' },
    ]
    const layers = dependencyLayers(tasks, item => item.id, item => item.dependencies ?? [], 'task')
    expect(layers).toHaveLength(2)
    expect(() => validateContextTaskReferences(tasks)).not.toThrow()

    const now = new Date()
    const outputs = new Map<string, CortexTaskOutput>([
      ['step-a', { completedAt: now, durationMs: 1, metadata: {}, output: 'alpha result', startedAt: now, status: 'succeeded', taskId: 'step-a' }],
    ])
    expect(taskContext(tasks[1]!, outputs)).toContain('alpha result')

    expect(() => dependencyLayers(
      [{ dependencies: ['   '], id: 'a' }],
      item => item.id,
      item => item.dependencies,
      'task',
    )).toThrow(TaskGraphError)
  })

  test('validateHybridPlan and parseHybridPlan reject cloud-placed tool_use steps and trim JSON identifiers', () => {
    const raw = {
      complexity: 'low',
      estimatedMinutes: 1,
      id: ' plan-trim ',
      objective: 'Test trimming',
      steps: [
        {
          action: 'research',
          arguments: {},
          dependencies: [],
          description: 'fetch topic',
          id: ' s1 ',
          inputs: [{ ref: ' public:doc ', sensitivity: 'shareable' }],
          kind: 'research',
          placement: 'either',
          produces: { ref: ' out:s1 ', sensitivity: 'shareable' },
        },
        {
          action: 'reason',
          arguments: {},
          dependencies: [' s1 '],
          description: 'derive',
          id: ' s2 ',
          inputs: [{ ref: ' out:s1 ', sensitivity: 'shareable' }],
          kind: 'reasoning',
          placement: 'either',
          produces: { ref: ' out:s2 ', sensitivity: 'shareable' },
        },
      ],
      version: 1,
    }
    const parsed = parseHybridPlan(raw)
    expect(parsed.id).toBe('plan-trim')
    expect(parsed.steps.map(item => item.id)).toEqual(['s1', 's2'])

    const cloudToolPlan: HybridPlan = {
      ...parsed,
      steps: [
        {
          ...parsed.steps[0]!,
          kind: 'tool_use',
          placement: 'cloud',
        },
        parsed.steps[1]!,
      ],
    }
    const verdict = validateHybridPlan(cloudToolPlan, 'hybrid')
    expect(verdict.ok).toBe(false)
    if (!verdict.ok) expect(verdict.errors.join(' ')).toContain('tool_use')

    // rebindHybridPlan rejects collisions with internal step output refs.
    expect(() =>
      rebindHybridPlan(parsed, new Map([['public:doc', { ref: 'out:s1', sensitivity: 'shareable' }]])),
    ).toThrow(ValidationError)
  })

  test('in-memory HybridPlan and ExecutionPlan normalize padded step/dependency IDs consistently without boundary bypass or false skips', async () => {
    // 1. Padded dependency [' s1 '] pointing to private step 's1' MUST fail closed in validateHybridPlan and runHybridPlan.
    const tamperedPaddedDep: HybridPlan = {
      complexity: 'low',
      estimatedMinutes: 1,
      id: 'padded-boundary',
      objective: 'test boundary with padded dependency',
      steps: [
        {
          action: 'read_file',
          arguments: {},
          dependencies: [],
          description: 'read secret',
          id: 's1',
          inputs: [{ ref: 'private:secret.csv', sensitivity: 'private' }],
          kind: 'tool_use',
          placement: 'local',
          produces: { ref: 'out:s1', sensitivity: 'private' },
        },
        {
          action: 'reason',
          arguments: {},
          dependencies: [' s1 '],
          description: 'cloud step with padded dependency on private step',
          id: 's2',
          inputs: [],
          kind: 'reasoning',
          placement: 'cloud',
          produces: { ref: 'out:s2', sensitivity: 'shareable' },
        },
      ],
      version: 1,
    }
    const boundaryVerdict = validateHybridPlan(tamperedPaddedDep, 'hybrid')
    expect(boundaryVerdict.ok).toBe(false)
    if (!boundaryVerdict.ok) {
      expect(boundaryVerdict.errors.join(' ')).toContain('s2')
      expect(boundaryVerdict.errors.join(' ')).toContain('private')
    }
    expect(analyzeSplit(tamperedPaddedDep).splittable).toBe(false)
    await expect(runHybridPlan(tamperedPaddedDep, {
      externalInputs: { 'private:secret.csv': 'TOP-SECRET' },
      execute: async () => ({ accepted: true, output: 'nope' }),
    })).rejects.toBeInstanceOf(DataBoundaryError)

    // 2. Valid in-memory HybridPlan with padded step.id (' s1 ') and padded dependencies ([' s1 ']) runs both layers cleanly.
    const validPaddedHybrid: HybridPlan = {
      complexity: 'low',
      estimatedMinutes: 1,
      id: 'padded-run',
      objective: 'run padded in-memory hybrid plan',
      steps: [
        {
          action: 'research',
          arguments: {},
          dependencies: [],
          description: 'step 1',
          id: ' s1 ',
          inputs: [{ ref: ' public:topic ', sensitivity: 'shareable' }],
          kind: 'research',
          placement: 'local',
          produces: { ref: ' out:s1 ', sensitivity: 'shareable' },
        },
        {
          action: 'reason',
          arguments: {},
          dependencies: [' s1 '],
          description: 'step 2',
          id: ' s2 ',
          inputs: [{ ref: 'out:s1', sensitivity: 'shareable' }],
          kind: 'reasoning',
          placement: 'local',
          produces: { ref: 'out:s2', sensitivity: 'shareable' },
        },
      ],
      version: 1,
    }
    expect(validateHybridPlan(validPaddedHybrid, 'hybrid')).toEqual({ ok: true })
    const hybridResult = await runHybridPlan(validPaddedHybrid, {
      externalInputs: { 'public:topic': 'open-data' },
      async execute(step, _location, inputs) {
        return { accepted: true, output: `${step.id.trim()}:${[...inputs.values()].join(',')}` }
      },
    })
    expect(hybridResult.accepted).toBe(true)
    expect(hybridResult.results.map(item => [item.stepId, item.status, item.output])).toEqual([
      ['s1', 'succeeded', 's1:open-data'],
      ['s2', 'succeeded', 's2:s1:open-data'],
    ])

    // 3. assignPlacements and executeExecutionPlan also normalize padded step.id / dependencies consistently.
    const paddedExecutionPlan: ExecutionPlan = {
      complexity: 'low',
      estimatedMinutes: 0,
      id: 'padded-exec',
      objective: 'padded plain plan',
      steps: [
        { action: 'read_file', arguments: { inputs: ' private:key ' }, dependencies: [], description: 'one', id: ' s1 ' },
        { action: 'reason', arguments: { prev: 'result_from_step_s1' }, dependencies: ['s1'], description: 'two', id: ' s2 ' },
      ],
    }
    const placed = assignPlacements(paddedExecutionPlan, {
      boundaryMode: 'hybrid',
      inputSensitivity: ref => (ref.startsWith('private:') ? 'private' : 'shareable'),
    }).plan
    expect(placed.steps.every(item => item.placement === 'local' && item.produces.sensitivity === 'private')).toBe(true)
    expect(validateHybridPlan(placed, 'hybrid')).toEqual({ ok: true })

    const plainResult = await executeExecutionPlan(paddedExecutionPlan, request => `done:${request.step.id.trim()}:${request.arguments.prev ?? ''}`)
    expect(plainResult.outputs.map(item => [item.stepId, item.status, item.output])).toEqual([
      ['s1', 'succeeded', 'done:s1:'],
      ['s2', 'succeeded', 'done:s2:done:s1:'],
    ])
  })

  test('runHybridPlan enforces maxParallel within a layer and skips unstarted same-layer steps on abort', async () => {
    const steps: PlanStep[] = Array.from({ length: 6 }, (_, index) => ({
      action: 'research',
      arguments: { inputs: 'public:seed' },
      dependencies: [],
      description: `step ${index}`,
      id: `p${index}`,
    }))
    const basePlan: ExecutionPlan = {
      complexity: 'medium',
      estimatedMinutes: 1,
      id: 'fanout-plan',
      objective: 'bounded fanout',
      steps,
    }
    const hybrid = assignPlacements(basePlan, {
      boundaryMode: 'hybrid',
      inputSensitivity: () => 'shareable',
    }).plan

    let active = 0
    let peakActive = 0
    const fullRun = await runHybridPlan(hybrid, {
      externalInputs: { 'public:seed': 'ok' },
      maxParallel: 2,
      async execute(step) {
        active += 1
        peakActive = Math.max(peakActive, active)
        await Bun.sleep(5)
        active -= 1
        return { accepted: true, output: step.id }
      },
    })
    expect(fullRun.accepted).toBe(true)
    expect(peakActive).toBe(2)

    // Abort on first step with maxParallel: 1 -> remaining 5 steps in the SAME layer never execute.
    const controller = new AbortController()
    const started: string[] = []
    const abortedRun = await runHybridPlan(
      hybrid,
      {
        externalInputs: { 'public:seed': 'ok' },
        maxParallel: 1,
        async execute(step) {
          started.push(step.id)
          controller.abort()
          return { accepted: true, output: step.id }
        },
      },
      controller.signal,
    )
    expect(started).toEqual(['p0'])
    expect(abortedRun.aborted).toBe(true)
    expect(abortedRun.accepted).toBe(false)
    expect(abortedRun.results.map(item => item.status)).toEqual([
      'succeeded',
      'skipped',
      'skipped',
      'skipped',
      'skipped',
      'skipped',
    ])

    await expect(runHybridPlan(hybrid, {
      externalInputs: { 'public:seed': 'ok' },
      maxParallel: 0,
      execute: async () => ({ accepted: true, output: 'x' }),
    })).rejects.toBeInstanceOf(ValidationError)
  })
})

describe('SubAgentManager wildcard tool patterns, retry spawn-budget enforcement, and worktree leak cleanup', () => {
  test('filterSubagentTools and definition intersection support plugin_* wildcards in allow-lists and exclude-lists', async () => {
    const filtered = filterSubagentTools({
      config: {
        _toolsAllowed: ['ReadFile', 'plugin_*'],
        _toolsExcluded: ['plugin_dangerous_*'],
        _toolsWhitelist: ['ReadFile', 'plugin_jira_*', 'plugin_dangerous_drop'],
      },
      isSubagent: true,
      toolExecutor: name => `ran:${name}`,
      toolSchemas: [
        { name: 'ReadFile' },
        { name: 'WriteFile' },
        { name: 'plugin_jira_search' },
        { name: 'plugin_dangerous_drop' },
      ],
    })
    expect(filtered.toolSchemas.map(schema => schema.name)).toEqual(['ReadFile', 'plugin_jira_search'])
    expect(await filtered.execute?.('plugin_jira_search', {})).toBe('ran:plugin_jira_search')
    expect(await filtered.execute?.('plugin_dangerous_drop', {})).toBe(
      "Error: tool 'plugin_dangerous_drop' is not allowed for this agent.",
    )

    // Intersecting caller concrete tool `plugin_jira_search` with definition `plugin_*` preserves `plugin_jira_search`.
    const observedConfigs: Readonly<Record<string, unknown>>[] = []
    const coderDef: AgentDefinition = {
      allowedTools: ['ReadFile', 'plugin_*'],
      description: 'coder',
      excludeTools: [],
      isolation: '',
      maxDepth: 2,
      model: 'test-model',
      name: 'coder',
      source: 'built-in',
      systemPrompt: 'You are coder.',
      tools: ['ReadFile', 'plugin_*'],
    }
    const manager = new SubAgentManager({
      runner: request => {
        observedConfigs.push(request.config)
        return 'ok'
      },
    })
    try {
      const child = await manager.spawn({
        agentDefinition: coderDef,
        config: { _toolsAllowed: ['ReadFile', 'plugin_jira_search'] },
        prompt: 'search issues',
      })
      await manager.wait(child.id, 1_000)
      expect(observedConfigs[0]?._toolsAllowed).toEqual(['ReadFile', 'plugin_jira_search'])
    } finally {
      await manager.close()
    }
  })

  test('retry enforces maxSpawnedAgents without corrupting archived state and cleans up allocated worktrees on handle failure', async () => {
    const gate = Promise.withResolvers<void>()
    let shouldFailFirst = true
    const manager = new SubAgentManager({
      maxRetainedTerminalTasks: 1,
      maxSpawnedAgents: 1,
      runner: async request => {
        if (shouldFailFirst && request.prompt === 'alpha') {
          shouldFailFirst = false
          throw new Error('alpha boom')
        }
        await gate.promise
        return 'done'
      },
    })
    try {
      const alpha = await manager.spawn({ name: 'alpha', prompt: 'alpha' })
      await manager.wait(alpha.id, 1_000)
      expect(alpha.status).toBe('failed')

      // Spawn beta, which stays running (consuming the 1 live slot).
      const beta = await manager.spawn({ name: 'beta', prompt: 'beta' })
      // Retrying terminal alpha while beta occupies the single live slot must fail with SpawnBudgetExhaustedError!
      await expect(manager.retry('alpha')).rejects.toBeInstanceOf(SpawnBudgetExhaustedError)
      expect(alpha.status).toBe('failed')

      gate.resolve()
      await manager.wait(beta.id, 1_000)

      // Now that beta is terminal, retrying alpha succeeds from the archive!
      expect(manager.findTask('alpha')).toMatchObject({ archived: true, id: alpha.id, status: 'failed' })
      const retried = await manager.retry('alpha', 'alpha retry')
      expect(retried?.id).toBe(alpha.id)
      await manager.wait(alpha.id, 1_000)
      expect(retried?.status).toBe('completed')
    } finally {
      gate.resolve()
      await manager.close()
    }

    // Verify retry cleans up a newly created worktree if handle dispatch throws after allocation.
    const created: string[] = []
    const removed: string[] = []
    const leakGuardManager = new SubAgentManager({
      runner: async () => {
        throw new Error('attempt 0 failed')
      },
      worktree: {
        async create() {
          const path = `/wt/${created.length + 1}`
          created.push(path)
          return { branch: `b-${created.length}`, path }
        },
        async isClean() {
          return true
        },
        async remove(tree) {
          removed.push(tree.path)
        },
      },
    })
    try {
      const task = await leakGuardManager.spawn({ isolation: 'worktree', name: 'wt-worker', prompt: 'run' })
      await leakGuardManager.wait(task.id, 1_000)
      expect(task.status).toBe('failed')
      // Wait for attempt 0 monitor cleanup
      await leakGuardManager.waitFor(() => removed.includes('/wt/1'), { timeoutMs: 1_000 })

      // Force handleManager.sendInput to reject with ValidationError after recreateRetryWorktree allocates /wt/2:
      const internalHandles = (leakGuardManager as unknown as {
        handleManager: { sendInput: () => Promise<never> }
      }).handleManager
      const originalSendInput = internalHandles.sendInput.bind(internalHandles)
      internalHandles.sendInput = async () => {
        throw new ValidationError('message', 'synthetic dispatch failure')
      }
      try {
        await expect(leakGuardManager.retry(task.id, 'retry prompt')).rejects.toBeInstanceOf(ValidationError)
      } finally {
        internalHandles.sendInput = originalSendInput
      }
      expect(created).toEqual(['/wt/1', '/wt/2'])
      expect(removed).toEqual(['/wt/1', '/wt/2'])
      expect(task.status).toBe('failed')
    } finally {
      await leakGuardManager.close()
    }
  })
})
