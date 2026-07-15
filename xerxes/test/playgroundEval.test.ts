// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import {
  BunWorkspaceModuleEvaluator,
  NativeEvaluationAgent,
  createEvaluationIsolation,
  diagnoseEvaluationFailure,
  formatHardReport,
  formatWarmupReport,
  runHardSuite,
  runWarmupSuite,
  sweepStaleEvaluationRuns,
  type EvaluationAgent,
  type EvaluationSessionPort,
  type EvaluationTurnOptions,
  type EvaluationTurnResult,
  type NativeHardTask,
} from '../playground/index.js'

test('evaluation isolation creates private home/workspace paths and only copies an explicit profile', async () => {
  const root = await temporaryDirectory('xerxes-playground-isolation-')
  const profileSource = join(root, 'profiles')
  await mkdir(profileSource)
  await writeFile(join(profileSource, 'profiles.json'), '{"active":"test"}\n', 'utf8')
  const originalHome = process.env.XERXES_HOME

  try {
    const isolation = await createEvaluationIsolation({
      profileSourceDirectory: profileSource,
      rootDirectory: join(root, 'runs'),
      runId: 'run-test-isolation',
    })
    expect(await readFile(join(isolation.homeDirectory, 'profiles.json'), 'utf8')).toContain('"test"')
    expect(isolation.workspaceDirectory).toContain('run-test-isolation/workspace')
    expect(process.env.XERXES_HOME).toBe(originalHome)
    await expect(createEvaluationIsolation({
      rootDirectory: join(root, 'runs'),
      runId: 'run-test-isolation',
    })).rejects.toThrow('still owned')
    await isolation.cleanup()

    const staleDirectory = join(root, 'runs', 'run-stale')
    await mkdir(staleDirectory, { recursive: true })
    await writeFile(join(staleDirectory, '.evaluation-owner.json'), '{"pid":0}\n', 'utf8')
    await sweepStaleEvaluationRuns(join(root, 'runs'))
    expect(await Bun.file(staleDirectory).exists()).toBeFalse()
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('native evaluation agent collects telemetry, approves requests, and retries only transient errors', async () => {
  let attempts = 0
  let approvals = 0
  let closed = false
  const transport: EvaluationSessionPort = {
    start: async () => ({ model: 'fake-eval-model' }),
    reset: async () => undefined,
    close: async () => {
      closed = true
    },
    approve: async () => {
      approvals += 1
    },
    submit: () => eventsForAttempt(++attempts),
  }
  const agent = new NativeEvaluationAgent({
    retryDelayMs: () => 0,
    sleep: async () => undefined,
    start: {
      homeDirectory: '/tmp/eval-home',
      permissionMode: 'accept-all',
      workspaceDirectory: '/tmp/eval-workspace',
    },
    transport,
  })

  await agent.start()
  const result = await agent.turn('calculate')
  await agent.close()

  expect(agent.model).toBe('fake-eval-model')
  expect(result).toMatchObject({
    contextTokens: 31,
    error: undefined,
    retries: 1,
    text: '391',
    tools: ['MathTool'],
  })
  expect(approvals).toBe(1)
  expect(closed).toBeTrue()
})

test('warm-up suite uses native TypeScript fixtures and scores behavioral edits without a provider', async () => {
  const workspaceDirectory = await temporaryDirectory('xerxes-playground-warmup-')
  const agent = new WarmupFakeAgent(workspaceDirectory)
  try {
    const report = await runWarmupSuite({
      agent,
      moduleEvaluator: new BunWorkspaceModuleEvaluator({ timeoutMs: 5_000 }),
      sandboxDirectory: workspaceDirectory,
    })

    expect(report.rows).toHaveLength(8)
    expect(report.rows.every(row => row.ok)).toBeTrue()
    expect(report.rows.find(row => row.name === 'file_edit')?.detail).toBe('greet() -> "goodbye"')
    expect(report.rows.find(row => row.name === 'bug_fix')?.detail).toBe('add(2,3) -> 5')
    expect(formatWarmupReport(report)).toContain('SCORE: 8/8 passed')
    expect(agent.starts).toBe(1)
    expect(agent.closed).toBe(1)
  } finally {
    await rm(workspaceDirectory, { force: true, recursive: true })
  }
})

test('hard suite uses typed native graders, task-local files, and reports diagnostics', async () => {
  const sandboxDirectory = await temporaryDirectory('xerxes-playground-hard-')
  const agent = new HardFakeAgent()
  const tasks: readonly NativeHardTask[] = [
    {
      category: 'coding',
      difficulty: 4,
      files: [{ path: 'fixture.txt', content: 'native fixture\n' }],
      name: 'native_pass',
      steps: [{ freshSession: true, prompt: 'Inspect {dir}/fixture.txt' }],
      whyHard: 'Task-local fixture and placeholder expansion must be preserved.',
      check: async context => ({
        detail: (await readFile(join(context.workspaceDirectory, 'fixture.txt'), 'utf8')).trim(),
        ok: context.results[0]?.text === 'done',
      }),
    },
    {
      category: 'reason',
      difficulty: 5,
      files: [],
      name: 'native_fail',
      steps: [{ freshSession: false, prompt: 'Answer precisely' }],
      whyHard: 'The response must meet a strict checker.',
      check: () => ({ detail: 'expected exact output', ok: false }),
    },
  ]

  try {
    const report = await runHardSuite({
      agent,
      judge: { diagnose: async request => `specific failure: ${request.graderDetail}` },
      sandboxDirectory,
      tasks,
    })
    expect(report.rows.map(row => row.ok)).toEqual([true, false])
    expect(report.rows[1]?.diagnosis).toBe('specific failure: expected exact output')
    expect(agent.prompts[0]).toContain('native_pass/fixture.txt')
    expect(formatHardReport(report)).toContain('by difficulty: d4=1/1  d5=0/1')
  } finally {
    await rm(sandboxDirectory, { force: true, recursive: true })
  }
})

test('diagnosis is deterministic without an injected judge and never reaches a provider itself', async () => {
  const diagnosis = await diagnoseEvaluationFailure({
    graderDetail: 'the expected value was 42',
    reply: '41',
    taskPrompt: 'compute a number',
    tools: [],
    whyHard: 'off-by-one errors are common',
  }, undefined, undefined)
  expect(diagnosis).toContain('no injected judge')
  expect(diagnosis).toContain('expected value was 42')
})

async function* eventsForAttempt(attempt: number) {
  if (attempt === 1) {
    yield { text: '[Error: service unavailable]', type: 'text' } as const
    yield { type: 'turn_end' } as const
    return
  }
  yield { id: 'approval-1', type: 'approval_request' } as const
  yield { contextTokens: 31, type: 'status' } as const
  yield { name: 'MathTool', type: 'tool_call' } as const
  yield { text: '391', type: 'text' } as const
  yield { type: 'turn_end' } as const
}

class WarmupFakeAgent implements EvaluationAgent {
  closed = 0
  readonly model = 'warmup-fake'
  starts = 0

  constructor(private readonly workspaceDirectory: string) {}

  async close(): Promise<void> {
    this.closed += 1
  }

  async freshSession(): Promise<void> {}

  async start(): Promise<void> {
    this.starts += 1
  }

  async turn(prompt: string, _options?: EvaluationTurnOptions): Promise<EvaluationTurnResult> {
    if (prompt.includes('greeting.ts')) {
      await writeFile(resolve(this.workspaceDirectory, 'greeting.ts'), "export function greet(): string { return 'goodbye' }\n", 'utf8')
      return turn('confirmed')
    }
    if (prompt.includes('calc.ts')) {
      await writeFile(resolve(this.workspaceDirectory, 'calc.ts'), 'export function add(a: number, b: number): number { return a + b }\n', 'utf8')
      return turn('fixed')
    }
    if (prompt.includes('17 times 23')) return turn('391')
    if (prompt.includes('API_VERSION')) return turn('4.2.0')
    if (prompt.includes('TREASURE')) return turn('b.txt', ['GrepTool'])
    if (prompt.includes('eval-ok-7')) return turn('eval-ok-7')
    if (prompt.includes('Multiply my favorite')) return turn('42')
    if (prompt.includes('deploy target I noted')) return turn('fly.io')
    return turn('acknowledged')
  }
}

class HardFakeAgent implements EvaluationAgent {
  closed = 0
  readonly model = 'hard-fake'
  readonly prompts: string[] = []

  async close(): Promise<void> {
    this.closed += 1
  }

  async freshSession(): Promise<void> {}

  async start(): Promise<void> {}

  async turn(prompt: string, _options?: EvaluationTurnOptions): Promise<EvaluationTurnResult> {
    this.prompts.push(prompt)
    return turn('done')
  }
}

function turn(text: string, tools: readonly string[] = []): EvaluationTurnResult {
  return {
    contextTokens: 0,
    error: undefined,
    latencyMs: 1,
    retries: 0,
    text,
    tools,
  }
}

async function temporaryDirectory(prefix: string): Promise<string> {
  return mkdtemp(join(tmpdir(), prefix))
}
