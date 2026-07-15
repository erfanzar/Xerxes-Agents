// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import { DaemonEvaluationSessionPort, NativeEvaluationAgent, parseWarmupCliOptions } from '../playground/index.js'
import { InMemoryDaemonRuntime, type DaemonEvent, type DaemonSession, type TurnRunner } from '../src/daemon/runtime.js'

const PLAYGROUND_CLI = resolve(import.meta.dir, '../playground/cli.ts')
const TRANSPORT_FIXTURE = resolve(import.meta.dir, 'playgroundTransportFixture.ts')
const PROJECT_ROOT = resolve(import.meta.dir, '../../..')

test('warm-up CLI argument parsing keeps the legacy short flags and requires explicit transport execution', () => {
  expect(parseWarmupCliOptions(['-v', '-k', 'memory', '--transport', './transport.ts', '--run-root', './runs'])).toMatchObject({
    help: false,
    keyword: 'memory',
    transportModule: './transport.ts',
    verbose: true,
  })
  expect(() => parseWarmupCliOptions(['--keyword'])).toThrow('--keyword requires a non-empty value')
  expect(() => parseWarmupCliOptions(['--unknown'])).toThrow('unknown argument')
})

test('root warm-up script exposes the native CLI without loading a provider', async () => {
  const child = Bun.spawn([process.execPath, 'run', 'playground:warmup', '--', '--help'], {
    cwd: PROJECT_ROOT,
    stderr: 'pipe',
    stdout: 'pipe',
  })
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ])

  expect(exitCode, stderr).toBe(0)
  expect(stdout).toContain('Usage: bun playground/cli.ts')
  expect(stdout).toContain('--transport <module>')
})

test('daemon evaluation transport runs the native runtime with isolated fresh sessions and normalized events', async () => {
  const root = await temporaryDirectory('xerxes-playground-daemon-')
  const runner = new FixtureTurnRunner()
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: root,
    model: 'native-fixture-model',
    sessionDirectory: join(root, 'sessions'),
    workspaceRoot: root,
  })
  const transport = new DaemonEvaluationSessionPort(runtime, { sessionKeyPrefix: 'warmup-test' })
  const agent = new NativeEvaluationAgent({
    start: {
      homeDirectory: join(root, 'home'),
      permissionMode: 'accept-all',
      workspaceDirectory: join(root, 'workspace'),
    },
    transport,
  })

  try {
    await agent.start()
    const first = await agent.turn('first native evaluation prompt', { retries: 0 })
    await agent.freshSession()
    const second = await agent.turn('second native evaluation prompt', { retries: 0 })

    expect(agent.model).toBe('native-fixture-model')
    expect(first).toMatchObject({ contextTokens: 27, text: 'native: first native evaluation prompt', tools: ['ReadFile'] })
    expect(second.text).toBe('native: second native evaluation prompt')
    expect(runner.sessionKeys).toHaveLength(2)
    expect(new Set(runner.sessionKeys).size).toBe(2)
  } finally {
    await agent.close()
    expect(runtime.listSessions()).toHaveLength(0)
    await rm(root, { force: true, recursive: true })
  }
})

test('standalone warm-up CLI runs the complete fixture battery, copies only an explicit profile, and cleans its private run', async () => {
  const root = await temporaryDirectory('xerxes-playground-cli-')
  const profileDirectory = join(root, 'profile')
  const runRoot = join(root, 'runs')
  const tracePath = join(root, 'trace.json')
  const untouchedHome = join(root, 'host-home-must-not-change')
  try {
    await mkdir(profileDirectory, { recursive: true })
    await writeFile(join(profileDirectory, 'profiles.json'), '{"active":"fixture"}\n', 'utf8')

    const result = await executeCli([
      '--transport', TRANSPORT_FIXTURE,
      '--profile-dir', profileDirectory,
      '--run-root', runRoot,
      '--verbose',
    ], {
      XERXES_HOME: untouchedHome,
      XERXES_PLAYGROUND_FIXTURE_TRACE: tracePath,
    })

    expect(result.exitCode, result.stderr).toBe(0)
    expect(result.stderr).toBe('')
    expect(result.stdout).toContain('Xerxes eval playground — model: warmup-fixture-model')
    expect(result.stdout).toContain('· [reasoning]')
    expect(result.stdout).toContain('SCORE: 8/8 passed')

    const trace = JSON.parse(await readFile(tracePath, 'utf8')) as FixtureTrace
    expect(trace.ambientHome).toBe(untouchedHome)
    expect(trace.profile).toBe('{"active":"fixture"}\n')
    expect(trace.starts).toBe(1)
    expect(trace.resets).toBe(9)
    expect(trace.prompts).toHaveLength(10)
    expect(await Bun.file(trace.runDirectory).exists()).toBeFalse()
  } finally {
    await rm(root, { force: true, recursive: true })
  }
}, 30_000)

class FixtureTurnRunner implements TurnRunner {
  readonly sessionKeys: string[] = []

  async *run(session: DaemonSession, text: string): AsyncGenerator<DaemonEvent> {
    this.sessionKeys.push(session.sessionKey)
    yield { type: 'tool_call', payload: { name: 'ReadFile' } }
    yield { type: 'text_part', payload: { text: `native: ${text}` } }
    yield { type: 'status_update', payload: { context_tokens: 27 } }
  }
}

interface FixtureTrace {
  readonly ambientHome: string
  readonly profile: string
  readonly prompts: readonly string[]
  readonly resets: number
  readonly runDirectory: string
  readonly starts: number
}

async function executeCli(
  args: readonly string[],
  environment: Readonly<Record<string, string>>,
): Promise<{ readonly exitCode: number; readonly stderr: string; readonly stdout: string }> {
  const child = Bun.spawn([process.execPath, PLAYGROUND_CLI, ...args], {
    cwd: PROJECT_ROOT,
    env: { ...process.env, ...environment },
    stderr: 'pipe',
    stdout: 'pipe',
  })
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ])
  return { exitCode, stderr, stdout }
}

async function temporaryDirectory(prefix: string): Promise<string> {
  return mkdtemp(join(tmpdir(), prefix))
}
