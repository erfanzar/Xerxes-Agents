// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'

import type {
  EvaluationEvent,
  EvaluationSessionPort,
  EvaluationStartRequest,
  EvaluationSubmitRequest,
  EvaluationTransportContext,
} from '../playground/types.js'

interface FixtureTrace {
  readonly ambientHome: string | undefined
  readonly homeDirectory: string
  readonly profile: string
  readonly prompts: readonly string[]
  readonly resets: number
  readonly runDirectory: string
  readonly starts: number
  readonly workspaceDirectory: string
}

/** Fixture transport loaded by the standalone CLI subprocess test. */
export async function createEvaluationSessionPort(context: EvaluationTransportContext): Promise<EvaluationSessionPort> {
  const prompts: string[] = []
  let resets = 0
  let starts = 0

  const writeTrace = async (): Promise<void> => {
    const tracePath = process.env.XERXES_PLAYGROUND_FIXTURE_TRACE
    if (!tracePath) return
    const profile = await profileText(context.homeDirectory)
    const trace: FixtureTrace = {
      ambientHome: process.env.XERXES_HOME,
      homeDirectory: context.homeDirectory,
      profile,
      prompts,
      resets,
      runDirectory: context.runDirectory,
      starts,
      workspaceDirectory: context.workspaceDirectory,
    }
    await writeFile(tracePath, JSON.stringify(trace), 'utf8')
  }

  return {
    approve: async () => undefined,
    close: writeTrace,
    reset: async () => {
      resets += 1
    },
    start: async request => {
      assertContext(request, context)
      starts += 1
      return { model: 'warmup-fixture-model' }
    },
    submit: request => eventsForPrompt(request, context, prompts),
  }
}

async function* eventsForPrompt(
  request: EvaluationSubmitRequest,
  context: EvaluationTransportContext,
  prompts: string[],
): AsyncGenerator<EvaluationEvent> {
  const prompt = request.prompt
  prompts.push(prompt)
  if (prompt.includes('greeting.ts')) {
    await writeFile(resolve(context.workspaceDirectory, 'greeting.ts'), "export function greet(): string { return 'goodbye' }\n", 'utf8')
    yield { type: 'text', text: 'confirmed' }
  } else if (prompt.includes('calc.ts')) {
    await writeFile(resolve(context.workspaceDirectory, 'calc.ts'), 'export function add(a: number, b: number): number { return a + b }\n', 'utf8')
    yield { type: 'text', text: 'fixed' }
  } else if (prompt.includes('17 times 23')) {
    yield { type: 'text', text: process.env.XERXES_PLAYGROUND_FIXTURE_FAIL === '1' ? '390' : '391' }
  } else if (prompt.includes('API_VERSION')) {
    yield { type: 'text', text: '4.2.0' }
  } else if (prompt.includes('TREASURE')) {
    yield { type: 'tool_call', name: 'GrepTool' }
    yield { type: 'text', text: 'b.txt' }
  } else if (prompt.includes('eval-ok-7')) {
    yield { type: 'text', text: 'eval-ok-7' }
  } else if (prompt.includes('Multiply my favorite')) {
    yield { type: 'text', text: '42' }
  } else if (prompt.includes('deploy target I noted')) {
    yield { type: 'text', text: 'fly.io' }
  } else {
    yield { type: 'text', text: 'acknowledged' }
  }
  yield { type: 'turn_end' }
}

function assertContext(request: EvaluationStartRequest, context: EvaluationTransportContext): void {
  if (request.homeDirectory !== context.homeDirectory || request.workspaceDirectory !== context.workspaceDirectory) {
    throw new Error('CLI did not forward the private evaluation paths')
  }
  if (request.permissionMode !== 'accept-all') throw new Error('CLI did not enforce accept-all evaluation permissions')
}

async function profileText(homeDirectory: string): Promise<string> {
  try {
    return await readFile(join(homeDirectory, 'profiles.json'), 'utf8')
  } catch {
    return ''
  }
}
