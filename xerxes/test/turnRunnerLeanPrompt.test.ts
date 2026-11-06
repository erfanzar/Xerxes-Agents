// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { parsePromptProfileConfig } from '../src/runtime/leanPrompt.js'
import type { DaemonSession } from '../src/daemon/runtime.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

function sessionFor(cwd: string, id: string): DaemonSession {
  return {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd, extra: {}, id,
    interactionMode: 'code', sessionKey: id, lastActive: 0, messages: [], metadata: {},
    model: 'bonsai-9b', planMode: false, status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: cwd,
  }
}

function tool(name: string): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description: `${name} does one thing. It also has a long second sentence that the lean profile drops.`,
      parameters: { type: 'object', properties: { path: { type: 'string', description: 'x'.repeat(200) } } },
    },
  }
}

const TOOLS = [tool('ReadFile'), tool('WriteFile'), tool('AgentTool'), tool('Workflow'), tool('exec_command')]
const BOOTSTRAP = 'FULL BOOTSTRAP PROMPT ' + 'with lots of doctrine '.repeat(200)

async function requestFor(profile: 'lean' | 'full' | undefined): Promise<CompletionRequest> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-lean-runner-'))
  try {
    const requests: CompletionRequest[] = []
    const llm: LlmClient = { async *stream(request) { requests.push(request); yield { content: 'done' } } }
    const runner = new AgentTurnRunner({
      llm,
      model: 'bonsai-9b',
      tools: TOOLS,
      bootstrapSystemPrompt: () => BOOTSTRAP,
      ...(profile === undefined ? {} : { promptProfileForSession: () => profile }),
    })
    for await (const event of runner.run(sessionFor(root, `lean-${profile ?? 'none'}`), 'List the files.', new AbortController().signal)) void event
    expect(requests.length).toBeGreaterThan(0)
    return requests[0]!
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

const systemText = (request: CompletionRequest): string =>
  request.messages.filter(message => message.role === 'system').map(message => String(message.content)).join('\n')
const toolNames = (request: CompletionRequest): string[] => (request.tools ?? []).map(entry => entry.function.name)

test('without a prompt profile the request is unchanged: full bootstrap and every tool', async () => {
  for (const profile of [undefined, 'full'] as const) {
    const request = await requestFor(profile)
    expect(systemText(request)).toContain('FULL BOOTSTRAP PROMPT')
    expect(toolNames(request)).toEqual(TOOLS.map(entry => entry.function.name))
  }
})

test('lean profile sends the terse prompt and only core tools with short descriptions', async () => {
  const request = await requestFor('lean')
  const system = systemText(request)
  expect(system).not.toContain('FULL BOOTSTRAP PROMPT')
  expect(system).toContain('Never say you checked or validated something you did not run')
  expect(toolNames(request)).toEqual(['ReadFile', 'WriteFile', 'exec_command'])
  for (const entry of request.tools ?? []) {
    expect(entry.function.description).not.toContain('second sentence')
  }
  expect(system.length).toBeLessThan(4_000)
})

test("the prompt-profile hook sees the turn's window, so a small window can choose lean on its own", async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-lean-runner-window-'))
  try {
    const seen: unknown[] = []
    const requests: CompletionRequest[] = []
    const llm: LlmClient = { async *stream(request) { requests.push(request); yield { content: 'done' } } }
    const runner = new AgentTurnRunner({
      llm,
      model: 'ternary-bonsai-27b',
      contextLimit: 16_384,
      tools: TOOLS,
      bootstrapSystemPrompt: () => BOOTSTRAP,
      promptProfileForSession: (_session, turn) => {
        seen.push(turn)
        return parsePromptProfileConfig({}, {}, undefined, turn.contextLimit === undefined ? {} : { contextLimit: turn.contextLimit })
      },
    })
    for await (const event of runner.run(sessionFor(root, 'lean-window'), 'List the files.', new AbortController().signal)) void event
    expect(seen[0]).toMatchObject({ contextLimit: 16_384 })
    expect(systemText(requests[0]!)).not.toContain('FULL BOOTSTRAP PROMPT')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
