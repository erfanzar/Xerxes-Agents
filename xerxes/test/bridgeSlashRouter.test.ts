// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  BridgeSlashRouter,
  parseBridgeSlashCommand,
  type BridgeProviderProfile,
} from '../src/bridge/slashRouter.js'
import { SkillRegistry, type Skill } from '../src/extensions/skills.js'
import { CostTracker } from '../src/runtime/costTracker.js'
import { createAgentState } from '../src/streaming/events.js'

const profile: BridgeProviderProfile = {
  name: 'local',
  api_key: 'local-key',
  base_url: 'http://localhost:11434/v1',
  model: 'llama3.3',
  provider: 'ollama',
  sampling: { temperature: 0.2 },
  active: true,
}

test('native bridge slash parsing rejects non-slash input and canonicalizes aliases without I/O', () => {
  expect(parseBridgeSlashCommand('hello')).toEqual({ raw: 'hello', error: 'Slash commands must start with `/`.' })
  expect(parseBridgeSlashCommand('/h')).toMatchObject({ name: 'help', args: '' })
  expect(parseBridgeSlashCommand('/thinking@XerxesBot high')).toMatchObject({ name: 'reasoning', args: 'high' })
  expect(parseBridgeSlashCommand('/custom arg value')).toMatchObject({ name: 'custom', args: 'arg value' })
})

test('router renders help, cost, history, context, config, and clears the caller-owned state', async () => {
  const state = createAgentState([
    { role: 'user', content: 'one' },
    { role: 'assistant', content: 'two' },
  ])
  state.thinkingContent.push('thought')
  state.toolExecutions.push({
    durationMs: 1,
    inputs: {},
    name: 'Read',
    permitted: true,
    result: 'ok',
    toolCallId: 'call_1',
  })
  state.totalInputTokens = 100
  state.totalOutputTokens = 20
  state.turnCount = 3
  const costs = new CostTracker()
  costs.recordTurn('gpt-4o', 100, 20)
  const router = new BridgeSlashRouter({
    cwd: '/workspace',
    state,
    config: { model: 'gpt-4o', verbose: false, _secret: 'do-not-render' },
    host: { costTracker: costs },
  })

  expect((await router.dispatch('/help')).output).toContain('/provider')
  expect((await router.dispatch('/cost')).output).toContain('Total cost:')
  expect((await router.dispatch('/history')).output).toBe('2 messages, 3 turns')
  expect((await router.dispatch('/context')).output).toContain('Provider: openai')
  expect((await router.dispatch('/context')).output).toContain('CWD: /workspace')
  expect((await router.dispatch('/config')).output).not.toContain('_secret')
  expect(await router.dispatch('/clear')).toMatchObject({ status: 'handled', output: 'Conversation cleared.' })
  expect(state.messages).toEqual([])
  expect(state.thinkingContent).toEqual([])
  expect(state.toolExecutions).toEqual([])
  expect(state.turnCount).toBe(0)
})

test('model and provider routing use explicit hosts and only confirm real host transitions', async () => {
  const state = createAgentState()
  const config: Record<string, unknown> = { model: 'gpt-4o', base_url: 'https://api.openai.com/v1', api_key: 'key' }
  const calls: string[] = []
  const router = new BridgeSlashRouter({
    cwd: '/workspace',
    state,
    config,
    host: {
      models: {
        async list(input) {
          calls.push(`list:${input.currentModel}`)
          return ['gpt-4o', 'gpt-4.1']
        },
        async switchModel(model) { calls.push(`model:${model}`) },
      },
      providers: {
        active: () => profile,
        list: () => [profile],
        saveSampling: () => profile,
        select: async name => name === 'local' ? profile : undefined,
      },
    },
  })

  expect((await router.dispatch('/model')).output).toContain('Available models (2):')
  expect(await router.dispatch('/model gpt-4.1')).toMatchObject({ status: 'handled', output: 'Model set to: gpt-4.1' })
  expect(config.model).toBe('gpt-4.1')
  expect((await router.dispatch('/provider')).output).toContain('* local')
  expect(await router.dispatch('/provider local')).toMatchObject({
    status: 'handled',
    output: "Switched to 'local'  (model: llama3.3)",
  })
  expect(config).toMatchObject({ model: 'llama3.3', provider: 'ollama', base_url: 'http://localhost:11434/v1' })
  expect(calls).toEqual(['list:gpt-4o', 'model:gpt-4.1'])
})

test('sampling, thinking, permissions, and yolo mutate caller config and persist through provider ports', async () => {
  const config: Record<string, unknown> = { model: 'gpt-4o', permission_mode: 'accept-all' }
  const saved: Array<Record<string, unknown>> = []
  const changes: Array<Record<string, unknown>> = []
  const router = new BridgeSlashRouter({
    cwd: '/workspace',
    state: createAgentState(),
    config,
    host: {
      configChanged: snapshot => { changes.push({ ...snapshot }) },
      providers: {
        active: () => profile,
        list: () => [profile],
        saveSampling: (_name, sampling) => {
          saved.push({ ...sampling })
          return profile
        },
        select: () => profile,
      },
    },
  })

  expect(await router.dispatch('/sampling temperature 0.4')).toMatchObject({
    status: 'handled',
    output: 'temperature = 0.4',
  })
  expect(config.temperature).toBe(0.4)
  expect(await router.dispatch('/thinking high')).toMatchObject({
    status: 'handled',
    output: 'Thinking effort set to: high',
  })
  expect(config).toMatchObject({ thinking: true, reasoning_effort: 'high' })
  expect(saved.at(-1)).toMatchObject({ thinking: true, reasoning_effort: 'high' })
  expect(await router.dispatch('/permissions')).toMatchObject({ output: 'Permission mode: manual' })
  expect(await router.dispatch('/yolo')).toMatchObject({ output: 'YOLO mode ON (accept-all)' })
  expect(await router.dispatch('/sampling nope 1')).toMatchObject({ status: 'invalid' })
  expect(await router.dispatch('/sampling max_tokens 10.5')).toMatchObject({ status: 'invalid' })
  expect(changes).toHaveLength(4)
})

test('compaction, listings, plan, and skills call real ports while absent ports stay unavailable', async () => {
  const state = createAgentState([
    { role: 'user', content: 'one' },
    { role: 'assistant', content: 'two' },
    { role: 'user', content: 'three' },
    { role: 'assistant', content: 'four' },
  ])
  const registry = new SkillRegistry()
  const skill: Skill = {
    sourcePath: '/skills/review/SKILL.md',
    instructions: 'Review changes.',
    metadata: {
      name: 'review',
      description: 'Review a patch',
      version: '1.0',
      tags: ['code'],
      resources: [],
      author: '',
      dependencies: [],
      requiredTools: [],
      platforms: ['linux'],
      configVars: [],
      trustLevel: 'local',
      source: 'local',
      setupCommand: '',
      subcommands: [],
    },
  }
  registry.register(skill)
  const invokes: string[] = []
  const router = new BridgeSlashRouter({
    cwd: '/workspace',
    platform: 'linux',
    state,
    config: { model: 'gpt-4o' },
    host: {
      compact: compactedState => {
        compactedState.messages.splice(0, 2)
        return { compacted: true, summarizedCount: 2, keptCount: 2 }
      },
      tools: () => [{ name: 'Read', description: 'Read a file', safe: true }],
      agents: () => [{ name: 'coder', description: 'Writes code' }],
      plan: objective => `Plan: ${objective}`,
      skills: {
        registry,
        invoke: input => {
          invokes.push(`${input.name}:${input.args}`)
          return 'Skill dispatched.'
        },
      },
    },
  })

  expect((await router.dispatch('/compact')).output).toContain('Compacted 4 messages -> 2 messages.')
  expect((await router.dispatch('/tools')).output).toContain('Read [safe]')
  expect((await router.dispatch('/agents')).output).toContain('coder — Writes code')
  expect(await router.dispatch('/plan add auth')).toMatchObject({ output: 'Plan: add auth' })
  expect(await router.dispatch('/skill review:carefully')).toMatchObject({ output: 'Skill dispatched.' })
  expect(invokes).toEqual(['review:carefully'])

  const unavailable = new BridgeSlashRouter({ cwd: '/workspace', state: createAgentState(), config: {} })
  expect(await unavailable.dispatch('/tools')).toMatchObject({ status: 'unavailable' })
  expect(await unavailable.dispatch('/exit')).toMatchObject({ status: 'unavailable' })
  expect(await unavailable.dispatch('/snapshot')).toMatchObject({ status: 'unavailable' })

  let compactCalled = false
  const noModel = new BridgeSlashRouter({
    cwd: '/workspace',
    state: createAgentState([
      { role: 'user', content: 'one' },
      { role: 'assistant', content: 'two' },
      { role: 'user', content: 'three' },
      { role: 'assistant', content: 'four' },
    ]),
    config: {},
    host: {
      compact: () => {
        compactCalled = true
        return { compacted: false }
      },
    },
  })
  expect(await noModel.dispatch('/compact')).toMatchObject({ output: 'No model configured. Run /provider first.' })
  expect(compactCalled).toBe(false)
})
