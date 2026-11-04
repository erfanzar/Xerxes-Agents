// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtempSync, realpathSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import type { ProviderProfile } from '../src/bridge/profiles.js'
import { createHybridPlacement } from '../src/daemon/hybridPlanTool.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { SpawnedAgentManager, type SpawnedAgentDescriptor } from '../src/operators/subagents.js'
import { registerClaudeWorkflowTools } from '../src/tools/claudeTools/index.js'
import type { JsonObject } from '../src/types/toolCalls.js'

const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function temp(): string {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-hybrid-plantool-')))
  roots.push(root)
  return root
}

function profile(name: string, baseUrl: string, model: string): ProviderProfile {
  return { name, provider: 'custom', base_url: baseUrl, api_key: '', model, sampling: {} }
}

const STEPS = [
  { id: 'one', agent: 'coder', description: 'Read the customer file and count rows', depends: [] },
  { id: 'two', agent: 'reviewer', description: 'Review the counts for mistakes', depends: ['one'] },
]

interface Setup {
  readonly registry: ToolRegistry
  readonly spawned: SpawnedAgentDescriptor[]
}

function setup(runtime: Record<string, unknown>, profiles: ReadonlyMap<string, ProviderProfile>): Setup {
  const cwd = temp()
  const spawned: SpawnedAgentDescriptor[] = []
  const manager = new SpawnedAgentManager({
    runner: async request => {
      spawned.push(request.agent)
      return { content: `done:${request.agent.id}` }
    },
  })
  const registry = new ToolRegistry()
  registerClaudeWorkflowTools(registry, {
    planGenerator: { generate: async () => STEPS },
    subagentManager: manager,
    hybridPlacement: createHybridPlacement({ runtime: () => runtime, env: {}, cwd: () => cwd, profiles }),
  })
  return { registry, spawned }
}

async function plan(registry: ToolRegistry, args: JsonObject): Promise<{ result?: Record<string, unknown>; error?: string }> {
  try {
    const raw = await registry.execute({ id: crypto.randomUUID(), type: 'function', function: { name: 'PlanTool', arguments: args } }, { metadata: {} })
    try {
      return { result: JSON.parse(raw) as Record<string, unknown> }
    } catch {
      return { error: raw }
    }
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) }
  }
}

const LOCAL_PROFILES = new Map([
  ['bonsai-local', profile('bonsai-local', 'http://127.0.0.1:11581/v1', 'bonsai-9b')],
])

test('with hybrid plans off PlanTool output and agents are unchanged', async () => {
  const { registry, spawned } = setup({}, LOCAL_PROFILES)
  const { result } = await plan(registry, { objective: 'Count rows', execute: true })
  expect(result?.executed).toBe(true)
  expect(result?.hybrid).toBeUndefined()
  const steps = result?.steps as Array<Record<string, unknown>>
  expect(steps.every(step => step.placement === undefined)).toBe(true)
  expect(spawned).toHaveLength(2)
  expect(spawned.every(agent => agent.providerProfile === undefined)).toBe(true)
})

test('with hybrid plans on, a tool step is placed local and pinned to the local model', async () => {
  const { registry, spawned } = setup({ hybrid_plans: 'on', data_boundary: 'hybrid' }, LOCAL_PROFILES)
  const { result, error } = await plan(registry, { objective: 'Count rows', execute: true })
  expect(error).toBeUndefined()
  const steps = result?.steps as Array<{ id: string; placement: string }>
  expect(steps.find(step => step.id === 'one')?.placement).toBe('local')
  // PlanTool nicknames each step's agent `plan-<step id>-<suffix>`.
  const first = spawned.find(agent => agent.id.startsWith('plan-one-'))
  expect(first?.providerProfile).toBe('bonsai-local')
  expect(first?.model).toBe('bonsai-9b')
  // A step not placed local keeps its own agent selection.
  const second = spawned.find(agent => agent.id.startsWith('plan-two-'))
  expect(second).toBeDefined()
  expect(second?.providerProfile).toBeUndefined()
})

test('a dry run reports placements without starting agents', async () => {
  const { registry, spawned } = setup({ hybrid_plans: 'on' }, LOCAL_PROFILES)
  const { result } = await plan(registry, { objective: 'Count rows', execute: false })
  expect(result?.executed).toBe(false)
  expect((result?.hybrid as { notes: string[] }).notes.length).toBeGreaterThan(0)
  expect(spawned).toEqual([])
})

test('under a boundary with no local model, execution is refused before any agent starts', async () => {
  const { registry, spawned } = setup({ hybrid_plans: 'on', data_boundary: 'hybrid' }, new Map())
  const dry = await plan(registry, { objective: 'Count rows', execute: false })
  expect(String((dry.result?.hybrid as { refusal?: string }).refusal)).toContain('local model')
  const run = await plan(registry, { objective: 'Count rows', execute: true })
  expect(run.error ?? JSON.stringify(run.result)).toContain('hybrid plan refused')
  expect(spawned).toEqual([])
})

test('an invalid hybrid_plans value fails loudly instead of running unplaced', async () => {
  const { registry, spawned } = setup({ hybrid_plans: 'sometimes' }, LOCAL_PROFILES)
  const run = await plan(registry, { objective: 'Count rows', execute: true })
  expect(run.error ?? JSON.stringify(run.result)).toContain('hybrid_plans')
  expect(spawned).toEqual([])
})
