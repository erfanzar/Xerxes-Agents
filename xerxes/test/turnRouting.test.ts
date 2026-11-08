// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import type { ProviderProfile } from '../src/bridge/profiles.js'
import { createTurnRouter, turnKind, type RoutedProvider } from '../src/daemon/turnRouting.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { DaemonEvent, DaemonSession } from '../src/daemon/runtime.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'
import type { DataBoundaryMode } from '../src/routing/types.js'

function profile(name: string, baseUrl: string, model: string): ProviderProfile {
  return { name, provider: 'custom', base_url: baseUrl, api_key: '', model, sampling: {} }
}

const PROFILES = new Map<string, ProviderProfile>([
  ['bonsai-local', profile('bonsai-local', 'http://127.0.0.1:11581/v1', 'bonsai-9b')],
  ['opus', profile('opus', 'https://api.example.com/v1', 'opus')],
])

const ROUTES = [
  { id: 'local', profile: 'bonsai-local', model: 'bonsai-9b', tier: 1 },
  { id: 'cloud', profile: 'opus', model: 'opus', tier: 3 },
]

function recording(label: string, sink: string[]): LlmClient {
  return { async *stream(request: CompletionRequest) { sink.push(`${label}:${request.model}`); yield { content: 'done' } } }
}

function sessionFor(cwd: string, id: string): DaemonSession {
  return {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd, extra: {}, id,
    interactionMode: 'code', sessionKey: id, lastActive: 0, messages: [], metadata: {},
    model: 'default-model', planMode: false, status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: cwd,
  }
}

async function turn(runtime: Record<string, unknown>, boundary: DataBoundaryMode, text: string): Promise<{ calls: string[]; error?: unknown; events: DaemonEvent[] }> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-turn-routing-'))
  const calls: string[] = []
  const events: DaemonEvent[] = []
  try {
    const runner = new AgentTurnRunner({
      llm: recording('default', calls),
      model: 'default-model',
      routeForSession: createTurnRouter<DaemonSession>({
        runtime: () => runtime,
        env: {},
        profiles: PROFILES,
        boundaryMode: () => boundary,
        buildProvider: (found, model): RoutedProvider => ({ llm: recording(found.name, calls), providerOverrides: { provider: found.provider, base_url: found.base_url } }),
      }),
    })
    try {
      for await (const event of runner.run(sessionFor(root, `route-${calls.length}-${text.length}`), text, new AbortController().signal)) events.push(event)
    } catch (error) {
      return { calls, error, events }
    }
    return { calls, events }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

test('turnKind reads a coarse step kind from the request', () => {
  expect(turnKind('please review this diff')).toBe('review')
  expect(turnKind('summarize the notes')).toBe('summarize')
  expect(turnKind('count the rows in data.csv')).toBe('analysis')
})

test('routing off keeps the configured model and client', async () => {
  const { calls, error } = await turn({ routing_routes: ROUTES }, 'off', 'review this')
  expect(error).toBeUndefined()
  expect(calls).toEqual(['default:default-model'])
})

test('rules sends analysis to the local route and review to the strongest one', async () => {
  expect((await turn({ routing: 'rules', routing_routes: ROUTES }, 'off', 'count the rows')).calls).toEqual(['bonsai-local:bonsai-9b'])
  expect((await turn({ routing: 'rules', routing_routes: ROUTES }, 'off', 'please review this')).calls).toEqual(['opus:opus'])
})

test('private boundary keeps review on the local route; with no local route the turn is refused', async () => {
  expect((await turn({ routing: 'rules', routing_routes: ROUTES }, 'private', 'please review this')).calls).toEqual(['bonsai-local:bonsai-9b'])
  const refused = await turn({ routing: 'rules', routing_routes: [ROUTES[1]] }, 'private', 'please review this')
  expect(refused.calls).toEqual([])
  const surfaced = `${String(refused.error ?? '')}${JSON.stringify(refused.events)}`
  expect(surfaced).toContain('Routing refused this turn')
})
