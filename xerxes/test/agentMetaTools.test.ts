// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { parseSkillMarkdown, SkillRegistry } from '../src/extensions/skills.js'
import {
  SkillBundleStore,
  mixtureOfAgents,
  registerAgentMetaTools,
  sessionSearchPortFromIndex,
  type SessionSearchRequest,
} from '../src/tools/agentMetaTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

async function execute(registry: ToolRegistry, name: string, arguments_: JsonObject): Promise<JsonObject> {
  return JSON.parse(await registry.execute(call(name, arguments_), { metadata: {} })) as JsonObject
}

test('mixture_of_agents uses configured real members, voting, synthesis, and explicit unavailable state', async () => {
  const registry = new ToolRegistry()
  registerAgentMetaTools(registry, {
    mixture: {
      members: {
        alpha: prompt => prompt === 'choose' ? 'same answer' : 'unexpected',
        beta: async () => ' same   answer ',
      },
      voting: true,
      synthesizer: prompt => prompt.includes('[alpha] same answer') ? 'combined outcome' : 'bad input',
    },
  })

  expect(await execute(registry, 'mixture_of_agents', {
    prompt: 'choose',
    members: ['alpha', 'beta', 'missing'],
  })).toEqual({
    members: ['alpha', 'beta', 'missing'],
    answers: {
      alpha: 'same answer',
      beta: ' same   answer ',
      missing: "[unknown member 'missing']",
    },
    voted: 'same answer',
    final: 'combined outcome',
  })
  expect(await mixtureOfAgents({ prompt: 'nothing' }, { members: {}, voting: false })).toEqual({
    error: 'no MoA members configured',
    members: [],
    answers: {},
  })
})

test('session_search only calls an injected search port and adapts existing indexed history', async () => {
  let received: SessionSearchRequest | undefined
  const registry = new ToolRegistry()
  registerAgentMetaTools(registry, {
    sessionSearch: {
      search: request => {
        received = request
        return { query: request.query, count: 1, hits: [{ session_id: 'session-1', text: 'real hit' }] }
      },
    },
  })
  expect(await execute(registry, 'session_search', {
    query: 'needle',
    limit: 3,
    agent_id: 'reviewer',
    session_id: 'session-1',
  })).toEqual({
    query: 'needle',
    count: 1,
    hits: [{ session_id: 'session-1', text: 'real hit' }],
  })
  expect(received).toEqual({ query: 'needle', limit: 3, agentId: 'reviewer', sessionId: 'session-1' })

  const unavailable = new ToolRegistry()
  registerAgentMetaTools(unavailable)
  expect(await execute(unavailable, 'session_search', { query: 'needle' })).toEqual({
    error: 'no session searcher configured',
    hits: [],
  })

  const port = sessionSearchPortFromIndex({
    search: () => [{
      sessionId: 'indexed-session',
      turnId: 'turn-1',
      agentId: null,
      prompt: 'Question',
      response: 'Answer',
      score: 0.87654,
      bm25Score: 1,
      semanticScore: 0,
      timestamp: '2026-07-13T00:00:00.000Z',
      metadata: {},
    }],
  })
  expect(await port.search({ query: 'Question', limit: 2 })).toEqual({
    query: 'Question',
    count: 1,
    hits: [{
      session_id: 'indexed-session',
      turn_id: 'turn-1',
      agent_id: null,
      prompt: 'Question',
      response: 'Answer',
      score: 0.8765,
      timestamp: '2026-07-13T00:00:00.000Z',
    }],
  })
})

test('skills_list and skill_view use the injected SkillRegistry with transparent lexical fallback', async () => {
  const skills = new SkillRegistry()
  skills.register(parseSkillMarkdown([
    '---',
    'name: deploy',
    'description: Release deployment workflow',
    'version: 2.0.0',
    'tags: [release, ci]',
    '---',
    '',
    'Verify health checks before rollout.',
  ].join('\n'), '/tmp/deploy/SKILL.md'))
  const registry = new ToolRegistry()
  registerAgentMetaTools(registry, { skillRegistry: skills })

  expect(await execute(registry, 'skills_list', {})).toEqual({
    count: 1,
    skills: [{
      name: 'deploy',
      version: '2.0.0',
      description: 'Release deployment workflow',
      tags: ['release', 'ci'],
    }],
  })
  const listed = await execute(registry, 'skills_list', { search: 'release deploy' })
  expect(listed).toMatchObject({
    count: 1,
    query: 'release deploy',
    match_strategy: 'lexical',
    skills: [expect.objectContaining({ name: 'deploy', score: expect.any(Number) })],
  })
  expect(await execute(registry, 'skill_view', { name: 'deploy' })).toMatchObject({
    name: 'deploy',
    instructions: 'Verify health checks before rollout.',
  })
  // A fuzzy miss must report not-found with candidate names, never another skill's body.
  expect(await execute(registry, 'skill_view', { name: 'release deploy' })).toEqual({
    candidates: ['deploy'],
    error: 'not_found',
    name: 'release deploy',
  })
  expect(await execute(registry, 'skill_view', { name: 'unrelated-zzz' })).toEqual({
    candidates: [],
    error: 'not_found',
    name: 'unrelated-zzz',
  })

  const unavailable = new ToolRegistry()
  registerAgentMetaTools(unavailable)
  expect(await execute(unavailable, 'skills_list', {})).toEqual({
    error: 'no skill registry configured',
    skills: [],
  })
})

test('skill_manage writes only through the explicit host-owned SkillBundleStore', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-meta-skills-'))
  try {
    const skills = new SkillRegistry()
    const store = new SkillBundleStore({ directory, registry: skills })
    const registry = new ToolRegistry()
    registerAgentMetaTools(registry, { skillStore: store })

    expect(await execute(registry, 'skill_manage', {
      action: 'create',
      name: 'incident-response',
      description: 'Handle incidents safely',
      version: '0.2.0',
      tags: ['ops', 'safety'],
      instructions: 'Escalate when impact is unknown.',
    })).toMatchObject({
      ok: true,
      name: 'incident-response',
      action: 'create',
      registry_updated: true,
    })
    expect(await execute(registry, 'skills_list', {})).toMatchObject({
      count: 1,
      skills: [expect.objectContaining({ name: 'incident-response', tags: ['ops', 'safety'] })],
    })
    expect(await execute(registry, 'skill_view', { name: 'incident-response' })).toMatchObject({
      name: 'incident-response',
      instructions: 'Escalate when impact is unknown.',
    })
    expect(await execute(registry, 'skill_manage', {
      action: 'delete',
      name: 'incident-response',
    })).toMatchObject({
      ok: true,
      name: 'incident-response',
      registry_updated: false,
      registry_error: expect.stringContaining('no public removal API'),
    })

    const unavailable = new ToolRegistry()
    registerAgentMetaTools(unavailable)
    expect(await execute(unavailable, 'skill_manage', {
      action: 'create',
      name: 'blocked',
      instructions: 'No implicit host writes.',
    })).toEqual({
      ok: false,
      error: 'no writable skill store configured; the host must provide SkillManagementStore explicitly',
    })
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})
