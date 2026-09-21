// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, realpath, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { hashSkillFile, saveTrustedHashes } from '../src/extensions/skillsGuard.js'

const CLI = join(import.meta.dir, '../src/cli.ts')

test('one-shot CLI exposes native subagents and their catalog to the main model', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-oneshot-'))
  const home = join(root, 'home')
  const project = join(root, 'project')
  const requests: Array<Record<string, unknown>> = []
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(request) {
      requests.push((await request.json()) as Record<string, unknown>)
      if (requests.length === 1) return sseResponse([{ choices: [{ delta: { tool_calls: [{ index: 0, id: 'catalog', function: { name: 'list_available_models', arguments: '{"include_usage":false,"provider_profile":"","query":"","offset":0,"limit":1,"revision":""}' } }] }, finish_reason: 'tool_calls' }] }])
      return sseResponse([
        { choices: [{ delta: { content: 'one-shot ready' }, finish_reason: 'stop' }] },
      ])
    },
  })
  try {
    await Promise.all([
      mkdir(join(home, 'daemon'), { recursive: true }),
      mkdir(join(project, '.agents', 'skills', 'demo-skill'), { recursive: true }),
    ])
    const skillPath = join(project, '.agents', 'skills', 'demo-skill', 'SKILL.md')
    await writeFile(skillPath, [
      '---',
      'name: demo-skill',
      'description: Demonstrate live skill discovery.',
      '---',
      'Follow the demo instructions.',
    ].join('\n'), 'utf8')
    const canonicalSkillPath = await realpath(skillPath)
    await saveTrustedHashes(
      { [canonicalSkillPath]: await hashSkillFile(canonicalSkillPath) },
      { skillsDirectory: join(home, 'skills') },
    )
    await writeFile(
      join(home, 'daemon', 'config.json'),
      JSON.stringify({
        project_directory: project,
        runtime: {
          model: 'gpt-4o',
          provider: 'openai',
          base_url: `${server.url}v1`,
          api_key: 'test-key',
          permission_mode: 'accept-all',
        },
      }),
      'utf8',
    )

    const child = Bun.spawn([process.execPath, CLI, 'inspect independent paths'], {
      cwd: project,
      env: { ...process.env, XERXES_HOME: home },
      stderr: 'pipe',
      stdout: 'pipe',
    })
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])

    expect(exitCode).toBe(0)
    expect(stderr).toBe('')
    expect(stdout).toBe('one-shot ready\n')
    expect(requests).toHaveLength(2)
    expect(JSON.stringify(requests[1]?.messages)).toContain('configured_profiles')
    expect(JSON.stringify(requests[1]?.messages)).not.toContain('Model inventory requires a session')
    const tools = Array.isArray(requests[0]?.tools) ? requests[0].tools : []
    expect(tools).toEqual(expect.arrayContaining([
      expect.objectContaining({ function: expect.objectContaining({ name: 'AgentTool' }) }),
      expect.objectContaining({ function: expect.objectContaining({ name: 'SpawnAgents' }) }),
      expect.objectContaining({ function: expect.objectContaining({ name: 'AwaitAgents' }) }),
      expect.objectContaining({ function: expect.objectContaining({ name: 'SkillTool' }) }),
    ]))
    const messages = Array.isArray(requests[0]?.messages) ? requests[0].messages : []
    const system = messages.find((message): message is { content: string; role: string } => (
      typeof message === 'object'
        && message !== null
        && (message as { role?: unknown }).role === 'system'
        && typeof (message as { content?: unknown }).content === 'string'
    ))
    expect(system?.content).toContain('On non-trivial turns, delegate only independent work that materially helps')
    expect(system?.content).toContain('Available subagent types:')
    expect(system?.content).toContain('- reviewer: Independent read-only code review')
    expect(system?.content).toContain('demo-skill: Demonstrate live skill discovery.')
  } finally {
    server.stop(true)
    await rm(root, { recursive: true, force: true })
  }
})

test('one-shot CLI waits for detached subagents and synthesizes their delivered output', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-oneshot-join-'))
  const home = join(root, 'home')
  const project = join(root, 'project')
  const requests: Array<Record<string, unknown>> = []
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(request) {
      const body = (await request.json()) as Record<string, unknown>
      requests.push(body)
      const messages = Array.isArray(body.messages) ? body.messages : []
      const context = JSON.stringify(messages)
      const userMessages = messages.flatMap(message => (
        typeof message === 'object'
        && message !== null
        && (message as { role?: unknown }).role === 'user'
        && typeof (message as { content?: unknown }).content === 'string'
          ? [(message as { content: string }).content]
          : []
      ))
      if (context.includes('[sub-agent events]')) {
        return completionResponse('Integrated CHILD_OK.')
      }
      if (userMessages.includes('Inspect independently and return CHILD_OK.')) {
        return completionResponse('Child independently found CHILD_OK.')
      }
      if (messages.some(message => (
        typeof message === 'object'
        && message !== null
        && (message as { role?: unknown }).role === 'tool'
        && (message as { name?: unknown }).name === 'SpawnAgents'
      ))) {
        return sseResponse([{
          choices: [{ delta: {}, finish_reason: 'stop' }],
          usage: { prompt_tokens: 1, completion_tokens: 0 },
        }])
      }
      return sseResponse([{
        choices: [{
          delta: {
            tool_calls: [{
              index: 0,
              id: 'spawn-review',
              function: {
                name: 'SpawnAgents',
                arguments: JSON.stringify({
                  agents: [{
                    name: 'review-one',
                    prompt: 'Inspect independently and return CHILD_OK.',
                    subagent_type: 'reviewer',
                    title: 'Independent review',
                  }],
                  wait: false,
                }),
              },
            }],
          },
          finish_reason: 'tool_calls',
        }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      }])
    },
  })

  try {
    await Promise.all([
      mkdir(join(home, 'daemon'), { recursive: true }),
      mkdir(project, { recursive: true }),
    ])
    await writeFile(
      join(home, 'daemon', 'config.json'),
      JSON.stringify({
        project_directory: project,
        runtime: {
          model: 'gpt-4o',
          provider: 'openai',
          base_url: `${server.url}v1`,
          api_key: 'test-key',
          permission_mode: 'accept-all',
        },
      }),
      'utf8',
    )

    const child = Bun.spawn([process.execPath, CLI, 'delegate an independent review'], {
      cwd: project,
      env: { ...process.env, XERXES_HOME: home },
      stderr: 'pipe',
      stdout: 'pipe',
    })
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])

    expect(exitCode).toBe(0)
    expect(stderr).toBe('')
    expect(stdout).toBe('Integrated CHILD_OK.\n')
    expect(requests.some(request => JSON.stringify(request).includes('[sub-agent events]'))).toBeTrue()
    expect(requests.some(request => JSON.stringify(request).includes('Child independently found CHILD_OK.'))).toBeTrue()
  } finally {
    server.stop(true)
    await rm(root, { recursive: true, force: true })
  }
})

test('one-shot CLI exits non-zero when the provider turn fails terminally', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-oneshot-fail-'))
  const home = join(root, 'home')
  const project = join(root, 'project')
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    fetch() {
      return new Response(JSON.stringify({ error: { message: 'invalid api key' } }), {
        headers: { 'Content-Type': 'application/json' },
        status: 401,
      })
    },
  })
  try {
    await Promise.all([
      mkdir(join(home, 'daemon'), { recursive: true }),
      mkdir(project, { recursive: true }),
    ])
    await writeFile(
      join(home, 'daemon', 'config.json'),
      JSON.stringify({
        project_directory: project,
        runtime: {
          model: 'gpt-4o',
          provider: 'openai',
          base_url: `${server.url}v1`,
          api_key: 'test-key',
          permission_mode: 'accept-all',
        },
      }),
      'utf8',
    )

    const child = Bun.spawn([process.execPath, CLI, 'say hello'], {
      cwd: project,
      env: { ...process.env, XERXES_HOME: home },
      stderr: 'pipe',
      stdout: 'pipe',
    })
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])

    // Scripts and CI must observe the failure through the exit code.
    expect(exitCode).toBe(1)
    expect(stderr).toContain('Provider error:')
    expect(stderr).toContain('401')
    expect(stdout).toContain('[Error:')
  } finally {
    server.stop(true)
    await rm(root, { recursive: true, force: true })
  }
})

function completionResponse(content: string): Response {
  return sseResponse([{
    choices: [{ delta: { content }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 1, completion_tokens: 1 },
  }])
}

function sseResponse(events: readonly Record<string, unknown>[]): Response {
  const body = events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n'
  return new Response(body, {
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

test('one-shot child uses TUI-saved tier provider credentials, model and effort', async () => {
  const { AgentSettingsStore } = await import('../src/agents/settingsStore.js')
  const { ProfileStore } = await import('../src/bridge/profiles.js')
  const root = await mkdtemp(join(tmpdir(), 'xerxes-tier-cli-'))
  const home = join(root, 'home'), project = join(root, 'project')
  let parentCalls = 0
  const discoveryCredentials: (string | null)[] = []
  const childRequests: { body: Record<string, unknown>; auth: string | null }[] = []
  const server = Bun.serve({ hostname: '127.0.0.1', port: 0, async fetch(request) {
    if (request.method === 'GET' && new URL(request.url).pathname === '/child/v1/models') {
      discoveryCredentials.push(request.headers.get('authorization'))
      return Response.json({ data: [{ id: 'gpt-5' }] })
    }
    const body = await request.json() as Record<string, unknown>
    if (new URL(request.url).pathname.startsWith('/child/')) {
      childRequests.push({ body, auth: request.headers.get('authorization') })
      return completionResponse('CHILD_PROVIDER_OK')
    }
    if (parentCalls++ > 0) return completionResponse('Delegated successfully.')
    return sseResponse([{ choices: [{ delta: { tool_calls: [{ index: 0, id: 'delegate', function: { name: 'SpawnAgents', arguments: JSON.stringify({ agents: [{ title: 'Tier test', prompt: 'Return CHILD_PROVIDER_OK', subagent_type: 'reviewer', intelligence: 'smart' }], wait: true }) } }] }, finish_reason: 'tool_calls' }] }])
  } })
  try {
    await mkdir(join(home, 'daemon'), { recursive: true }); await mkdir(project)
    await Bun.write(join(home, 'daemon', 'config.json'), JSON.stringify({ runtime: { model: 'gpt-4o', provider: 'openai', base_url: `${server.url}parent/v1`, api_key: 'parent-key', permission_mode: 'accept-all' } }))
    new ProfileStore(join(home, 'profiles.json')).save({ name: 'child-profile', provider: 'openai', model: 'gpt-5', baseUrl: `${server.url}child/v1`, apiKey: 'child-key' })
    new AgentSettingsStore(join(home, 'daemon', 'agent-settings.sqlite')).save({ smart: { model: 'gpt-5', provider_profile: 'child-profile', reasoning_effort: 'high' } }, 0)
    const child = Bun.spawn([process.execPath, CLI, 'Delegate this test'], { cwd: project, env: { ...process.env, XERXES_HOME: home }, stdout: 'pipe', stderr: 'pipe' })
    const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
    expect(code).toBe(0)
    expect(stderr).toBe('')
    expect(stdout).toContain('Delegated successfully')
    expect(childRequests).toHaveLength(1)
    expect(discoveryCredentials).toEqual(['Bearer child-key', 'Bearer child-key'])
    expect(childRequests[0]?.auth).toBe('Bearer child-key')
    expect(childRequests[0]?.body).toMatchObject({ model: 'gpt-5', reasoning_effort: 'high' })
  } finally { server.stop(true); await rm(root, { recursive: true, force: true }) }
})
