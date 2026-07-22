// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

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
    await writeFile(join(project, '.agents', 'skills', 'demo-skill', 'SKILL.md'), [
      '---',
      'name: demo-skill',
      'description: Demonstrate live skill discovery.',
      '---',
      'Follow the demo instructions.',
    ].join('\n'), 'utf8')
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
    expect(requests).toHaveLength(1)
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
