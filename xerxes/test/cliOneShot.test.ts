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
    expect(system?.content).toContain('On every non-trivial turn, decide internally whether delegation helps')
    expect(system?.content).toContain('Available subagent types:')
    expect(system?.content).toContain('- reviewer: Independent read-only code review')
    expect(system?.content).toContain('demo-skill: Demonstrate live skill discovery.')
  } finally {
    server.stop(true)
    await rm(root, { recursive: true, force: true })
  }
})

function sseResponse(events: readonly Record<string, unknown>[]): Response {
  const body = events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n'
  return new Response(body, {
    headers: { 'Content-Type': 'text/event-stream' },
  })
}
