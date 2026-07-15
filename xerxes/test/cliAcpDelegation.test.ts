// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const CLI = join(import.meta.dir, '../src/cli.ts')

test('ACP CLI advertises the native delegation surface', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-acp-'))
  const home = join(root, 'home')
  const project = join(root, 'project')
  try {
    await Promise.all([
      mkdir(join(home, 'daemon'), { recursive: true }),
      mkdir(project, { recursive: true }),
    ])
    await writeFile(
      join(home, 'daemon', 'config.json'),
      JSON.stringify({
        runtime: {
          model: 'gpt-4o',
          provider: 'openai',
          base_url: 'http://127.0.0.1:1/v1',
          api_key: 'test-key',
          permission_mode: 'accept-all',
        },
      }),
      'utf8',
    )

    const child = Bun.spawn([
      process.execPath,
      CLI,
      'acp',
      '--project-dir',
      project,
    ], {
      env: { ...process.env, XERXES_HOME: home },
      stdin: 'pipe',
      stderr: 'pipe',
      stdout: 'pipe',
    })
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/list', params: {} })}\n`)
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'shutdown', params: {} })}\n`)
    child.stdin.end()

    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])
    expect(exitCode).toBe(0)
    expect(stderr).toBe('')
    const frames = stdout.trim().split('\n').map(line => JSON.parse(line) as Record<string, unknown>)
    const toolsFrame = frames.find(frame => frame.id === 1)
    const tools = Array.isArray(toolsFrame?.result) ? toolsFrame.result : []
    expect(tools).toEqual(expect.arrayContaining([
      expect.objectContaining({ function: expect.objectContaining({ name: 'AgentTool' }) }),
      expect.objectContaining({ function: expect.objectContaining({ name: 'SpawnAgents' }) }),
      expect.objectContaining({ function: expect.objectContaining({ name: 'AwaitAgents' }) }),
      expect.objectContaining({ function: expect.objectContaining({ name: 'SkillTool' }) }),
    ]))
    expect(frames.find(frame => frame.id === 2)?.result).toEqual({ ok: true })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
