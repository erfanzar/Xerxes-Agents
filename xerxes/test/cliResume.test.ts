// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const CLI = join(import.meta.dir, '../src/cli.ts')

test('CLI keeps a bare --resume interactive when standard input is not a terminal', async () => {
  const child = Bun.spawn([process.execPath, CLI, '--resume', 'deadbeef'], {
    stderr: 'pipe',
    stdout: 'pipe',
  })
  const [stderr, exitCode] = await Promise.all([new Response(child.stderr).text(), child.exited])

  expect(exitCode).not.toBe(0)
  expect(stderr).toContain('The interactive TUI requires a terminal')
})

test('CLI --resume submits a supplied prompt to the persisted native session without interactive waits', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-resume-'))
  const home = join(root, 'home')
  const project = join(root, 'project')
  const sessions = join(home, 'sessions')
  const sessionId = 'feedface'
  const requests: Array<Record<string, unknown>> = []
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(request) {
      requests.push((await request.json()) as Record<string, unknown>)
      if (requests.length === 1) {
        return sseResponse([
          {
            choices: [
              {
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      id: 'write-1',
                      function: {
                        name: 'WriteFile',
                        arguments: '{"file_path":"accepted.txt","content":"accepted without a TUI"}',
                      },
                    },
                  ],
                },
                finish_reason: 'tool_calls',
              },
            ],
          },
        ])
      }
      return sseResponse([
        {
          choices: [{ delta: { content: 'resumed' }, finish_reason: null }],
        },
        {
          choices: [{ delta: { content: ' reply' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 8, completion_tokens: 2 },
        },
      ])
    },
  })
  try {
    await Promise.all([
      mkdir(join(home, 'daemon'), { recursive: true }),
      mkdir(project, { recursive: true }),
      mkdir(sessions, { recursive: true }),
    ])
    await writeFile(
      join(home, 'daemon', 'config.json'),
      JSON.stringify({
        runtime: {
          model: 'gpt-4o',
          provider: 'openai',
          base_url: `${server.url}v1`,
          api_key: 'test-key',
          permission_mode: 'manual',
        },
      }),
      'utf8',
    )
    await writeFile(
      join(sessions, `${sessionId}.json`),
      JSON.stringify({
        session_id: sessionId,
        key: sessionId,
        cwd: project,
        agent_id: 'default',
        updated_at: '2026-07-14T00:00:00.000Z',
        turn_count: 1,
        messages: [
          { role: 'user', content: 'remember this persisted context' },
          { role: 'assistant', content: 'I remembered it.' },
        ],
      }),
      'utf8',
    )

    const child = Bun.spawn([process.execPath, CLI, '--resume', sessionId, 'continue the saved work'], {
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
    expect(stdout).toBe('resumed reply\n')
    expect(requests).toHaveLength(2)
    const firstRequest = requests[0]
    const messages = Array.isArray(firstRequest?.messages) ? firstRequest.messages : []
    expect(messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          role: 'user',
          content: 'remember this persisted context',
        }),
        expect.objectContaining({
          role: 'assistant',
          content: 'I remembered it.',
        }),
        expect.objectContaining({
          role: 'user',
          content: 'continue the saved work',
        }),
      ]),
    )
    const tools = Array.isArray(firstRequest?.tools) ? firstRequest.tools : []
    expect(tools).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          function: expect.objectContaining({ name: 'WriteFile' }),
        }),
      ]),
    )
    expect(tools).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          function: expect.objectContaining({ name: 'AskUserQuestionTool' }),
        }),
      ]),
    )
    expect(await Bun.file(join(project, 'accepted.txt')).text()).toBe('accepted without a TUI')

    const persisted = JSON.parse(await readFile(join(sessions, `${sessionId}.json`), 'utf8')) as {
      messages: Array<{ content?: string; role: string }>
      turn_count: number
    }
    expect(persisted.turn_count).toBe(2)
    expect(persisted.messages).toEqual(
      expect.arrayContaining([
        { role: 'user', content: 'continue the saved work' },
        { role: 'assistant', content: 'resumed reply' },
      ]),
    )
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
