// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

test('Bun CLI exports a persisted session trace without starting Python', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-export-'))
  const sessions = join(root, 'sessions')
  const project = join(root, 'project')
  try {
    await Promise.all([mkdir(sessions), mkdir(project)])
    await writeFile(join(sessions, 'cli-session.json'), JSON.stringify({
      session_id: 'cli-session',
      key: 'cli-session',
      cwd: project,
      updated_at: '2026-07-13T00:00:00.000Z',
      turn_count: 1,
      messages: [{ role: 'user', content: 'export this session' }],
    }), 'utf8')

    const child = Bun.spawn([
      process.execPath,
      join(import.meta.dir, '../src/cli.ts'),
      'export',
      'cli-session',
      '--all-projects',
      '--store-dir',
      sessions,
      '--format',
      'json',
    ], {
      env: { ...process.env, XERXES_HOME: join(root, 'home') },
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
    expect(JSON.parse(stdout)).toMatchObject({
      schema: 'xerxes.session.export.v1',
      session: { id: 'cli-session' },
      messages: [{ role: 'user', content: 'export this session' }],
    })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
