// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, realpath, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import {
  EXPORT_SCHEMA,
  SessionExportError,
  buildSessionExport,
  formatSessionExport,
  listSavedSessions,
  savedSessionSummary,
  selectSavedSession,
} from '../src/runtime/sessionExport.js'

interface SessionFixture {
  readonly agentId?: string
  readonly key?: string
  readonly messages?: readonly unknown[]
  readonly metadata?: Record<string, unknown>
  readonly project: string
  readonly sessionId: string
  readonly toolExecutions?: readonly unknown[]
  readonly turnCount?: number
  readonly updatedAt: string
}

test('session export discovers real JSON records, filters projects, and selects exact or unique matches', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-session-export-'))
  const sessions = join(root, 'sessions')
  const project = join(root, 'project')
  const otherProject = join(root, 'other')
  try {
    await Promise.all([mkdir(sessions), mkdir(project), mkdir(otherProject)])
    await writeSession(sessions, {
      sessionId: 'old11111',
      project,
      key: 'worker-old',
      metadata: { title: 'old session' },
      updatedAt: '2026-06-27T09:00:00.000Z',
    })
    await writeSession(sessions, {
      sessionId: 'new22222',
      project,
      key: 'worker-key',
      metadata: { title: 'target session' },
      updatedAt: '2026-06-27T12:00:00.000Z',
    })
    await writeSession(sessions, {
      sessionId: 'other333',
      project: otherProject,
      updatedAt: '2026-06-27T13:00:00.000Z',
    })
    await writeFile(join(sessions, '.hidden.json'), JSON.stringify({ messages: [{ role: 'user', content: 'hidden' }] }), 'utf8')
    await writeFile(join(sessions, 'broken.json'), '{not-json', 'utf8')
    await writeFile(join(sessions, 'empty.json'), JSON.stringify({ session_id: 'empty', messages: [] }), 'utf8')

    const canonicalProject = await realpath(project)
    const found = await listSavedSessions({ storeDir: sessions, projectDir: project })
    expect(found.map(saved => savedSessionSummary(saved).id)).toEqual(['new22222', 'old11111'])
    expect(savedSessionSummary(found[0]!).project_dir).toBe(canonicalProject)
    expect((await selectSavedSession('', { storeDir: sessions, projectDir: project })).record.session_id).toBe('new22222')
    expect((await selectSavedSession('WORKER-key', { storeDir: sessions })).record.session_id).toBe('new22222')
    await expect(selectSavedSession('worker', { storeDir: sessions, projectDir: project }))
      .rejects.toThrow('matched multiple sessions')
    await expect(selectSavedSession('old', { storeDir: sessions, projectDir: project })).resolves.toMatchObject({
      record: { session_id: 'old11111' },
    })
    await expect(selectSavedSession('new', {
      storeDir: sessions,
      projectDir: 'virtual-project',
      pathResolver: path => path === 'virtual-project' || path === project ? canonicalProject : resolve(path),
    })).resolves.toMatchObject({ record: { session_id: 'new22222' } })
    await expect(selectSavedSession('session', { storeDir: sessions, projectDir: project }))
      .rejects.toThrow('No saved Xerxes session matched')
    await expect(selectSavedSession('', { storeDir: sessions, projectDir: join(root, 'missing') }))
      .rejects.toThrow('No saved Xerxes sessions found')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('session exports join archive and live data and render every supported trace format', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-session-trace-'))
  const sessions = join(root, 'sessions')
  const project = join(root, 'lovely-pirate')
  try {
    await Promise.all([mkdir(sessions), mkdir(project)])
    const path = await writeSession(sessions, {
      sessionId: 'abc12345',
      project,
      key: 'abc12345',
      updatedAt: '2026-06-27T10:00:00.000Z',
      metadata: {
        title: 'clone lovely pirate',
        provider: 'claude-code',
        model: 'claude-code/opus',
        version: '0.3.0',
        tools: ['ReadFile'],
      },
      toolExecutions: [{ name: 'ReadFile', ok: true }],
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'read lovely pirate' }] },
        {
          role: 'assistant',
          content: 'I will inspect the repo.',
          reasoning_content: 'brief plan',
          tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'ReadFile', arguments: '{"file":"README.md"}' } }],
        },
        { role: 'tool', tool_call_id: 'call_1', name: 'ReadFile', content: { text: 'README body' } },
      ],
    })
    await writeFile(
      path.replace(/\.json$/, '.archive.jsonl'),
      JSON.stringify({ role: 'system', content: 'archived setup' }) + '\nnot-json\n',
      'utf8',
    )

    const saved = await selectSavedSession('abc12345', { storeDir: sessions, projectDir: project })
    const canonicalProject = await realpath(project)
    const exported = await buildSessionExport(saved, {
      now: () => new Date('2026-06-28T01:02:03.000Z'),
    })
    expect(exported).toMatchObject({
      schema: EXPORT_SCHEMA,
      exported_at: '2026-06-28T01:02:03.000Z',
      archive_included: true,
      session: { id: 'abc12345', project_dir: canonicalProject },
      usage: { total_input_tokens: 10, total_output_tokens: 20 },
      runtime: { model: 'claude-code/opus', model_provider: 'claude-code' },
    })
    expect(exported.messages).toHaveLength(4)
    expect(exported.archive_messages).toEqual([{ role: 'system', content: 'archived setup' }])
    expect(exported.live_messages[0]).toEqual({ role: 'user', content: [{ type: 'text', text: 'read lovely pirate' }] })
    expect(exported.tool_executions).toEqual([{ name: 'ReadFile', ok: true }])

    const json = JSON.parse(formatSessionExport(exported, 'json')) as { schema: string }
    expect(json.schema).toBe(EXPORT_SCHEMA)
    const jsonl = formatSessionExport(exported, 'jsonl').trim().split('\n').map(line => JSON.parse(line) as Record<string, unknown>)
    expect(jsonl.map(row => row.type)).toEqual(['session', 'message', 'message', 'message', 'message', 'tool_execution'])
    expect(jsonl[1]).toMatchObject({ source: 'archive', index: 0 })
    expect(jsonl[2]).toMatchObject({ source: 'live', index: 1 })

    const markdown = formatSessionExport(exported, 'md')
    expect(markdown).toContain('# Xerxes Session Export: abc12345')
    expect(markdown).toContain('### 2. user (live)')
    expect(markdown).toContain('ReadFile')

    const pirate = formatSessionExport(exported, 'lovely-pirate').trim().split('\n')
      .map(line => JSON.parse(line) as Record<string, unknown>)
    expect(pirate.map(row => row.type)).toEqual([
      'external_session_meta',
      'external_message',
      'external_message',
      'external_message',
      'external_message',
    ])
    expect(pirate[0]).toMatchObject({
      payload: {
        source: 'xerxes',
        cwd: canonicalProject,
        model: 'claude-code/opus',
        total_tokens: 30,
      },
    })
    expect(pirate[2]).toMatchObject({ role: 'user', content: 'read lovely pirate', source: 'live' })
    expect(pirate[3]).toMatchObject({ reasoning_content: 'brief plan', tool_calls: [{ id: 'call_1' }] })
    expect(pirate[4]).toMatchObject({ role: 'tool', tool_call_id: 'call_1', name: 'ReadFile' })
    await expect(Promise.resolve().then(() => formatSessionExport(exported, 'yaml'))).rejects.toBeInstanceOf(SessionExportError)

    const withoutArchive = await buildSessionExport(saved, { includeArchive: false, now: () => new Date('2026-06-28T01:02:03.000Z') })
    expect(withoutArchive).toMatchObject({ archive_included: false, archive_messages: [] })
    expect(withoutArchive.archive_path).toContain('abc12345.archive.jsonl')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

async function writeSession(directory: string, fixture: SessionFixture): Promise<string> {
  const path = join(directory, fixture.sessionId + '.json')
  await writeFile(path, JSON.stringify({
    session_id: fixture.sessionId,
    key: fixture.key ?? fixture.sessionId,
    agent_id: fixture.agentId ?? 'default',
    cwd: fixture.project,
    workspace: join(directory, 'workspace'),
    updated_at: fixture.updatedAt,
    messages: fixture.messages ?? [
      { role: 'user', content: 'read project' },
      { role: 'assistant', content: 'report complete' },
    ],
    turn_count: fixture.turnCount ?? 1,
    interaction_mode: 'code',
    plan_mode: false,
    total_input_tokens: 10,
    total_output_tokens: 20,
    metadata: fixture.metadata ?? {},
    thinking_content: ['thought'],
    tool_executions: fixture.toolExecutions ?? [],
  }, null, 2), 'utf8')
  return path
}
