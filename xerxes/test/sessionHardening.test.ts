// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { Database } from 'bun:sqlite'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { SessionRecord, TurnRecord } from '../src/session/models.js'
import { SessionIndex } from '../src/session/search.js'
import { SnapshotManager } from '../src/session/snapshots.js'

function turn(turnId: string, prompt = 'hello world'): TurnRecord {
  return new TurnRecord({
    turnId,
    prompt,
    responseContent: 'a useful response',
    startedAt: '2026-01-01T00:00:00.000Z',
  })
}

test('shadow snapshots never track the broader secret denylist', async () => {
  if (!Bun.which('git')) return
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-snapshot-secret-denylist-'))
  const workspace = join(directory, 'workspace')
  try {
    mkdirSync(join(workspace, '.ssh'), { recursive: true })
    writeFileSync(join(workspace, 'a.txt'), 'tracked content', 'utf8')
    const secrets = [
      '.env.local',
      '.npmrc',
      '.netrc',
      join('.ssh', 'id_ed25519'),
      'secrets.yaml',
      'cert.p12',
      'store.keystore',
      'kubeconfig',
    ]
    for (const secret of secrets) writeFileSync(join(workspace, secret), 'secret-material', 'utf8')

    const manager = new SnapshotManager(workspace, { shadowRoot: join(directory, 'shadow') })
    await manager.snapshot('base')

    const tracked = await manager.runGit(['ls-files'])
    expect(tracked).toContain('a.txt')
    for (const secret of secrets) {
      expect(tracked.split(/\r?\n/)).not.toContain(secret.split('\\').join('/'))
    }

    // Secrets are also preserved across a full-tree rollback.
    writeFileSync(join(workspace, 'a.txt'), 'changed', 'utf8')
    const records = manager.list()
    await manager.rollback(records[0]!.id)
    for (const secret of secrets) {
      expect(readFileSync(join(workspace, secret), 'utf8')).toBe('secret-material')
    }
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('rollback removes ignored build outputs and captures a pre-rollback snapshot', async () => {
  if (!Bun.which('git')) return
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-snapshot-rollback-safety-'))
  const workspace = join(directory, 'workspace')
  try {
    mkdirSync(join(workspace, 'build'), { recursive: true })
    writeFileSync(join(workspace, '.gitignore'), 'build/\n', 'utf8')
    writeFileSync(join(workspace, 'a.txt'), 'first version', 'utf8')
    const manager = new SnapshotManager(workspace, { shadowRoot: join(directory, 'shadow') })
    const first = await manager.snapshot('first')

    // Ignored build output created after the snapshot must not survive rollback.
    writeFileSync(join(workspace, 'build', 'output.js'), 'post-snapshot artifact', 'utf8')
    writeFileSync(join(workspace, 'a.txt'), 'dirty version', 'utf8')
    await manager.rollback(first.id)

    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('first version')
    expect(existsSync(join(workspace, 'build', 'output.js'))).toBe(false)

    // The pre-rollback snapshot can itself restore the pre-rollback tree.
    // (Ignored build output is never tracked by design, so only tracked
    // files return; the second clean still removes the ignored artifact.)
    const preRollback = manager.list().at(-1)
    expect(preRollback?.label).toBe(`pre-rollback:${first.id}`)
    await manager.rollback(preRollback!.id)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('dirty version')
    expect(existsSync(join(workspace, 'build', 'output.js'))).toBe(false)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('snapshot refs reject empty and short prefixes and raise on ambiguity', async () => {
  if (!Bun.which('git')) return
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-snapshot-ref-guard-'))
  const workspace = join(directory, 'workspace')
  try {
    mkdirSync(workspace)
    writeFileSync(join(workspace, 'a.txt'), 'content', 'utf8')
    const manager = new SnapshotManager(workspace, { shadowRoot: join(directory, 'shadow') })
    await manager.snapshot('base')

    // Empty and short refs must not silently match the first record.
    expect(manager.get('')).toBeUndefined()
    expect(manager.get('ab1')).toBeUndefined()
    await expect(manager.rollback('')).rejects.toThrow('snapshot not found')
    await expect(manager.rollback('ab1')).rejects.toThrow('snapshot not found')

    // Two records sharing a commit-SHA prefix make the prefix ambiguous.
    writeFileSync(
      join(manager.shadowDirectory, '_records.txt'),
      [
        ['id-one', 'one', `aaaa1111${'0'.repeat(32)}`, '2026-01-01T00:00:00.000Z', workspace].join('\t'),
        ['id-two', 'two', `aaaa2222${'0'.repeat(32)}`, '2026-01-02T00:00:00.000Z', workspace].join('\t'),
        '',
      ].join('\n'),
      'utf8',
    )
    expect(manager.get('aaaa1')?.id).toBe('id-one')
    expect(() => manager.get('aaaa')).toThrow(/ambiguous snapshot ref/)
    await expect(manager.rollback('aaaa')).rejects.toThrow(/ambiguous snapshot ref/)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

/** Borrowed-connection fake whose FTS MATCH query fails with a caller-chosen error. */
function matchFailingDatabase(matchError: Error, likeRows: Array<Record<string, unknown>> = []): Database {
  return {
    run() {},
    close() {},
    query(sql: string) {
      return {
        run() {},
        get() { return null },
        all() {
          if (sql.includes('MATCH')) throw matchError
          return likeRows
        },
      }
    },
  } as unknown as Database
}

test('SessionIndex degrades to LIKE only for malformed MATCH expressions', () => {
  const database = matchFailingDatabase(new Error('fts5: syntax error near "%"'), [{
    session_id: 'fallback',
    turn_id: 't1',
    agent_id: null,
    prompt: 'hello world',
    response: 'a useful response',
    started_at: '2026-01-01T00:00:00.000Z',
    metadata: '{}',
    embedding: '',
  }])
  const index = new SessionIndex({ database })
  expect(index.ftsAvailable).toBe(true)
  const hits = index.search('100%')
  expect(hits).toHaveLength(1)
  expect(hits[0]?.sessionId).toBe('fallback')
})

test('SessionIndex surfaces FTS storage failures instead of silently degrading to LIKE', () => {
  // A storage failure that merely mentions fts5 is not a user syntax error.
  const database = matchFailingDatabase(new Error('fts5: database disk image is malformed'))
  const index = new SessionIndex({ database })
  expect(() => index.search('hello')).toThrow(/disk image is malformed/)
})

test('SessionIndex surfaces a missing FTS table instead of downgrading to LIKE', () => {
  const database = new Database(':memory:')
  try {
    const index = new SessionIndex({ database })
    index.indexSession(new SessionRecord({ sessionId: 's', turns: [turn('t1')] }))
    expect(index.search('hello')).toHaveLength(1)
    database.run('DROP TABLE session_turns_fts')
    expect(() => index.search('hello')).toThrow(/no such table/)
  } finally {
    database.close()
  }
})
