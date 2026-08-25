// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, mock, test } from 'bun:test'
import { Database } from 'bun:sqlite'
import { createHash } from 'node:crypto'
import * as nodeFs from 'node:fs'
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, utimesSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import { FileStorage, SQLiteStorage } from '../src/memory/index.js'

// ESM module namespace objects expose live bindings, so a snapshot of the
// original fs exports is needed so mock.module can be restored after tests.
const originalFs = Object.fromEntries(Object.entries(nodeFs)) as typeof nodeFs

function hashOf(key: string): string {
  return createHash('md5').update(key).digest('hex')
}

test('file storage round-trips through a non-normalized directory and stores basenames in the index', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-nondenorm-'))
  try {
    const raw = join(directory, 'store') + '/'
    const storage = new FileStorage(raw)
    expect(storage.directory).toBe(resolve(raw))

    expect(storage.save('alpha', { value: 1 })).toBeTrue()
    expect(storage.save('beta', 'two')).toBeTrue()

    // The index must hold plain `<md5>.json` basenames, not sliced paths.
    const index = JSON.parse(readFileSync(join(storage.directory, '_index.json'), 'utf8')) as Record<string, string>
    expect(index).toEqual({ alpha: `${hashOf('alpha')}.json`, beta: `${hashOf('beta')}.json` })

    // A fresh instance over the same directory resolves every record.
    const restored = new FileStorage(raw)
    expect(restored.load('alpha')).toEqual({ value: 1 })
    expect(restored.load('beta')).toBe('two')
    expect(restored.listKeys().sort()).toEqual(['alpha', 'beta'])
    expect(restored.delete('alpha')).toBeTrue()
    expect(restored.load('alpha')).toBeUndefined()
    expect(restored.clear()).toBe(1)
    expect(readdirSync(storage.directory).filter(entry => entry.endsWith('.json') && !entry.startsWith('_index'))).toEqual([])
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('two file storage instances on one directory merge their index writes instead of orphaning records', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-merge-'))
  const root = join(directory, 'store')
  try {
    // Both instances open before either writes, so each holds a stale index.
    const first = new FileStorage(root)
    const second = new FileStorage(root)
    expect(first.save('from-first', 1)).toBeTrue()
    expect(second.save('from-second', 2)).toBeTrue()

    const merged = new FileStorage(root)
    expect(merged.load('from-first')).toBe(1)
    expect(merged.load('from-second')).toBe(2)
    expect(merged.listKeys().sort()).toEqual(['from-first', 'from-second'])

    // A delete through a fresh instance removes the data file; a stale
    // sibling rewriting afterwards cannot resurrect the record's payload.
    const deleter = new FileStorage(root)
    expect(deleter.delete('from-second')).toBeTrue()
    expect(second.save('from-second-again', 3)).toBeTrue()
    const after = new FileStorage(root)
    expect(after.load('from-second')).toBeUndefined()
    expect(after.load('from-first')).toBe(1)
    expect(after.load('from-second-again')).toBe(3)
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('long-lived file storage instances see each other\'s writes through read operations', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-stale-read-'))
  const root = join(directory, 'store')
  try {
    const first = new FileStorage(root)
    const second = new FileStorage(root)

    // Each instance starts with an empty in-memory index.
    expect(first.exists('shared')).toBeFalse()
    expect(second.listKeys()).toEqual([])

    // A write through one instance becomes visible to the other without
    // creating a fresh FileStorage object.
    expect(first.save('shared', { writer: 'first' })).toBeTrue()
    expect(second.exists('shared')).toBeTrue()
    expect(second.load('shared')).toEqual({ writer: 'first' })
    expect(second.listKeys()).toEqual(['shared'])

    // Overwriting a key through the second instance is reflected in the first.
    expect(second.save('shared', { writer: 'second' })).toBeTrue()
    expect(first.load('shared')).toEqual({ writer: 'second' })
    expect(first.listKeys()).toEqual(['shared'])

    // New keys and deletes are also observed on read operations.
    expect(second.save('only-second', 2)).toBeTrue()
    expect(first.exists('only-second')).toBeTrue()
    expect(first.listKeys().sort()).toEqual(['only-second', 'shared'])

    expect(first.delete('shared')).toBeTrue()
    expect(second.exists('shared')).toBeFalse()
    expect(second.load('shared')).toBeUndefined()
    expect(second.listKeys()).toEqual(['only-second'])
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('concurrent file storage index writers preserve every disjoint entry', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-concurrent-'))
  const root = join(directory, 'store')
  const workerPath = join(directory, 'writer.ts')
  try {
    const storageModule = pathToFileURL(resolve(import.meta.dir, '../src/memory/storage.ts')).href
    writeFileSync(workerPath, `
      import { FileStorage } from ${JSON.stringify(storageModule)}
      const [directory, prefix] = process.argv.slice(2)
      const storage = new FileStorage(directory)
      for (let index = 0; index < 40; index += 1) storage.save(prefix + index, index)
    `, 'utf8')
    const workers = Array.from({ length: 4 }, (_, index) => Bun.spawn({
      cmd: [process.execPath, 'run', workerPath, root, `writer-${index}-`],
      stdout: 'pipe',
      stderr: 'pipe',
    }))
    const results = await Promise.all(workers.map(async worker => ({
      exitCode: await worker.exited,
      stderr: await new Response(worker.stderr).text(),
    })))
    expect(results).toEqual(Array.from({ length: 4 }, () => ({ exitCode: 0, stderr: '' })))
    const restored = new FileStorage(root)
    expect(restored.listKeys()).toHaveLength(160)
    for (let writer = 0; writer < 4; writer += 1) {
      for (let index = 0; index < 40; index += 1) {
        expect(restored.load(`writer-${writer}-${index}`)).toBe(index)
      }
    }
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('concurrent same-key file storage saves leave one complete indexed payload', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-same-key-'))
  const root = join(directory, 'store')
  const workerPath = join(directory, 'same-key-writer.ts')
  const barrier = join(directory, 'start')
  try {
    const storageModule = pathToFileURL(resolve(import.meta.dir, '../src/memory/storage.ts')).href
    writeFileSync(workerPath, `
      import { existsSync } from 'node:fs'
      import { FileStorage } from ${JSON.stringify(storageModule)}
      const [directory, barrier, writer] = process.argv.slice(2)
      while (!existsSync(barrier)) await Bun.sleep(1)
      const storage = new FileStorage(directory)
      const value = { writer, content: writer.repeat(100_000) }
      if (!storage.save('shared-key', value)) process.exit(2)
    `, 'utf8')
    const writers = ['a', 'b', 'c', 'd']
    const workers = writers.map(writer => Bun.spawn({
      cmd: [process.execPath, 'run', workerPath, root, barrier, writer],
      stdout: 'pipe',
      stderr: 'pipe',
    }))
    writeFileSync(barrier, 'start', 'utf8')
    const results = await Promise.all(workers.map(async worker => ({
      exitCode: await worker.exited,
      stderr: await new Response(worker.stderr).text(),
    })))
    expect(results).toEqual(writers.map(() => ({ exitCode: 0, stderr: '' })))

    const restored = new FileStorage(root)
    expect(restored.listKeys()).toEqual(['shared-key'])
    const value = restored.load('shared-key') as { writer: string; content: string }
    expect(writers).toContain(value.writer)
    expect(value.content).toBe(value.writer.repeat(100_000))
    expect(JSON.parse(readFileSync(join(root, '_index.json'), 'utf8'))).toEqual({
      'shared-key': `${hashOf('shared-key')}.json`,
    })
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('file storage recovers a stale lock and cleans payloads when index persistence fails', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-stale-lock-'))
  const root = join(directory, 'store')
  try {
    const storage = new FileStorage(root, { lockTimeoutMs: 50, staleLockMs: 1_000 })
    const lock = join(root, '_index.json.lock')
    mkdirSync(lock)
    const old = new Date(Date.now() - 60_000)
    utimesSync(lock, old, old)
    expect(storage.save('recovered', 1)).toBeTrue()
    expect(new FileStorage(root).load('recovered')).toBe(1)

    // A live lock cannot be stolen. The failed save must remove the payload
    // written before index acquisition instead of leaving an orphaned record.
    mkdirSync(lock)
    const before = readdirSync(root).filter(entry => /^[0-9a-f]{32}\.json$/.test(entry)).sort()
    expect(storage.save('blocked', 2)).toBeFalse()
    expect(readdirSync(root).filter(entry => /^[0-9a-f]{32}\.json$/.test(entry)).sort()).toEqual(before)
    expect(new FileStorage(root).load('blocked')).toBeUndefined()
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('a corrupt file index is backed up and rebuilt from scanned data files', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-corrupt-index-'))
  const root = join(directory, 'store')
  try {
    const storage = new FileStorage(root)
    expect(storage.save('key-a', { a: 1 })).toBeTrue()
    expect(storage.save('key-b', 'b')).toBeTrue()

    writeFileSync(join(root, '_index.json'), '{{{ corrupt', 'utf8')
    const rebuilt = new FileStorage(root)

    // The corrupt index is preserved as a timestamped backup.
    expect(readdirSync(root).filter(entry => entry.startsWith('_index.json.corrupt-'))).toHaveLength(1)

    // Records are recovered under their hash stems and remain loadable.
    const keys = rebuilt.listKeys().sort()
    expect(keys).toEqual([hashOf('key-a'), hashOf('key-b')].sort())
    expect(rebuilt.load(hashOf('key-a'))).toEqual({ a: 1 })
    expect(rebuilt.load(hashOf('key-b'))).toBe('b')

    // Recovered records are not orphaned: clear() deletes their data files.
    expect(rebuilt.clear()).toBe(2)
    expect(readdirSync(root).filter(entry => /^[0-9a-f]{32}\.json$/.test(entry))).toEqual([])
    expect(rebuilt.save('fresh', true)).toBeTrue()
    expect(new FileStorage(root).load('fresh')).toBeTrue()
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('a wrong-shaped file index is also backed up and rebuilt', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-shape-index-'))
  const root = join(directory, 'store')
  try {
    const storage = new FileStorage(root)
    expect(storage.save('key', 42)).toBeTrue()
    writeFileSync(join(root, '_index.json'), JSON.stringify(['not', 'a', 'map']), 'utf8')
    const rebuilt = new FileStorage(root)
    expect(rebuilt.listKeys()).toEqual([hashOf('key')])
    expect(rebuilt.load(hashOf('key'))).toBe(42)
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('SQLite storage warns and returns undefined for a corrupt row instead of throwing', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-sqlite-corrupt-row-'))
  const path = join(directory, 'memory.db')
  try {
    const storage = new SQLiteStorage({ dbPath: path, writeEnabled: true })
    expect(storage.save('good', { value: 1 })).toBeTrue()
    expect(storage.save('bad', 'will be corrupted')).toBeTrue()
    storage.close()

    const raw = new Database(path)
    raw.query('UPDATE memory SET data = ? WHERE key = ?').run('not json', 'bad')
    raw.close()

    const warnings: unknown[][] = []
    const original = console.warn
    console.warn = (...args: unknown[]) => {
      warnings.push(args)
    }
    try {
      const reopened = new SQLiteStorage({ dbPath: path, writeEnabled: true })
      expect(reopened.load('bad')).toBeUndefined()
      expect(reopened.load('good')).toEqual({ value: 1 })
      expect(reopened.exists('bad')).toBeTrue()
      reopened.close()
    } finally {
      console.warn = original
    }
    expect(warnings).toHaveLength(1)
    expect(String(warnings[0]?.[0])).toContain('bad')
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('SQLite access updates wait through transient migration contention and remain atomic', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-sqlite-access-contention-'))
  const path = join(directory, 'memory.db')
  let blocker: Database | undefined
  try {
    const initialized = new SQLiteStorage({ dbPath: path, writeEnabled: true })
    expect(initialized.save('memory', { access_count: 4, content: 'preserved' })).toBeTrue()
    initialized.close()

    blocker = new Database(path)
    blocker.run('PRAGMA user_version = 0')
    blocker.run('BEGIN IMMEDIATE')

    const writer = Bun.spawn([
      process.execPath,
      '-e',
      `import { SQLiteStorage } from './src/memory/storage.ts';
       const storage = new SQLiteStorage({ dbPath: process.env.XERXES_CONTENTION_DB, writeEnabled: true });
       try {
         const result = storage.updateAccessState('memory', 3, '2026-08-03T00:00:00.000Z');
         console.log(JSON.stringify({ result, value: storage.load('memory') }));
       } finally { storage.close(); }`,
    ], {
      cwd: join(import.meta.dir, '..'),
      env: { ...process.env, XERXES_CONTENTION_DB: path },
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'pipe',
    })

    await Bun.sleep(100)
    expect(writer.exitCode).toBeNull()
    blocker.run('COMMIT')
    blocker.close()
    blocker = undefined

    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(writer.stdout).text(),
      new Response(writer.stderr).text(),
      writer.exited,
    ])
    expect(exitCode, stderr).toBe(0)
    expect(JSON.parse(stdout)).toEqual({
      result: 'updated',
      value: {
        access_count: 7,
        content: 'preserved',
        last_accessed: '2026-08-03T00:00:00.000Z',
      },
    })
  } finally {
    if (blocker !== undefined) {
      try {
        blocker.run('ROLLBACK')
      } finally {
        blocker.close()
      }
    }
    rmSync(directory, { force: true, recursive: true })
  }
})

test('SQLite access updates fail atomically when contention outlasts the busy timeout', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-sqlite-access-persistent-lock-'))
  const path = join(directory, 'memory.db')
  let blocker: Database | undefined
  let storage: SQLiteStorage | undefined
  try {
    const initialized = new SQLiteStorage({ dbPath: path, writeEnabled: true })
    expect(initialized.save('memory', { access_count: 4, content: 'preserved' })).toBeTrue()
    initialized.close()

    blocker = new Database(path)
    blocker.run('BEGIN IMMEDIATE')
    storage = new SQLiteStorage({ dbPath: path, writeEnabled: true, busyTimeoutMs: 25 })
    expect(storage.updateAccessState('memory', 3, '2026-08-03T00:00:00.000Z')).toBe('failed')
    expect(storage.load('memory')).toEqual({ access_count: 4, content: 'preserved' })
  } finally {
    storage?.close()
    if (blocker !== undefined) {
      try {
        blocker.run('ROLLBACK')
      } finally {
        blocker.close()
      }
    }
    rmSync(directory, { force: true, recursive: true })
  }
})

test('SQLite storage applies ordered user_version migrations and skips them on reopen', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-sqlite-migration-'))
  const path = join(directory, 'memory.db')
  const userVersion = (dbPath: string): number => {
    const database = new Database(dbPath)
    const row = database.query('PRAGMA user_version').get() as { user_version: number }
    database.close()
    return row.user_version
  }
  try {
    // A brand-new database is migrated to the current schema version.
    const storage = new SQLiteStorage({ dbPath: path, writeEnabled: true })
    expect(storage.save('persisted', 'value')).toBeTrue()
    storage.close()
    expect(userVersion(path)).toBe(1)

    // Reopening does not re-run or downgrade migrations; data survives.
    const reopened = new SQLiteStorage({ dbPath: path, writeEnabled: true })
    expect(reopened.load('persisted')).toBe('value')
    reopened.close()
    expect(userVersion(path)).toBe(1)

    // A database stamped with a newer version is left untouched.
    const future = new Database(path)
    future.run('PRAGMA user_version = 99')
    future.close()
    const forward = new SQLiteStorage({ dbPath: path, writeEnabled: true })
    expect(forward.load('persisted')).toBe('value')
    forward.close()
    expect(userVersion(path)).toBe(99)
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('SQLite storage migrates a legacy user_version=0 database with an existing memory table', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-sqlite-legacy-'))
  const path = join(directory, 'memory.db')
  try {
    // Simulate a pre-versioning database: schema present, user_version 0.
    const legacy = new Database(path)
    legacy.run(`
      CREATE TABLE memory (
        key TEXT PRIMARY KEY,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    `)
    legacy.query('INSERT INTO memory (key, data, created_at, updated_at) VALUES (?, ?, ?, ?)')
      .run('legacy', JSON.stringify({ old: true }), 'then', 'then')
    legacy.close()

    const storage = new SQLiteStorage({ dbPath: path, writeEnabled: true })
    expect(storage.load('legacy')).toEqual({ old: true })
    expect(storage.save('new', [1, 2, 3])).toBeTrue()
    storage.close()

    const check = new Database(path)
    expect((check.query('PRAGMA user_version').get() as { user_version: number }).user_version).toBe(1)
    check.close()
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('file storage save keeps its boolean contract when finally cleanup throws', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-finally-cleanup-'))
  const root = join(directory, 'store')
  const originalRmSync = nodeFs.rmSync
  let cleanupAttempts = 0
  try {
    mock.module('node:fs', () => ({
      ...originalFs,
      rmSync: (path: string, options?: { force?: boolean; recursive?: boolean }) => {
        // Target only payload temp/backup cleanup, not index persistence cleanup.
        if (
          typeof path === 'string' &&
          !path.includes('_index.json') &&
          (path.endsWith('.tmp') || path.endsWith('.replaced'))
        ) {
          cleanupAttempts += 1
          throw new Error(`cleanup refused: ${path}`)
        }
        return originalRmSync(path, options)
      },
    }))
    const storage = new FileStorage(root)

    // The save must succeed (data and index are committed) and return a boolean
    // even though the defensive finally cleanup rejects every rmSync call.
    expect(storage.save('key', 'value')).toBeTrue()
    expect(cleanupAttempts).toBeGreaterThan(0)
    expect(storage.load('key')).toBe('value')

    const reopened = new FileStorage(root)
    expect(reopened.load('key')).toBe('value')
  } finally {
    // mock.restore() does not reset mock.module overrides; restore explicitly.
    mock.module('node:fs', () => originalFs)
    originalRmSync(directory, { force: true, recursive: true })
  }
})

test('file storage delete and clear keep their committed index when backup cleanup fails', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-file-delete-cleanup-'))
  const root = join(directory, 'store')
  const originalRmSync = nodeFs.rmSync
  let cleanupAttempts = 0
  try {
    mock.module('node:fs', () => ({
      ...originalFs,
      rmSync: (path: string, options?: { force?: boolean; recursive?: boolean }) => {
        // Target only post-commit payload backup cleanup, not index persistence.
        if (typeof path === 'string' && path.endsWith('.deleted')) {
          cleanupAttempts += 1
          throw new Error(`cleanup refused: ${path}`)
        }
        return originalRmSync(path, options)
      },
    }))
    const storage = new FileStorage(root)
    expect(storage.save('a', 1)).toBeTrue()
    expect(storage.save('b', 2)).toBeTrue()

    // Delete must commit even though backup removal fails.
    expect(storage.delete('a')).toBeTrue()
    expect(cleanupAttempts).toBeGreaterThan(0)
    expect(storage.exists('a')).toBeFalse()
    expect(storage.load('a')).toBeUndefined()

    const afterDelete = new FileStorage(root)
    expect(afterDelete.listKeys().sort()).toEqual(['b'])
    expect(afterDelete.load('a')).toBeUndefined()
    expect(afterDelete.load('b')).toBe(2)

    // Clear must also commit even though backup removal fails.
    expect(storage.clear()).toBe(1)
    expect(storage.listKeys()).toEqual([])
    expect(storage.exists('b')).toBeFalse()

    const afterClear = new FileStorage(root)
    expect(afterClear.listKeys()).toEqual([])
  } finally {
    // mock.restore() does not reset mock.module overrides; restore explicitly.
    mock.module('node:fs', () => originalFs)
    originalRmSync(directory, { force: true, recursive: true })
  }
})

test('file storage validates lock ownership before removing a stale-looking lock', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-live-lock-'))
  const root = join(directory, 'store')
  try {
    const storage = new FileStorage(root, { lockTimeoutMs: 2_000, staleLockMs: 100 })
    const lock = join(root, '_index.json.lock')
    mkdirSync(lock)
    const old = new Date(Date.now() - 60_000)
    utimesSync(lock, old, old)

    // Simulate a live owner that keeps refreshing the lock even though its
    // mtime started out stale. The save must wait for ownership to expire,
    // not steal the lock and interleave with a live transaction.
    const refresher = Bun.spawn(
      [
        process.execPath,
        // NOTE: `bun run -e SCRIPT ARG` treats ARG as a script name ("Script
        // not found") and dies instantly; plain `bun -e SCRIPT ARG` passes
        // ARG as process.argv[1].
        '-e',
        `const fs = require('node:fs'); const lock = process.argv[1]; const id = setInterval(() => { try { fs.utimesSync(lock, new Date(), new Date()); } catch { process.exit(0); } }, 25); setTimeout(() => { clearInterval(id); process.exit(0); }, 300);`,
        lock,
      ],
      { stdout: 'ignore', stderr: 'ignore' },
    )

    // Give the refresher a moment to start so the lock looks live during validation.
    await Bun.sleep(20)

    const before = Date.now()
    expect(storage.save('after-live', 42)).toBeTrue()
    const elapsed = Date.now() - before
    expect(elapsed).toBeGreaterThanOrEqual(200)

    await refresher.exited

    // The lock was not stolen mid-refresh: the payload and index are consistent.
    expect(new FileStorage(root).load('after-live')).toBe(42)
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})

test('file storage refreshes its lock during a long transaction so it is not stolen', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-lock-heartbeat-'))
  const root = join(directory, 'store')
  const workerPath = join(directory, 'holder.ts')
  try {
    const storage = new FileStorage(root, { lockTimeoutMs: 2_000, staleLockMs: 100 })
    const storageModule = pathToFileURL(resolve(import.meta.dir, '../src/memory/storage.ts')).href
    writeFileSync(
      workerPath,
      `
      import { FileStorage } from ${JSON.stringify(storageModule)}
      const [root] = process.argv.slice(2)
      const storage = new FileStorage(root, { lockTimeoutMs: 2_000, staleLockMs: 100 })
      const lock = (storage as any).withIndexLock.bind(storage)
      lock((touchLock: () => void) => {
        const end = Date.now() + 350
        while (Date.now() < end) {
          touchLock()
          Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10)
        }
      })
    `,
      'utf8',
    )

    const worker = Bun.spawn(
      [process.execPath, 'run', workerPath, root],
      { stdout: 'pipe', stderr: 'pipe' },
    )
    await Bun.sleep(50)

    const before = Date.now()
    expect(storage.save('key', 1)).toBeTrue()
    const elapsed = Date.now() - before
    expect(elapsed).toBeGreaterThanOrEqual(250)

    const [stderr, exitCode] = await Promise.all([
      new Response(worker.stderr).text(),
      worker.exited,
    ])
    expect(exitCode, stderr).toBe(0)
    expect(new FileStorage(root).load('key')).toBe(1)
  } finally {
    rmSync(directory, { force: true, recursive: true })
  }
})
