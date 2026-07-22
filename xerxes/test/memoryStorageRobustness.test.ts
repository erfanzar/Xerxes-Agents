// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { Database } from 'bun:sqlite'
import { createHash } from 'node:crypto'
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import { FileStorage, SQLiteStorage } from '../src/memory/index.js'

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
