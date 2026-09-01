// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmodSync, readdirSync, readFileSync, statSync, utimesSync, writeFileSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import {
  describeChange,
  FileStateTracker,
  fileReadsForMetadata,
  fileStateTracker,
  guardedCreate,
  guardedWrite,
  hydrateFileReadsFromMetadata,
  isFileFreshnessEnforced,
  recordFileRead,
  setFileFreshnessEnforcement,
} from '../src/tools/fileState.js'

const SESSION = 'session-a'

async function inWorkspace(run: (workspace: string) => Promise<void> | void): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-file-state-'))
  try {
    await run(workspace)
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

/** Record the file exactly as a read tool would after showing it to the model. */
function recordCurrent(tracker: FileStateTracker, path: string, partialView = false, session = SESSION): void {
  const stats = statSync(path)
  recordFileRead(
    { sessionId: session },
    path,
    readFileSync(path, 'utf8'),
    { mtimeMs: stats.mtimeMs, partialView, size: stats.size },
    tracker,
  )
}

/** Simulate another writer: change the bytes and push the mtime past the recorded one. */
function externalWrite(path: string, content: string): void {
  writeFileSync(path, content)
  const future = new Date(Date.now() + 5_000)
  utimesSync(path, future, future)
}

afterEach(() => {
  fileStateTracker.clear()
  setFileFreshnessEnforcement(undefined)
})

test('guardedWrite refuses a file the session never read and points at the kill switch', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'a.txt')
    writeFileSync(path, 'one\n')
    const tracker = new FileStateTracker()

    expect(() => guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: () => 'two\n',
    }, tracker)).toThrow('requires reading "a.txt" first — read the file, then retry')
    expect(() => guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: () => 'two\n',
    }, tracker)).toThrow('XERXES_FILE_FRESHNESS=off')
    expect(readFileSync(path, 'utf8')).toBe('one\n')
  })
})

test('a recorded read unblocks the write and the post-write state unblocks the next one', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'a.txt')
    writeFileSync(path, 'one\n')
    const tracker = new FileStateTracker()
    recordCurrent(tracker, path)

    const first = guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current + 'two\n',
    }, tracker)
    expect(first.changed).toBe(true)
    expect(first.staleNotice).toBeUndefined()

    // Without recording the post-write state the tool's own edit would look like drift.
    const second = guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current + 'three\n',
    }, tracker)
    expect(second.previous).toBe('one\ntwo\n')
    expect(readFileSync(path, 'utf8')).toBe('one\ntwo\nthree\n')
  })
})

test('a rewrite that produced identical bytes is not drift even with a newer mtime', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'a.txt')
    writeFileSync(path, 'stable\n')
    const tracker = new FileStateTracker()
    recordCurrent(tracker, path)
    // A formatter no-op or a checkout of the revision already on disk.
    externalWrite(path, 'stable\n')

    const result = guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'overwrite',
      sessionId: SESSION,
      toolName: 'WriteFile',
      transform: () => 'rewritten\n',
    }, tracker)
    expect(result.staleNotice).toBeUndefined()
    expect(readFileSync(path, 'utf8')).toBe('rewritten\n')
  })
})

test('a targeted edit on a drifted file proceeds and reports what changed', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'a.txt')
    writeFileSync(path, 'alpha\nbeta\ngamma\n')
    const tracker = new FileStateTracker()
    recordCurrent(tracker, path)
    externalWrite(path, 'alpha\nBETA CHANGED\ngamma\n')

    const result = guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current.replace('gamma', 'GAMMA'),
    }, tracker)
    expect(result.staleNotice).toContain('[stale-read] a.txt changed on disk')
    expect(result.staleNotice).toContain('at line 2: -1 +1')
    expect(result.staleNotice).toContain('-beta')
    expect(result.staleNotice).toContain('+BETA CHANGED')
    // The other writer's line survives; the edit applied to the current bytes.
    expect(readFileSync(path, 'utf8')).toBe('alpha\nBETA CHANGED\nGAMMA\n')
  })
})

test('a whole-file overwrite of a drifted file is refused and carries the same report', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'a.txt')
    writeFileSync(path, 'alpha\nbeta\n')
    const tracker = new FileStateTracker()
    recordCurrent(tracker, path)
    externalWrite(path, 'alpha\nbeta rewritten by someone else\n')

    expect(() => guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'overwrite',
      sessionId: SESSION,
      toolName: 'WriteFile',
      transform: () => 'clobbered\n',
    }, tracker)).toThrow('a whole-file write would discard those changes')
    expect(() => guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'overwrite',
      sessionId: SESSION,
      toolName: 'WriteFile',
      transform: () => 'clobbered\n',
    }, tracker)).toThrow('+beta rewritten by someone else')
    expect(readFileSync(path, 'utf8')).toBe('alpha\nbeta rewritten by someone else\n')
  })
})

test('drift after a partial or binary read is refused because no cheap report exists', async () => {
  await inWorkspace(workspace => {
    const ranged = join(workspace, 'ranged.txt')
    writeFileSync(ranged, 'one\ntwo\nthree\n')
    const tracker = new FileStateTracker()
    recordCurrent(tracker, ranged, true)
    externalWrite(ranged, 'one\ntwo\nthree\nfour\n')

    expect(() => guardedWrite({
      absolutePath: ranged,
      displayPath: 'ranged.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current.replace('one', 'ONE'),
    }, tracker)).toThrow('your read covered only part of the file')

    const binary = join(workspace, 'blob.bin')
    writeFileSync(binary, 'head\u0000tail')
    recordCurrent(tracker, binary)
    externalWrite(binary, 'head\u0000tail changed')
    expect(() => guardedWrite({
      absolutePath: binary,
      displayPath: 'blob.bin',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current + '!',
    }, tracker)).toThrow('cannot be summarised here')
  })
})

test('a file over the snapshot ceiling is still guarded but reported without a diff', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'big.txt')
    writeFileSync(path, 'x'.repeat(200) + '\n')
    const tracker = new FileStateTracker({ maxSnapshotBytes: 32 })
    recordCurrent(tracker, path)
    externalWrite(path, 'y'.repeat(300) + '\n')

    expect(() => guardedWrite({
      absolutePath: path,
      displayPath: 'big.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current + 'z',
    }, tracker)).toThrow('cannot be summarised here')
  })
})

test('the kill switch and a missing session both disable the freshness requirement', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'a.txt')
    writeFileSync(path, 'one\n')
    const tracker = new FileStateTracker()

    // No session: nothing was read in a conversation, so there is nothing to be stale against.
    expect(guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'overwrite',
      sessionId: undefined,
      toolName: 'WriteFile',
      transform: () => 'two\n',
    }, tracker).changed).toBe(true)

    setFileFreshnessEnforcement(false)
    expect(guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'overwrite',
      sessionId: SESSION,
      toolName: 'WriteFile',
      transform: () => 'three\n',
    }, tracker).changed).toBe(true)
    expect(readFileSync(path, 'utf8')).toBe('three\n')
  })
})

test('freshness enforcement resolves environment over the runtime setting', () => {
  expect(isFileFreshnessEnforced({})).toBe(true)
  setFileFreshnessEnforcement(false)
  expect(isFileFreshnessEnforced({})).toBe(false)
  expect(isFileFreshnessEnforced({ XERXES_FILE_FRESHNESS: 'on' })).toBe(true)
  expect(isFileFreshnessEnforced({ XERXES_FILE_FRESHNESS: 'nonsense' })).toBe(false)
  setFileFreshnessEnforcement(undefined)
  expect(isFileFreshnessEnforced({ XERXES_FILE_FRESHNESS: '0' })).toBe(false)
})

test('an unchanged transform leaves the file alone', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'a.txt')
    writeFileSync(path, 'one\n')
    const tracker = new FileStateTracker()
    recordCurrent(tracker, path)
    const before = statSync(path).mtimeMs

    const result = guardedWrite({
      absolutePath: path,
      displayPath: 'a.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'find_and_replace',
      transform: current => current,
    }, tracker)
    expect(result.changed).toBe(false)
    expect(statSync(path).mtimeMs).toBe(before)
  })
})

test('guardedCreate fails atomically when the path appeared after the existence check', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'race.txt')
    writeFileSync(path, 'winner\n')

    expect(() => guardedCreate({
      absolutePath: path,
      content: 'loser\n',
      displayPath: 'race.txt',
      sessionId: SESSION,
    })).toThrow('already exists; pass overwrite=true to replace it')
    expect(readFileSync(path, 'utf8')).toBe('winner\n')

    const fresh = join(workspace, 'new.txt')
    guardedCreate({ absolutePath: fresh, content: 'made\n', displayPath: 'new.txt', sessionId: SESSION })
    // The creator knows the contents, so an immediate edit must not be treated as blind.
    expect(guardedWrite({
      absolutePath: fresh,
      displayPath: 'new.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current + 'again\n',
    }).changed).toBe(true)
  })
})

test('the tracker bounds itself, keeps sessions apart, and exposes the session read list', async () => {
  await inWorkspace(workspace => {
    const tracker = new FileStateTracker({ maxEntries: 2 })
    const paths = ['a.txt', 'b.txt', 'c.txt'].map(name => join(workspace, name))
    for (const path of paths) {
      writeFileSync(path, path + '\n')
      recordCurrent(tracker, path)
    }
    expect(tracker.size).toBe(2)
    expect(tracker.peek(SESSION, paths[0] ?? '')).toBeUndefined()
    expect(tracker.pathsForSession(SESSION).sort()).toEqual([paths[1] ?? '', paths[2] ?? ''].sort())

    // Another session's read says nothing about this one's beliefs.
    recordCurrent(tracker, paths[1] ?? '', false, 'session-b')
    expect(tracker.pathsForSession('session-b')).toEqual([paths[1] ?? ''])
    expect(tracker.forget('session-b', paths[1] ?? '')).toBe(true)
    expect(tracker.clearSession(SESSION)).toBeGreaterThan(0)
    expect(tracker.size).toBe(0)
  })
})

test('describeChange trims the common head and tail and caps a long side', () => {
  expect(describeChange('a\nb\nc\n', 'a\nB\nc\n')).toBe('at line 2: -1 +1\n-b\n+B')

  const added = ['head', ...Array.from({ length: 20 }, (_value, index) => 'line ' + index), 'tail'].join('\n')
  const report = describeChange('head\ntail', added)
  expect(report).toContain('at line 2: -0 +20')
  expect(report).toContain('+… (8 more lines)')

  const long = 'head\n' + 'z'.repeat(400)
  expect(describeChange('head\nshort', long)).toContain('…')
})

test('guardedWrite leaves the original intact when the transform throws', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'precious.txt')
    writeFileSync(path, 'original bytes\n')
    const tracker = new FileStateTracker()
    recordCurrent(tracker, path)

    expect(() => guardedWrite({
      absolutePath: path,
      displayPath: 'precious.txt',
      mode: 'overwrite',
      sessionId: SESSION,
      toolName: 'WriteFile',
      transform: () => {
        throw new Error('boom after read')
      },
    }, tracker)).toThrow('boom after read')

    expect(readFileSync(path, 'utf8')).toBe('original bytes\n')
    // No temp residue next to the target.
    expect(readdirSync(workspace).filter(name => name !== 'precious.txt')).toEqual([])
  })
})

test('a successful guardedWrite replaces content atomically without stray temp files', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'swap.txt')
    writeFileSync(path, 'before\n')
    const tracker = new FileStateTracker()
    recordCurrent(tracker, path)

    const result = guardedWrite({
      absolutePath: path,
      displayPath: 'swap.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current.replace('before', 'after'),
    }, tracker)

    expect(result.changed).toBe(true)
    expect(readFileSync(path, 'utf8')).toBe('after\n')
    // Freshness semantics are unchanged: the post-write state is recorded, so a
    // second edit in the same session is not refused as self-inflicted drift.
    const followUp = guardedWrite({
      absolutePath: path,
      displayPath: 'swap.txt',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current + 'again\n',
    }, tracker)
    expect(followUp.changed).toBe(true)
    expect(readFileSync(path, 'utf8')).toBe('after\nagain\n')
    expect(readdirSync(workspace).filter(name => name !== 'swap.txt')).toEqual([])
  })
})

test('guardedWrite preserves permission bits through the atomic rename', async () => {
  await inWorkspace(workspace => {
    // An executable script edited through the guarded path must stay
    // executable: rename-over used to reset it to the process umask.
    const scriptPath = join(workspace, 'run.sh')
    writeFileSync(scriptPath, '#!/bin/sh\necho one\n')
    chmodSync(scriptPath, 0o755)
    const tracker = new FileStateTracker()
    recordCurrent(tracker, scriptPath)

    guardedWrite({
      absolutePath: scriptPath,
      displayPath: 'run.sh',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current.replace('one', 'two'),
    }, tracker)

    expect(readFileSync(scriptPath, 'utf8')).toBe('#!/bin/sh\necho two\n')
    expect(statSync(scriptPath).mode & 0o777).toBe(0o755)
    expect(readdirSync(workspace).filter(name => name !== 'run.sh')).toEqual([])

    // And a private file stays private.
    const secretPath = join(workspace, 'secret.env')
    writeFileSync(secretPath, 'TOKEN=a\n')
    chmodSync(secretPath, 0o600)
    recordCurrent(tracker, secretPath)
    guardedWrite({
      absolutePath: secretPath,
      displayPath: 'secret.env',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current.replace('a', 'b'),
    }, tracker)
    expect(statSync(secretPath).mode & 0o777).toBe(0o600)
  })
})

test('read state serializes into session metadata without pinning file contents', () => {
  const tracker = new FileStateTracker()
  tracker.record(SESSION, '/tmp/one.ts', 'one', { mtimeMs: 111, partialView: false, size: 3 })
  tracker.record(SESSION, '/tmp/big.ts', 'huge', { mtimeMs: 222, partialView: true, size: 4 })

  const persisted = fileReadsForMetadata(SESSION, tracker)
  expect(persisted.map(entry => entry.path).sort()).toEqual(['/tmp/big.ts', '/tmp/one.ts'])
  const one = persisted.find(entry => entry.path === '/tmp/one.ts')
  expect(one).toMatchObject({ digest: expect.any(String), mtime_ms: 111, partial: false, size: 3 })
  // Sessions with no reads, or no session at all, persist nothing.
  expect(fileReadsForMetadata('other-session', tracker)).toEqual([])
  expect(fileReadsForMetadata(undefined, tracker)).toEqual([])
})

test('hydrated metadata restores the guard across a restart, minus the diff snapshots', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'resume.ts')
    writeFileSync(path, 'const original = 1\n')
    const first = new FileStateTracker()
    recordCurrent(first, path)
    const saved = fileReadsForMetadata(SESSION, first)

    // A fresh process: empty tracker, state restored from session metadata.
    const rehydrated = new FileStateTracker()
    expect(hydrateFileReadsFromMetadata(SESSION, { file_reads: saved }, rehydrated)).toBe(1)
    const record = rehydrated.peek(SESSION, path)
    expect(record?.digest).toBe(first.peek(SESSION, path)?.digest)
    expect(record?.snapshot).toBeUndefined()

    // The file is untouched since the read, so an edit proceeds.
    expect(() => guardedWrite({
      absolutePath: path,
      displayPath: 'resume.ts',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current.replace('original', 'resumed'),
    }, rehydrated)).not.toThrow()
    expect(readFileSync(path, 'utf8')).toBe('const resumed = 1\n')

    // A file the resumed session never read is still refused.
    writeFileSync(join(workspace, 'never.ts'), 'x\n')
    expect(() => guardedWrite({
      absolutePath: join(workspace, 'never.ts'),
      displayPath: 'never.ts',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: () => 'y\n',
    }, rehydrated)).toThrow('requires reading "never.ts" first')
  })
})

test('malformed persisted read state is ignored instead of poisoning the tracker', () => {
  const tracker = new FileStateTracker()
  expect(hydrateFileReadsFromMetadata(SESSION, undefined, tracker)).toBe(0)
  expect(hydrateFileReadsFromMetadata(SESSION, { file_reads: 'junk' }, tracker)).toBe(0)
  expect(hydrateFileReadsFromMetadata(SESSION, {
    file_reads: [
      null,
      42,
      { path: '/tmp/no-digest.ts', mtime_ms: 1, size: 1 },
      { path: '/tmp/ok.ts', digest: 'abc', mtime_ms: 5, partial: false, size: 10 },
    ],
  }, tracker)).toBe(1)
  expect(tracker.peek(SESSION, '/tmp/ok.ts')?.size).toBe(10)
})

test('clearing read state — what compaction does — forces a fresh read before editing', async () => {
  await inWorkspace(workspace => {
    const path = join(workspace, 'compact.ts')
    writeFileSync(path, 'before\n')
    const tracker = new FileStateTracker()
    recordCurrent(tracker, path)

    // Compaction retires the session's read records…
    expect(tracker.clearSession(SESSION)).toBe(1)

    // …so the model must read the file again before editing it.
    expect(() => guardedWrite({
      absolutePath: path,
      displayPath: 'compact.ts',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current.replace('before', 'after'),
    }, tracker)).toThrow('requires reading "compact.ts" first')
    expect(readFileSync(path, 'utf8')).toBe('before\n')

    recordCurrent(tracker, path)
    expect(() => guardedWrite({
      absolutePath: path,
      displayPath: 'compact.ts',
      mode: 'targeted',
      sessionId: SESSION,
      toolName: 'FileEditTool',
      transform: current => current.replace('before', 'after'),
    }, tracker)).not.toThrow()
  })
})
