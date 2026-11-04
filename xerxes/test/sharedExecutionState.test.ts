// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { appendFileSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { SharedExecutionState, SharedStateError } from '../src/cortex/sharedState.js'
import type { DataBoundaryMode } from '../src/routing/types.js'
import { DataBoundaryError } from '../src/security/boundaryTypes.js'

let dir: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'xerxes-shared-state-'))
})

afterEach(() => {
  rmSync(dir, { recursive: true, force: true })
})

function fixedClock(): () => Date {
  let n = 0
  return () => new Date(Date.UTC(2026, 10, 3, 0, 0, n++))
}

function make(mode: DataBoundaryMode, extra: { journalPath?: string; persistValues?: boolean } = {}): SharedExecutionState {
  const s = new SharedExecutionState({ runId: 'run-1', boundaryMode: mode, now: fixedClock(), ...extra })
  s.registerWorker({ id: 'local-1', trust: 'local', role: 'analyst' })
  s.registerWorker({ id: 'local-2', trust: 'local' })
  s.registerWorker({ id: 'cloud-1', trust: 'cloud', role: 'planner' })
  return s
}

function thrown(fn: () => unknown): unknown {
  try {
    fn()
  } catch (err) {
    return err
  }
  return undefined
}

describe('access control', () => {
  test('cloud worker is denied a private artifact under hybrid', () => {
    const s = make('hybrid')
    s.putArtifact({ ref: 'ledger', value: 'acct 1234', sensitivity: 'private', producedBy: 'local-1' })
    const err = thrown(() => s.read('cloud-1', 'ledger'))
    expect(err).toBeInstanceOf(DataBoundaryError)
    expect((err as DataBoundaryError).code).toBe('DATA_BOUNDARY_DENIED')
    expect(s.read('local-2', 'ledger')).toBe('acct 1234')
    expect(s.snapshot().reads).toEqual({ total: 2, denied: 1 })
  })

  test('the same read is allowed (and recorded) under mode off', () => {
    const s = make('off')
    s.putArtifact({ ref: 'ledger', value: 'acct 1234', sensitivity: 'private', producedBy: 'local-1' })
    expect(s.read('cloud-1', 'ledger')).toBe('acct 1234')
    expect(s.snapshot().reads).toEqual({ total: 1, denied: 0 })
  })

  test('a denied read does not taint the cloud worker', () => {
    const s = make('private')
    s.putArtifact({ ref: 'p', value: 'x', sensitivity: 'private', producedBy: 'local-1' })
    expect(thrown(() => s.read('cloud-1', 'p'))).toBeInstanceOf(DataBoundaryError)
    const info = s.putArtifact({ ref: 'plan', value: 'steps', sensitivity: 'shareable', producedBy: 'cloud-1' })
    expect(info.sensitivity).toBe('shareable')
  })

  test('view lists only refs the worker may read', () => {
    const s = make('hybrid')
    s.putArtifact({ ref: 'secret', value: 's', sensitivity: 'private', producedBy: 'local-1' })
    s.putArtifact({ ref: 'public', value: 'p', sensitivity: 'shareable', producedBy: 'local-1' })
    expect(s.view('cloud-1').map(v => v.ref)).toEqual(['public'])
    expect(s.view('local-2').map(v => v.ref).sort()).toEqual(['public', 'secret'])
    expect(s.snapshot().reads.total).toBe(0)
  })

  test('unknown worker and unknown artifact are typed errors', () => {
    const s = make('hybrid')
    expect((thrown(() => s.read('ghost', 'x')) as SharedStateError).code).toBe('UNKNOWN_WORKER')
    expect((thrown(() => s.read('local-1', 'nope')) as SharedStateError).code).toBe('UNKNOWN_ARTIFACT')
    expect((thrown(() => s.putArtifact({ ref: 'a', value: 'v', sensitivity: 'shareable', producedBy: 'ghost' })) as SharedStateError).code)
      .toBe('UNKNOWN_WORKER')
  })
})

describe('sensitivity propagation', () => {
  test('taint: output of a worker that read private data is private', () => {
    const s = make('hybrid')
    s.putArtifact({ ref: 'raw', value: 'ssn 999', sensitivity: 'private', producedBy: 'local-2' })
    expect(s.read('local-1', 'raw')).toBe('ssn 999')
    const info = s.putArtifact({ ref: 'summary', value: 'one customer', sensitivity: 'shareable', producedBy: 'local-1' })
    expect(info.sensitivity).toBe('private')
    expect(info.forced).toBe('taint')
    expect(thrown(() => s.read('cloud-1', 'summary'))).toBeInstanceOf(DataBoundaryError)
    expect(s.snapshot().workers.find(w => w.id === 'local-1')?.tainted).toBe(true)
  })

  test('a worker that only read shareable data stays clean', () => {
    const s = make('hybrid')
    s.putArtifact({ ref: 'doc', value: 'public', sensitivity: 'shareable', producedBy: 'local-2' })
    s.read('local-1', 'doc')
    const info = s.putArtifact({ ref: 'out', value: 'o', sensitivity: 'shareable', producedBy: 'local-1' })
    expect(info.sensitivity).toBe('shareable')
    expect(info.forced).toBeUndefined()
  })

  test('sticky: a ref once private stays private for later versions', () => {
    const s = make('hybrid')
    const v1 = s.putArtifact({ ref: 'r', value: 'one', sensitivity: 'private', producedBy: 'local-1' })
    const v2 = s.putArtifact({ ref: 'r', value: 'two', sensitivity: 'shareable', producedBy: 'local-2' })
    expect(v1.version).toBe(1)
    expect(v2.version).toBe(2)
    expect(v2.sensitivity).toBe('private')
    expect(v2.forced).toBe('sticky')
    expect(thrown(() => s.read('cloud-1', 'r'))).toBeInstanceOf(DataBoundaryError)
    expect(s.read('local-1', 'r')).toBe('two')
  })

  test('versions are content-hashed', () => {
    const s = make('off')
    const a = s.putArtifact({ ref: 'h', value: 'abc', sensitivity: 'shareable', producedBy: 'local-1' })
    expect(a.sha256).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
    expect(a.bytes).toBe(3)
  })
})

describe('journal', () => {
  test('restore rebuilds the exact state, including taint and steps', () => {
    const path = join(dir, 'nested', 'journal.jsonl')
    const live = make('hybrid', { journalPath: path, persistValues: true })
    live.putArtifact({ ref: 'raw', value: 'secret', sensitivity: 'private', producedBy: 'local-2', check: { passed: true, detail: 'ok' } })
    live.read('local-1', 'raw')
    live.putArtifact({ ref: 'sum', value: 'sum', sensitivity: 'shareable', producedBy: 'local-1' })
    live.stepStarted('s1', 'local-1')
    live.stepCompleted('s1', { outputs: ['sum'], accepted: true })
    live.stepStarted('s2', 'cloud-1')
    live.externalAction('s2', { key: 'mail-1', description: 'sent report' })
    live.stepFailed('s2', 'timeout')
    expect(thrown(() => live.read('cloud-1', 'sum'))).toBeInstanceOf(DataBoundaryError)

    const restored = SharedExecutionState.restore(path, { runId: 'run-1', boundaryMode: 'hybrid', persistValues: true })
    expect(restored.snapshot()).toEqual(live.snapshot())
    expect(restored.read('local-2', 'raw')).toBe('secret')
    expect(restored.hasExternalAction('mail-1')).toBe(true)
    expect(restored.restoreReport).toEqual({ events: expect.any(Number), truncatedTail: false, droppedBytes: 0 })

    // restored state keeps appending, and a second restore sees it
    restored.putArtifact({ ref: 'later', value: 'l', sensitivity: 'shareable', producedBy: 'local-2' })
    const again = SharedExecutionState.restore(path, { runId: 'run-1', boundaryMode: 'hybrid', persistValues: true })
    expect(again.snapshot().artifacts.map(a => a.ref)).toContain('later')
  })

  test('journal file is private (0600) and its directory 0700', () => {
    const path = join(dir, 'priv', 'j.jsonl')
    make('off', { journalPath: path })
    expect(statSync(path).mode & 0o777).toBe(0o600)
    expect(statSync(join(dir, 'priv')).mode & 0o777).toBe(0o700)
  })

  test('values are not in the journal by default (canary absent)', () => {
    const path = join(dir, 'j.jsonl')
    const canary = 'CANARY-7f3a9c-do-not-persist'
    const s = make('hybrid', { journalPath: path })
    s.putArtifact({ ref: 'c', value: canary, sensitivity: 'private', producedBy: 'local-1' })
    const text = readFileSync(path, 'utf8')
    expect(text).not.toContain(canary)
    expect(text).toContain('"sha256"')

    const restored = SharedExecutionState.restore(path, { runId: 'run-1', boundaryMode: 'hybrid' })
    expect(restored.snapshot().artifacts[0]?.valueAvailable).toBe(false)
    const err = thrown(() => restored.read('local-1', 'c')) as SharedStateError
    expect(err.code).toBe('VALUE_UNAVAILABLE')
  })

  test('persistValues: true writes the value', () => {
    const path = join(dir, 'j.jsonl')
    const s = make('off', { journalPath: path, persistValues: true })
    s.putArtifact({ ref: 'c', value: 'PERSISTED-VALUE', sensitivity: 'shareable', producedBy: 'local-1' })
    expect(readFileSync(path, 'utf8')).toContain('PERSISTED-VALUE')
  })

  test('truncated last line is tolerated, reported and repaired', () => {
    const path = join(dir, 'j.jsonl')
    const s = make('hybrid', { journalPath: path, persistValues: true })
    s.putArtifact({ ref: 'a', value: 'one', sensitivity: 'shareable', producedBy: 'local-1' })
    appendFileSync(path, '{"t":"artifact","at":"2026-11-03T00:00:00.000Z","ref":"b","vers')

    const restored = SharedExecutionState.restore(path, { runId: 'run-1', boundaryMode: 'hybrid', persistValues: true })
    expect(restored.restoreReport?.truncatedTail).toBe(true)
    expect(restored.restoreReport?.droppedBytes).toBeGreaterThan(0)
    expect(restored.snapshot().artifacts.map(a => a.ref)).toEqual(['a'])
    expect(readFileSync(path, 'utf8').endsWith('\n')).toBe(true)

    restored.putArtifact({ ref: 'b', value: 'two', sensitivity: 'shareable', producedBy: 'local-1' })
    const again = SharedExecutionState.restore(path, { runId: 'run-1', boundaryMode: 'hybrid', persistValues: true })
    expect(again.restoreReport?.truncatedTail).toBe(false)
    expect(again.snapshot().artifacts.map(a => a.ref)).toEqual(['a', 'b'])
  })

  test('a malformed middle line is a typed JOURNAL_CORRUPT error with its line', () => {
    const path = join(dir, 'j.jsonl')
    const s = make('off', { journalPath: path })
    s.putArtifact({ ref: 'a', value: 'one', sensitivity: 'shareable', producedBy: 'local-1' })
    const lines = readFileSync(path, 'utf8').split('\n').filter(l => l.length > 0)
    lines.splice(2, 0, '{not json')
    writeFileSync(path, `${lines.join('\n')}\n`)
    const err = thrown(() => SharedExecutionState.restore(path, { runId: 'run-1', boundaryMode: 'off' })) as SharedStateError
    expect(err).toBeInstanceOf(SharedStateError)
    expect(err.code).toBe('JOURNAL_CORRUPT')
    expect(err.line).toBe(3)
  })

  test('a malformed final line that ends with a newline is corruption, not truncation', () => {
    const path = join(dir, 'j.jsonl')
    make('off', { journalPath: path })
    appendFileSync(path, '{broken}\n')
    const err = thrown(() => SharedExecutionState.restore(path, { runId: 'run-1', boundaryMode: 'off' })) as SharedStateError
    expect(err.code).toBe('JOURNAL_CORRUPT')
  })

  test('restore refuses a different run id, a missing file, and an existing journal on construct', () => {
    const path = join(dir, 'j.jsonl')
    make('off', { journalPath: path })
    expect((thrown(() => SharedExecutionState.restore(path, { runId: 'other', boundaryMode: 'off' })) as SharedStateError).code)
      .toBe('JOURNAL_CORRUPT')
    expect((thrown(() => SharedExecutionState.restore(join(dir, 'none.jsonl'), { runId: 'run-1', boundaryMode: 'off' })) as SharedStateError).code)
      .toBe('JOURNAL_NOT_FOUND')
    expect((thrown(() => make('off', { journalPath: path })) as SharedStateError).code).toBe('JOURNAL_EXISTS')
  })
})

describe('progress and recovery', () => {
  function build(): SharedExecutionState {
    const s = make('hybrid')
    // a: done. b: failed. c: done but depends on b. d: done, independent. e: never started.
    s.stepStarted('a', 'local-1')
    s.putArtifact({ ref: 'a.out', value: 'A', sensitivity: 'shareable', producedBy: 'local-1' })
    s.stepCompleted('a', { outputs: ['a.out'], accepted: true })

    s.stepStarted('b', 'local-1')
    s.stepFailed('b', 'model timed out')

    s.stepStarted('c', 'cloud-1')
    s.putArtifact({ ref: 'c.out', value: 'C', sensitivity: 'shareable', producedBy: 'cloud-1' })
    s.externalAction('c', { key: 'email-42', description: 'sent summary email' })
    s.stepCompleted('c', { outputs: ['c.out'], accepted: true })

    s.stepStarted('d', 'local-2')
    s.putArtifact({ ref: 'd.out', value: 'D', sensitivity: 'shareable', producedBy: 'local-2' })
    s.stepCompleted('d', { outputs: ['d.out'], accepted: true })
    return s
  }

  const ids = ['a', 'b', 'c', 'd', 'e']
  const deps = new Map<string, readonly string[]>([
    ['b', ['a']],
    ['c', ['b']],
    ['e', ['d']],
  ])

  test('skips completed work, reruns failed, flags committed external actions on dependents', () => {
    const plan = build().recoveryPlan(ids, deps)
    expect(plan.skip).toEqual(['a', 'd'])
    expect(plan.rerun).toEqual(['b'])
    expect(plan.blockedExternal).toEqual([{ stepId: 'c', keys: ['email-42'] }])
    expect(plan.pending).toEqual(['e'])
    expect(plan.reasons['b']).toContain('model timed out')
    expect(plan.reasons['c']).toContain('upstream step b')
    expect(plan.reasons['c']).toContain('email-42')
  })

  test('a step that failed after committing an external action is blocked, not rerun', () => {
    const s = make('off')
    s.stepStarted('x', 'local-1')
    s.externalAction('x', { key: 'upload-1', description: 'wrote remote file' })
    s.stepFailed('x', 'crashed after upload')
    const plan = s.recoveryPlan(['x'], new Map())
    expect(plan.rerun).toEqual([])
    expect(plan.blockedExternal).toEqual([{ stepId: 'x', keys: ['upload-1'] }])
  })

  test('interrupted, rejected and failed-check steps rerun; accepted is forced false on a failed check', () => {
    const s = make('off')
    s.stepStarted('run', 'local-1') // never finished
    s.stepStarted('rej', 'local-1')
    s.putArtifact({ ref: 'rej.out', value: 'r', sensitivity: 'shareable', producedBy: 'local-1' })
    s.stepCompleted('rej', { outputs: ['rej.out'], accepted: false })
    s.stepStarted('chk', 'local-1')
    s.putArtifact({ ref: 'chk.out', value: 'c', sensitivity: 'shareable', producedBy: 'local-1', check: { passed: false, detail: 'bad totals' } })
    const done = s.stepCompleted('chk', { outputs: ['chk.out'], accepted: true })
    expect(done.accepted).toBe(false)
    const plan = s.recoveryPlan(['run', 'rej', 'chk'], new Map())
    expect(plan.skip).toEqual([])
    expect(plan.rerun).toEqual(['run', 'rej', 'chk'])
  })

  test('an accepted step whose output is missing reruns', () => {
    const s = make('off')
    s.stepStarted('m', 'local-1')
    s.stepCompleted('m', { outputs: ['ghost.out'], accepted: true })
    const plan = s.recoveryPlan(['m'], new Map())
    expect(plan.rerun).toEqual(['m'])
    expect(plan.reasons['m']).toContain('missing')
  })

  test('recovery works from a restored journal (values persisted)', () => {
    const path = join(dir, 'j.jsonl')
    const live = make('hybrid', { journalPath: path, persistValues: true })
    live.stepStarted('a', 'local-1')
    live.putArtifact({ ref: 'a.out', value: 'A', sensitivity: 'shareable', producedBy: 'local-1' })
    live.stepCompleted('a', { outputs: ['a.out'], accepted: true })
    live.stepStarted('b', 'local-1')
    const restored = SharedExecutionState.restore(path, { runId: 'run-1', boundaryMode: 'hybrid', persistValues: true })
    const plan = restored.recoveryPlan(['a', 'b'], new Map([['b', ['a']]]))
    expect(plan.skip).toEqual(['a'])
    expect(plan.rerun).toEqual(['b'])
  })

  test('without persisted values, completed steps cannot be skipped', () => {
    const path = join(dir, 'j.jsonl')
    const live = make('hybrid', { journalPath: path })
    live.stepStarted('a', 'local-1')
    live.putArtifact({ ref: 'a.out', value: 'A', sensitivity: 'shareable', producedBy: 'local-1' })
    live.stepCompleted('a', { outputs: ['a.out'], accepted: true })
    const restored = SharedExecutionState.restore(path, { runId: 'run-1', boundaryMode: 'hybrid' })
    const plan = restored.recoveryPlan(['a'], new Map())
    expect(plan.skip).toEqual([])
    expect(plan.rerun).toEqual(['a'])
  })

  test('external action keys are idempotent per step and unique across steps', () => {
    const s = make('off')
    s.stepStarted('p', 'local-1')
    s.stepStarted('q', 'local-1')
    expect(s.externalAction('p', { key: 'k', description: 'd' })).toBe(true)
    expect(s.externalAction('p', { key: 'k', description: 'd' })).toBe(false)
    expect((thrown(() => s.externalAction('q', { key: 'k', description: 'd' })) as SharedStateError).code).toBe('INVALID_STATE')
  })

  test('invalid graphs are rejected', () => {
    const s = make('off')
    expect(thrown(() => s.recoveryPlan(['a', 'a'], new Map()))).toBeInstanceOf(SharedStateError)
    expect(thrown(() => s.recoveryPlan(['a'], new Map([['a', ['zzz']]])))).toBeInstanceOf(SharedStateError)
    const cyc = new Map<string, readonly string[]>([['a', ['b']], ['b', ['a']]])
    expect((thrown(() => s.recoveryPlan(['a', 'b'], cyc)) as SharedStateError).code).toBe('INVALID_ARGUMENT')
  })

  test('step transitions are validated', () => {
    const s = make('off')
    expect((thrown(() => s.stepCompleted('nope', { outputs: [], accepted: true })) as SharedStateError).code).toBe('UNKNOWN_STEP')
    s.stepStarted('t', 'local-1')
    s.stepFailed('t', 'x')
    expect((thrown(() => s.stepFailed('t', 'y')) as SharedStateError).code).toBe('INVALID_STATE')
    s.stepStarted('t', 'local-1') // restart is allowed
    expect(s.snapshot().steps[0]?.attempts).toBe(2)
  })
})

describe('snapshot', () => {
  test('never contains artifact values and is JSON-serializable', () => {
    const s = make('hybrid')
    s.putArtifact({ ref: 'v', value: 'TOP-SECRET-VALUE', sensitivity: 'private', producedBy: 'local-1' })
    s.putArtifact({ ref: 'v', value: 'TOP-SECRET-VALUE-2', sensitivity: 'shareable', producedBy: 'local-1' })
    const snap = s.snapshot()
    const json = JSON.stringify(snap)
    expect(json).not.toContain('TOP-SECRET')
    expect(snap.artifacts[0]).toMatchObject({ ref: 'v', latestVersion: 2, versions: 2, sensitivity: 'private', valueAvailable: true })
    expect(snap.workers.map(w => w.id)).toEqual(['local-1', 'local-2', 'cloud-1'])
  })
})
