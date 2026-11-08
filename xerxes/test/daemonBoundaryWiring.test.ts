// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import { boundaryCommand, sourcesCommand } from '../src/daemon/boundaryCommand.js'
import { createSessionBoundaryFactory, type BoundarySession } from '../src/daemon/sessionBoundary.js'
import type { DataTrust } from '../src/security/boundaryTypes.js'
import { DisclosureLog } from '../src/security/disclosureLog.js'
import { SourceLog } from '../src/tools/sourceLog.js'
import type { JsonObject } from '../src/types/toolCalls.js'

const CANARY = 'CANARY-daemon-wiring-4411'
const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function workspace(): string {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-boundary-wiring-')))
  roots.push(root)
  mkdirSync(join(root, 'secret'), { recursive: true })
  writeFileSync(join(root, 'secret', 'customers.csv'), `name,ssn\nalice,${CANARY}\n`)
  return root
}

const call = (name: string, args: JsonObject) => ({ function: { name, arguments: args } })
const session = (cwd: string): BoundarySession => ({ cwd, model: 'm', metadata: {} })

test('session boundary factory returns nothing when the mode is off', () => {
  const root = workspace()
  const factory = createSessionBoundaryFactory({ runtime: () => ({}), env: {}, trustOf: () => 'cloud' })
  expect(factory(session(root))).toBeUndefined()
})

test('session boundary denies a cloud read of a private path with a reason', () => {
  const root = workspace()
  const factory = createSessionBoundaryFactory({
    runtime: () => ({ data_boundary: 'hybrid', boundary: { private_paths: ['secret/**'] } }),
    env: {},
    trustOf: () => 'cloud' as DataTrust,
  })
  const port = factory(session(root))
  expect(port).toBeDefined()
  const read = call('ReadFile', { file_path: 'secret/customers.csv' })
  expect(port!.toolDisposition(read)).toBe('deny')
  expect(port!.reason(read)).toContain('private')
  expect(port!.toolDisposition(call('ReadFile', { file_path: 'README.md' }))).toBeUndefined()
})

test('a trust resolver that throws is treated as cloud', () => {
  const root = workspace()
  const factory = createSessionBoundaryFactory({
    runtime: () => ({ data_boundary: 'hybrid', boundary: { private_paths: ['secret/**'] } }),
    env: {},
    trustOf: () => { throw new Error('profile missing') },
  })
  expect(factory(session(root))!.toolDisposition(call('ReadFile', { file_path: 'secret/customers.csv' }))).toBe('deny')
})

test('a malformed boundary setting fails closed', () => {
  const root = workspace()
  const factory = createSessionBoundaryFactory({ runtime: () => ({ data_boundary: 'sometimes' }), env: {}, trustOf: () => 'local' })
  expect(() => factory(session(root))).toThrow()
})

test('/boundary shows status, switches mode and respects the env override', () => {
  const root = workspace()
  let mode = ''
  const status = boundaryCommand('', { cwd: root, runtime: {}, env: {}, setMode: next => { mode = next } })
  expect(status.ok).toBe(true)
  expect(status.report).toContain('OFF')
  const switched = boundaryCommand('hybrid', { cwd: root, runtime: {}, env: {}, setMode: next => { mode = next } })
  expect(switched.ok).toBe(true)
  expect(mode).toBe('hybrid')
  mode = ''
  const refused = boundaryCommand('off', { cwd: root, runtime: {}, env: { XERXES_DATA_BOUNDARY: 'private' }, setMode: next => { mode = next } })
  expect(refused.ok).toBe(false)
  expect(mode).toBe('')
  expect(boundaryCommand('bogus', { cwd: root, runtime: {}, env: {}, setMode: () => {} }).ok).toBe(false)
})

test('/boundary log lists this session only and never prints sent text', () => {
  const root = workspace()
  const log = new DisclosureLog({ path: join(root, 'log', 'disclosures.jsonl') })
  const summary = { provider: 'p', model: 'm', trust: 'cloud' as const, purpose: 'turn' as const, text: CANARY, bytes: CANARY.length }
  log.record({ ...summary, sessionId: 's1' }, 'sent')
  log.record({ ...summary, sessionId: 's2' }, 'denied', 'private')
  const mine = boundaryCommand('log', { cwd: root, sessionId: 's1', runtime: {}, env: {}, setMode: () => {}, log })
  expect(mine.ok).toBe(true)
  expect(mine.count).toBe(1)
  const all = boundaryCommand('log 10 --all', { cwd: root, sessionId: 's1', runtime: {}, env: {}, setMode: () => {}, log })
  expect(all.count).toBe(2)
  expect(all.report).not.toContain(CANARY)
  expect(readFileSync(log.path, 'utf8')).not.toContain(CANARY)
})

test('/sources lists and clears one session', () => {
  const log = new SourceLog()
  log.record({ sessionId: 's1', url: 'https://example.com/a', title: 'A', tool: 'WebScraper' })
  log.record({ sessionId: 's2', url: 'https://example.com/b', tool: 'WebScraper' })
  const listed = sourcesCommand('', 's1', log)
  expect(listed.count).toBe(1)
  expect(listed.report).toContain('https://example.com/a')
  expect(sourcesCommand('clear', 's1', log).ok).toBe(true)
  expect(sourcesCommand('', 's1', log).count).toBe(0)
  expect(sourcesCommand('', 's2', log).count).toBe(1)
  expect(sourcesCommand('nope', 's1', log).ok).toBe(false)
})
