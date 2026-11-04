// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { appendFileSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import type { OutboundRequestSummary } from '../src/security/boundaryTypes.js'
import { DisclosureLog, defaultDisclosureLogPath, renderDisclosures } from '../src/security/disclosureLog.js'

const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function temp(): string {
  const root = mkdtempSync(join(tmpdir(), 'xerxes-disclosure-'))
  roots.push(root)
  return root
}

const summary = (text: string, sessionId = 's1'): OutboundRequestSummary => ({
  provider: 'openai', model: 'gpt-5', trust: 'cloud', purpose: 'turn', sessionId, agentId: 'a', text, bytes: Buffer.byteLength(text),
})

test('records digest and size, never the text; private permissions', () => {
  const root = temp()
  const path = join(root, 'boundary', 'disclosures.jsonl')
  let clock = 0
  const log = new DisclosureLog({ path, now: () => new Date(Date.UTC(2026, 10, 2, 0, 0, clock++)) })
  const record = log.record(summary('secret text body'), 'sent')
  expect(record.digest).toBe(createHash('sha256').update('secret text body').digest('hex'))
  expect(record.bytes).toBe(16)
  log.record(summary('other', 's2'), 'denied', 'user declined')
  const raw = readFileSync(path, 'utf8')
  expect(raw).not.toContain('secret text body')
  expect(raw.trim().split('\n')).toHaveLength(2)
  expect(statSync(path).mode & 0o777).toBe(0o600)
  expect(statSync(join(root, 'boundary')).mode & 0o777).toBe(0o700)
  const newest = log.read()
  expect(newest.map(entry => entry.decision)).toEqual(['denied', 'sent'])
  expect(log.read({ sessionId: 's1' }).map(entry => entry.sessionId)).toEqual(['s1'])
  expect(log.read({ limit: 1 })).toHaveLength(1)
})

test('read skips corrupt lines and a missing file is empty', () => {
  const root = temp()
  const path = join(root, 'disclosures.jsonl')
  const log = new DisclosureLog({ path })
  expect(log.read()).toEqual([])
  log.record(summary('x'), 'approved')
  appendFileSync(path, 'not json\n{"at":"x"}\n')
  expect(log.read().map(entry => entry.decision)).toEqual(['approved'])
})

test('renderDisclosures prints a compact table', () => {
  expect(renderDisclosures([])).toBe('No outbound requests recorded.')
  const root = temp()
  const log = new DisclosureLog({ path: join(root, 'd.jsonl'), now: () => new Date('2026-11-02T10:00:00.123Z') })
  log.record(summary('hello'), 'denied', 'data boundary is private')
  const table = renderDisclosures(log.read())
  const lines = table.split('\n')
  expect(lines[0]).toMatch(/^time\s+decision\s+trust\s+purpose\s+provider\/model\s+bytes\s+digest\s+reason$/)
  expect(lines[2]).toContain('2026-11-02 10:00:00Z')
  expect(lines[2]).toContain('openai/gpt-5')
  expect(lines[2]).toContain('data boundary is private')
})

test('default path lives under XERXES_HOME/boundary', () => {
  expect(defaultDisclosureLogPath({ XERXES_HOME: '/tmp/xh' })).toBe('/tmp/xh/boundary/disclosures.jsonl')
})
