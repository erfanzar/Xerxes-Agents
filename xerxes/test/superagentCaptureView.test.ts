// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { workspaceCaptureView } from '../src/daemon/superagentCaptureView.js'

test('a capture view projects only exact-root task summaries and never reads transcripts or account overlays', async () => {
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: '/fixture/b' })
  const b = await runtime.openSession('b', undefined, { cwd: '/fixture/b' }); const a = await runtime.openSession('a', undefined, { cwd: '/fixture/a' })
  b.model = 'fixture-visible-model'; b.status = 'working'; a.model = 'private-A-marker'
  b.messages.push({ role: 'user', content: 'private-transcript-marker' })
  b.metadata.account = 'private-account-marker'
  Object.defineProperty(b, 'messages', { get() { throw new Error('capture must not hydrate/read a transcript') } })
  Object.defineProperty(b, 'metadata', { get() { throw new Error('capture must not inspect account metadata') } })
  const view = workspaceCaptureView(runtime, '/fixture/b', 1234)
  expect(view.html).toContain('fixture-visible-model')
  expect(view.html).not.toContain('private-A-marker')
  expect(view.html).not.toContain('private-transcript-marker')
  expect(view.html).not.toContain('private-account-marker')
  expect(view.root).toBe('/fixture/b')
  expect(view.viewId).toMatch(/^[a-f0-9]{32}$/)
  expect(Object.isFrozen(view)).toBe(true)
  b.model = 'later-model'
  expect(view.html).not.toContain('later-model')
})

test('nested sibling roots do not enter the immutable capture scope', async () => {
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: '/fixture/b' })
  const outer = await runtime.openSession('outer', undefined, { cwd: '/fixture/b' }); const nested = await runtime.openSession('nested', undefined, { cwd: '/fixture/b/nested' })
  outer.model = 'outer-visible'; nested.model = 'nested-private'
  const view = workspaceCaptureView(runtime, '/fixture/b', 1234)
  expect(view.html).toContain('outer-visible'); expect(view.html).not.toContain('nested-private')
})

test('capture markup escapes untrusted summaries and cannot execute scripts, navigate or load external assets', async () => {
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: '/fixture/b' })
  const session = await runtime.openSession('b', undefined, { cwd: '/fixture/b' })
  session.model = '<img src=https://elsewhere.invalid/ onerror=alert(1)>'
  const view = workspaceCaptureView(runtime, '/fixture/b', 1234)
  expect(view.html).not.toContain('<img')
  expect(view.html).not.toContain('<script')
  expect(view.html).not.toContain('<a ')
  expect(view.html).toContain('&lt;img')
  expect(view.headers['Content-Security-Policy']).toContain("default-src 'none'")
  expect(view.headers['Content-Security-Policy']).toContain("frame-ancestors 'none'")
  expect(view.headers['Cache-Control']).toBe('no-store')
  expect(view.headers['X-Content-Type-Options']).toBe('nosniff')
})

test('capture roots and timestamps must be explicit; no ambient project or clock fallback', () => {
  const runtime = new InMemoryDaemonRuntime()
  expect(() => workspaceCaptureView(runtime, '', 1234)).toThrow()
  expect(() => workspaceCaptureView(runtime, '../b', 1234)).toThrow()
  expect(() => workspaceCaptureView(runtime, '/fixture/b', NaN)).toThrow()
})
