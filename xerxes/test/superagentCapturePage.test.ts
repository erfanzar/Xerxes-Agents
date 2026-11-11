// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { openWorkspaceCapturePage } from '../src/daemon/superagentCapturePage.js'

test('an owned ephemeral loopback page serves one immutable exact-root view and no image or ambient UI route', async () => {
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: '/fixture/b' })
  const b = await runtime.openSession('b', undefined, { cwd: '/fixture/b' }); b.model = 'B-visible'
  const a = await runtime.openSession('a', undefined, { cwd: '/fixture/a' }); a.model = 'A-private'
  const page = openWorkspaceCapturePage(runtime, '/fixture/b', 1234)
  try {
    const url = new URL(page.url)
    expect(url.hostname).toBe('127.0.0.1')
    expect(page.root).toBe('/fixture/b')
    expect(page.current()).toBe(true)
    const response = await fetch(page.url)
    const html = await response.text()
    expect(response.status).toBe(200)
    expect(html).toContain(`<meta name="xerxes-capture-view" content="${page.viewId}">`)
    expect(html).toContain('B-visible'); expect(html).not.toContain('A-private')
    expect(response.headers.get('cache-control')).toBe('no-store')
    expect(response.headers.get('content-security-policy')).toContain("default-src 'none'")
    b.model = 'B-later'
    expect(await (await fetch(page.url)).text()).toBe(html)
    expect((await fetch(new URL('/api/sessions', url))).status).toBe(404)
    expect((await fetch(`${page.url}?workspace=A`)).status).toBe(404)
    expect((await fetch(page.url, { method: 'POST' })).status).toBe(405)
  } finally { page.close() }
  expect(page.current()).toBe(false)
})

test('capture page cancellation closes only its owned ephemeral listener and refuses a pre-aborted request', async () => {
  const runtime = new InMemoryDaemonRuntime(); const abort = new AbortController()
  const page = openWorkspaceCapturePage(runtime, '/fixture/b', 1234, abort.signal)
  abort.abort()
  expect(page.current()).toBe(false)
  await expect(fetch(page.url)).rejects.toThrow()
  expect(() => openWorkspaceCapturePage(runtime, '/fixture/b', 1234, abort.signal)).toThrow()
  page.close()
})
