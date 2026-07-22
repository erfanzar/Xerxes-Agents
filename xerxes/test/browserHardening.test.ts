// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ClientError, ValidationError } from '../src/core/errors.js'
import {
  BrowserManager,
  type BrowserAdapter,
} from '../src/operators/browser.js'
import {
  CdpBrowserAdapter,
  resolveCdpWebSocketUrl,
  type CdpConnection,
  type CdpConnectionFactory,
} from '../src/operators/cdpBrowser.js'

test('CDP discovery pins the resolved WebSocket to the supplied endpoint host and port', async () => {
  const inits: Array<RequestInit | undefined> = []
  const fetchImplementation = async (_url: string, init?: RequestInit): Promise<Response> => {
    inits.push(init)
    return Response.json({ webSocketDebuggerUrl: 'ws://169.254.169.254/latest/meta-data' })
  }

  await expect(resolveCdpWebSocketUrl(new URL('http://127.0.0.1:9222'), fetchImplementation))
    .rejects.toBeInstanceOf(ClientError)
  expect(inits).toEqual([{ redirect: 'error' }])

  await expect(resolveCdpWebSocketUrl(
    new URL('http://127.0.0.1:9222'),
    async () => Response.json({ webSocketDebuggerUrl: 'ws://127.0.0.1:9999/devtools/browser/id' }),
  )).rejects.toThrow('does not match the supplied endpoint')

  const pinned = await resolveCdpWebSocketUrl(
    new URL('http://127.0.0.1:9222'),
    async () => Response.json({ webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/id' }),
  )
  expect(pinned).toBe('ws://127.0.0.1:9222/devtools/browser/id')

  const allowlisted = await resolveCdpWebSocketUrl(
    new URL('http://127.0.0.1:9222'),
    async () => Response.json({ webSocketDebuggerUrl: 'ws://browser.internal:9222/devtools/browser/id' }),
    { allowedHosts: ['browser.internal'] },
  )
  expect(allowlisted).toBe('ws://browser.internal:9222/devtools/browser/id')

  const adapterRejects = CdpBrowserAdapter.connect('http://127.0.0.1:9222', {
    connectionFactory: factory(new FakeCdpConnection()),
    fetchImplementation,
  })
  await expect(adapterRejects).rejects.toBeInstanceOf(ClientError)
})

test('CDP adapter closes an owned target when attach fails and closes owned pages on disconnect', async () => {
  const failingAttach = new FakeCdpConnection({ failAttach: true })
  const failed = await CdpBrowserAdapter.connect('http://127.0.0.1:9222', {
    connectionFactory: factory(failingAttach),
    fetchImplementation: async () => Response.json({
      webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/id',
    }),
    screenshotDirectory: await mkdtemp(join(tmpdir(), 'xerxes-cdp-hardening-')),
  })
  await expect(failed.open({ url: 'https://example.test', waitMs: 0 })).rejects.toThrow('attach refused')
  expect(failingAttach.commands.filter(command => command.method === 'Target.closeTarget'))
    .toEqual([{ method: 'Target.closeTarget', params: { targetId: 'target-1' }, sessionId: undefined }])
  await failed.close()

  const connection = new FakeCdpConnection()
  const adapter = await CdpBrowserAdapter.connect('http://127.0.0.1:9222', {
    connectionFactory: factory(connection),
    fetchImplementation: async () => Response.json({
      webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/id',
    }),
    screenshotDirectory: await mkdtemp(join(tmpdir(), 'xerxes-cdp-hardening-')),
  })
  const page = await adapter.open({ url: 'https://example.test', waitMs: 0 })
  expect(refIdOf(page)).toMatch(/^page_/)
  await adapter.close()
  const closedTargets = connection.commands.filter(command => command.method === 'Target.closeTarget')
  expect(closedTargets).toEqual([
    { method: 'Target.closeTarget', params: { targetId: 'target-1' }, sessionId: undefined },
  ])
  expect(connection.commands.map(command => command.method)).toContain('Target.detachFromTarget')
})

test('browser manager caps collected links and the in-page inspection caps anchors at the source', async () => {
  const manyLinks = Array.from({ length: 500 }, (_, index) => ({ url: `https://example.test/${index}` }))
  const manager = new BrowserManager({
    adapter: {
      open: async request => ({
        ...(request.refId === undefined ? {} : { refId: request.refId }),
        url: request.url ?? 'https://example.test',
        title: 'links',
        links: manyLinks,
      }),
      click: async request => ({ refId: request.refId, url: 'https://example.test', title: 'links' }),
      find: async (refId, pattern) => ({ refId, pattern, matchCount: 0, matches: [] }),
      screenshot: async (refId, options) => ({ refId, path: options.path ?? '/tmp/fake.png', fullPage: options.fullPage }),
    },
  })
  const observation = await manager.open({ url: 'https://example.test' })
  expect(observation.links).toHaveLength(200)
  expect(observation.links[199]).toEqual({ id: 199, url: 'https://example.test/199' })

  const connection = new FakeCdpConnection()
  const adapter = await CdpBrowserAdapter.connect('http://127.0.0.1:9222', {
    connectionFactory: factory(connection),
    fetchImplementation: async () => Response.json({
      webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/id',
    }),
    screenshotDirectory: await mkdtemp(join(tmpdir(), 'xerxes-cdp-hardening-')),
  })
  await adapter.open({ url: 'https://example.test', waitMs: 0 })
  const inspection = connection.commands.find(command =>
    command.method === 'Runtime.evaluate' && String(command.params?.expression).includes('contentPreview'))
  expect(String(inspection?.params?.expression)).toContain("querySelectorAll('a[href]')).slice(0, 200)")
  await adapter.close()
})

test('browser find rejects catastrophic-backtracking patterns before they reach the page', async () => {
  const connection = new FakeCdpConnection()
  const adapter = await CdpBrowserAdapter.connect('http://127.0.0.1:9222', {
    connectionFactory: factory(connection),
    fetchImplementation: async () => Response.json({
      webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/id',
    }),
    screenshotDirectory: await mkdtemp(join(tmpdir(), 'xerxes-cdp-hardening-')),
  })
  const page = await adapter.open({ url: 'https://example.test', waitMs: 0 })
  const refId = refIdOf(page)
  const evaluationsBefore = connection.commands.filter(command => command.method === 'Runtime.evaluate').length

  for (const pattern of ['(a+)+', '(\\w*)*', '.*.*', 'x'.repeat(300)]) {
    await expect(adapter.find(refId, pattern)).rejects.toBeInstanceOf(ValidationError)
  }
  // Rejected patterns never compile or run on the page main thread.
  expect(connection.commands.filter(command => command.method === 'Runtime.evaluate'))
    .toHaveLength(evaluationsBefore)

  await expect(adapter.find(refId, '(https?://)?example')).resolves.toMatchObject({ matchCount: 1 })
  await adapter.close()
})

test('browser wait_ms is clamped to sixty seconds with a validation error above it', async () => {
  const manager = new BrowserManager({ adapter: new FakeAdapter() })
  await expect(manager.open({ url: 'https://example.test', waitMs: 60_001 }))
    .rejects.toBeInstanceOf(ValidationError)
  await expect(manager.open({ url: 'https://example.test', waitMs: 60_001 }))
    .rejects.toThrow('at most 60000')
  const accepted = await manager.open({ url: 'https://example.test', waitMs: 60_000 })
  expect(accepted.refId).toMatch(/^page_/)
  const clicked = await manager.click(accepted.refId, { selector: '#go', waitMs: 60_000 })
  expect(clicked.refId).toBe(accepted.refId)
  await expect(manager.click(accepted.refId, { selector: '#go', waitMs: 86_400_000 }))
    .rejects.toBeInstanceOf(ValidationError)
})

test('browser manager setAdapter closes the previous adapter before swapping', async () => {
  const first = new FakeAdapter()
  const second = new FakeAdapter()
  const manager = new BrowserManager({ adapter: first })

  await manager.setAdapter(second)
  expect(first.closeCalls).toBe(1)
  expect(second.closeCalls).toBe(0)

  await manager.setAdapter(second)
  expect(second.closeCalls).toBe(0)

  await manager.setAdapter(undefined)
  expect(second.closeCalls).toBe(1)
  await expect(manager.open({ url: 'https://example.test' })).rejects.toThrow('no browser adapter is configured')
})

test('CDP screenshots reject oversized base64 payloads before decoding or writing', async () => {
  const screenshots = await mkdtemp(join(tmpdir(), 'xerxes-cdp-hardening-'))
  const connection = new FakeCdpConnection({ screenshotData: 'A'.repeat(50 * 1024 * 1024 + 1) })
  try {
    const adapter = await CdpBrowserAdapter.connect('http://127.0.0.1:9222', {
      connectionFactory: factory(connection),
      fetchImplementation: async () => Response.json({
        webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/id',
      }),
      screenshotDirectory: screenshots,
    })
    const page = await adapter.open({ url: 'https://example.test', waitMs: 0 })
    const refId = refIdOf(page)
    await expect(adapter.screenshot(refId, { fullPage: true }))
      .rejects.toThrow('base64 capture limit')
    expect(await Bun.file(join(screenshots, `${refId}.png`)).exists()).toBeFalse()
    await adapter.close()
  } finally {
    await rm(screenshots, { recursive: true, force: true })
  }
})

test('CDP screenshots refuse a symlinked screenshot directory and keep the directory private', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-cdp-hardening-'))
  const realDirectory = join(root, 'real')
  const linkedDirectory = join(root, 'linked')
  try {
    await Bun.write(join(realDirectory, '.keep'), '')
    await symlink(realDirectory, linkedDirectory)

    const connection = new FakeCdpConnection()
    const adapter = await CdpBrowserAdapter.connect('http://127.0.0.1:9222', {
      connectionFactory: factory(connection),
      fetchImplementation: async () => Response.json({
        webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/id',
      }),
      screenshotDirectory: linkedDirectory,
    })
    const page = await adapter.open({ url: 'https://example.test', waitMs: 0 })
    await expect(adapter.screenshot(refIdOf(page), { fullPage: false }))
      .rejects.toThrow('not a real directory')
    await adapter.close()

    const safeConnection = new FakeCdpConnection()
    const safeAdapter = await CdpBrowserAdapter.connect('http://127.0.0.1:9222', {
      connectionFactory: factory(safeConnection),
      fetchImplementation: async () => Response.json({
        webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/id',
      }),
      screenshotDirectory: join(root, 'safe'),
    })
    const safePage = await safeAdapter.open({ url: 'https://example.test', waitMs: 0 })
    const captured = await safeAdapter.screenshot(refIdOf(safePage), { fullPage: false })
    expect(await Bun.file(captured.path).exists()).toBeTrue()
    await safeAdapter.close()
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

function factory(connection: FakeCdpConnection): CdpConnectionFactory {
  return { connect: async () => connection }
}

function refIdOf(observation: { readonly refId?: string }): string {
  const refId = observation.refId
  if (refId === undefined) throw new Error('observation did not include a refId')
  return refId
}

class FakeAdapter implements BrowserAdapter {
  closeCalls = 0

  async open(request: { readonly refId?: string; readonly url?: string }) {
    return {
      ...(request.refId === undefined ? {} : { refId: request.refId }),
      url: request.url ?? 'https://example.test',
      title: 'Fake page',
      links: [],
    }
  }

  async click(request: { readonly refId: string }) {
    return { refId: request.refId, url: 'https://example.test', title: 'Fake page', links: [] }
  }

  async find(refId: string, pattern: string) {
    return { refId, pattern, matchCount: 1, matches: [pattern] }
  }

  async screenshot(refId: string, options: { readonly fullPage: boolean; readonly path?: string }) {
    return { refId, path: options.path ?? '/tmp/fake.png', fullPage: options.fullPage }
  }

  async close(): Promise<void> {
    this.closeCalls += 1
  }
}

class FakeCdpConnection implements CdpConnection {
  readonly commands: Array<{
    method: string
    params: Readonly<Record<string, unknown>> | undefined
    sessionId: string | undefined
  }> = []
  closed = false
  private url = 'about:blank'

  constructor(private readonly options: {
    readonly failAttach?: boolean
    readonly screenshotData?: string
  } = {}) {}

  close(): void {
    this.closed = true
  }

  async command(
    method: string,
    params?: Readonly<Record<string, unknown>>,
    sessionId?: string,
  ): Promise<unknown> {
    this.commands.push({ method, params, sessionId })
    switch (method) {
      case 'Browser.getVersion':
        return { product: 'Fake Chromium' }
      case 'Target.createTarget':
        return { targetId: 'target-1' }
      case 'Target.attachToTarget':
        if (this.options.failAttach) throw new ClientError('cdp', 'attach refused')
        return { sessionId: 'session-1' }
      case 'Target.detachFromTarget':
      case 'Target.closeTarget':
        return {}
      case 'Page.navigate':
        this.url = String(params?.url)
        return { frameId: 'frame-1' }
      case 'Page.captureScreenshot':
        return { data: this.options.screenshotData ?? 'iVBORw0KGgo=' }
      case 'Runtime.evaluate':
        return this.evaluate(String(params?.expression ?? ''))
      default:
        throw new Error(`unexpected CDP command: ${method}`)
    }
  }

  private evaluate(expression: string): unknown {
    if (expression.includes('contentPreview')) {
      return {
        result: {
          value: {
            url: this.url,
            title: 'Example browser page',
            contentPreview: 'Example rendered page',
            links: [{ url: 'https://example.test/next' }],
          },
        },
      }
    }
    if (expression.includes('new RegExp')) {
      return { result: { value: { ok: true, matchCount: 1, matches: ['Example'] } } }
    }
    return { result: { value: { ok: true } } }
  }
}
