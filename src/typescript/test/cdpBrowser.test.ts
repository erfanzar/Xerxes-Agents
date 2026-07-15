// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ValidationError } from '../src/core/errors.js'
import {
  CdpBrowserAdapter,
  redactedCdpEndpoint,
  resolveCdpWebSocketUrl,
  type CdpConnection,
  type CdpConnectionFactory,
} from '../src/operators/cdpBrowser.js'
import { BrowserManager, registerBrowserManagerTools } from '../src/operators/browser.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

test('CDP browser adapter discovers an explicit endpoint and executes open, click, find, and screenshot without Playwright', async () => {
  const screenshots = await mkdtemp(join(tmpdir(), 'xerxes-cdp-browser-'))
  const connection = new FakeCdpConnection()
  const fetches: string[] = []
  const manager = new BrowserManager()
  try {
    const status = await manager.connectCdp('http://127.0.0.1:9222?token=do-not-display', {
      connectionFactory: factory(connection),
      fetchImplementation: async url => {
        fetches.push(url)
        return Response.json({ webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/browser-id?token=private' })
      },
      screenshotDirectory: screenshots,
      sleep: async () => undefined,
    })

    expect(fetches).toEqual(['http://127.0.0.1:9222/json/version'])
    expect(status).toEqual({ connected: true, endpoint: 'http://127.0.0.1:9222', kind: 'cdp' })
    expect(connection.endpoints).toEqual(['ws://127.0.0.1:9222/devtools/browser/browser-id?token=private'])

    const page = await manager.open({ url: 'https://example.test/start', waitMs: 0 })
    const refId = page.refId
    expect(page).toMatchObject({
      url: 'https://example.test/start',
      title: 'Example browser page',
      links: [{ id: 0, url: 'https://example.test/next' }],
    })
    expect(refId).toMatch(/^page_/)

    const clicked = await manager.click(refId, { selector: '#continue', waitMs: 0 })
    expect(clicked.url).toBe('https://example.test/start')
    expect(await manager.find(refId, 'example')).toEqual({
      refId,
      pattern: 'example',
      matchCount: 2,
      matches: ['Example', 'example'],
    })

    const screenshot = await manager.screenshot(refId, { fullPage: true })
    expect(await Bun.file(screenshot.path).exists()).toBeTrue()
    await expect(manager.screenshot(refId, { fullPage: true, path: '/tmp/outside.png' }))
      .rejects.toBeInstanceOf(ValidationError)
    expect(connection.commands.map(command => command.method)).toEqual(expect.arrayContaining([
      'Browser.getVersion',
      'Target.createTarget',
      'Target.attachToTarget',
      'Page.navigate',
      'Runtime.evaluate',
      'Page.captureScreenshot',
    ]))
  } finally {
    await manager.disconnect()
    await rm(screenshots, { recursive: true, force: true })
  }
  expect(connection.closed).toBeTrue()
  expect(connection.commands.map(command => command.method)).toContain('Target.detachFromTarget')
})

test('browser manager tool registration forwards real CDP manager calls and leaves unconfigured sessions actionable', async () => {
  const connected = new BrowserManager({ adapter: new FakeBrowserAdapter() })
  const registry = new ToolRegistry()
  registerBrowserManagerTools(registry, connected)

  expect(await execute(registry, 'web.open', { url: 'https://example.test', wait_ms: 0 })).toEqual({
    ref_id: 'page_fake',
    url: 'https://example.test',
    title: 'Fake page',
    content_preview: 'fake content',
    links: [{ id: 0, url: 'https://example.test/next' }],
  })

  const unconfigured = new ToolRegistry()
  registerBrowserManagerTools(unconfigured, new BrowserManager())
  await expect(unconfigured.execute(call('web.open', { url: 'https://example.test' }), { metadata: {} }))
    .rejects.toThrow('no browser adapter is configured')
})

test('CDP endpoint discovery preserves the secret transport endpoint but redacts display metadata', async () => {
  const discovered = await resolveCdpWebSocketUrl(
    new URL('https://browser.example.test/control?token=private'),
    async () => Response.json({ webSocketDebuggerUrl: 'wss://browser.example.test/devtools/browser/id?token=private' }),
  )
  expect(discovered).toBe('wss://browser.example.test/devtools/browser/id?token=private')
  expect(redactedCdpEndpoint('wss://name:password@browser.example.test/devtools/browser/id?token=private'))
    .toBe('wss://browser.example.test/devtools/browser/id')
})

function factory(connection: FakeCdpConnection): CdpConnectionFactory {
  return {
    async connect(endpoint) {
      connection.endpoints.push(endpoint)
      return connection
    },
  }
}

async function execute(registry: ToolRegistry, name: string, arguments_: JsonObject): Promise<JsonObject> {
  return JSON.parse(await registry.execute(call(name, arguments_), { metadata: {} })) as JsonObject
}

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

class FakeCdpConnection implements CdpConnection {
  readonly commands: Array<{ method: string; params: Readonly<Record<string, unknown>> | undefined; sessionId: string | undefined }> = []
  readonly endpoints: string[] = []
  closed = false
  private url = 'about:blank'

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
        return { sessionId: 'session-1' }
      case 'Target.detachFromTarget':
        return {}
      case 'Page.navigate':
        this.url = String(params?.url)
        return { frameId: 'frame-1' }
      case 'Page.captureScreenshot':
        return { data: 'iVBORw0KGgo=' }
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
    if (expression.includes('matchAll')) {
      return { result: { value: { ok: true, matchCount: 2, matches: ['Example', 'example'] } } }
    }
    return { result: { value: { ok: true } } }
  }
}

class FakeBrowserAdapter {
  async click(request: { readonly refId: string }): Promise<{ readonly url: string; readonly title: string }> {
    return { url: 'https://example.test/next', title: request.refId }
  }

  async find(refId: string, pattern: string) {
    return { refId, pattern, matchCount: 0, matches: [] }
  }

  async open(request: { readonly refId?: string; readonly url?: string }): Promise<{
    readonly contentPreview: string
    readonly links: readonly { readonly url: string }[]
    readonly refId: string
    readonly title: string
    readonly url: string
  }> {
    return {
      refId: request.refId ?? 'page_fake',
      url: request.url ?? 'https://example.test',
      title: 'Fake page',
      contentPreview: 'fake content',
      links: [{ url: 'https://example.test/next' }],
    }
  }

  async screenshot(refId: string, options: { readonly fullPage: boolean; readonly path?: string }) {
    return { refId, fullPage: options.fullPage, path: options.path ?? '/tmp/fake.png' }
  }
}
