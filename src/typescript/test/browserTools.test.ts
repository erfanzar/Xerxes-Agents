// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ClientError, ValidationError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  BrowserSession,
  registerBrowserTools,
  type BrowserDocument,
  type BrowserPort,
} from '../src/tools/browserTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

test('browser tool handlers delegate their full surface through an injected port', async () => {
  const port = new FakeBrowserPort()
  const session = new BrowserSession({ port })
  const registry = new ToolRegistry()
  registerBrowserTools(registry, { session })

  expect(await execute(registry, 'browser_navigate', { url: 'https://example.test/next' })).toEqual({
    url: 'https://example.test/next',
    title: 'Page next',
    elements: 2,
  })
  const snapshot = await execute(registry, 'browser_snapshot', {})
  expect(snapshot.url).toBe('https://example.test/next')
  expect((snapshot.text as string).length).toBe(4_000)
  expect(snapshot.elements).toEqual([
    { ref: 'e1', tag: 'a', role: 'link', name: 'Next', href: 'https://example.test/next' },
    { ref: 'e2', tag: 'input', role: 'textbox', name: 'Search', href: '' },
  ])

  expect(await execute(registry, 'browser_vision', {})).toEqual({
    url: 'https://example.test/next',
    image_b64: 'UE5H',
    format: 'png',
    summary: 'Page next',
    text: 'visual summary',
  })
  expect(await execute(registry, 'browser_get_images', {})).toEqual({
    url: 'https://example.test/next',
    images: [{ src: 'https://example.test/image.png', alt: 'Example image' }],
  })
  expect(await execute(registry, 'browser_console', {})).toEqual({
    url: 'https://example.test/next',
    console: ['[log] ready'],
  })
  expect(await execute(registry, 'browser_click', { ref: 'e1' })).toEqual({
    ok: true,
    ref: 'e1',
    note: 'clicked e1',
  })
  expect(await execute(registry, 'browser_type', { ref: 'e2', text: 'query', submit: true })).toEqual({
    ok: true,
    ref: 'e2',
    submitted: true,
  })
  expect(await execute(registry, 'browser_press', { key: 'Enter' })).toEqual({ ok: true, key: 'Enter' })
  expect(await execute(registry, 'browser_scroll', {})).toEqual({ ok: true, scroll_y: 400 })
  expect(await execute(registry, 'browser_back', {})).toEqual({
    url: 'https://example.test/start',
    title: 'Page start',
    elements: 2,
    ok: true,
  })

  expect(port.events).toEqual([
    'navigate:https://example.test/next',
    'snapshot',
    'vision',
    'images',
    'console',
    'click:e1',
    'type:e2:query:true',
    'press:Enter',
    'scroll:400',
    'back',
  ])
})

test('browser session blocks local input and rejects a port that reports an unsafe redirect', async () => {
  const port = new FakeBrowserPort()
  const session = new BrowserSession({ port })

  await expect(session.navigate('file:///etc/passwd')).rejects.toBeInstanceOf(ValidationError)
  await expect(session.navigate('http://127.0.0.1/private')).rejects.toBeInstanceOf(ValidationError)
  expect(port.events).toEqual([])

  port.nextUrl = 'http://127.0.0.1/redirected'
  await expect(session.navigate('https://example.test/public')).rejects.toBeInstanceOf(ClientError)
  expect(port.events).toEqual(['navigate:https://example.test/public'])
})

test('browser action failures remain explicit and unconfigured browser tools do not pretend to succeed', async () => {
  const port = new FakeBrowserPort()
  const configured = new ToolRegistry()
  registerBrowserTools(configured, { session: new BrowserSession({ port }) })

  expect(await execute(configured, 'browser_click', { ref: 'missing' })).toEqual({
    ok: false,
    reason: 'unknown ref missing',
  })
  expect(await execute(configured, 'browser_type', { ref: 'missing', text: 'query' })).toEqual({
    ok: false,
    reason: 'unknown ref missing',
  })

  const unconfigured = new ToolRegistry()
  registerBrowserTools(unconfigured, {})
  await expect(unconfigured.execute(call('browser_navigate', { url: 'https://example.test' }), { metadata: {} }))
    .rejects.toThrow('no BrowserSession is configured')
})

test('browser session reports missing adapter capabilities rather than synthesizing state', async () => {
  const session = new BrowserSession({
    port: {
      async navigate(url) {
        return documentFor(url)
      },
    },
  })
  await expect(session.snapshot()).rejects.toThrow('does not support snapshot')
})

async function execute(registry: ToolRegistry, name: string, arguments_: JsonObject): Promise<JsonObject> {
  return JSON.parse(await registry.execute(call(name, arguments_), { metadata: {} })) as JsonObject
}

function call(name: string, arguments_: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

class FakeBrowserPort implements BrowserPort {
  readonly events: string[] = []
  nextUrl: string | undefined
  private current = documentFor('https://example.test/start')

  async navigate(url: string): Promise<BrowserDocument> {
    this.events.push('navigate:' + url)
    this.current = documentFor(this.nextUrl ?? url)
    this.nextUrl = undefined
    return this.current
  }

  async back() {
    this.events.push('back')
    this.current = documentFor('https://example.test/start')
    return { ok: true, document: this.current }
  }

  async snapshot(): Promise<BrowserDocument> {
    this.events.push('snapshot')
    return this.current
  }

  async vision() {
    this.events.push('vision')
    return {
      url: this.current.url,
      imageB64: 'UE5H',
      format: 'png',
      summary: this.current.title,
      text: 'visual summary',
    }
  }

  async getImages() {
    this.events.push('images')
    return { url: this.current.url, images: this.current.images }
  }

  async consoleLog() {
    this.events.push('console')
    return { url: this.current.url, console: this.current.console }
  }

  async click(ref: string) {
    this.events.push('click:' + ref)
    if (ref !== 'e1') return { ok: false, reason: 'unknown ref ' + ref }
    return { ok: true, note: 'clicked e1' }
  }

  async typeText(ref: string, text: string, options: { readonly submit: boolean }) {
    this.events.push('type:' + ref + ':' + text + ':' + options.submit)
    if (ref !== 'e2') return { ok: false, reason: 'unknown ref ' + ref }
    return { ok: true }
  }

  async press(key: string) {
    this.events.push('press:' + key)
    return { ok: true }
  }

  async scroll(dy: number) {
    this.events.push('scroll:' + dy)
    this.current = { ...this.current, scrollY: Math.max(0, this.current.scrollY + dy) }
    return { ok: true, scrollY: this.current.scrollY, document: this.current }
  }
}

function documentFor(url: string): BrowserDocument {
  const path = new URL(url).pathname
  const title = path === '/next' ? 'Page next' : 'Page start'
  return {
    url,
    title,
    text: 'text '.repeat(1_000),
    elements: [
      { ref: 'e1', tag: 'a', role: 'link', name: 'Next', href: 'https://example.test/next' },
      { ref: 'e2', tag: 'input', role: 'textbox', name: 'Search' },
    ],
    images: [{ src: 'https://example.test/image.png', alt: 'Example image' }],
    console: ['[log] ready'],
    scrollY: 0,
  }
}
