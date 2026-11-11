// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { createWorkspaceCaptureHost } from '../src/daemon/superagentCaptureHost.js'
import type { CdpConnectionFactory, CdpConnectionOptions } from '../src/operators/cdpBrowser.js'
import { validateCaptureImage } from '../src/hub/supervisor/abilities/captureImage.js'
import { png } from './support/captureImage.js'

function fixture() {
  const methods: string[] = []; let closed = 0; let url = 'about:blank'; let width = 390; let events: CdpConnectionOptions | undefined
  let rootFrame = 'frame'; let live = true; let onCapture = (): void => {}; let sendLoad = true
  const factory: CdpConnectionFactory = { async connect(_endpoint, options) { events = options; return {
    close() { closed += 1 },
    async command(method, params) {
      methods.push(method)
      switch (method) {
        case 'Target.getTargetInfo': return { targetInfo: { targetId: 'selected', type: 'page', url } }
        case 'Target.attachToTarget': return { sessionId: 'attached' }
        case 'Page.getFrameTree': return { frameTree: { frame: { id: rootFrame, url } } }
        case 'Page.navigate':
          url = typeof params?.url === 'string' ? params.url : ''
          if (sendLoad) events?.onEvent?.({ method: 'Page.lifecycleEvent', sessionId: 'attached', params: { frameId: rootFrame, loaderId: 'loader', name: 'load' } })
          return { frameId: rootFrame, loaderId: 'loader' }
        case 'Emulation.setDeviceMetricsOverride': width = typeof params?.width === 'number' ? params.width : 0; return {}
        case 'DOM.getDocument': return { root: { nodeId: 1 } }
        case 'DOM.querySelector': return { nodeId: 2 }
        case 'DOM.getAttributes': return { attributes: ['name', 'xerxes-capture-view', 'content', new URL(url).pathname.split('/').pop()] }
        case 'Page.captureScreenshot': onCapture(); return { data: png(width).toString('base64') }
        default: return {}
      }
    },
  } } }
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: '/fixture/b' })
  return { runtime, factory, methods, get closed() { return closed }, get url() { return url }, get events() { return events },
    current: () => live, revoke() { live = false }, navigate() { rootFrame = 'foreign'; events?.onEvent?.({ method: 'Page.frameNavigated', sessionId: 'attached', params: { frame: { id: 'foreign', url: 'https://elsewhere.invalid' } } }) },
    duringCapture(callback: () => void) { onCapture = callback }, targetUrl(value: string) { url = value }, suppressLoad() { sendLoad = false }, resetBlank() { url = 'about:blank'; sendLoad = true } }
}

test('explicit selected dedicated target captures only the immutable root view and admits pixels before returning', async () => {
  const f = fixture(); let admitted = 0
  const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
    targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: f.factory,
    async admit(image, view) { expect(image.viewId).toBe(view.viewId); expect(view.root).toBe('/fixture/b'); expect(Object.keys(view).sort()).toEqual(['root', 'viewId']); admitted += 1 } })
  const raw = await host(390)
  const image = validateCaptureImage(raw)
  expect(image.width).toBe(390); expect(image.height).toBe(900)
  expect(admitted).toBe(1); expect(f.closed).toBe(1)
  expect(f.methods).not.toContain('Target.createTarget')
  expect(f.methods).not.toContain('Target.getTargets')
  await expect(fetch(f.url)).rejects.toThrow()
})

test('missing explicit capture or dedicated-navigation consent never contacts CDP', async () => {
  const f = fixture()
  const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
    targetId: 'selected', enabled: false, allowViewNavigation: false, current: f.current, now: () => 1234, connectionFactory: f.factory,
    async admit() {} })
  await expect(host(390)).rejects.toThrow()
  expect(f.methods).toHaveLength(0)
})

test('navigation or authority drift while screenshot bytes are pending cannot reach disclosure', async () => {
  for (const change of ['navigate', 'revoke']) {
    const f = fixture(); let admitted = 0
    f.duringCapture(() => change === 'navigate' ? f.navigate() : f.revoke())
    const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
      targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: f.factory,
      async admit() { admitted += 1 } })
    await expect(host(390)).rejects.toThrow()
    expect(admitted).toBe(0); expect(f.closed).toBe(1)
  }
})

test('a connection that ignores cancellation keeps capacity until settlement and is then closed', async () => {
  const f = fixture(); const abort = new AbortController()
  let release = (): void => {}; let started = (): void => {}
  const held = new Promise<void>(resolve => { release = resolve })
  const connecting = new Promise<void>(resolve => { started = resolve })
  const factory: CdpConnectionFactory = { async connect(endpoint, options) { started(); await held; return f.factory.connect(endpoint, options) } }
  const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
    targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: factory, async admit() {} })
  const pending = host(390, abort.signal)
  await connecting; abort.abort()
  await expect(pending).rejects.toThrow()
  await expect(host(390)).rejects.toThrow()
  release(); await Bun.sleep(0)
  expect(f.closed).toBe(1)
  expect(f.methods).toHaveLength(0)
  expect(validateCaptureImage(await host(390)).width).toBe(390)
})

test('invalid endpoint diagnostics never echo private connection configuration', async () => {
  const f = fixture()
  const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'bad-fixture-secret',
    targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: f.factory, async admit() {} })
  let message = ''
  try { await host(390) } catch (error) { message = error instanceof Error ? error.message : String(error) }
  expect(message).not.toBe('')
  expect(message).not.toContain('fixture-secret')
  expect(f.methods).toHaveLength(0)
})

test('cancelling the internal load wait settles the operation instead of permanently occupying the selected target', async () => {
  const f = fixture(); f.suppressLoad(); const abort = new AbortController()
  const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
    targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: f.factory, async admit() {} })
  const pending = host(390, abort.signal)
  await Bun.sleep(0)
  expect(f.methods).toContain('Page.navigate')
  abort.abort(); await expect(pending).rejects.toThrow()
  await Bun.sleep(0); f.resetBlank()
  await expect(host(390)).resolves.toMatchObject({ width: 390 })
})

test('an ordinary selected browser tab is never navigated or used to generate a workspace view', async () => {
  const f = fixture(); f.targetUrl('https://elsewhere.invalid/private-tab'); let projected = 0
  const host = createWorkspaceCaptureHost({ runtime: { listSessions() { projected += 1; return f.runtime.listSessions() } }, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
    targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: f.factory, async admit() {} })
  await expect(host(390)).rejects.toThrow()
  expect(projected).toBe(0)
  expect(f.methods).not.toContain('Page.navigate')
  expect(f.methods).not.toContain('Page.captureScreenshot')
})

test('the selected dedicated target can produce each supported fixed viewport without arbitrary evaluation or target discovery', async () => {
  const f = fixture()
  const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
    targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: f.factory, async admit() {} })
  for (const width of [390, 768, 1280, 1440]) expect(validateCaptureImage(await host(width)).width).toBe(width)
  expect(f.closed).toBe(4)
  expect(f.methods).not.toContain('Runtime.evaluate')
  expect(f.methods).not.toContain('Target.getTargets')
})

test('failed admission leaves the actual owned target reusable without resetting its URL', async () => {
  const f = fixture(); let attempts = 0
  const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
    targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: f.factory,
    async admit() { if (++attempts === 1) throw new Error('declined') } })
  await expect(host(390)).rejects.toThrow()
  expect(f.url).not.toBe('about:blank')
  await expect(host(390)).resolves.toMatchObject({ width: 390 })
})

test('authority expiring during target-info lookup prevents debugger attachment', async () => {
  const f = fixture()
  const factory: CdpConnectionFactory = { async connect(endpoint, options) {
    const connection = await f.factory.connect(endpoint, options)
    return { close: () => connection.close(), async command(method, params, session) {
      const result = await connection.command(method, params, session)
      if (method === 'Target.getTargetInfo') f.revoke()
      return result
    } }
  } }
  const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
    targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: factory, async admit() {} })
  await expect(host(390)).rejects.toThrow()
  expect(f.methods).not.toContain('Target.attachToTarget')
})

test('service-worker interception is bypassed before navigating the owned capture page', async () => {
  const f = fixture(); let bypassed = false; let intercepted = 0
  const factory: CdpConnectionFactory = { async connect(endpoint, options) {
    const connection = await f.factory.connect(endpoint, options)
    return { close: () => connection.close(), async command(method, params, session) {
      if (method === 'Network.setBypassServiceWorker') bypassed = params?.bypass === true
      if (method === 'Page.navigate' && !bypassed) { intercepted += 1; throw new Error('service worker intercepted private HTML') }
      return connection.command(method, params, session)
    } }
  } }
  const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine: 'B', endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
    targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: factory, async admit() {} })
  await expect(host(390)).resolves.toMatchObject({ width: 390 })
  expect(bypassed).toBe(true); expect(intercepted).toBe(0)
})

test('loopback CDP does not establish a local-browser machine binding', async () => {
  for (const browserMachine of ['', 'A']) {
    const f = fixture()
    const host = createWorkspaceCaptureHost({ runtime: f.runtime, root: '/fixture/b', sourceMachine: 'B', browserMachine, endpoint: 'ws://127.0.0.1:1/devtools/browser/fixture',
      targetId: 'selected', enabled: true, allowViewNavigation: true, current: f.current, now: () => 1234, connectionFactory: f.factory, async admit() {} })
    await expect(host(390)).rejects.toThrow()
    expect(f.methods).toHaveLength(0)
  }
})
