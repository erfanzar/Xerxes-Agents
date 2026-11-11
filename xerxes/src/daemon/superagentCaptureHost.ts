// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { BunCdpConnectionFactory, type CdpConnection, type CdpConnectionFactory } from '../operators/cdpBrowser.js'
import { validateCaptureImage, type CaptureImage } from '../hub/supervisor/abilities/captureImage.js'
import type { DaemonRuntime } from './runtime.js'
import { openWorkspaceCapturePage } from './superagentCapturePage.js'

interface CaptureHostOptions {
  readonly runtime: Pick<DaemonRuntime, 'listSessions'>
  readonly root: string
  /** Trusted owner/native binding, never inferred from loopback or provided by a model. */
  readonly sourceMachine: string
  readonly browserMachine: string
  /** Explicit owner-selected browser endpoint and dedicated existing target. No discovery or browser launch. */
  readonly endpoint: string
  readonly targetId: string
  readonly enabled: boolean
  readonly allowViewNavigation: boolean
  readonly current: () => boolean
  readonly now: () => number
  readonly connectionFactory?: CdpConnectionFactory
  /** Source-side policy admission of these validated pixels before any release. */
  readonly admit: (image: CaptureImage, view: { readonly root: string; readonly viewId: string }, signal?: AbortSignal) => Promise<void>
}

function record(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value) }
function unavailable(): never { throw new Error('Workspace capture unavailable, cancelled or authority changed') }

/** Captures only its owned inert task-summary page. It is not a screenshot of the ordinary desktop/mobile app. */
export function createWorkspaceCaptureHost(options: CaptureHostOptions) {
  const { runtime, root, sourceMachine, browserMachine, endpoint, targetId, enabled, allowViewNavigation, current, now, admit } = options
  const factory = options.connectionFactory ?? new BunCdpConnectionFactory()
  let busy = false
  let previousUrl: string | undefined
  return async (width: number, signal?: AbortSignal) => {
    if (typeof sourceMachine !== 'string' || !sourceMachine || browserMachine !== sourceMachine || enabled !== true || allowViewNavigation !== true || typeof admit !== 'function' || !targetId || ![390, 768, 1280, 1440].includes(width) || busy) unavailable()
    let address: URL
    try { address = new URL(endpoint) } catch { return unavailable() }
    if (!['ws:', 'wss:'].includes(address.protocol) || !['127.0.0.1', '[::1]'].includes(address.hostname)
      || address.username || address.password || address.search || address.hash || !address.pathname.startsWith('/devtools/browser/')) unavailable()
    busy = true
    const deadline = new AbortController()
    const timer = setTimeout(() => deadline.abort(), 10_000)
    const bounded = AbortSignal.any([deadline.signal, ...(signal === undefined ? [] : [signal])])
    let connection: CdpConnection | undefined
    let page: ReturnType<typeof openWorkspaceCapturePage> | undefined
    let sessionId: string | undefined
    let frameId: string | undefined
    let armed = false
    const loaded = new Set<string>()
    let loadChanged = (): void => {}
    const live = (): void => {
      if (bounded.aborted || !current() || (page !== undefined && !page.current())) unavailable()
    }
    let closing: Promise<void> | undefined
    const cleanup = (): Promise<void> => {
      page?.close()
      if (closing !== undefined) return closing
      if (connection === undefined) return Promise.resolve()
      const owned = connection
      closing = (async () => { try { await owned.close() } catch { unavailable() } })()
      return closing
    }
    const work = (async () => {
      live()
      connection = await factory.connect(endpoint, { signal: bounded, maxMessageBytes: 384 * 1024, onEvent(event) {
        if ((event.method === 'Target.detachedFromTarget' && (event.params.sessionId === sessionId || event.params.targetId === targetId))
          || (['Target.targetDestroyed', 'Target.targetCrashed'].includes(event.method) && event.params.targetId === targetId)) { deadline.abort(); return }
        if (event.sessionId !== sessionId || sessionId === undefined) return
        if (event.method === 'Page.lifecycleEvent' && event.params.name === 'load' && event.params.frameId === frameId && typeof event.params.loaderId === 'string') {
          if (loaded.size >= 8) { deadline.abort(); return }
          loaded.add(event.params.loaderId); loadChanged(); return
        }
        if (['Inspector.detached', 'Inspector.targetCrashed', 'Page.frameDetached', 'Page.frameAttached', 'Page.javascriptDialogOpening'].includes(event.method)
          || (armed && ['Page.frameNavigated', 'Page.frameStartedNavigating', 'Page.navigatedWithinDocument', 'DOM.documentUpdated'].includes(event.method))) deadline.abort()
      } })
      live()
      const target = await connection.command('Target.getTargetInfo', { targetId })
      live()
      if (!record(target) || !record(target.targetInfo) || target.targetInfo.targetId !== targetId || target.targetInfo.type !== 'page'
        || (target.targetInfo.url !== 'about:blank' && target.targetInfo.url !== previousUrl)) unavailable()
      live()
      const attached = await connection.command('Target.attachToTarget', { targetId, flatten: true })
      live()
      if (!record(attached) || typeof attached.sessionId !== 'string' || !attached.sessionId) unavailable()
      sessionId = attached.sessionId
      const command = async (method: string, params: Readonly<Record<string, unknown>> = {}): Promise<unknown> => {
        live(); const result = await connection?.command(method, params, sessionId); live(); return result
      }
      const frame = async (url: string): Promise<void> => {
        const tree = await command('Page.getFrameTree')
        if (!record(tree) || !record(tree.frameTree) || !record(tree.frameTree.frame) || typeof tree.frameTree.frame.id !== 'string'
          || tree.frameTree.frame.url !== url || (frameId !== undefined && tree.frameTree.frame.id !== frameId)
          || (Array.isArray(tree.frameTree.childFrames) && tree.frameTree.childFrames.length > 0)) unavailable()
        frameId = tree.frameTree.frame.id
      }
      await command('Page.enable'); await command('Page.setLifecycleEventsEnabled', { enabled: true })
      await frame(String(target.targetInfo.url))
      await command('Network.setBypassServiceWorker', { bypass: true })
      page = openWorkspaceCapturePage(runtime, root, now(), bounded)
      // Remember our attempted navigation even if later capture/admission fails.
      previousUrl = page.url
      const navigated = await command('Page.navigate', { url: page.url })
      if (!record(navigated) || navigated.frameId !== frameId || typeof navigated.loaderId !== 'string' || navigated.errorText !== undefined) unavailable()
      const loader = navigated.loaderId
      if (!loaded.has(loader)) await new Promise<void>((resolve, reject) => {
        const clean = (): void => { bounded.removeEventListener('abort', stop); loadChanged = () => {} }
        const stop = (): void => { clean(); reject(new Error('Workspace capture load cancelled')) }
        loadChanged = () => { if (loaded.has(loader)) { clean(); resolve() } }
        bounded.addEventListener('abort', stop, { once: true })
        if (bounded.aborted) stop(); else loadChanged()
      })
      live(); await frame(page.url)
      const identity = async (): Promise<void> => {
        const document = await command('DOM.getDocument', { depth: 0 })
        if (!record(document) || !record(document.root) || typeof document.root.nodeId !== 'number') unavailable()
        const marker = await command('DOM.querySelector', { nodeId: document.root.nodeId, selector: 'meta[name="xerxes-capture-view"]' })
        if (!record(marker) || typeof marker.nodeId !== 'number' || marker.nodeId <= 0) unavailable()
        const attributes = await command('DOM.getAttributes', { nodeId: marker.nodeId })
        if (!record(attributes) || !Array.isArray(attributes.attributes)) unavailable()
        const at = attributes.attributes.indexOf('content')
        if (at < 0 || at % 2 !== 0 || attributes.attributes[at + 1] !== page?.viewId) unavailable()
      }
      await identity()
      await command('Emulation.setDeviceMetricsOverride', { width, height: 900, deviceScaleFactor: 1, mobile: false })
      armed = true
      const captured = await command('Page.captureScreenshot', { format: 'png', fromSurface: true, captureBeyondViewport: false, clip: { x: 0, y: 0, width, height: 900, scale: 1 } })
      await frame(page.url); await identity()
      if (!record(captured) || typeof captured.data !== 'string' || captured.data.length > 349528) unavailable()
      const bytes = Buffer.from(captured.data, 'base64')
      const raw = { kind: 'xerxes-ui-capture/v1', view_id: page.viewId, mime: 'image/png', width, height: 900,
        captured_at: now(), sha256: new Bun.CryptoHasher('sha256').update(bytes).digest('hex'), png_base64: captured.data }
      const image = validateCaptureImage(raw)
      await admit(image, Object.freeze({ root: page.root, viewId: page.viewId }), bounded); live()
      return Object.freeze({ ...raw, sha256: image.sha256, png_base64: image.imageUrl.slice('data:image/png;base64,'.length) })
    })()
    const operation = work.finally(cleanup)
    let cancel = (): void => {}
    const cancelled = new Promise<never>((_resolve, reject) => { cancel = () => reject(new Error('Workspace capture unavailable, cancelled or authority changed')); bounded.addEventListener('abort', cancel, { once: true }); if (bounded.aborted) cancel() })
    try { return await Promise.race([operation, cancelled]) }
    catch { unavailable() }
    finally {
      clearTimeout(timer); bounded.removeEventListener('abort', cancel)
      // Operation settlement also cleans up a connection delivered after caller cancellation.
      void cleanup().catch(() => {})
      // A noncooperative host retains the single-target reservation until its operation settles.
      void operation.then(() => { busy = false }, () => { busy = false })
    }
  }
}
