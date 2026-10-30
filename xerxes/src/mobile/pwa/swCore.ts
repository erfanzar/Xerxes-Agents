// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Service worker logic, written against a minimal scope interface so it can be
 * driven by a fake in tests. Policy:
 * - only versioned shell assets (hashed bundle files and index.html) are cached;
 * - nothing under /api/ is ever cached or even intercepted;
 * - a push shows a notification with minimal text and carries no actions, so a
 *   decision can never be made from a notification;
 * - clicking a notification opens the exact task URL (validated, same-origin).
 */

export interface ExtendableLike {
  waitUntil(promise: Promise<unknown>): void
}

export interface FetchEventLike {
  readonly request: Request
  respondWith(response: Promise<Response> | Response): void
}

export interface PushEventLike extends ExtendableLike {
  readonly data: { json(): unknown; text(): string } | null
}

export interface NotificationClickLike extends ExtendableLike {
  readonly notification: { close(): void; readonly data: unknown }
}

/** The browser rotated or dropped this worker's push subscription (endpoint expiry, key change). */
export interface PushSubscriptionChangeLike extends ExtendableLike {
  readonly oldSubscription?: { readonly options?: { readonly applicationServerKey?: BufferSource | null } } | null
  readonly newSubscription?: unknown
}

interface SwEventMap {
  install: ExtendableLike
  activate: ExtendableLike
  fetch: FetchEventLike
  push: PushEventLike
  notificationclick: NotificationClickLike
  pushsubscriptionchange: PushSubscriptionChangeLike
}

/**
 * Posted to every open window after `pushsubscriptionchange`. The worker cannot
 * register a subscription with the Hub itself: `push/subscribe` needs the
 * `x-xerxes-csrf` token, which only the page holds (getting one here would mean
 * rotating the page's refresh token from the worker). The page answers by
 * calling `push.resync()`; with no window open, the next boot's resync does it.
 */
export const PUSH_CHANGED_MESSAGE = 'xerxes.push.subscription_changed'

export interface CacheLike {
  addAll(urls: readonly string[]): Promise<void>
  match(request: Request | string): Promise<Response | undefined>
  put(request: Request | string, response: Response): Promise<void>
}

export interface CacheStorageLike {
  open(name: string): Promise<CacheLike>
  keys(): Promise<string[]>
  delete(name: string): Promise<boolean>
}

export interface SwClient {
  readonly url: string
  focus(): Promise<unknown>
  navigate?(url: string): Promise<unknown>
  postMessage?(message: unknown): void
}

export interface SwPushManager {
  getSubscription(): Promise<unknown>
  subscribe(options: { readonly userVisibleOnly: boolean; readonly applicationServerKey: BufferSource }): Promise<unknown>
}

export interface SwScope {
  readonly location: { readonly origin: string }
  readonly registration: {
    showNotification(title: string, options: NotificationOptions): Promise<void>
    readonly pushManager?: SwPushManager | undefined
  }
  readonly clients: {
    claim(): Promise<void>
    matchAll(options: { readonly type: 'window'; readonly includeUncontrolled: boolean }): Promise<readonly SwClient[]>
    openWindow(url: string): Promise<unknown>
  }
  readonly caches: CacheStorageLike
  /** The worker's own fetch; tests inject a fake. Defaults to the global fetch. */
  readonly fetch?: ((request: Request, init?: RequestInit) => Promise<Response>) | undefined
  skipWaiting(): Promise<void>
  addEventListener<K extends keyof SwEventMap>(type: K, listener: (event: SwEventMap[K]) => void): void
}

export interface SwConfig {
  readonly version: string
  /** Absolute paths of the versioned shell: `/index.html` and hashed assets. */
  readonly shell: readonly string[]
  /**
   * How long a navigation waits for the network before the cached shell answers.
   * Default 3.5 s: a slow launch shows the app, never a blank white screen.
   */
  readonly navigationTimeoutMs?: number
}

const DEFAULT_NAVIGATION_TIMEOUT_MS = 3_500

export const CACHE_PREFIX = 'xerxes-shell-'

export function cacheName(version: string): string {
  return `${CACHE_PREFIX}${version}`
}

export type FetchDecision = 'shell' | 'navigation' | 'network'

export function fetchDecision(input: {
  readonly method: string
  readonly url: string
  readonly origin: string
  readonly mode: string
  readonly shell: ReadonlySet<string>
}): FetchDecision {
  if (input.method !== 'GET') return 'network'
  let url: URL
  try { url = new URL(input.url) } catch { return 'network' }
  if (url.origin !== input.origin) return 'network'
  // The API, the socket and every identity route are never touched by the worker.
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/voice-models/')) return 'network'
  if (input.shell.has(url.pathname)) return 'shell'
  if (input.mode === 'navigate') return 'navigation'
  return 'network'
}

const TASK_PATH = /^\/w\/[^/]+\/s\/[^/]+$/

/** A notification target must be the Inbox, the XERXES tab or one exact task; anything else opens the Inbox. */
export function safeTaskPath(value: unknown, origin: string): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > 600) return '/'
  let url: URL
  try { url = new URL(value, origin) } catch { return '/' }
  if (url.origin !== origin) return '/'
  if (url.pathname === '/xerxes') return '/xerxes'
  return TASK_PATH.test(url.pathname) ? url.pathname : '/'
}

export interface PushDisplay {
  readonly title: string
  readonly options: NotificationOptions
}

const clip = (text: string, max: number): string => (text.length > max ? `${text.slice(0, max - 1)}…` : text)

/**
 * Lock-screen text is minimal by design: a short title and body, never a
 * transcript or tool arguments. Payloads that do not parse still show a
 * generic notice rather than nothing (a silent push is not allowed to stay silent).
 */
export function pushDisplay(data: unknown, origin: string): PushDisplay {
  const record = typeof data === 'object' && data !== null && !Array.isArray(data) ? (data as Record<string, unknown>) : {}
  const title = typeof record.title === 'string' && record.title.trim() ? clip(record.title.trim(), 60) : 'Xerxes'
  const body = typeof record.body === 'string' && record.body.trim() ? clip(record.body.trim(), 120) : 'Open Xerxes to see what needs you.'
  const tag = typeof record.tag === 'string' && record.tag.length <= 100 ? record.tag : undefined
  return {
    title,
    options: {
      body,
      ...(tag === undefined ? {} : { tag }),
      data: { url: safeTaskPath(record.url, origin) },
    },
  }
}

export function installServiceWorker(scope: SwScope, config: SwConfig): void {
  const name = cacheName(config.version)
  const shell: ReadonlySet<string> = new Set(config.shell)
  const origin = scope.location.origin
  const fetchFn = (request: Request, init?: RequestInit): Promise<Response> => (scope.fetch ? scope.fetch(request, init) : fetch(request, init))
  const navigationTimeoutMs = config.navigationTimeoutMs ?? DEFAULT_NAVIGATION_TIMEOUT_MS

  scope.addEventListener('install', event => {
    event.waitUntil(scope.caches.open(name).then(cache => cache.addAll(config.shell)).then(() => scope.skipWaiting()))
  })

  scope.addEventListener('activate', event => {
    event.waitUntil((async () => {
      for (const key of await scope.caches.keys()) {
        if (key.startsWith(CACHE_PREFIX) && key !== name) await scope.caches.delete(key)
      }
      await scope.clients.claim()
    })())
  })

  scope.addEventListener('fetch', event => {
    const decision = fetchDecision({ method: event.request.method, url: event.request.url, origin, mode: event.request.mode, shell })
    if (decision === 'network') return
    if (decision === 'shell') {
      event.respondWith((async () => {
        const cache = await scope.caches.open(name)
        const hit = await cache.match(event.request)
        if (hit) return hit
        const response = await fetchFn(event.request)
        if (response.ok && response.type === 'basic') await cache.put(event.request, response.clone())
        return response
      })())
      return
    }
    // Navigation: the network wins when it answers in time. When it fails, or is slower than
    // the deadline, the cached shell answers (the app then connects on its own). With no cached
    // shell the network is simply awaited: a late page beats an error page.
    event.respondWith((async () => {
      const network = fetchFn(event.request)
      // The race below may leave this promise unobserved; its failure is handled there.
      network.catch(() => undefined)
      const cachedShell = async (): Promise<Response | undefined> => (await scope.caches.open(name)).match('/index.html')
      let timer: ReturnType<typeof setTimeout> | undefined
      const timedOut = new Promise<'timeout'>(resolve => { timer = setTimeout(() => resolve('timeout'), navigationTimeoutMs) })
      try {
        const first = await Promise.race([network, timedOut])
        if (first !== 'timeout') return first
        const fallback = await cachedShell()
        return fallback ?? await network
      } catch (error) {
        const fallback = await cachedShell()
        if (fallback) return fallback
        throw error
      } finally {
        if (timer !== undefined) clearTimeout(timer)
      }
    })())
  })

  scope.addEventListener('push', event => {
    let payload: unknown
    try { payload = event.data ? event.data.json() : undefined } catch { payload = undefined }
    const display = pushDisplay(payload, origin)
    event.waitUntil(scope.registration.showNotification(display.title, display.options))
  })

  scope.addEventListener('pushsubscriptionchange', event => {
    event.waitUntil((async () => {
      const manager = scope.registration.pushManager
      const key = event.oldSubscription?.options?.applicationServerKey
      // Re-subscribe in the browser with the same VAPID key when it gave no replacement.
      // Subscribing is browser-local; telling the Hub needs the page (see PUSH_CHANGED_MESSAGE).
      if (manager !== undefined && (event.newSubscription === undefined || event.newSubscription === null) && key) {
        try {
          if (await manager.getSubscription() === null) await manager.subscribe({ userVisibleOnly: true, applicationServerKey: key })
        } catch {
          // Refused (permission gone): the app's notification setting offers to turn it on again.
        }
      }
      const windows = await scope.clients.matchAll({ type: 'window', includeUncontrolled: true })
      for (const client of windows) {
        if (client.url.startsWith(origin)) client.postMessage?.({ type: PUSH_CHANGED_MESSAGE })
      }
    })())
  })

  scope.addEventListener('notificationclick', event => {
    event.notification.close()
    const data = event.notification.data
    const target = safeTaskPath(typeof data === 'object' && data !== null ? (data as Record<string, unknown>).url : undefined, origin)
    event.waitUntil((async () => {
      const windows = await scope.clients.matchAll({ type: 'window', includeUncontrolled: true })
      const mine = windows.find(client => client.url.startsWith(origin))
      if (mine) {
        await mine.focus()
        if (mine.navigate) {
          try {
            await mine.navigate(target)
            return
          } catch {
            // A client that refuses navigation gets a fresh window instead.
          }
        }
      }
      await scope.clients.openWindow(target)
    })())
  })
}
