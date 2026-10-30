// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createRoot } from 'react-dom/client'

import { createDraftStore, type DraftStore } from '../client/drafts.js'
import { createHubController } from '../client/hubController.js'
import { HubConnection, hubWebSocketUrl } from '../client/hubConnection.js'
import { createHubStore } from '../client/hubStore.js'
import { createHubHttp } from './api.js'
import { createScrollMemory, type AppServices } from './context.js'
import { MobileApp } from './MobileApp.js'
import { browserPairStorage, deviceLabelFromUserAgent, readPairRoute } from './pairing.js'
import { createPushService } from './pwa/push.js'
import { createResumeGate } from './pwa/resumeGate.js'
import { PUSH_CHANGED_MESSAGE } from './pwa/swCore.js'
import { createBrowserRouter, parseRoute } from './routes.js'
import { createVoiceControl } from './voice/index.js'

/** While visible and live, a link that carried no frame for this long is probed. */
const HEARTBEAT_MS = 25_000
/** How often a visible app asks the service worker registration for a new version. */
const UPDATE_CHECK_MS = 30 * 60_000

/** Drafts with a live count, so an app update never reloads over unsent text. */
function countingDrafts(inner: DraftStore): DraftStore & { pending(): number } {
  const filled = new Set<string>()
  return {
    get: key => inner.get(key),
    set(key, text) {
      inner.set(key, text)
      if (text === '') filled.delete(key)
      else filled.add(key)
    },
    clear(key) {
      inner.clear(key)
      filled.delete(key)
    },
    pending: () => filled.size,
  }
}

const isVisible = (): boolean => document.visibilityState !== 'hidden'

/**
 * Entry point. Order matters on the pair route: the URL fragment is read and
 * removed from the address bar before anything else runs.
 */
function boot(): void {
  const container = document.getElementById('root')
  if (container === null) throw new Error('index.html has no #root element')

  const onPairRoute = parseRoute(window.location.pathname).name === 'pair'
  const read = onPairRoute ? readPairRoute(window, browserPairStorage(), Date.now()) : ({ kind: 'missing' } as const)

  const http = createHubHttp((input, init) => fetch(input, init))
  const store = createHubStore(undefined, { coalesceMs: 50 })
  const connection = new HubConnection({
    url: hubWebSocketUrl(window.location),
    heartbeatMs: HEARTBEAT_MS,
    ensureSession: async () => {
      const outcome = await http.refresh()
      // 'pending' (202) means the owner has not confirmed this device: not signed in yet, not revoked.
      return outcome.status === 'ok' ? 'ok' : outcome.status === 'denied' ? 'denied' : 'unavailable'
    },
  })
  connection.setVisible(isVisible())
  // The Hub router forwards { media_type, data } images on submit and steer; the daemon validates them.
  const controller = createHubController({ connection, store, imagesSupported: true })
  const router = createBrowserRouter(window)
  const drafts = countingDrafts(createDraftStore())

  const hasPush = 'serviceWorker' in navigator && 'PushManager' in window
  const push = createPushService({
    http,
    ready: hasPush ? () => navigator.serviceWorker.ready : undefined,
    permission: () => (typeof Notification === 'undefined' ? 'unsupported' : Notification.permission),
    requestPermission: () => Notification.requestPermission(),
  })

  const root = createRoot(container)
  let services: AppServices
  const render = (): void => { root.render(<MobileApp services={services} />) }

  // The talk button appears only once the Hub reports voice capability. The probe waits for a live
  // connection (an unauthenticated probe would be refused) and runs once; it never opens the microphone.
  let voiceProbeStarted = false
  function startVoiceProbe(): void {
    if (voiceProbeStarted) return
    voiceProbeStarted = true
    let probed = false
    const run = (): void => {
      if (probed || store.getSnapshot().connection.state !== 'live') return
      probed = true
      stopWatching()
      void createVoiceControl({ location: window.location, conversationId: () => store.getSnapshot().supervisor.threadId })
        .then(voice => {
          if (voice === undefined) return
          services = { ...services, voice }
          render()
        })
        .catch(() => undefined)
    }
    const stopWatching = store.subscribe(run)
    run()
  }

  // A re-pair or a device revoke deletes the Hub's push subscription while the browser keeps its
  // own; re-register it (idempotent, never prompts) once the connection is live.
  let pushSynced = false
  function syncPushWhenLive(): void {
    const run = (): void => {
      if (pushSynced || store.getSnapshot().connection.state !== 'live') return
      pushSynced = true
      stopWatching()
      void push.resync().catch(() => undefined)
    }
    const stopWatching = store.subscribe(run)
    run()
  }

  services = {
    store,
    controller,
    router,
    drafts,
    scroll: createScrollMemory(),
    now: () => Date.now(),
    push,
    pair: {
      read,
      http,
      deviceLabel: deviceLabelFromUserAgent(navigator.userAgent),
      onPaired: () => {
        router.navigate({ name: 'inbox' }, { replace: true })
        controller.start()
        startVoiceProbe()
        pushSynced = false
        syncPushWhenLive()
      },
    },
  }

  if (!onPairRoute) controller.start()
  render()
  if (!onPairRoute) {
    startVoiceProbe()
    syncPushWhenLive()
  }

  // The 5-minute browser session is renewed at ~70% of its life while the page is visible, so
  // push, voice and HTTP calls never meet an expired session while the socket is up.
  http.keepAlive(isVisible)

  // ── service worker and app updates ──────────────────────────────────
  let registration: ServiceWorkerRegistration | undefined
  let reloadWanted = false
  const idle = (): boolean => {
    if (drafts.pending() > 0) return false
    if (document.querySelector('.voice-screen') !== null) return false
    const active = document.activeElement
    return !(active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement || (active instanceof HTMLElement && active.isContentEditable))
  }
  const reloadIfIdle = (): void => {
    if (reloadWanted && idle()) window.location.reload()
  }
  const checkForUpdate = (): void => {
    if (registration !== undefined) void registration.update().catch(() => undefined)
  }
  if ('serviceWorker' in navigator) {
    // Only a replacement counts: the first install claiming this page is not a new version.
    let hadController = navigator.serviceWorker.controller !== null
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      if (!hadController) { hadController = true; return }
      // A new bundle took over. Reload now when nothing would be lost, otherwise on the next resume.
      reloadWanted = true
      reloadIfIdle()
    })
    window.addEventListener('load', () => {
      void navigator.serviceWorker.register('/sw.js', { scope: '/' })
        .then(found => { registration = found })
        .catch(() => {
          // The app works without the worker; it only adds offline shell and push.
        })
    })
    window.setInterval(() => { if (isVisible()) checkForUpdate() }, UPDATE_CHECK_MS)
  }

  // The service worker saw the browser rotate the push subscription; only the page holds the CSRF
  // token `push/subscribe` needs, so it re-registers here (a closed app does it on its next boot).
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('message', event => {
      const data: unknown = event.data
      if (typeof data === 'object' && data !== null && (data as { type?: unknown }).type === PUSH_CHANGED_MESSAGE) {
        void push.resync().catch(() => undefined)
      }
    })
  }

  // ── resume: foreground, restored page, network back ─────────────────
  // One iOS foreground fires visibilitychange and pageshow together; window focus is left out
  // (on a desktop browser every window switch fired a full resync).
  const resume = createResumeGate(Date.now, (): void => {
    connection.resume()
    void http.ensureFresh().catch(() => undefined)
    checkForUpdate()
    reloadIfIdle()
  })
  const onForeground = (): void => { if (isVisible()) resume.trigger() }
  document.addEventListener('visibilitychange', () => {
    connection.setVisible(isVisible())
    onForeground()
  })
  window.addEventListener('pageshow', onForeground)
  window.addEventListener('online', () => { if (isVisible()) resume.force() })
}

boot()
