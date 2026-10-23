// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createRoot } from 'react-dom/client'

import { createDraftStore } from '../client/drafts.js'
import { createHubController } from '../client/hubController.js'
import { HubConnection, hubWebSocketUrl } from '../client/hubConnection.js'
import { createHubStore } from '../client/hubStore.js'
import { createHubHttp } from './api.js'
import { createScrollMemory, type AppServices } from './context.js'
import { MobileApp } from './MobileApp.js'
import { deviceLabelFromUserAgent, takePairingFragment } from './pairing.js'
import { createPushService } from './pwa/push.js'
import { createBrowserRouter, parseRoute } from './routes.js'
import { createVoiceControl } from './voice/index.js'

/**
 * Entry point. Order matters on the pair route: the URL fragment is read and
 * removed from the address bar before anything else runs.
 */
function boot(): void {
  const container = document.getElementById('root')
  if (container === null) throw new Error('index.html has no #root element')

  const onPairRoute = parseRoute(window.location.pathname).name === 'pair'
  const read = onPairRoute ? takePairingFragment(window) : ({ kind: 'missing' } as const)

  const http = createHubHttp((input, init) => fetch(input, init))
  const store = createHubStore(undefined, { coalesceMs: 50 })
  const connection = new HubConnection({
    url: hubWebSocketUrl(window.location),
    ensureSession: async () => {
      const outcome = await http.refresh()
      // 'pending' (202) means the owner has not confirmed this device: not signed in yet, not revoked.
      return outcome.status === 'ok' ? 'ok' : outcome.status === 'denied' ? 'denied' : 'unavailable'
    },
  })
  // The Hub router forwards { media_type, data } images on submit and steer; the daemon validates them.
  const controller = createHubController({ connection, store, imagesSupported: true })
  const router = createBrowserRouter(window)

  const hasPush = 'serviceWorker' in navigator && 'PushManager' in window
  const services: AppServices = {
    store,
    controller,
    router,
    drafts: createDraftStore(),
    scroll: createScrollMemory(),
    now: () => Date.now(),
    push: createPushService({
      http,
      ready: hasPush ? () => navigator.serviceWorker.ready : undefined,
      permission: () => (typeof Notification === 'undefined' ? 'unsupported' : Notification.permission),
      requestPermission: () => Notification.requestPermission(),
    }),
    pair: {
      read,
      http,
      deviceLabel: deviceLabelFromUserAgent(navigator.userAgent),
      onPaired: () => {
        router.navigate({ name: 'inbox' }, { replace: true })
        controller.start()
      },
    },
  }

  const root = createRoot(container)
  if (!onPairRoute) controller.start()
  root.render(<MobileApp services={services} />)

  // The talk button appears only once the Hub reports voice capability. The probe waits for a live
  // connection (an unauthenticated probe would be refused) and runs once; it never opens the microphone.
  if (!onPairRoute) {
    let probed = false
    const stopWatching = store.subscribe(() => {
      if (probed || store.getSnapshot().connection.state !== 'live') return
      probed = true
      stopWatching()
      void createVoiceControl({ location: window.location, conversationId: () => store.getSnapshot().supervisor.threadId })
        .then(voice => { if (voice !== undefined) root.render(<MobileApp services={{ ...services, voice }} />) })
        .catch(() => undefined)
    })
  }

  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      void navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(() => {
        // The app works without the worker; it only adds offline shell and push.
      })
    })
  }
}

boot()
