// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react'

import { connectionBanner, inboxCounts } from '../client/hubStore.js'
import { AppLink } from './components/AppLink.js'
import { AppCallHost } from './components/CallControls.js'
import { browserVoiceCallService } from './voice/call/browser.js'
import { InboxTab } from './components/Inbox.js'
import { PairPage } from './components/PairPage.js'
import { SignedOut } from './components/SignedOut.js'
import { InstallHint } from './components/InstallHint.js'
import { trackKeyboardInset } from './components/keyboardInset.js'
import { TaskTab } from './components/TaskTab.js'
import { UsageTab } from './components/UsageTab.js'
import { XerxesTab } from './components/XerxesTab.js'
import { ServicesProvider, useHubState, useRoute, useServices, type AppServices } from './context.js'
import { routeHref, taskRoute, type Route } from './routes.js'

function routeKey(route: Route): string {
  return route.name === 'task' ? `task:${route.workspaceId ?? ''}:${route.sessionId ?? ''}` : route.name
}

/**
 * Remember each screen's scroll position and move focus to the new screen after a navigation.
 * The position is recorded while the screen scrolls, not in the cleanup: by the time a
 * cleanup runs the next screen has already rendered and window.scrollY belongs to it.
 */
function useScreenMemory(key: string, main: React.RefObject<HTMLElement | null>): void {
  const { scroll } = useServices()
  const first = useRef(true)
  useEffect(() => {
    // TaskTranscript owns bottom-following and history prepend anchors. A
    // shell-level restore would override its initial bottom scroll.
    // The XERXES chat follows its newest message the same way, so neither is restored here.
    const restores = !key.startsWith('task:') && key !== 'xerxes'
    if (restores) window.scrollTo(0, scroll.get(key) ?? 0)
    if (!first.current) main.current?.focus({ preventScroll: true })
    first.current = false
    if (!restores) return undefined
    const record = (): void => { scroll.set(key, window.scrollY) }
    window.addEventListener('scroll', record, { passive: true })
    return () => window.removeEventListener('scroll', record)
  }, [key, scroll, main])
}

/**
 * The controller's "reconnect now" (HubConnection.reconnectNow behind it), when this build has one.
 * Retry is only offered when it really reconnects; a button that does nothing is worse than none.
 */
export function reconnectOf(controller: object): (() => void) | undefined {
  if ('reconnectNow' in controller && typeof controller.reconnectNow === 'function') {
    const method = controller.reconnectNow
    return () => { method.call(controller) }
  }
  // resync() probes the link and reconnects at once when it is down (HubConnection.resume).
  if ('resync' in controller && typeof controller.resync === 'function') {
    const method = controller.resync
    return () => { void Promise.resolve(method.call(controller)).catch(() => undefined) }
  }
  return undefined
}

/** Keeps `--keyboard-inset` on the root element while the iOS keyboard covers the page. */
function useKeyboardInset(): void {
  useEffect(() => trackKeyboardInset(window, document.documentElement), [])
}

function Shell(): ReactElement {
  const route = useRoute()
  const state = useHubState()
  const services = useServices()
  const main = useRef<HTMLElement>(null)
  useScreenMemory(routeKey(route), main)
  useKeyboardInset()

  if (route.name === 'pair') {
    return services.pair === undefined
      ? <main className="pair"><h1 className="page-title">Pair this phone</h1><p>Pairing is not available here.</p></main>
      : <PairPage env={services.pair} />
  }

  const banner = connectionBanner(state.connection)
  const revoked = state.connection.state === 'revoked'
  // A dropped link offers Retry: it reconnects at once instead of waiting out the backoff.
  const retryable = state.connection.state === 'reconnecting' || state.connection.state === 'closed'
  const reconnect = reconnectOf(services.controller)
  const counts = inboxCounts(state)
  const taskTarget: Route = state.selected === undefined ? { name: 'task' } : taskRoute(state.selected.workspaceId, state.selected.sessionId)
  const tabs: readonly { readonly name: Route['name']; readonly label: string; readonly route: Route }[] = [
    { name: 'inbox', label: 'Inbox', route: { name: 'inbox' } },
    { name: 'task', label: 'Task', route: taskTarget },
    { name: 'xerxes', label: 'Xerxes', route: { name: 'xerxes' } },
    { name: 'usage', label: 'Usage', route: { name: 'usage' } },
  ]

  return (
    <div className="app">
      <header className="app-head" role="banner">
        <span className="app-name">Xerxes</span>
        {/* Status changes are announced here; streamed content never is. */}
        <div role="status" aria-live="polite" className={banner === undefined ? 'banner is-empty' : `banner banner-${banner.tone}`}>
          {banner?.text ?? ''}
        </div>
        {retryable && reconnect !== undefined ? <button type="button" className="btn banner-retry" onClick={reconnect}>Retry</button> : null}
      </header>
      <main id="main" ref={main} tabIndex={-1} className="app-main">
        <InstallHint env={services.install} />
        {revoked ? (
          <SignedOut reason={state.connection.state === 'revoked' ? state.connection.reason : ''} />
        ) : route.name === 'inbox' ? <InboxTab />
          : route.name === 'task' ? <TaskTab workspaceId={route.workspaceId} sessionId={route.sessionId} />
            : route.name === 'xerxes' ? <XerxesTab />
              : <UsageTab />}
      </main>
      <nav aria-label="Main" className="tabs">
        <ul>
          {tabs.map(tab => (
            <li key={tab.name}>
              <AppLink route={tab.route} className="tab" current={route.name === tab.name}>
                {tab.label}
                {tab.name === 'inbox' && counts.needsYou > 0 ? <span className="badge badge-attention">{counts.needsYou}<span className="visually-hidden"> need you</span></span> : null}
              </AppLink>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  )
}

export function MobileApp({ services }: { services: AppServices }): ReactElement {
  const [call] = useState(() => services.call ?? browserVoiceCallService(services.controller, services.store, services.voice))
  const appServices = useMemo(() => ({ ...services, call }), [services, call])
  useEffect(() => () => call.dispose(), [call])
  return (
    <ServicesProvider services={appServices}>
      <Shell />
      <AppCallHost />
    </ServicesProvider>
  )
}

export { routeHref }
