// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react'

import { connectionBanner, inboxCounts, workspaceAvailable } from '../client/hubStore.js'
import { AppLink } from './components/AppLink.js'
import { AppCallHost } from './components/CallControls.js'
import { HubStatusSheet } from './components/HubStatusSheet.js'
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

function TabIcon({ name }: { readonly name: Route['name'] }): ReactElement {
  switch (name) {
    case 'inbox':
      return (
        <svg className="tab-icon" viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M3 11.5L5.2 4.5h9.6L17 11.5v4a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 3 15.5v-4Z" />
          <path d="M3 11.5h4.2l1.2 2h3.2l1.2-2H17" />
        </svg>
      )
    case 'task':
      return (
        <svg className="tab-icon" viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <rect x="3" y="3.5" width="14" height="13" rx="2.5" />
          <path d="M6.5 8.2l2 1.8-2 1.8M10.5 12h3" />
        </svg>
      )
    case 'xerxes':
      return (
        <svg className="tab-icon" viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M10 2.5c.6 4.1 3.4 6.9 7.5 7.5-4.1.6-6.9 3.4-7.5 7.5-.6-4.1-3.4-6.9-7.5-7.5 4.1-.6 6.9-3.4 7.5-7.5Z" />
        </svg>
      )
    default:
      return (
        <svg className="tab-icon" viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M4.5 15.5V10.5M8.2 15.5V6M11.8 15.5V8.8M15.5 15.5V4.5" />
        </svg>
      )
  }
}

function Shell(): ReactElement {
  const route = useRoute()
  const state = useHubState()
  const services = useServices()
  const main = useRef<HTMLElement>(null)
  const [hubSheet, setHubSheet] = useState(false)
  useScreenMemory(routeKey(route), main)
  useKeyboardInset()

  if (route.name === 'pair') {
    return services.pair === undefined
      ? <main className="pair"><h1 className="page-title">Pair this phone</h1><p>Pairing is not available here.</p></main>
      : <PairPage env={services.pair} />
  }

  const banner = connectionBanner(state.connection)
  const revoked = state.connection.state === 'revoked'
  const live = state.connection.state === 'live'
  const retryable = state.connection.state === 'reconnecting' || state.connection.state === 'closed'
  const reconnect = reconnectOf(services.controller)
  const counts = inboxCounts(state)
  const onlineWorkspaces = state.workspaceOrder.filter(id => workspaceAvailable(state.workspaces[id])).length
  const totalWorkspaces = state.workspaceOrder.length
  const taskTarget: Route = state.selected === undefined ? { name: 'task' } : taskRoute(state.selected.workspaceId, state.selected.sessionId)
  const tabs: readonly { readonly name: Route['name']; readonly label: string; readonly route: Route }[] = [
    { name: 'inbox', label: 'Inbox', route: { name: 'inbox' } },
    { name: 'task', label: 'Task', route: taskTarget },
    { name: 'xerxes', label: 'Xerxes', route: { name: 'xerxes' } },
    { name: 'usage', label: 'Usage', route: { name: 'usage' } },
  ]

  return (
    <div className="app">
      <a className="skip-link" href="#main">Skip to main content</a>
      <header className="app-head" role="banner">
        <span className="app-brand">
          <span className="app-logo" aria-hidden="true" />
          <span className="app-name">Xerxes</span>
        </span>
        {/* Status changes are announced here; streamed content never is. */}
        <div role="status" aria-live="polite" className={banner === undefined ? 'banner is-empty' : `banner banner-${banner.tone}`}>
          {banner?.text ?? ''}
        </div>
        {retryable && reconnect !== undefined ? (
          <button type="button" className="btn banner-retry" onClick={reconnect}>Retry</button>
        ) : !revoked ? (
          <button
            type="button"
            className="btn head-hub-btn"
            aria-haspopup="dialog"
            aria-expanded={hubSheet}
            aria-label="Hub and paired devices"
            onClick={() => setHubSheet(true)}
          >
            <span className={live ? 'head-dot is-active' : 'head-dot'} aria-hidden="true" />
            <span>{totalWorkspaces > 0 ? `${onlineWorkspaces}/${totalWorkspaces} online` : 'Hub'}</span>
          </button>
        ) : null}
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
      {hubSheet && !revoked ? <HubStatusSheet onClose={() => setHubSheet(false)} /> : null}
      <nav aria-label="Main" className="tabs">
        <ul>
          {tabs.map(tab => (
            <li key={tab.name}>
              <AppLink route={tab.route} className="tab" current={route.name === tab.name}>
                <TabIcon name={tab.name} />
                <span className="tab-label">{tab.label}</span>
                {tab.name === 'inbox' && counts.needsYou > 0 ? <span className="badge badge-attention">{counts.needsYou}<span className="visually-hidden"> need you</span></span> : null}
                {tab.name === 'xerxes' && state.supervisor.working === true ? <span className="tab-dot" aria-hidden="true" /> : null}
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
