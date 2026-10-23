// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, type ReactElement } from 'react'

import { connectionBanner, inboxCounts } from '../client/hubStore.js'
import { AppLink } from './components/AppLink.js'
import { InboxTab } from './components/Inbox.js'
import { PairPage } from './components/PairPage.js'
import { TaskTab } from './components/TaskTab.js'
import { UsageTab } from './components/UsageTab.js'
import { XerxesTab } from './components/XerxesTab.js'
import { ServicesProvider, useHubState, useRoute, useServices, type AppServices } from './context.js'
import { routeHref, taskRoute, type Route } from './routes.js'

function routeKey(route: Route): string {
  return route.name === 'task' ? `task:${route.workspaceId ?? ''}:${route.sessionId ?? ''}` : route.name
}

/** Remember each screen's scroll position and move focus to the new screen after a navigation. */
function useScreenMemory(key: string, main: React.RefObject<HTMLElement | null>): void {
  const { scroll } = useServices()
  const first = useRef(true)
  useEffect(() => {
    window.scrollTo(0, scroll.get(key) ?? 0)
    if (!first.current) main.current?.focus({ preventScroll: true })
    first.current = false
    return () => { scroll.set(key, window.scrollY) }
  }, [key, scroll, main])
}

function Shell(): ReactElement {
  const route = useRoute()
  const state = useHubState()
  const services = useServices()
  const main = useRef<HTMLElement>(null)
  useScreenMemory(routeKey(route), main)

  if (route.name === 'pair') {
    return services.pair === undefined
      ? <main className="pair"><h1 className="page-title">Pair this phone</h1><p>Pairing is not available here.</p></main>
      : <PairPage env={services.pair} />
  }

  const banner = connectionBanner(state.connection)
  const revoked = state.connection.state === 'revoked'
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
      </header>
      <main id="main" ref={main} tabIndex={-1} className="app-main">
        {revoked ? (
          <section aria-labelledby="revoked-title">
            <h1 id="revoked-title" className="page-title">Signed out</h1>
            <p>{state.connection.state === 'revoked' ? state.connection.reason : ''}</p>
            <p>On the desktop, create a new pairing code and scan it with this phone. This page will not reconnect on its own.</p>
          </section>
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
  return (
    <ServicesProvider services={services}>
      <Shell />
    </ServicesProvider>
  )
}

export { routeHref }
