// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * History-based routes. Deep links name opaque ids: `/w/<workspace>/s/<session>`.
 * Unknown paths fall back to the Inbox (the Hub serves the shell for them).
 */

export type Route =
  | { readonly name: 'inbox' }
  | { readonly name: 'task'; readonly workspaceId?: string; readonly sessionId?: string }
  | { readonly name: 'xerxes' }
  | { readonly name: 'usage' }
  | { readonly name: 'pair' }

export const TAB_ROUTES = ['inbox', 'task', 'xerxes', 'usage'] as const
export type TabName = (typeof TAB_ROUTES)[number]

export function parseRoute(pathname: string): Route {
  const parts = pathname.split('/').filter(part => part.length > 0)
  const head = parts[0]
  if (head === undefined) return { name: 'inbox' }
  if (parts.length === 1) {
    if (head === 'task') return { name: 'task' }
    if (head === 'xerxes') return { name: 'xerxes' }
    if (head === 'usage') return { name: 'usage' }
    if (head === 'pair') return { name: 'pair' }
    return { name: 'inbox' }
  }
  if (head === 'w' && parts.length === 4 && parts[2] === 's') {
    try {
      const workspaceId = decodeURIComponent(parts[1] ?? '')
      const sessionId = decodeURIComponent(parts[3] ?? '')
      if (workspaceId && sessionId) return { name: 'task', workspaceId, sessionId }
    } catch {
      return { name: 'inbox' }
    }
  }
  return { name: 'inbox' }
}

export function routeHref(route: Route): string {
  switch (route.name) {
    case 'inbox': return '/'
    case 'xerxes': return '/xerxes'
    case 'usage': return '/usage'
    case 'pair': return '/pair'
    case 'task':
      return route.workspaceId !== undefined && route.sessionId !== undefined
        ? `/w/${encodeURIComponent(route.workspaceId)}/s/${encodeURIComponent(route.sessionId)}`
        : '/task'
  }
}

export function taskRoute(workspaceId: string, sessionId: string): Route {
  return { name: 'task', workspaceId, sessionId }
}

export interface Router {
  path(): string
  subscribe(listener: () => void): () => void
  navigate(route: Route, options?: { readonly replace?: boolean }): void
}

export function createMemoryRouter(initial = '/'): Router {
  let current = initial
  const listeners = new Set<() => void>()
  return {
    path: () => current,
    subscribe(listener) {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    },
    navigate(route) {
      const next = routeHref(route)
      if (next === current) return
      current = next
      for (const listener of [...listeners]) listener()
    },
  }
}

export function createBrowserRouter(win: Window): Router {
  const listeners = new Set<() => void>()
  const notify = (): void => { for (const listener of [...listeners]) listener() }
  let attached = false
  const onPop = (): void => notify()
  return {
    path: () => win.location.pathname,
    subscribe(listener) {
      listeners.add(listener)
      if (!attached) {
        attached = true
        win.addEventListener('popstate', onPop)
      }
      return () => {
        listeners.delete(listener)
        if (listeners.size === 0 && attached) {
          attached = false
          win.removeEventListener('popstate', onPop)
        }
      }
    },
    navigate(route, options = {}) {
      const next = routeHref(route)
      if (next === win.location.pathname) return
      if (options.replace === true) win.history.replaceState(null, '', next)
      else win.history.pushState(null, '', next)
      notify()
    },
  }
}
