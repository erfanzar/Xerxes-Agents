// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { MouseEvent, ReactElement, ReactNode } from 'react'

import { useServices } from '../context.js'
import { routeHref, type Route } from '../routes.js'

/** A real anchor (deep links open in a new tab) that navigates in place on a plain tap. */
export function AppLink({ route, className, current, children }: { route: Route; className?: string; current?: boolean; children: ReactNode }): ReactElement {
  const { router } = useServices()
  const onClick = (event: MouseEvent<HTMLAnchorElement>): void => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    router.navigate(route)
  }
  return (
    <a href={routeHref(route)} className={className} onClick={onClick} {...(current === true ? { 'aria-current': 'page' as const } : {})}>
      {children}
    </a>
  )
}
