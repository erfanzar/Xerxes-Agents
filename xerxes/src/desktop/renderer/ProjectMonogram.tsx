// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Real project logo / favicon badge.
 *
 * Resolves the workspace's real logo/favicon on disk (`favicon.svg`, `assets/logo*`,
 * `public/favicon.*`, `<link rel="icon">` in `index.html`, etc.) via Electron IPC.
 * When a project has no logo file, renders `null` (pure clean text — no fake
 * 2-letter placeholder tiles).
 */

import { useEffect, useState, type ReactElement } from 'react'
import type { ProjectMonogramColor } from './threadInbox.js'

const resolvedCache = new Map<string, string | null>()
const inflight = new Map<string, Promise<string | null>>()

function lookupProjectIcon(cwd: string): Promise<string | null> {
  const existing = inflight.get(cwd)
  if (existing) return existing
  const promise = (async () => {
    try {
      const fn = typeof window !== 'undefined' ? window.xerxes?.projectIcon : undefined
      if (typeof fn !== 'function') return null
      const url = await fn(cwd)
      const value = typeof url === 'string' && url.startsWith('data:image/') ? url : null
      resolvedCache.set(cwd, value)
      return value
    } catch {
      resolvedCache.set(cwd, null)
      return null
    } finally {
      inflight.delete(cwd)
    }
  })()
  inflight.set(cwd, promise)
  return promise
}

/** Seed or clear the renderer project logo cache (used by deterministic tests). */
export function setProjectIconCacheForTests(entries?: Readonly<Record<string, string | null>>): void {
  resolvedCache.clear()
  inflight.clear()
  if (!entries) return
  for (const [cwd, url] of Object.entries(entries)) {
    resolvedCache.set(cwd, url)
  }
}

export function ProjectMonogram({
  name,
  seed,
  size = 16,
  className,
}: {
  readonly name: string
  readonly seed?: string
  readonly size?: number
  readonly color?: ProjectMonogramColor
  readonly className?: string
}): ReactElement | null {
  const cwd = seed && seed.startsWith('/') ? seed : ''
  const [url, setUrl] = useState<string | null>(() => (cwd ? (resolvedCache.get(cwd) ?? null) : null))

  useEffect(() => {
    if (!cwd) {
      setUrl(null)
      return
    }
    if (resolvedCache.has(cwd)) {
      setUrl(resolvedCache.get(cwd) ?? null)
      return
    }
    let cancelled = false
    void lookupProjectIcon(cwd).then(found => {
      if (!cancelled) setUrl(found)
    })
    return () => {
      cancelled = true
    }
  }, [cwd])

  if (!url) return null

  return (
    <img
      src={url}
      alt=""
      aria-hidden="true"
      width={size}
      height={size}
      draggable={false}
      data-slot="project-favicon"
      data-project={name}
      className={className ? `project-favicon ${className}` : 'project-favicon'}
      onError={() => {
        if (cwd) resolvedCache.set(cwd, null)
        setUrl(null)
      }}
    />
  )
}
