// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** "just now", "5m ago", "2h ago", "3d ago", then an ISO date. Browser-safe and pure. */
export function relativeTime(iso: string, now: number): string {
  const then = Date.parse(iso)
  if (!Number.isFinite(then)) return 'unknown'
  const seconds = Math.max(0, Math.round((now - then) / 1000))
  if (seconds < 45) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.round(hours / 24)
  if (days < 7) return `${days}d ago`
  return new Date(then).toISOString().slice(0, 10)
}

export function plural(count: number, one: string, many: string = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`
}
