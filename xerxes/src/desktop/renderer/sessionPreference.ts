// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

const key = (workspace: string) => `xerxes.desktop.session.v1:${workspace}`
export function selectedSession(workspace: string): string | null {
  try {
    const id = localStorage.getItem(key(workspace))
    return id && /^[a-zA-Z0-9_-]{1,128}$/.test(id) ? id : null
  } catch { return null }
}
export function rememberSession(workspace: string, id: string): void {
  if (!workspace || !/^[a-zA-Z0-9_-]{1,128}$/.test(id)) return
  try { localStorage.setItem(key(workspace), id) } catch { /* Selection remains usable without browser storage. */ }
}
