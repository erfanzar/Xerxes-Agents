// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { DaemonRuntime } from './runtime.js'
import { workspaceCaptureView } from './superagentCaptureView.js'

/** An owned, private, short-lived rendering endpoint; never mounted on the Hub or a public asset route. */
export function openWorkspaceCapturePage(runtime: Pick<DaemonRuntime, 'listSessions'>, root: string, capturedAt: number, signal?: AbortSignal) {
  signal?.throwIfAborted()
  const view = workspaceCaptureView(runtime, root, capturedAt)
  let closed = false
  let origin = ''
  const server = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch(request) {
    const url = new URL(request.url)
    if (closed || url.origin !== origin || url.pathname !== `/capture/${view.viewId}` || url.search || url.hash) return new Response('Not found', { status: 404 })
    if (request.method !== 'GET' && request.method !== 'HEAD') return new Response('Method not allowed', { status: 405 })
    return new Response(request.method === 'HEAD' ? null : view.html, { headers: view.headers })
  } })
  origin = `http://127.0.0.1:${server.port}`
  const close = (): void => {
    if (closed) return
    closed = true
    clearTimeout(timer)
    signal?.removeEventListener('abort', close)
    void server.stop(true)
  }
  const timer = setTimeout(close, 10_000)
  signal?.addEventListener('abort', close, { once: true })
  if (signal?.aborted) close()
  return Object.freeze({ root: view.root, viewId: view.viewId, capturedAt: view.capturedAt,
    url: `${origin}/capture/${view.viewId}`, current: () => !closed, close })
}
