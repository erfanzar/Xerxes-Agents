// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomBytes } from 'node:crypto'
import { basename, isAbsolute, resolve } from 'node:path'
import type { DaemonRuntime } from './runtime.js'
import { redactString } from '../security/redact.js'

const STYLE = `:root{color-scheme:dark;font-family:ui-monospace,monospace;background:#15191d;color:#e6e9ec}*{box-sizing:border-box}body{margin:0;padding:24px;max-width:1440px}h1{font-size:22px;margin:8px 0}p{color:#a6b0ba;font-size:12px;overflow-wrap:anywhere}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.stat,article{border:1px solid #35404b;border-radius:10px;padding:14px;background:#1e252b}.stat{font-size:13px}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}article{min-width:0}h2{font-size:13px;margin:0 0 8px;overflow-wrap:anywhere}.state{color:#9ed5b2}footer{margin-top:18px;color:#a6b0ba;font-size:11px}@media(max-width:600px){body{padding:16px}main{grid-template-columns:1fr}.stats{gap:8px}.stat{padding:10px}article{padding:11px}}`
const CSP = `default-src 'none'; style-src 'sha256-${new Bun.CryptoHasher('sha256').update(STYLE).digest('base64')}'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; sandbox`

function escaped(value: string): string {
  return redactString(value).slice(0, 160).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;')
}

/** Pure immutable, exact-workspace projection. No transcript, account state, actions or browser access. */
export function workspaceCaptureView(runtime: Pick<DaemonRuntime, 'listSessions'>, root: string, capturedAt: number) {
  if (!isAbsolute(root) || resolve(root) !== root || !Number.isSafeInteger(capturedAt) || capturedAt < 0) throw new Error('An explicit canonical capture workspace and timestamp are required')
  const inventory = runtime.listSessions()
  if (inventory.length > 2000) throw new Error('Capture session inventory exceeds budget')
  const sessions = inventory.filter(session => session.cwd === root)
  const viewId = randomBytes(16).toString('hex')
  const working = sessions.filter(session => session.status === 'working' || session.status === 'starting').length
  const waiting = sessions.filter(session => session.status === 'waiting').length
  const rows = sessions.slice().sort((a, b) => a.id.localeCompare(b.id)).slice(0, 8).map(session =>
    `<article><h2>Task ${escaped(session.id.slice(0, 16))}</h2><span class="state">${escaped(session.status)}</span><p>${escaped(session.model)}</p></article>`).join('')
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Xerxes workspace capture</title><style>${STYLE}</style></head><body><header><p>XERXES · READ-ONLY CAPTURE VIEW</p><h1>${escaped(basename(root) || root)}</h1><p>Workspace-only summaries. No account overlays or transcript content.</p></header><section class="stats"><div class="stat">${sessions.length} tasks</div><div class="stat">${working} working</div><div class="stat">${waiting} waiting</div></section><main>${rows || '<p>No active task summaries in this workspace.</p>'}</main><footer>Fixed viewport preview · ${Math.min(sessions.length, 8)} of ${sessions.length} summaries · ${capturedAt}</footer></body></html>`
  return Object.freeze({ root, viewId, capturedAt, html, headers: Object.freeze({
    'Content-Type': 'text/html; charset=utf-8', 'Content-Security-Policy': CSP, 'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', 'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
  }) })
}
