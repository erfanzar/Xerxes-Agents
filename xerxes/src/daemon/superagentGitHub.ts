// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isRecord } from '../hub/protocol.js'
import type { ObservationHosts } from '../hub/supervisor/abilities/observations.js'
import { REDACTED, redactString } from '../security/redact.js'
import type { DnsLookup } from '../security/urlSafety.js'
import { PublicWebClient, type WebFetch } from '../tools/webTools.js'

/** Host-owned selection only. Never derive repository or credentials from model/task/repository text. */
export function createGitHubObservationHost(options: {
  readonly repository: string
  readonly token?: string
  readonly fetcher?: WebFetch
  readonly dnsLookup?: DnsLookup
}): NonNullable<ObservationHosts['github']> {
  const token = options.token
  const parts = options.repository.split('/')
  const [owner, name] = parts
  if (parts.length !== 2 || owner === undefined || name === undefined || !/^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$/.test(owner)
    || !/^[A-Za-z0-9_.-]{1,100}$/.test(name) || name === '.' || name === '..') throw new Error('GitHub repository must be an explicit owner/name')
  if (token !== undefined && (token.length === 0 || token.length > 512 || /[\r\n\0]/.test(token))) throw new Error('Invalid GitHub credential header')
  const repository = `${owner}/${name}`
  const client = new PublicWebClient({ maxRedirects: 0, maxResponseBytes: 128 * 1024, timeoutMs: 5000,
    ...(options.fetcher === undefined ? {} : { fetcher: options.fetcher }),
    ...(options.dnsLookup === undefined ? {} : { urlSafety: { dnsLookup: options.dnsLookup } }) })
  const headers: Record<string, string> = { accept: 'application/vnd.github+json', 'x-github-api-version': '2022-11-28', 'user-agent': 'Xerxes-Observations' }
  if (token !== undefined) headers.authorization = `Bearer ${token}`
  return async (view, signal) => {
    let status: number | undefined
    let truncated = false
    const text = (value: unknown, max: number): string => {
      if (typeof value !== 'string') throw new Error('Malformed GitHub text')
      const safe = redactString(token === undefined ? value : value.replaceAll(token, REDACTED))
      const bytes = Buffer.from(safe)
      if (bytes.length > max) truncated = true
      return bytes.subarray(0, max).toString('utf8')
    }
    const id = (value: unknown): number => {
      if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) throw new Error('Malformed GitHub identifier')
      return value
    }
    try {
      if (signal?.aborted) throw new Error('cancelled')
      if (view !== 'issues' && view !== 'prs' && view !== 'ci') throw new Error('Invalid GitHub view')
      const suffix = view === 'ci' ? 'actions/runs?per_page=30' : `${view === 'prs' ? 'pulls' : 'issues'}?state=all&per_page=30`
      const { response } = await client.fetch(`https://api.github.com/repos/${repository}/${suffix}`,
        { method: 'GET', headers, credentials: 'omit' }, { allowedDomains: ['api.github.com'], ...(signal === undefined ? {} : { signal }) })
      status = response.status
      if (!response.ok) { void response.body?.cancel().catch(() => {}); throw new Error('HTTP failure') }
      const raw: unknown = JSON.parse(await client.text(response, { signal }))
      const rows: unknown = view === 'ci' && isRecord(raw) ? raw.workflow_runs : raw
      if (!Array.isArray(rows)) throw new Error('Malformed GitHub list')
      const items: Record<string, unknown>[] = []
      for (const row of rows.slice(0, 30)) {
        if (!isRecord(row)) throw new Error('Malformed GitHub item')
        if (view === 'issues' && row.pull_request !== undefined) continue
        if (view === 'ci') {
          const run = id(row.id)
          items.push({ id: run, name: text(row.name, 256), status: text(row.status, 32), conclusion: row.conclusion === null ? null : text(row.conclusion, 32),
            url: `https://github.com/${repository}/actions/runs/${run}` })
        } else {
          const number = id(row.number)
          const item: Record<string, unknown> = { number, title: text(row.title, 256), state: text(row.state, 32), body: row.body === null ? null : text(row.body, 768),
            url: `https://github.com/${repository}/${view === 'prs' ? 'pull' : 'issues'}/${number}` }
          if (view === 'prs') { if (typeof row.draft !== 'boolean') throw new Error('Malformed GitHub draft'); item.draft = row.draft }
          items.push(item)
        }
      }
      const result = { repository, view, items, has_more: /rel="next"/.test(response.headers.get('link') ?? '') || rows.length > 30,
        coverage: 'first_page', truncated, untrusted: true }
      if (Buffer.byteLength(JSON.stringify(result)) > 60 * 1024) throw new Error('GitHub projection too large')
      return result
    } catch {
      // No upstream body, URL, transport message or credential reaches the model.
      if (signal?.aborted) throw new Error('GitHub observation cancelled')
      throw new Error(`GitHub observation unavailable${status !== undefined && (status < 200 || status >= 300) ? ` (HTTP ${status})` : ''}`)
    }
  }
}
