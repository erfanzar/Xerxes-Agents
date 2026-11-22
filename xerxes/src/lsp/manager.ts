// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { realpath } from 'node:fs/promises'
import { basename } from 'node:path'
import type { LspAdapter, LspRequest } from '../tools/claudeTools/search.js'
import { WorkspacePathResolver } from '../tools/pathSafety.js'
import { parseLspConfig, type LspServerConfig } from './config.js'
import { LspHost, type LspCompletionItem, type LspHostOptions } from './host.js'

type Host = LspAdapter & { readonly connected?: boolean; close(): Promise<void>; complete?: LspHost['complete'] }
interface Entry { controller: AbortController; host: Promise<Host>; resolved?: Host; stopping?: Promise<void>; cleanupFailed?: boolean }
export interface LspServerHealth {
  readonly name: string
  readonly languageId: string
  readonly extensions: readonly string[]
  readonly state: 'disabled' | 'idle' | 'starting' | 'stopping' | 'ready' | 'failed' | 'closed'
  readonly detail?: string
}
const cancelled = () => new DOMException('LSP request cancelled', 'AbortError')

/** One host per canonical workspace/server. A caller cancellation never owns shared startup. */
export class LspManager {
  private servers: readonly LspServerConfig[]
  private readonly entries = new Map<string, Entry>()
  private closing: Promise<void> | undefined
  private reconfiguring = false
  private readonly failedStarts = new Set<string>()
  constructor(config: unknown, private readonly start: (options: LspHostOptions, signal: AbortSignal) => Promise<Host> = LspHost.start) {
    this.servers = parseLspConfig(config)
  }
  async reconfigure(value: unknown, commit: () => undefined, signal?: AbortSignal): Promise<void> {
    const servers = parseLspConfig(value)
    if (this.closing) throw new Error('LSP manager is closed')
    if (this.reconfiguring) throw new Error('LSP settings update already in progress')
    if (signal?.aborted) throw cancelled()
    this.reconfiguring = true
    try {
      const previousNames = new Set(this.servers.map(server => server.name))
      const changed = new Set([
        ...this.servers.filter(before => {
          const after = servers.find(server => server.name === before.name)
          return JSON.stringify(before) !== JSON.stringify(after)
        }).map(server => server.name),
        ...servers.filter(after => !previousNames.has(after.name)).map(server => server.name),
      ])
      const targets = [...this.entries.keys()].filter(key => changed.has((JSON.parse(key) as [string, string])[1]))
      const results = await Promise.allSettled(targets.map(key => this.releaseKey(key)))
      if (results.some(result => result.status === 'rejected')) throw new Error('Language server cleanup failed; settings were not saved')
      if (this.closing) throw new Error('LSP manager closed during settings update')
      if (signal?.aborted) throw cancelled()
      // This synchronous boundary must commit disk before publishing the configuration.
      commit()
      this.servers = servers
      for (const key of this.failedStarts) if (changed.has((JSON.parse(key) as [string, string])[1])) this.failedStarts.delete(key)
    } finally { this.reconfiguring = false }
  }
  get configured(): boolean { return this.servers.some(server => server.enabled) }
  async health(root: string): Promise<readonly LspServerHealth[]> {
    const canonical = await realpath(root)
    return this.servers.map(server => {
      const key = JSON.stringify([canonical, server.name])
      const entry = this.entries.get(key)
      const state = this.closing ? 'closed' : !server.enabled ? 'disabled'
        : entry?.stopping ? 'stopping' : this.failedStarts.has(key) || entry?.cleanupFailed || entry?.resolved?.connected === false ? 'failed'
        : entry?.resolved ? 'ready' : entry ? 'starting' : 'idle'
      return { name: server.name, languageId: server.languageId, extensions: [...server.extensions], state,
        ...(state === 'failed' ? { detail: entry?.cleanupFailed ? 'Language server cleanup failed; retry release before reconnecting.' : entry?.resolved ? 'Language server disconnected; release its host before retrying.' : 'Language server initialization failed; check executable configuration and retry.' } : {}),
      }
    })
  }
  /** Stop only this workspace/server. The next explicit request may initialize a fresh host. */
  async release(root: string, name: string): Promise<void> {
    if (this.closing) throw new Error('LSP manager is closed')
    if (!this.servers.some(server => server.name === name)) throw new Error('Unknown LSP server')
    const canonical = await realpath(root)
    if (this.closing) throw new Error('LSP manager is closed')
    const key = JSON.stringify([canonical, name])
    return this.releaseKey(key)
  }
  private releaseKey(key: string): Promise<void> {
    const entry = this.entries.get(key)
    if (!entry) { this.failedStarts.delete(key); return Promise.resolve() }
    if (entry.stopping) return entry.stopping
    entry.controller.abort()
    const stopping = Promise.resolve().then(async () => {
      let host: Host | undefined
      try { host = await entry.host } catch { /* Failed initialization owns its startup cleanup. */ }
      if (host) await host.close()
      if (this.entries.get(key) === entry) this.entries.delete(key)
      this.failedStarts.delete(key)
    }).catch(() => {
      entry.cleanupFailed = true
      throw new Error('Language server cleanup failed; no replacement was started')
    }).finally(() => { delete entry.stopping })
    entry.stopping = stopping
    return stopping
  }
  forWorkspace(root: string): LspAdapter {
    return { execute: (request, signal) => this.execute(root, request, signal) }
  }
  /** Completion on the editor's unsaved text; the path only selects and scopes the server. */
  async complete(root: string, request: { readonly filePath: string; readonly text: string; readonly line: number; readonly character: number }, signal?: AbortSignal): Promise<{ readonly items: readonly LspCompletionItem[]; readonly incomplete: boolean }> {
    const { host, filePath } = await this.hostFor(root, request.filePath, signal)
    if (!host.complete) throw new Error('Language server does not support completion')
    return host.complete({ ...request, filePath }, signal)
  }
  private async execute(root: string, request: LspRequest, signal?: AbortSignal): Promise<unknown> {
    const { host, filePath } = await this.hostFor(root, request.filePath, signal)
    return host.execute({ ...request, filePath }, signal)
  }
  private async hostFor(root: string, requestedPath: string, signal?: AbortSignal): Promise<{ host: Host; filePath: string }> {
    if (this.closing) throw new Error('LSP manager is closed')
    if (signal?.aborted) throw cancelled()
    if (this.reconfiguring) throw new Error('LSP settings update in progress; retry shortly')
    const canonical = await realpath(root)
    const filePath = await new WorkspacePathResolver(canonical).resolve(requestedPath)
    if (this.reconfiguring) throw new Error('LSP settings update in progress; retry shortly')
    const name = basename(filePath).toLowerCase()
    const matchingExtLengths = (candidate: LspServerConfig) => candidate.extensions
      .filter(ext => name.endsWith(ext.toLowerCase()))
      .map(ext => ext.length)
    const server = this.servers
      .filter(candidate => candidate.enabled && matchingExtLengths(candidate).length > 0)
      .sort((a, b) => Math.max(...matchingExtLengths(b)) - Math.max(...matchingExtLengths(a)))[0]
    if (!server) throw new Error('No enabled LSP server is configured for this file type')
    if (this.closing) throw new Error('LSP manager is closed')
    if (signal?.aborted) throw cancelled()
    const key = JSON.stringify([canonical, server.name])
    let entry = this.entries.get(key)
    if (!entry) {
      if (this.entries.size >= 32) throw new Error('LSP workspace/server limit reached; close the runtime to release hosts')
      this.failedStarts.delete(key)
      const controller = new AbortController()
      const host = Promise.resolve().then(() => this.start({ ...server, cwd: canonical }, controller.signal))
      entry = { controller, host }
      this.entries.set(key, entry)
      const owned = entry
      void host.then(resolved => { owned.resolved = resolved }, () => {
        if (this.entries.get(key) === owned && !owned.stopping) {
          this.entries.delete(key)
          if (!this.closing) {
            this.failedStarts.add(key)
            if (this.failedStarts.size > 32) this.failedStarts.delete(this.failedStarts.values().next().value!)
          }
        }
      })
    }
    if (entry.stopping || entry.cleanupFailed) throw new Error('Language server is stopping or requires cleanup; retry after release completes')
    const host = await waitForHost(entry.host, signal)
    if (entry.stopping || entry.cleanupFailed || this.entries.get(key) !== entry) throw new Error('Language server was released during this request; retry')
    if (this.closing) throw new Error('LSP manager is closed')
    if (signal?.aborted) throw cancelled()
    return { host, filePath }
  }
  close(): Promise<void> {
    if (this.closing) return this.closing
    const entries = [...this.entries.values()]
    for (const entry of entries) entry.controller.abort()
    this.closing = Promise.allSettled(entries.map(async entry => {
      if (entry.stopping) {
        try { await entry.stopping; return } catch { /* Retry failed cleanup during final shutdown. */ }
      }
      let host: Host
      try { host = await entry.host } catch { return }
      await host.close()
    })).then(results => {
      this.entries.clear()
      this.failedStarts.clear()
      const errors = results.filter(result => result.status === 'rejected').map(result => result.reason)
      if (errors.length) throw new AggregateError(errors, 'Failed to close language servers')
    })
    return this.closing
  }
}
function waitForHost(host: Promise<Host>, signal?: AbortSignal): Promise<Host> {
  if (!signal) return host
  if (signal.aborted) return Promise.reject(cancelled())
  return new Promise((resolve, reject) => {
    const abort = () => { signal.removeEventListener('abort', abort); reject(cancelled()) }
    signal.addEventListener('abort', abort, { once: true })
    host.then(value => { signal.removeEventListener('abort', abort); resolve(value) }, error => { signal.removeEventListener('abort', abort); reject(error) })
  })
}
