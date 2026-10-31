// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { cleanFolder, contextKey, contextLabel, contextSessions, latestSessionIn, sameRemote, sessionsFor, type ContextSession, type WorkspaceContext } from './contextNavigation.js'
import { probeRemote, remoteTarget, type RemoteRunner, type RemoteTarget } from './remote.js'
import { remoteAttachAddress, remoteAttachProgram } from './remoteAttach.js'

export function savedHostResult(output: string): Pick<WorkspaceContext, 'availability' | 'reason' | 'sessions'> {
  if (!remoteAttachAddress(output)) return { availability: 'no-runtime', reason: 'No running Xerxes runtime', sessions: [] }
  const lines = output.split('\n').filter(line => line.startsWith('XERXES_REMOTE_SESSIONS '))
  if (lines.length !== 1) throw new Error('Missing remote session list')
  const value: unknown = JSON.parse(lines[0]!.slice('XERXES_REMOTE_SESSIONS '.length))
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid remote session list')
  const row = value as Record<string, unknown>
  if (!Array.isArray(row.saved) || !Array.isArray(row.live)) throw new Error('Invalid remote session list')
  return { availability: 'online', sessions: contextSessions(row.saved, row.live) }
}

interface HostEntry {
  machine: RemoteTarget
  context: WorkspaceContext
  lastAttempt: number
  pending?: Promise<void> | undefined
  controller?: AbortController | undefined
}
export interface SavedHostPorts {
  list(): Promise<Record<string, unknown>>
  run?: RemoteRunner
  now?: () => number
  timeoutMs?: number
}
const REFRESH_MS = 60000
/** Cached, read-only discovery. No tunnels, bootstrap, credentials, or runtime mutations. */
export class SavedHostContexts {
  private readonly entries = new Map<string, HostEntry>()
  private nextId = -1
  private lastRegistry = -Infinity
  private registryPending: Promise<void> | undefined
  private disposed = false
  constructor(private readonly ports: SavedHostPorts) {}

  async refresh(open: readonly RemoteTarget[] = []): Promise<void> {
    if (this.disposed) return
    const now = (this.ports.now ?? Date.now)()
    if (!this.registryPending && now - this.lastRegistry >= REFRESH_MS) {
      this.lastRegistry = now
      this.registryPending = (async () => {
        const result = await this.ports.list()
        if (result.ok !== true || !Array.isArray(result.machines) || result.machines.length > 100) throw new Error('Could not read saved SSH hosts')
        const machines = result.machines.map(remoteTarget)
        if (this.disposed) return
        const retained = new Set<string>()
        for (const machine of machines) {
          // One context per host AND folder: two saved folders on one host
          // (EasyDeL and prismcore on v5p-128-dev) are two projects.
          const key = contextKey(machine)
          if (retained.has(key)) continue
          retained.add(key)
          const previous = this.entries.get(key)
          const label = contextLabel(machine.alias, machine.workspacePath)
          if (previous && sameRemote(previous.machine, machine)) {
            previous.machine = machine
            previous.context.label = label
            previous.context.alias = machine.alias
          } else {
            previous?.controller?.abort()
            this.entries.set(key, { machine, lastAttempt: -Infinity,
              context: { id: previous?.context.id ?? this.nextId--, label,
                workspace: machine.workspacePath, workspacePath: machine.workspacePath, alias: machine.alias,
                remote: true, saved: true, availability: 'offline', reason: 'Checking connection…', sessions: [] },
            })
          }
        }
        for (const [key, entry] of this.entries) if (!retained.has(key)) {
          entry.controller?.abort()
          this.entries.delete(key)
        }
      })().catch(() => { /* Keep cached navigation; never block a window on registry/SSH failures. */ })
        .finally(() => { this.registryPending = undefined })
    }
    await this.registryPending
    this.contexts(open)
  }

  /** Synchronous cached snapshot; a stale host refresh never delays IPC rendering. */
  contexts(open: readonly RemoteTarget[] = []): WorkspaceContext[] {
    const keys = new Set(open.map(contextKey))
    const contexts: WorkspaceContext[] = []
    for (const entry of this.entries.values()) {
      if (keys.has(contextKey(entry.machine))) continue
      this.schedule(entry)
      contexts.push({ ...entry.context, sessions: this.sessionsOf(entry, open) })
    }
    return contexts
  }

  /** Every project folder known on `target`: saved ones plus folders open in views. */
  folders(target: string, open: readonly RemoteTarget[] = []): string[] {
    const folders = [...this.entries.values(), ...open.map(machine => ({ machine }))]
      .filter(entry => entry.machine.target === target)
      .map(entry => cleanFolder(entry.machine.workspacePath))
    return [...new Set(folders)]
  }

  /** The host lists every project's tasks; keep the ones of this entry's folder. */
  private sessionsOf(entry: HostEntry, open: readonly RemoteTarget[] = []): ContextSession[] {
    return sessionsFor(entry.context.sessions, entry.machine.workspacePath, this.folders(entry.machine.target, open))
  }

  /** Resolve clicks against our validated cache, never a renderer-supplied SSH target/path. */
  selection(id: number, sessionId?: unknown): { machine: RemoteTarget; sessionId?: string } {
    const entry = [...this.entries.values()].find(entry => entry.context.id === id)
    if (!entry) throw new Error('Saved SSH host is unavailable')
    if (sessionId === undefined) {
      // Opening the folder itself resumes its newest task; a blank chat made
      // a project with history look empty.
      const latest = latestSessionIn(this.sessionsOf(entry), entry.machine.workspacePath)
      return latest === undefined ? { machine: { ...entry.machine } } : { machine: { ...entry.machine }, sessionId: latest }
    }
    if (typeof sessionId !== 'string' || !/^[a-zA-Z0-9_-]{1,256}$/.test(sessionId)) throw new Error('Invalid remote session id')
    const row = entry.context.sessions.find(row => row.id === sessionId)
    if (!row) throw new Error('Session is no longer available on this host')
    return { machine: remoteTarget({ ...entry.machine, workspacePath: row.cwd }), sessionId: row.id }
  }

  dispose(): void {
    this.disposed = true
    for (const entry of this.entries.values()) entry.controller?.abort()
  }

  private schedule(entry: HostEntry): void {
    const now = (this.ports.now ?? Date.now)()
    if (this.disposed || entry.pending || now - entry.lastAttempt < REFRESH_MS) return
    entry.lastAttempt = now
    const controller = new AbortController()
    entry.controller = controller
    let timer: ReturnType<typeof setTimeout>
    const timeout = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => { controller.abort(); reject(new Error('Read-only SSH timeout')) }, this.ports.timeoutMs ?? 20000)
    })
    entry.pending = Promise.race([
      probeRemote(entry.machine, remoteAttachProgram(entry.machine.workspacePath, true), controller.signal, this.ports.run), timeout,
    ]).then(output => {
      const result = savedHostResult(output)
      const { reason: _previousReason, ...context } = entry.context
      entry.context = { ...context, ...result }
    }).catch(() => {
      // SSH stderr may include private host/config information: never forward it.
      entry.context = { ...entry.context, availability: 'offline', reason: controller.signal.aborted ? 'SSH probe timed out' : 'SSH host unavailable or read-only probe failed' }
    }).finally(() => {
      clearTimeout(timer)
      entry.pending = undefined
      entry.controller = undefined
    })
  }
}
