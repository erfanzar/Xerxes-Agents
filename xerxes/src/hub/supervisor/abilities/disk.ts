// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { isRecord } from '../../protocol.js'

interface DiskMonitorOptions {
  readonly workspaces: () => readonly string[]
  readonly read: (workspace: string, signal: AbortSignal) => Promise<unknown>
  readonly observe: (event: { id: string; workspace: string; kind: 'disk_threshold'; digest: string; value: number }) => Promise<void>
  readonly paused: () => boolean
  readonly intervalMs?: number
  readonly onError?: (error: unknown) => void
}

/** Model-free, subscription-only sampling. No ambient network or provider client. */
export class DiskThresholdMonitor {
  private readonly stopped = new AbortController()
  private readonly epoch = randomUUID()
  private sequence = 0
  private timer: ReturnType<typeof setTimeout> | undefined
  private active: Promise<void> | undefined
  private readonly intervalMs: number

  constructor(private readonly options: DiskMonitorOptions) {
    this.intervalMs = options.intervalMs ?? 60_000
    if (!Number.isSafeInteger(this.intervalMs) || this.intervalMs < 1 || this.intervalMs > 3_600_000) throw new Error('Invalid disk sampling interval')
    this.schedule()
  }

  tick(): Promise<void> {
    if (this.active !== undefined) return this.active
    const run = async () => {
      if (this.stopped.signal.aborted || this.options.paused()) return
      const workspaces = [...new Set(this.options.workspaces())]
      if (workspaces.length > 100) throw new Error('Disk subscription sampling exceeds 100 workspaces')
      for (let offset = 0; offset < workspaces.length; offset += 4) {
        if (this.stopped.signal.aborted || this.options.paused()) return
        await Promise.all(workspaces.slice(offset, offset + 4).map(async workspace => {
          try {
            const signal = AbortSignal.any([this.stopped.signal, AbortSignal.timeout(5000)])
            const result = await this.options.read(workspace, signal)
            if (this.stopped.signal.aborted || this.options.paused()) return
            signal.throwIfAborted()
            const observation = isRecord(result) && result.ok === true && isRecord(result.observation) ? result.observation : undefined
            const disk = observation !== undefined && isRecord(observation.disk) ? observation.disk : undefined
            const available = disk?.available_bytes, total = disk?.total_bytes
            if (typeof available !== 'number' || typeof total !== 'number' || !Number.isFinite(available) || !Number.isFinite(total)
              || total <= 0 || available < 0 || available > total) throw new Error('Invalid disk observation')
            const value = 100 * (1 - available / total)
            await this.options.observe({ id: `disk:${this.epoch}:${++this.sequence}`, workspace, kind: 'disk_threshold',
              digest: `disk used ${value.toFixed(1)}%`, value })
          } catch (error) {
            // Owner shutdown is quiet. Ordinary timeouts/provider/RPC faults remain observable.
            if (!this.stopped.signal.aborted) this.options.onError?.(error)
          }
        }))
      }
    }
    this.active = run().finally(() => { this.active = undefined })
    return this.active
  }

  dispose(): void { this.stopped.abort(); clearTimeout(this.timer); this.timer = undefined }

  private schedule(): void {
    if (this.stopped.signal.aborted) return
    this.timer = setTimeout(() => {
      void this.tick().catch(error => { if (!this.stopped.signal.aborted) this.options.onError?.(error) }).finally(() => this.schedule())
    }, this.intervalMs)
    this.timer.unref?.()
  }
}
