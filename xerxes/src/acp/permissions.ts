// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject } from '../types/toolCalls.js'

export interface AcpPermissionRequest {
  allowed: boolean
  decided: boolean
  readonly description: string
  readonly id: string
  readonly inputs: JsonObject
  readonly metadata: Record<string, unknown>
  readonly sessionId: string
  readonly toolName: string
}

interface PendingDecision {
  readonly promise: Promise<boolean>
  readonly resolve: (allowed: boolean) => void
}

/** Make a fresh permission request suitable for transport over ACP. */
export function routePermission(options: {
  readonly description: string
  readonly inputs: JsonObject
  readonly sessionId: string
  readonly toolName: string
}): AcpPermissionRequest {
  return {
    id: crypto.randomUUID().replaceAll('-', ''),
    sessionId: options.sessionId,
    toolName: options.toolName,
    description: options.description,
    inputs: { ...options.inputs },
    decided: false,
    allowed: false,
    metadata: {},
  }
}

/**
 * Holds ACP approval requests while a streamed agent turn awaits an editor
 * response. The returned promise is intentionally one-shot and idempotent.
 */
export class AcpPermissionBoard {
  private readonly decisions = new Map<string, PendingDecision>()
  private readonly requests = new Map<string, AcpPermissionRequest>()

  submit(request: AcpPermissionRequest): Promise<boolean> {
    const existing = this.decisions.get(request.id)
    if (existing) {
      return existing.promise
    }
    const decision = deferredDecision()
    this.requests.set(request.id, request)
    this.decisions.set(request.id, decision)
    return decision.promise
  }

  resolve(requestId: string, allow: boolean): boolean {
    const request = this.requests.get(requestId)
    if (!request || request.decided) {
      return false
    }
    request.decided = true
    request.allowed = allow
    this.decisions.get(requestId)?.resolve(allow)
    return true
  }

  snapshotPending(): readonly AcpPermissionRequest[] {
    return [...this.requests.values()].filter(request => !request.decided)
  }

  get(requestId: string): AcpPermissionRequest | undefined {
    return this.requests.get(requestId)
  }

  drop(requestId: string): void {
    const request = this.requests.get(requestId)
    if (request && !request.decided) {
      request.decided = true
      request.allowed = false
    }
    this.requests.delete(requestId)
    const decision = this.decisions.get(requestId)
    this.decisions.delete(requestId)
    decision?.resolve(false)
  }

  async awaitDecision(request: AcpPermissionRequest, signal?: AbortSignal): Promise<boolean> {
    const decision = this.submit(request)
    if (!signal) {
      return decision
    }
    if (signal.aborted) {
      this.drop(request.id)
      return false
    }

    return new Promise(resolve => {
      const onAbort = () => {
        this.drop(request.id)
        resolve(false)
      }
      signal.addEventListener('abort', onAbort, { once: true })
      void decision.then(allowed => {
        signal.removeEventListener('abort', onAbort)
        resolve(allowed)
      })
    })
  }
}

function deferredDecision(): PendingDecision {
  let resolve!: (allowed: boolean) => void
  const promise = new Promise<boolean>(complete => {
    resolve = complete
  })
  return { promise, resolve }
}
