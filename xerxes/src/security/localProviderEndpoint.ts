// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { LlmDelta } from '../llms/client.js'
import { LocalProviderRelay, LocalProviderRelayError, type RelayPeer, type RelayFailureCode } from './localProviderRelay.js'
import { decodeRelayDelta, decodeRelayRequest, MAX_RELAY_DELTA_BYTES, type ProviderRelayReply } from './providerRelayProtocol.js'

interface Call {
  readonly controller: AbortController
  readonly iterator: AsyncGenerator<LlmDelta>
  timer?: ReturnType<typeof setTimeout>
  pending: boolean
  prefetched?: Promise<IteratorResult<LlmDelta>>
  failure?: RelayFailureCode
}

/** One endpoint belongs to one consented grant and authenticated transport peer.
 * Pull replies contain bounded batches. The remote side never receives the
 * local grant token, provider endpoint, credentials or client object. */
export class LocalProviderEndpoint {
  private readonly calls = new Map<string, Call>()
  private closed = false
  private closedReason: RelayFailureCode | undefined
  constructor(private readonly relay: LocalProviderRelay, private readonly peer: RelayPeer, private readonly token: string,
    private readonly idleTimeoutMs = 60_000, private readonly batchSize = 32) {
    if (!Number.isFinite(idleTimeoutMs) || idleTimeoutMs <= 0 || !Number.isSafeInteger(batchSize) || batchSize < 1 || batchSize > 256) throw new LocalProviderRelayError('invalid_request')
  }

  async handle(value: unknown): Promise<ProviderRelayReply> {
    if (this.closed) return { error: this.closedReason ?? 'grant_unavailable' }
    let id: string | undefined
    let call: Call | undefined
    let ownsPull = false
    const deltas: LlmDelta[] = []
    try {
      const frame = decodeRelayRequest(value)
      id = frame.id
      call = this.calls.get(id)
      if (frame.op === 'cancel') { if (call) this.remove(id, call); return { done: true } }
      if (call && (call.pending || frame.request !== undefined)) return { error: 'invalid_request' }
      if (call?.failure) { const error = call.failure; this.remove(id, call); return { error } }
      if (!call) {
        if (!frame.request) return { error: 'invalid_request' }
        if (this.calls.size >= this.relay.inspect(this.peer, this.token).maxConcurrent) return { error: 'concurrency_limit' }
        const controller = new AbortController()
        call = { controller, iterator: this.relay.stream(this.peer, this.token, frame.request, controller.signal), pending: false }
        this.calls.set(id, call)
      }
      clearTimeout(call.timer)
      const current = call
      call.timer = setTimeout(() => this.remove(id!, current), this.idleTimeoutMs)
      call.timer.unref?.()
      call.pending = true
      ownsPull = true
      let bytes = 0, deadline = 0
      // Wait for the first delta, then collect available output for at most
      // four milliseconds. At most one prefetch survives a response boundary.
      while (deltas.length < this.batchSize) {
        if (deltas.length && (bytes >= 64 * 1024 || Date.now() >= deadline)) break
        call.prefetched ??= call.iterator.next()
        const pending = abortableNext(call.prefetched, call.controller.signal)
        const result = deltas.length ? await within(pending, Math.max(0, deadline - Date.now())) : await pending
        if (result === undefined) break
        delete call.prefetched
        if (this.closed || this.calls.get(id) !== call) return { error: this.closedReason ?? 'cancelled' }
        if (result.done) { this.remove(id, call); return { done: true, ...(deltas.length ? { deltas } : {}) } }
        let delta: LlmDelta
        try { delta = decodeRelayDelta(result.value) } catch { throw new LocalProviderRelayError('provider_failed') }
        const size = Buffer.byteLength(JSON.stringify(delta))
        if (deltas.length && bytes + size > MAX_RELAY_DELTA_BYTES) { call.prefetched = Promise.resolve(result); break }
        deltas.push(delta); bytes += size
        if (deltas.length === 1) deadline = Date.now() + 4
      }
      return { done: false, deltas }
    } catch (error) {
      const code = this.closedReason ?? (error instanceof LocalProviderRelayError ? error.code : 'provider_failed')
      // Preserve already received text when a later provider chunk fails.
      // Revocation/expiry/cancellation must still stop delivery immediately.
      if (ownsPull && call && deltas.length && code === 'provider_failed' && !call.controller.signal.aborted) {
        call.failure = code
        return { done: false, deltas }
      }
      if (id && call) this.remove(id, call)
      return { error: code }
    } finally { if (ownsPull && call) call.pending = false }
  }

  close(reason?: RelayFailureCode): void {
    this.closed = true
    this.closedReason = reason
    for (const [id, call] of this.calls) this.remove(id, call)
    this.relay.disconnect(this.peer)
  }

  private remove(id: string, call: Call): void {
    if (this.calls.get(id) !== call) return
    this.calls.delete(id)
    clearTimeout(call.timer)
    call.controller.abort()
    // A noncooperative provider may never settle next(). Do not let its queued
    // return block disconnect; the grant keeps counting that active backend
    // request until it settles, preventing extra concurrency from being granted.
    void call.iterator.return(undefined).catch(() => {})
  }
}

function abortableNext(pending: Promise<IteratorResult<LlmDelta>>, signal: AbortSignal): Promise<IteratorResult<LlmDelta>> {
  return new Promise((resolve, reject) => {
    const cancel = () => { signal.removeEventListener('abort', cancel); reject(new LocalProviderRelayError('cancelled')) }
    if (signal.aborted) { cancel(); return }
    signal.addEventListener('abort', cancel, { once: true })
    pending.then(value => { signal.removeEventListener('abort', cancel); resolve(value) }, error => {
      signal.removeEventListener('abort', cancel); reject(error)
    })
  })
}

function within<T>(pending: Promise<T>, ms: number): Promise<T | undefined> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => resolve(undefined), ms)
    pending.then(value => { clearTimeout(timer); resolve(value) }, error => { clearTimeout(timer); reject(error) })
  })
}
