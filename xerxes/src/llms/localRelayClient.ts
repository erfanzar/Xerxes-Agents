// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomBytes } from 'node:crypto'
import { LocalProviderRelayError, isRelayFailureCode } from '../security/localProviderRelay.js'
import { decodeRelayCompletion, decodeRelayDelta, type ProviderRelayRequest } from '../security/providerRelayProtocol.js'
import type { CompletionRequest, LlmClient, LlmDelta } from './client.js'

export type ProviderRelayCall = (frame: ProviderRelayRequest, signal?: AbortSignal) => Promise<unknown>

/** Remote-side model client. The injected transport is bound to a specific
 * approved local endpoint; it accepts no URL, key or fallback provider. */
export class LocalRelayClient implements LlmClient {
  constructor(private readonly call: ProviderRelayCall) {}
  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    if (signal?.aborted) throw new LocalProviderRelayError('cancelled')
    const decoded = decodeRelayCompletion(request)
    const id = randomBytes(16).toString('hex')
    let complete = false
    try {
      let first = true
      for (;;) {
        if (signal?.aborted) throw new LocalProviderRelayError('cancelled')
        const value = await withCancellation(this.call({ op: 'next', id, ...(first ? { request: decoded } : {}) }, signal), signal)
        first = false
        if (signal?.aborted) throw new LocalProviderRelayError('cancelled')
        if (!value || typeof value !== 'object' || Array.isArray(value)) throw new LocalProviderRelayError('provider_failed')
        const reply = value as Record<string, unknown>
        if (Object.hasOwn(reply, 'error')) {
          if (Object.keys(reply).length !== 1 || !isRelayFailureCode(reply.error)) throw new LocalProviderRelayError('provider_failed')
          throw new LocalProviderRelayError(reply.error)
        }
        if (typeof reply.done !== 'boolean' || Object.keys(reply).some(key => key !== 'done' && key !== 'deltas') ||
          (reply.deltas !== undefined && !Array.isArray(reply.deltas)) ||
          (reply.done === false && (!Array.isArray(reply.deltas) || !reply.deltas.length)) ||
          (Array.isArray(reply.deltas) && reply.deltas.length > 256)) throw new LocalProviderRelayError('provider_failed')
        const deltas = reply.deltas === undefined ? [] : (reply.deltas as unknown[]).map(decodeRelayDelta)
        if (Buffer.byteLength(JSON.stringify(deltas)) > 1024 * 1024 + 1024) throw new LocalProviderRelayError('provider_failed')
        for (const delta of deltas) { if (signal?.aborted) throw new LocalProviderRelayError('cancelled'); yield delta }
        if (reply.done) { complete = true; return }
      }
    } catch (error) {
      if (signal?.aborted) throw new LocalProviderRelayError('cancelled')
      if (error instanceof LocalProviderRelayError) throw new LocalProviderRelayError(error.code)
      throw new LocalProviderRelayError('provider_failed')
    } finally {
      // Explicit cancellation is best effort on a failed transport; the local
      // endpoint also has a bounded idle deadline and disconnect cleanup.
      if (!complete) void this.call({ op: 'cancel', id }).catch(() => {})
    }
  }
}

function withCancellation<T>(pending: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return pending
  return new Promise((resolve, reject) => {
    const cancel = () => { signal.removeEventListener('abort', cancel); reject(new LocalProviderRelayError('cancelled')) }
    signal.addEventListener('abort', cancel, { once: true })
    pending.then(value => { signal.removeEventListener('abort', cancel); resolve(value) }, error => {
      signal.removeEventListener('abort', cancel); reject(error)
    })
    if (signal.aborted) cancel()
  })
}
