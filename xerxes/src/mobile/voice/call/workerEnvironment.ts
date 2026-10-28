// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { env } from '@huggingface/transformers'

import { createCallFetch } from './logic.js'
import type { CallWorkerRequest, CallWorkerResponse } from './protocol.js'

/** Module-local scope declarations keep worker globals out of the phone's DOM compilation. */
export interface CallWorkerScope {
  readonly location: { readonly origin: string }
  readonly navigator: { readonly gpu?: unknown }
  onmessage: ((event: MessageEvent<CallWorkerRequest>) => void) | null
  postMessage(message: CallWorkerResponse, transfer?: Transferable[]): void
}

/**
 * A short, user-visible reason for a failed worker job: the error's own message with URLs
 * and paths removed, clipped. Enough to diagnose a phone without leaking where files live.
 */
export function workerErrorDetail(error: unknown): string {
  const raw = error instanceof Error ? `${error.name}: ${error.message}` : typeof error === 'string' ? error : 'unknown error'
  const cleaned = raw.replace(/\b[a-z][a-z0-9+.-]*:\/\/\S+/giu, '<url>').replace(/(?:^|\s)\/[^\s:]+/gu, ' <path>').replace(/\s+/gu, ' ').trim()
  return cleaned.length > 160 ? `${cleaned.slice(0, 157)}...` : cleaned || 'unknown error'
}

/** Each worker owns its own single-threaded session; both use the same pinned runtime assets. */
export function configureCallWorker(): CallWorkerScope {
  const scope = globalThis as unknown as CallWorkerScope
  const origin = scope.location.origin
  const nativeFetch = globalThis.fetch.bind(globalThis)
  globalThis.fetch = Object.assign(createCallFetch(origin, nativeFetch), nativeFetch)
  env.allowLocalModels = false
  env.remoteHost = `${origin}/voice-models/`
  env.remotePathTemplate = '{model}/resolve/{revision}/'
  const wasm = env.backends.onnx.wasm
  if (!wasm) throw new Error('ONNX browser runtime unavailable.')
  wasm.wasmPaths = `${origin}/voice-ort/`
  wasm.numThreads = 1
  wasm.proxy = false
  return scope
}
