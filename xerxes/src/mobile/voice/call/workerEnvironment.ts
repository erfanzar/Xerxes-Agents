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
