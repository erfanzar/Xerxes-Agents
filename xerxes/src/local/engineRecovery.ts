// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Crash recovery for a local engine Xerxes started with `xerxes local start`.
 *
 * A local model can go away mid-task: LM Studio unloads a model that ran out
 * of memory, and llama-server exits. The next model request then fails with a
 * refused or dropped connection, or with LM Studio's "model crashed" or "no
 * models loaded" reply. When that happens, the turn loop asks the hook built
 * here, once per turn, to put the model back. The engine reloads it with the
 * window recorded when Xerxes started it, and the loop retries the step once
 * and says so.
 *
 * An engine Xerxes did not start, one serving another port or model, and one
 * that is still serving the model are left alone, so the failure is reported
 * as it was.
 */
import { errorMessage } from '../executors/toolRegistry.js'
import { isRecoverableNetworkError } from '../runtime/errorClassifier.js'
import { loopbackOrigin } from './loadedWindow.js'

/** The endpoint a failed request went to. */
export interface EngineTarget {
  readonly port: number
  /** The model id the request named. */
  readonly model: string
}

export type EngineRecovery =
  | {
      readonly recovered: true
      /** One sentence for the user: what was reloaded, with which window. */
      readonly message: string
    }
  | {
      readonly recovered: false
      /**
       * `not-managed`: Xerxes did not start this engine, or has no record of
       * the window it used. `other-endpoint`: it serves another port or model.
       * `still-running`: the model is still loaded, so the failure was not a
       * crash.
       */
      readonly reason: 'not-managed' | 'other-endpoint' | 'still-running'
    }

export interface RecoverableEngine {
  recover(target: EngineTarget): Promise<EngineRecovery>
}

/**
 * Reload the crashed model. Resolves with what was reloaded, or undefined when
 * no engine Xerxes started serves this route. Rejects when a reload was tried
 * and failed.
 */
export type RecoverEngine = () => Promise<string | undefined>

/** LM Studio's replies when the model is gone, and an engine out of memory. */
const ENGINE_GONE =
  /model (?:has )?crashed|model (?:was |is |has been )?unloaded|no models? (?:are )?loaded|model\b.{0,80}\bnot (?:been )?loaded|failed to load (?:the )?model|out of memory|insufficient memory/i

/** Whether a failed model request looks like the engine lost its model. */
export function isEngineFailure(error: unknown): boolean {
  return isRecoverableNetworkError(error) || ENGINE_GONE.test(errorMessage(error))
}

/**
 * The recovery hook for one route, or undefined when the route is not a
 * loopback endpoint (a remote provider is never Xerxes's own engine).
 */
export function engineRecoveryFor(
  engines: readonly RecoverableEngine[],
  route: { readonly baseUrl: string | undefined; readonly model: string },
): RecoverEngine | undefined {
  const origin = route.baseUrl === undefined ? undefined : loopbackOrigin(route.baseUrl)
  if (origin === undefined) return undefined
  const url = new URL(origin)
  const port = url.port === '' ? (url.protocol === 'https:' ? 443 : 80) : Number(url.port)
  return async () => {
    for (const engine of engines) {
      const result = await engine.recover({ port, model: route.model })
      if (result.recovered) return result.message
    }
    return undefined
  }
}
