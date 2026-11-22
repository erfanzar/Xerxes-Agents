// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useReducer, useRef, useState, type ReactElement } from 'react'

import type { PairEnv } from '../context.js'
import { browserPairStorage, initialPairState, reducePair, startPairingSession, type PairState } from '../pairing.js'

/** Presentation of one pairing state; the page below drives it. */
export function PairView({ state, onRetry }: { state: PairState; onRetry: () => void }): ReactElement {
  const comparison = 'comparison' in state ? state.comparison : undefined
  return (
    <main className="pair" aria-labelledby="pair-title">
      <h1 id="pair-title" className="page-title">Pair this phone</h1>
      {state.phase === 'missing' ? (
        <div>
          <p>This page needs a pairing link. On the desktop, create a pairing code and scan its QR code with this phone.</p>
          <p><a className="btn" href="/">Open Xerxes</a></p>
        </div>
      ) : null}
      {state.phase === 'invalid' ? <p role="alert" className="field-error">That pairing link is incomplete. Create a new pairing code on the desktop and scan it again.</p> : null}
      {state.phase === 'rejected' ? <p role="alert" className="field-error">{state.message}</p> : null}
      {state.phase === 'unavailable' ? (
        <div role="alert" className="notice is-error">
          <p>{state.message}</p>
          <button type="button" className="btn" onClick={onRetry}>Try again</button>
        </div>
      ) : null}
      {state.phase === 'submitting' ? <p aria-busy="true">Contacting the Hub…</p> : null}
      {state.phase === 'waiting' ? <p role="status">Waiting for confirmation on the desktop. Check that the desktop shows the same code, then confirm it there.</p> : null}
      {state.phase === 'timeout' ? <p role="alert" className="field-error">The desktop did not confirm in time. Create a new pairing code and scan it again.</p> : null}
      {state.phase === 'paired' ? <p role="status">Paired. Opening your inbox…</p> : null}
      {comparison === undefined ? null : (
        <div className="comparison">
          <p>Comparison code</p>
          <p className="comparison-code" aria-label={`Comparison code ${comparison.split('').join(' ')}`}>{comparison}</p>
        </div>
      )}
    </main>
  )
}

/**
 * Pair page. The URL fragment was read and removed before the first render
 * (see takePairingFragment); this page only ever holds the secret in memory
 * and sends it once, in the POST body. It then waits until the desktop owner
 * confirms, which is when the session refresh starts succeeding.
 */
export function PairPage({ env }: { env: PairEnv }): ReactElement {
  const [state, dispatch] = useReducer(reducePair, env.read, initialPairState)
  const [attempt, setAttempt] = useState(0)
  const consumed = useRef(env.read.kind === 'resume')

  useEffect(() => startPairingSession({
    read: env.read,
    consumed,
    http: env.http,
    deviceLabel: env.deviceLabel,
    storage: browserPairStorage(),
    dispatch,
    onPaired: env.onPaired,
    pollMs: env.pollMs,
    maxWaitMs: env.maxWaitMs,
  // One consume per attempt; env is fixed for the page's lifetime.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [attempt])

  return <PairView state={state} onRetry={() => { dispatch({ type: 'retry' }); setAttempt(count => count + 1) }} />
}
