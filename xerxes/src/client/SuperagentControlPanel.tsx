// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, useSyncExternalStore, type ReactElement } from 'react'
import { CONTROL_METRICS, parseDailyLimits, type ControlMetric, type DailyLimitsDraft, type SuperagentControlClient, type SuperagentControlSnapshot } from './superagentControl.js'

const LABELS: Readonly<Record<ControlMetric, string>> = { actions: 'actions', wakes: 'wakes', modelCalls: 'model calls', tokens: 'tokens', usd: 'USD' }
function draft(data?: SuperagentControlSnapshot): DailyLimitsDraft {
  return { actions: String(data?.limits.actions ?? ''), wakes: String(data?.limits.wakes ?? ''),
    modelCalls: String(data?.limits.modelCalls ?? ''), tokens: String(data?.limits.tokens ?? ''), usd: String(data?.limits.usd ?? '') }
}

/** Shared human-only surface. The Hub independently checks current ownership on every request. */
export function SuperagentControlPanel({ client, online }: { readonly client: SuperagentControlClient; readonly online: boolean }): ReactElement {
  const view = useSyncExternalStore(client.subscribe, client.getSnapshot, client.getSnapshot)
  const [fields, setFields] = useState<DailyLimitsDraft>(() => draft(view.data))
  const [confirmKill, setConfirmKill] = useState(false)
  const [formError, setFormError] = useState<string>()
  useEffect(() => { void client.load() }, [client, online])
  useEffect(() => { setFields(draft(view.data)); setConfirmKill(false) }, [view.data])
  const ready = online && view.phase === 'ready'
  const busy = view.phase === 'loading' || view.phase === 'saving'
  const data = view.data
  return <section className="superagent-controls" aria-label="Superagent safety controls" aria-busy={busy}>
    <h3>Owner controls</h3>
    <p>Pause admissions, kill the current generation, or set daily budgets. These controls never approve a task’s permissions.</p>
    <button type="button" className="btn" disabled={!online || busy} onClick={() => { setFormError(undefined); void client.load() }}>Reload controls</button>
    {view.error === undefined ? null : <p role="alert">{view.error}</p>}
    {!online ? <p>Reconnect before changing controls.</p> : null}
    {data === undefined ? <p>{busy ? 'Loading owner controls…' : 'Load the current Hub state before making changes.'}</p> : <>
      <p>Last confirmed status: <strong>{data.killed ? 'Killed' : data.paused ? 'Paused' : 'Active'}</strong></p>
      <fieldset disabled={!ready}>
        <legend>Admission</legend>
        <button type="button" className="btn" disabled={data.paused || data.killed} onClick={() => { void client.save({ paused: true }) }}>Pause</button>
        <button type="button" className="btn" disabled={!data.paused && !data.killed} onClick={() => { void client.save({ paused: false, killed: false }) }}>Resume new work</button>
        <button type="button" className="btn" disabled={data.killed} onClick={() => setConfirmKill(true)}>Kill</button>
        {confirmKill ? <div role="group" aria-label="Confirm superagent kill">
          <p>Kill cancels this generation and blocks further dispatch. It cannot undo an operation already sent. Resuming will not revive killed work.</p>
          <button type="button" className="btn" onClick={() => { setConfirmKill(false); void client.save({ killed: true }) }}>Confirm kill</button>
          <button type="button" className="btn" onClick={() => setConfirmKill(false)}>Cancel</button>
        </div> : null}
      </fieldset>
      <form onSubmit={event => {
        event.preventDefault()
        if (!ready) return
        try { const limits = parseDailyLimits(fields); setFormError(undefined); void client.save({ limits }) }
        catch (error) { setFormError(error instanceof Error ? error.message : 'Invalid daily budget') }
      }}>
        <fieldset disabled={!ready}>
          <legend>Daily budgets</legend>
          <p>Blank means no cap; zero blocks that metric. Token and USD caps require trustworthy per-call quotes, otherwise admission is refused. Usage resets by UTC day; unknown usage is never zero.</p>
          {CONTROL_METRICS.map(key => <label key={key} style={{ display: 'block', marginBlock: '0.5rem' }}>
            <span>Daily {LABELS[key]} limit</span>{' '}
            <input aria-label={`Daily ${LABELS[key]} limit`} type="number" min="0" step={key === 'usd' ? 'any' : '1'} value={fields[key]}
              onChange={event => { const value = event.currentTarget.value; setFields(current => ({ ...current, [key]: value })) }} />
            <span> Used/reserved: {data.usage[key] === null ? 'Unknown' : String(data.usage[key])}</span>
          </label>)}
          {formError === undefined ? null : <p role="alert">{formError}</p>}
          <button type="submit" className="btn">Save daily budgets</button>
        </fieldset>
      </form>
    </>}
  </section>
}
