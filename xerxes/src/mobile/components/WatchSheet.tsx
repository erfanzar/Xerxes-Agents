// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import {
  AUTOPILOT_EXPIRY_LAPSED_NOTE, autopilotExpiresAt, autopilotExpiryOptions, defaultAutopilotExpiry, effectiveExpiryChoice, isExpiryLapsed,
  type AutopilotExpiryChoice,
} from '../../client/autopilot.js'
import type { WatchInput } from '../../client/hubController.js'
import { inboxTasks, watchKey } from '../../client/hubStore.js'
import type { WatchRow } from '../../client/hubTypes.js'
import {
  WATCH_EXPLANATION, WATCH_NO_TASKS_NOTE, WATCH_READ_ONLY_NOTE, WATCH_STRIP_NOTE, activityAgo, anomalyLabel, reviewMinutesOf, reviewText, reviewValueOf,
  watchCardRows, watchFacts, watchReviewOptions, watchStateText,
} from '../../client/watch.js'
import { useHubState, useNow, useServices } from '../context.js'
import { Sheet } from './Sheet.js'

/**
 * Watch from the phone. A watcher on the Hub reads every step of a watched run
 * at no model cost; XERXES wakes only on a problem or at the review interval.
 * Cards are compact; the owner starts, tunes and stops a watch in the sheet.
 * Everything that came from a run (title, test names, details, summary line)
 * is untrusted and only ever rendered as text.
 */

export function WatchCard({ row, now }: { row: WatchRow; now: number }): ReactElement {
  const stats = row.stats
  const anomalies = stats?.anomalies ?? []
  return (
    <li className={`card${anomalies.length > 0 ? ' card-attention' : ''}`} aria-label={`Watching ${row.title}`} data-task={row.taskId}>
      <h3 className="card-title">{row.title}</h3>
      <p className="block-meta">{row.workspaceLabel} · {watchStateText(row)}</p>
      {stats === null ? <p className="block-meta">Waiting for the first step of the run…</p> : (
        <>
          <p className="block-meta">
            {stats.status} · last activity{' '}
            {stats.lastActivityAt === null ? activityAgo(null, now) : <time dateTime={stats.lastActivityAt}>{activityAgo(stats.lastActivityAt, now)}</time>}
          </p>
          {stats.summaryLine === '' ? null : <p className="block-text">{stats.summaryLine}</p>}
          <ul className="chips" aria-label="Counters">
            {watchFacts(stats).map(fact => <li key={fact.key} className={`chip${fact.tone === 'bad' ? ' chip-bad' : ''}`}>{fact.text}</li>)}
          </ul>
          {stats.tests.last === 'fail' && stats.tests.lastFailing.length > 0 ? <p className="field-error">Failing: {stats.tests.lastFailing.join(', ')}</p> : null}
          {anomalies.length === 0 ? null : (
            <ul className="chips" aria-label="Problems found">
              {anomalies.map(anomaly => (
                <li key={anomaly.kind} className="chip chip-anomaly" data-kind={anomaly.kind}>
                  <strong>{anomalyLabel(anomaly.kind)}</strong>{anomaly.detail === '' ? '' : ` ${anomaly.detail}`}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
      <p className="block-meta">
        {reviewText(row.reviewIntervalMinutes)}
        {row.expiresAt === null ? '' : <> · until <time dateTime={row.expiresAt}>{new Date(row.expiresAt).toLocaleString()}</time></>}
      </p>
    </li>
  )
}

/** Live cards on the Xerxes tab; nothing while no watch is on. */
export function WatchCards(): ReactElement | null {
  const { watch } = useHubState()
  const now = useNow(1_000)
  const cards = watch.unsupported ? [] : watchCardRows(watch.rows)
  if (cards.length === 0) return null
  return (
    <section aria-label="Watching">
      <h2 className="section-title">Watching</h2>
      <p className="block-meta">{WATCH_STRIP_NOTE}</p>
      <ul className="cards">{cards.map(row => <WatchCard key={watchKey(row.workspaceId, row.taskId)} row={row} now={now} />)}</ul>
    </section>
  )
}

function ReadOnlyRow({ row }: { row: WatchRow }): ReactElement {
  return (
    <li className="card" aria-label={`Watch for ${row.title}`}>
      <h3 className="card-title">{row.title} · {watchStateText(row)}</h3>
      <p className="block-meta">{row.workspaceLabel} · {reviewText(row.reviewIntervalMinutes)}</p>
      {row.enabledBy === null ? null : <p className="block-meta">Turned on by {row.enabledBy.name}.</p>}
      {row.expiresAt === null ? null : <p className="block-meta">Until <time dateTime={row.expiresAt}>{new Date(row.expiresAt).toLocaleString()}</time>.</p>}
    </li>
  )
}

function Selects({ review, onReview, expiry, onExpiry, current }: {
  review: string
  onReview: (value: string) => void
  expiry: AutopilotExpiryChoice
  onExpiry: (value: AutopilotExpiryChoice) => void
  current: Pick<WatchRow, 'expiresAt' | 'reviewIntervalMinutes'>
}): ReactElement {
  const nowMs = Date.now()
  return (
    <>
      <label className="field">
        <span>XERXES reviews the run</span>
        <select value={review} onChange={event => onReview(event.target.value)}>
          {watchReviewOptions(current.reviewIntervalMinutes).map(option => <option key={option.value} value={option.value}>{option.label === 'Off' ? 'Off' : `Every ${option.label}`}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Stop watching automatically</span>
        <select value={effectiveExpiryChoice(expiry, current, nowMs)} onChange={event => onExpiry(event.target.value as AutopilotExpiryChoice)}>
          {autopilotExpiryOptions(current, nowMs).map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
      </label>
    </>
  )
}

function EditableRow({ row }: { row: WatchRow }): ReactElement {
  const state = useHubState()
  const { controller } = useServices()
  const [review, setReview] = useState(() => reviewValueOf(row.reviewIntervalMinutes))
  const [expiry, setExpiry] = useState<AutopilotExpiryChoice>(() => defaultAutopilotExpiry(row, Date.now()))
  const live = state.connection.state === 'live'
  const saving = state.watch.savingKey === watchKey(row.workspaceId, row.taskId)
  const nowMs = Date.now()
  const choice = effectiveExpiryChoice(expiry, row, nowMs)
  const lapsed = isExpiryLapsed(row.expiresAt, nowMs)
  const save = async (): Promise<void> => {
    if (!live || saving) return
    const minutes = reviewMinutesOf(review)
    const expiresAt = autopilotExpiresAt(choice, Date.now())
    const ok = await controller.setWatch({
      workspaceId: row.workspaceId,
      taskId: row.taskId,
      enabled: true,
      ...(minutes === row.reviewIntervalMinutes ? {} : { reviewIntervalMinutes: minutes }),
      ...(expiresAt === undefined ? {} : { expiresAt }),
    })
    if (ok) setExpiry(defaultAutopilotExpiry({ expiresAt: expiresAt === undefined ? row.expiresAt : expiresAt }, Date.now()))
  }
  return (
    <li className="card" aria-label={`Watch for ${row.title}`}>
      <form className="model-form" onSubmit={event => { event.preventDefault(); void save() }}>
        <h3 className="card-title">{row.title} · {watchStateText(row)}</h3>
        <p className="block-meta">{row.workspaceLabel}</p>
        <Selects review={review} onReview={setReview} expiry={expiry} onExpiry={setExpiry} current={row} />
        {row.expiresAt === null ? null : lapsed
          ? <p role="status" className="block-meta">{AUTOPILOT_EXPIRY_LAPSED_NOTE}</p>
          : <p className="block-meta">Currently until <time dateTime={row.expiresAt}>{new Date(row.expiresAt).toLocaleString()}</time>.</p>}
        <div className="composer-row">
          <button type="submit" className="btn btn-primary" disabled={!live || saving}>{saving ? 'Saving…' : 'Save'}</button>
          <button type="button" className="btn" disabled={!live || saving} onClick={() => { const input: WatchInput = { workspaceId: row.workspaceId, taskId: row.taskId, enabled: false }; void controller.setWatch(input) }}>Stop watching</button>
        </div>
      </form>
    </li>
  )
}

function StartWatch({ watched }: { watched: ReadonlySet<string> }): ReactElement {
  const state = useHubState()
  const { controller } = useServices()
  const [picked, setPicked] = useState('')
  const [review, setReview] = useState('off')
  const [expiry, setExpiry] = useState<AutopilotExpiryChoice>('none')
  const candidates = inboxTasks(state).filter(task => !watched.has(watchKey(task.workspaceId, task.id))).slice(0, 100)
  const task = candidates.find(candidate => watchKey(candidate.workspaceId, candidate.id) === picked)
  const live = state.connection.state === 'live'
  const saving = state.watch.savingKey !== undefined && !watched.has(state.watch.savingKey)
  if (candidates.length === 0) return <p role="status" className="block-meta">{WATCH_NO_TASKS_NOTE}</p>
  const start = async (): Promise<void> => {
    if (task === undefined || !live || saving) return
    const expiresAt = autopilotExpiresAt(expiry, Date.now())
    const ok = await controller.setWatch({
      workspaceId: task.workspaceId,
      taskId: task.id,
      enabled: true,
      reviewIntervalMinutes: reviewMinutesOf(review),
      ...(expiresAt === undefined ? {} : { expiresAt }),
    })
    if (ok) setPicked('')
  }
  return (
    <form className="model-form card" aria-label="Watch another task" onSubmit={event => { event.preventDefault(); void start() }}>
      <label className="field">
        <span>Task to watch</span>
        <select value={picked} onChange={event => setPicked(event.target.value)}>
          <option value="">Choose a task…</option>
          {candidates.map(candidate => (
            <option key={watchKey(candidate.workspaceId, candidate.id)} value={watchKey(candidate.workspaceId, candidate.id)}>
              {candidate.title} — {state.workspaces[candidate.workspaceId]?.label ?? candidate.workspaceId}
            </option>
          ))}
        </select>
      </label>
      <Selects review={review} onReview={setReview} expiry={expiry} onExpiry={setExpiry} current={{ expiresAt: null, reviewIntervalMinutes: null }} />
      <div className="composer-row">
        <button type="submit" className="btn btn-primary" disabled={!live || saving || task === undefined}>{saving ? 'Saving…' : 'Start watching'}</button>
      </div>
    </form>
  )
}

/** The watch rows plus the start form: owners edit, everyone else reads. */
export function WatchBody(): ReactElement {
  const { watch } = useHubState()
  const { controller } = useServices()
  const enabled = watch.rows.filter(row => row.enabled)
  const watched = new Set(enabled.map(row => watchKey(row.workspaceId, row.taskId)))
  return (
    <>
      <p className="summary">{WATCH_EXPLANATION}</p>
      {watch.phase === 'loading' && watch.rows.length === 0 ? <p role="status" aria-busy="true">Reading Watch settings…</p> : null}
      {watch.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not read Watch: {watch.error ?? 'unknown error'}</p>
          <button type="button" className="btn" onClick={() => { void controller.loadWatches() }}>Try again</button>
        </div>
      ) : null}
      {watch.phase === 'ready' || watch.rows.length > 0 ? (
        <>
          {watch.canEdit ? null : <p role="status" className="block-meta">{WATCH_READ_ONLY_NOTE}</p>}
          {enabled.length === 0 ? <p role="status" className="notice">No task is being watched.</p> : (
            <ul className="cards" aria-label="Watched tasks">
              {enabled.map(row => watch.canEdit ? <EditableRow key={watchKey(row.workspaceId, row.taskId)} row={row} /> : <ReadOnlyRow key={watchKey(row.workspaceId, row.taskId)} row={row} />)}
            </ul>
          )}
          {watch.canEdit ? <StartWatch watched={watched} /> : null}
        </>
      ) : null}
      {watch.saveError === undefined ? null : <p role="alert" className="field-error">{watch.saveError}</p>}
    </>
  )
}

export function WatchSheet({ onClose }: { onClose: () => void }): ReactElement {
  const { controller } = useServices()
  useEffect(() => { void controller.loadWatches() }, [controller])
  return (
    <Sheet title="Watch" onClose={onClose} tall>
      <WatchBody />
    </Sheet>
  )
}
