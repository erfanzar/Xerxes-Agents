// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import {
  AUTOPILOT_EXPIRY_LAPSED_NOTE, autopilotExpiresAt, autopilotExpiryOptions, defaultAutopilotExpiry, effectiveExpiryChoice, isExpiryLapsed,
  type AutopilotExpiryChoice,
} from '../../client/autopilot.js'
import type { WatchInput } from '../../client/hubController.js'
import { watchKey, type WatchState } from '../../client/hubStore.js'
import type { TaskSummary, WatchRow, WorkspaceView } from '../../client/hubTypes.js'
import {
  WATCH_EXPLANATION, WATCH_NO_TASKS_NOTE, WATCH_READ_ONLY_NOTE, WATCH_STRIP_NOTE, activityAgo, anomalyLabel, reviewMinutesOf, reviewText, reviewValueOf,
  watchCardRows, watchFacts, watchReviewOptions, watchStateText,
} from '../../client/watch.js'

/**
 * Watch, on the XERXES tab. The strip shows one live card per watched task;
 * the panel lets the Hub owner start, tune and stop a watch. Everything that
 * came from a run (title, test names, anomaly details, the summary line) is
 * untrusted: it is only ever rendered as text. `active` is the Hub's verdict.
 */

const slug = (text: string): string => text.replace(/[^\w-]/g, '_')

// ── cards ─────────────────────────────────────────────────────────────

export function WatchCard({ row, now }: { readonly row: WatchRow; readonly now: number }): ReactElement {
  const stats = row.stats
  const anomalies = stats?.anomalies ?? []
  return (
    <li className={`watchcard${anomalies.length > 0 ? ' has-anomaly' : ''}`} aria-label={`Watching ${row.title}`} data-task={row.taskId}>
      <div className="watchcard__head">
        <strong className="watchcard__title">{row.title}</strong>
        <span className="watchcard__ws">{row.workspaceLabel}</span>
        <span className={`autopilot__state${row.active ? ' is-on' : ''}`}>{watchStateText(row)}</span>
      </div>
      {stats === null ? (
        <p className="studio-muted">Waiting for the first step of the run…</p>
      ) : (
        <>
          <p className="watchcard__status">
            <span className="watchcard__run">{stats.status}</span> · last activity{' '}
            {stats.lastActivityAt === null ? <span>{activityAgo(null, now)}</span> : <time dateTime={stats.lastActivityAt}>{activityAgo(stats.lastActivityAt, now)}</time>}
          </p>
          {stats.summaryLine === '' ? null : <p className="watchcard__summary">{stats.summaryLine}</p>}
          <ul className="watchcard__facts" aria-label="Counters">
            {watchFacts(stats).map(fact => <li key={fact.key} className={fact.tone === 'bad' ? 'is-bad' : undefined}>{fact.text}</li>)}
          </ul>
          {stats.tests.last === 'fail' && stats.tests.lastFailing.length > 0 ? (
            <p className="watchcard__failing">Failing: {stats.tests.lastFailing.join(', ')}</p>
          ) : null}
          {anomalies.length === 0 ? null : (
            <ul className="watchcard__anomalies" aria-label="Problems found">
              {anomalies.map(anomaly => (
                <li key={anomaly.kind} className="watchcard__chip is-anomaly" data-kind={anomaly.kind}>
                  <strong>{anomalyLabel(anomaly.kind)}</strong>
                  {anomaly.detail === '' ? null : <span> {anomaly.detail}</span>}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
      <p className="studio-muted watchcard__review">
        {reviewText(row.reviewIntervalMinutes)}
        {row.expiresAt === null ? '' : <> · until <time dateTime={row.expiresAt}>{new Date(row.expiresAt).toLocaleString()}</time></>}
      </p>
    </li>
  )
}

/** The live cards, or nothing while no watch is on. "Last activity" ticks every second. */
export function WatchStrip({ rows, now }: { readonly rows: readonly WatchRow[]; readonly now: number }): ReactElement | null {
  const cards = watchCardRows(rows)
  const [tick, setTick] = useState(now)
  const any = cards.length > 0
  useEffect(() => {
    if (!any) return
    const timer = setInterval(() => setTick(Date.now()), 1_000)
    return () => clearInterval(timer)
  }, [any])
  if (!any) return null
  return (
    <section className="watchstrip" aria-label="Watching">
      <div className="hubphone__row">
        <strong>Watching</strong>
        <span className="studio-muted">{WATCH_STRIP_NOTE}</span>
      </div>
      <ul className="watchstrip__cards">{cards.map(row => <WatchCard key={watchKey(row.workspaceId, row.taskId)} row={row} now={tick} />)}</ul>
    </section>
  )
}

// ── controls ──────────────────────────────────────────────────────────

function ReviewSelect({ id, value, current, onChange }: { readonly id: string; readonly value: string; readonly current: number | null; readonly onChange: (value: string) => void }): ReactElement {
  return (
    <div className="field">
      <label htmlFor={id}>XERXES reviews the run</label>
      <select id={id} value={value} onChange={event => onChange(event.target.value)}>
        {watchReviewOptions(current).map(option => <option key={option.value} value={option.value}>{option.label === 'Off' ? 'Off' : `Every ${option.label}`}</option>)}
      </select>
    </div>
  )
}

function ExpirySelect({ id, value, row, nowMs, onChange }: {
  readonly id: string
  readonly value: AutopilotExpiryChoice
  readonly row: Pick<WatchRow, 'expiresAt'>
  readonly nowMs: number
  readonly onChange: (value: AutopilotExpiryChoice) => void
}): ReactElement {
  return (
    <div className="field">
      <label htmlFor={id}>Stop watching automatically</label>
      <select id={id} value={value} onChange={event => onChange(event.target.value as AutopilotExpiryChoice)}>
        {autopilotExpiryOptions(row, nowMs).map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </div>
  )
}

function ReadOnlyWatch({ row }: { readonly row: WatchRow }): ReactElement {
  return (
    <li className="autopilot__row" aria-label={`Watch for ${row.title}`}>
      <div className="hubphone__row">
        <strong>{row.title}</strong>
        <span className={`autopilot__state${row.active ? ' is-on' : ''}`}>{watchStateText(row)}</span>
      </div>
      <p className="studio-muted">{row.workspaceLabel} · {reviewText(row.reviewIntervalMinutes)}</p>
      {row.enabledBy === null ? null : <p className="studio-muted">Turned on by {row.enabledBy.name}.</p>}
      {row.expiresAt === null ? null : <p className="studio-muted">Until <time dateTime={row.expiresAt}>{new Date(row.expiresAt).toLocaleString()}</time>.</p>}
    </li>
  )
}

function EditableWatch({ row, live, saving, onSave }: {
  readonly row: WatchRow
  readonly live: boolean
  readonly saving: boolean
  readonly onSave: (input: WatchInput) => Promise<boolean>
}): ReactElement {
  const [review, setReview] = useState(() => reviewValueOf(row.reviewIntervalMinutes))
  const [expiry, setExpiry] = useState<AutopilotExpiryChoice>(() => defaultAutopilotExpiry(row, Date.now()))
  const id = `watch-${slug(row.workspaceId)}-${slug(row.taskId)}`
  const nowMs = Date.now()
  const choice = effectiveExpiryChoice(expiry, row, nowMs)
  const lapsed = isExpiryLapsed(row.expiresAt, nowMs)
  const save = async (): Promise<void> => {
    const minutes = reviewMinutesOf(review)
    const expiresAt = autopilotExpiresAt(choice, Date.now())
    const ok = await onSave({
      workspaceId: row.workspaceId,
      taskId: row.taskId,
      enabled: true,
      // Unchanged fields are left out so the Hub keeps what it holds.
      ...(minutes === row.reviewIntervalMinutes ? {} : { reviewIntervalMinutes: minutes }),
      ...(expiresAt === undefined ? {} : { expiresAt }),
    })
    if (ok) setExpiry(defaultAutopilotExpiry({ expiresAt: expiresAt === undefined ? row.expiresAt : expiresAt }, Date.now()))
  }
  return (
    <li className="autopilot__row" aria-label={`Watch for ${row.title}`}>
      <form onSubmit={event => { event.preventDefault(); void save() }}>
        <div className="hubphone__row">
          <strong>{row.title}</strong>
          <span className={`autopilot__state${row.active ? ' is-on' : ''}`}>{watchStateText(row)}</span>
        </div>
        <p className="studio-muted">{row.workspaceLabel}</p>
        <ReviewSelect id={`${id}-review`} value={review} current={row.reviewIntervalMinutes} onChange={setReview} />
        <ExpirySelect id={`${id}-expiry`} value={choice} row={row} nowMs={nowMs} onChange={setExpiry} />
        {row.expiresAt === null ? null : lapsed
          ? <p role="status" className="studio-muted">{AUTOPILOT_EXPIRY_LAPSED_NOTE}</p>
          : <p className="studio-muted">Currently until <time dateTime={row.expiresAt}>{new Date(row.expiresAt).toLocaleString()}</time>.</p>}
        <div className="hubphone__row">
          <button type="submit" className="btn btn--solid" disabled={!live || saving}>{saving ? 'Saving…' : 'Save'}</button>
          <button type="button" className="btn" disabled={!live || saving} onClick={() => { void onSave({ workspaceId: row.workspaceId, taskId: row.taskId, enabled: false }) }}>Stop watching</button>
        </div>
      </form>
    </li>
  )
}

/** Pick one of the tasks this app knows and start watching it. */
function StartWatch({ candidates, workspaces, live, saving, onSave }: {
  readonly candidates: readonly TaskSummary[]
  readonly workspaces: Readonly<Record<string, WorkspaceView>>
  readonly live: boolean
  readonly saving: boolean
  readonly onSave: (input: WatchInput) => Promise<boolean>
}): ReactElement {
  const [picked, setPicked] = useState('')
  const [review, setReview] = useState('off')
  const [expiry, setExpiry] = useState<AutopilotExpiryChoice>('none')
  const task = candidates.find(candidate => watchKey(candidate.workspaceId, candidate.id) === picked)
  const start = async (): Promise<void> => {
    if (task === undefined) return
    const expiresAt = autopilotExpiresAt(expiry, Date.now())
    const ok = await onSave({
      workspaceId: task.workspaceId,
      taskId: task.id,
      enabled: true,
      reviewIntervalMinutes: reviewMinutesOf(review),
      ...(expiresAt === undefined ? {} : { expiresAt }),
    })
    if (ok) setPicked('')
  }
  if (candidates.length === 0) return <p role="status" className="studio-muted">{WATCH_NO_TASKS_NOTE}</p>
  return (
    <form className="autopilot__row" aria-label="Watch another task" onSubmit={event => { event.preventDefault(); void start() }}>
      <div className="field">
        <label htmlFor="watch-pick">Task to watch</label>
        <select id="watch-pick" value={picked} onChange={event => setPicked(event.target.value)}>
          <option value="">Choose a task…</option>
          {candidates.map(candidate => (
            <option key={watchKey(candidate.workspaceId, candidate.id)} value={watchKey(candidate.workspaceId, candidate.id)}>
              {candidate.title} — {workspaces[candidate.workspaceId]?.label ?? candidate.workspaceId}
            </option>
          ))}
        </select>
      </div>
      <ReviewSelect id="watch-pick-review" value={review} current={null} onChange={setReview} />
      <ExpirySelect id="watch-pick-expiry" value={expiry} row={{ expiresAt: null }} nowMs={Date.now()} onChange={setExpiry} />
      <div className="hubphone__row">
        <button type="submit" className="btn btn--solid" disabled={!live || saving || task === undefined}>{saving ? 'Saving…' : 'Start watching'}</button>
      </div>
    </form>
  )
}

export interface WatchPanelProps {
  readonly watch: WatchState
  /** Tasks this app knows, in the order they should be offered. */
  readonly tasks: readonly TaskSummary[]
  readonly workspaces: Readonly<Record<string, WorkspaceView>>
  /** False while the Hub link is down: saving is disabled. */
  readonly live: boolean
  readonly onSave: (input: WatchInput) => Promise<boolean>
  readonly onRetry: () => void
  readonly onClose: () => void
}

export function WatchPanel({ watch, tasks, workspaces, live, onSave, onRetry, onClose }: WatchPanelProps): ReactElement {
  const enabled = watch.rows.filter(row => row.enabled)
  const watched = new Set(enabled.map(row => watchKey(row.workspaceId, row.taskId)))
  const candidates = tasks.filter(task => !watched.has(watchKey(task.workspaceId, task.id))).slice(0, 100)
  const savingNew = watch.savingKey !== undefined && !watched.has(watch.savingKey)
  return (
    <section className="autopilot watchpanel" aria-label="Watch">
      <div className="hubphone__row">
        <strong>Watch</strong>
        <button type="button" className="btn" onClick={onClose}>Close</button>
      </div>
      <p className="studio-muted">{WATCH_EXPLANATION}</p>
      {watch.phase === 'loading' && watch.rows.length === 0 ? <p role="status" aria-busy="true" className="studio-muted">Reading Watch settings…</p> : null}
      {watch.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not read Watch: {watch.error ?? 'unknown error'}</p>
          <button type="button" className="btn" onClick={onRetry}>Try again</button>
        </div>
      ) : null}
      {watch.phase === 'ready' || watch.rows.length > 0 ? (
        <>
          {watch.canEdit ? null : <p role="status" className="studio-muted">{WATCH_READ_ONLY_NOTE}</p>}
          {enabled.length === 0 ? <p role="status" className="notice">No task is being watched.</p> : (
            <ul className="autopilot__rows" aria-label="Watched tasks">
              {enabled.map(row => watch.canEdit
                ? <EditableWatch key={watchKey(row.workspaceId, row.taskId)} row={row} live={live} saving={watch.savingKey === watchKey(row.workspaceId, row.taskId)} onSave={onSave} />
                : <ReadOnlyWatch key={watchKey(row.workspaceId, row.taskId)} row={row} />)}
            </ul>
          )}
          {watch.canEdit ? <StartWatch candidates={candidates} workspaces={workspaces} live={live} saving={savingNew} onSave={onSave} /> : null}
        </>
      ) : null}
      {watch.saveError === undefined ? null : <p role="alert" className="action-card__error">{watch.saveError}</p>}
    </section>
  )
}
