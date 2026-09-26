// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The rail's Usage tab: this session's statistics (always shown, live while
 * a turn runs), then every imported provider profile with its plan windows —
 * 5-hour, weekly, per-model — its API balance, or the plain reason there is
 * nothing to show. Quota numbers come from the runtime's `usage.report`, the
 * same answer the TUI's /usage panel renders.
 */

import { useCallback, useEffect, useRef, useState, type ReactElement } from 'react'

import {
  parseUsageReport,
  resetIn,
  usageSourceLabel,
  windowLabel,
  windowNote,
  type UsageProfileView,
  type UsageReportView,
  type UsageWindowView,
} from '../../auth/usageView.js'
import { desktopCall } from './desktopRpc.js'
import { Icon } from './Icon.js'
import type { Snapshot } from './store.js'

/** Plan windows move slowly; the runtime caches each answer for a minute. */
const REFRESH_MS = 60_000

/** Only a nearly spent window gets colour; amber stays reserved for "you must act". */
function level(percent: number): 'ok' | 'full' {
  return percent >= 90 ? 'full' : 'ok'
}

function ago(at: number, now: number): string {
  const seconds = Math.max(0, Math.round((now - at) / 1_000))
  if (seconds < 10) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  return `${Math.round(seconds / 60)}m ago`
}

function Meter({ label, percent, note, reset }: { label: string; percent: number; note?: string | undefined; reset?: string | undefined }): ReactElement {
  const rounded = Math.round(percent)
  return <div className="usage-meter" data-level={level(percent)}>
    <div className="usage-meter__row"><span className="usage-meter__label">{label}</span><span className="usage-meter__pct">{rounded}%</span></div>
    <div className="usage-meter__bar" role="meter" aria-label={`${label} used`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={rounded}><i style={{ width: `${Math.max(percent > 0 ? 1.5 : 0, Math.min(100, percent))}%` }} /></div>
    {(note || reset) && <div className="usage-meter__meta">{note && <span>{note}</span>}{reset && <span>Resets in {reset}</span>}</div>}
  </div>
}

function windowMeter(window: UsageWindowView, profile: UsageProfileView, now: number): ReactElement {
  return <Meter key={window.label + (window.detail ?? '')} label={windowLabel(window)} percent={window.usedPercent} note={windowNote(window)} reset={resetIn(window, profile.fetchedAt, now)} />
}

export function ProfileCard({ profile, now }: { profile: UsageProfileView; now: number }): ReactElement {
  const account = [usageSourceLabel(profile.source, profile.provider), profile.plan].filter(Boolean).join(' · ')
  return <li className="usage-plan" data-status={profile.status} data-active={profile.active || undefined}>
    <div className="usage-plan__head">
      <span className="usage-plan__dot" aria-hidden="true" />
      <strong title={profile.label === profile.profile ? profile.model : `${profile.profile} · ${profile.model}`}>{profile.label}</strong>
      <span className="usage-plan__account">{account}</span>
      {profile.active && <span className="usage-plan__badge">In use</span>}
      {profile.balance && <span className="usage-plan__balance">{profile.balance}</span>}
    </div>
    {profile.windows.length > 0 && <div className="usage-plan__windows">{profile.windows.map(window => windowMeter(window, profile, now))}</div>}
    {profile.status === 'error' && <p className="usage-plan__message" role="status">{profile.message || 'Usage is unavailable right now.'}</p>}
    {profile.status === 'unsupported' && <p className="usage-plan__message usage-plan__message--quiet">This provider doesn't publish usage limits or a balance.</p>}
    {profile.status === 'ok' && !profile.windows.length && !profile.balance && <p className="usage-plan__message usage-plan__message--quiet">No limits reported for this plan.</p>}
  </li>
}

export function UsagePanel({ snap }: { snap: Snapshot }): ReactElement {
  const [report, setReport] = useState<UsageReportView | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const alive = useRef(true)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  const online = snap.connection === 'online'

  const load = useCallback(async (refresh: boolean) => {
    setLoading(true)
    try {
      const result = await desktopCall(window.xerxes, snap.sessionKey, 'usage.report', refresh ? { refresh: true } : {})
      if (!alive.current) return
      setReport(parseUsageReport(result)); setError('')
    } catch (failure) {
      if (!alive.current) return
      const message = failure instanceof Error ? failure.message : String(failure)
      setError(/unknown method/i.test(message) ? 'Plan limits need the latest runtime. Update the runtime to see them.' : message)
    } finally { if (alive.current) setLoading(false) }
  }, [snap.sessionKey])

  useEffect(() => {
    if (!online) return
    void load(false)
    const timer = setInterval(() => { if (document.visibilityState === 'visible') void load(false) }, REFRESH_MS)
    return () => clearInterval(timer)
  }, [load, online])
  // Countdowns and "updated 20s ago" stay true between fetches.
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 15_000); return () => clearInterval(timer) }, [])

  const profiles = report?.profiles ?? []

  // Account-level only: plans, keys and limits. The current task's numbers
  // are in the Activity card's session statistics.
  return <div className="usage">
    <section className="usage__section" aria-labelledby="usage-plans" aria-busy={loading || undefined}>
      <header className="usage__head">
        <h3 id="usage-plans">Plans &amp; keys</h3>
        {report && <span className="usage__sub">Updated {ago(report.fetchedAt, now)}</span>}
        <button className="usage__refresh" disabled={loading || !online} onClick={() => void load(true)} aria-label="Refresh plan usage" title="Refresh plan usage"><Icon name="retry" size={13} /></button>
      </header>
      {error && <p className="usage__error" role="alert">{error}</p>}
      {!report && !error && <p className="usage__empty" role="status">{online ? 'Checking your plans…' : 'Plan usage appears when the runtime is connected.'}</p>}
      {report && !profiles.length && <p className="usage__empty">No provider profiles yet. Add one from the model picker to see its limits here.</p>}
      {profiles.length > 0 && <ul className="usage-plans">{profiles.map(profile => <ProfileCard key={profile.profile} profile={profile} now={now} />)}</ul>}
    </section>
  </div>
}
