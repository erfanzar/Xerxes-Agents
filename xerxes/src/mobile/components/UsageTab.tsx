// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, type ReactElement } from 'react'

import { compactTokens, costLabel, modelUsageSummary, resetIn, usageSourceLabel, usd, windowLabel, windowNote, type UsageProfileView } from '../../auth/usageView.js'
import { useHubState, useServices } from '../context.js'
import { PushOptIn } from './PushOptIn.js'

function ProfileCard({ profile, now }: { profile: UsageProfileView; now: number }): ReactElement {
  const head = [profile.label, usageSourceLabel(profile.source, profile.provider), profile.plan].filter(Boolean).join(' · ')
  return (
    <li className="card card-plain">
      <h3 className="card-title">{head}</h3>
      {profile.balance === undefined ? null : <p>{profile.balance}</p>}
      {profile.status !== 'ok' ? <p className="block-meta">{profile.status === 'unsupported' ? 'No usage limits published.' : profile.message ?? 'Unavailable.'}</p> : null}
      {(profile.facts ?? []).map(fact => <p key={fact.label} className="block-meta">{fact.label}: {fact.value}</p>)}
      {profile.windows.map(window => {
        const reset = resetIn(window, profile.fetchedAt, now)
        const note = windowNote(window)
        const percent = Math.round(window.usedPercent)
        return (
          <div key={window.label + (window.detail ?? '')} className="usage-window">
            <p><span>{windowLabel(window)}</span> <strong>{percent}% used</strong></p>
            <progress max={100} value={percent} aria-label={`${windowLabel(window)} usage`} />
            <p className="block-meta">{[note, reset ? `resets in ${reset}` : undefined].filter(Boolean).join(' · ')}</p>
          </div>
        )
      })}
    </li>
  )
}

export function UsageTab(): ReactElement {
  const state = useHubState()
  const { controller, now } = useServices()
  const usage = state.usage

  useEffect(() => {
    if (state.connection.state === 'live') void controller.loadUsage()
  }, [controller, state.connection.state])

  const report = usage.phase === 'idle' ? undefined : usage.report
  const extras = usage.phase === 'idle' ? undefined : usage.extras
  const stamp = now()

  return (
    <section aria-labelledby="usage-title">
      <h1 id="usage-title" className="page-title">Usage</h1>
      {usage.phase === 'loading' && report === undefined ? <p aria-busy="true">Loading usage…</p> : null}
      {usage.phase === 'idle' && state.connection.state !== 'live' ? <p>Usage loads once the connection is live.</p> : null}
      {usage.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not load usage{report === undefined ? '' : ' (showing the last result)'}: {usage.message}</p>
          <button type="button" className="btn" onClick={() => { void controller.loadUsage() }}>Try again</button>
        </div>
      ) : null}
      {extras === undefined || extras.unavailable.length === 0 ? null : <p className="notice">Usage is unavailable for: {extras.unavailable.join(', ')}.</p>}

      {report !== undefined && report.profiles.length > 0 ? (
        <>
          <h2 className="section-title">Plans and keys</h2>
          <p className="block-meta">A plan shared by several workspaces is listed once.</p>
          <ul className="cards">{report.profiles.map(profile => <ProfileCard key={profile.profile} profile={profile} now={stamp} />)}</ul>
        </>
      ) : null}

      {report !== undefined && (report.models?.length ?? 0) > 0 ? (
        <>
          <h2 className="section-title">By model</h2>
          <ul className="cards">
            {(report.models ?? []).map(model => (
              <li key={`${model.profile ?? ''}/${model.model}`} className="card card-plain">
                <h3 className="card-title">{model.model}</h3>
                <p><strong>{costLabel(model.costUsd, model.costBasis)}</strong></p>
                <p className="block-meta">{modelUsageSummary(model)}</p>
              </li>
            ))}
          </ul>
          {extras === undefined ? null : (
            <p className="summary">
              Total {compactTokens(extras.inputTokens)} in · {compactTokens(extras.outputTokens)} out
              {extras.billableCostUsd !== undefined ? ` · ${usd(extras.billableCostUsd)} billed` : ''}
              {extras.subscriptionLabel !== undefined ? ` · ${extras.subscriptionLabel}` : ''}
            </p>
          )}
        </>
      ) : null}

      {usage.phase === 'ready' && (report?.profiles.length ?? 0) === 0 && (report?.models?.length ?? 0) === 0 ? <p className="empty">No usage recorded yet.</p> : null}
      <PushOptIn />
    </section>
  )
}
